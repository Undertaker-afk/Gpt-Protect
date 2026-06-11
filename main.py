"""
main.py — HuggingFace Space app (Gradio UI + realtime training)
==============================================================
Runs on free Space hardware (2 vCPU / 16 GB). It:

  * loads / builds the AI-text detector and resumes from /data if a checkpoint
    exists (so a Space restart instantly continues where it left off);
  * serves a Gradio UI where users paste text, get a HUMAN/AI verdict + the
    stylometric breakdown, and can optionally label the sample;
  * every submission is appended to /data and folded into a background
    realtime-training loop;
  * a live dashboard shows step, loss, accuracy, samples seen and a loss curve;
  * model + optimizer + training state + collected data are continuously
    persisted to the /data bucket so nothing is lost on restart.

Entry point launched by app.py (or directly: `python main.py`).
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from collections import deque

import torch

from config import preset, TrainConfig
from model import build_model
from muon import build_optimizer
from preprocessor import Preprocessor, stylometric_features
from augmentation import Augmenter
from data_loader import get_tokenizer


# --------------------------------------------------------------------------- #
#  Persistent storage (the mounted bucket lives at /data on HF Spaces)
# --------------------------------------------------------------------------- #
def _pick_data_dir() -> str:
    cand = os.environ.get("DATA_DIR", "/data")
    try:
        os.makedirs(cand, exist_ok=True)
        t = os.path.join(cand, ".write_test")
        with open(t, "w") as f:
            f.write("ok")
        os.remove(t)
        return cand
    except Exception:
        local = os.path.join(os.getcwd(), "data")
        os.makedirs(local, exist_ok=True)
        print(f"[storage] /data not writable -> using {local}")
        return local


DATA_DIR = _pick_data_dir()
CKPT_DIR = os.path.join(DATA_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(CKPT_DIR, "last.pt")
STATE_PATH = os.path.join(DATA_DIR, "state.json")
COLLECTED_PATH = os.path.join(DATA_DIR, "collected.jsonl")
LOG_PATH = os.path.join(DATA_DIR, "train_log.jsonl")
BASE_CACHE_PATH = os.path.join(DATA_DIR, "base_cache.jsonl")

PRESET = os.environ.get("MODEL_PRESET", "tiny")        # free CPU -> tiny
MAX_SEQ = int(os.environ.get("MAX_SEQ_LEN", "192"))
BATCH = int(os.environ.get("BATCH_SIZE", "8"))
BASE_SAMPLES = int(os.environ.get("BASE_SAMPLES", "4000"))
# incremental harvester: per-source in-memory target, synchronous seed per
# source, round-robin chunk size, and a pacing delay to dodge HF rate limits.
DATASET_TARGET = int(os.environ.get("DATASET_TARGET", "1500"))
SEED_PER_SOURCE = int(os.environ.get("SEED_PER_SOURCE", "25"))
HARVEST_CHUNK = int(os.environ.get("HARVEST_CHUNK", "40"))
HARVEST_PACE_SEC = float(os.environ.get("HARVEST_PACE_SEC", "0.4"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "25"))   # steps between ckpts
TRAIN_ENABLED = os.environ.get("TRAIN", "1") == "1"
# how long the SIGTERM handler waits for the training thread to finish a step
UPDATE_GRACE_MARGIN = int(os.environ.get("UPDATE_GRACE_MARGIN", "30"))
UPDATER_STATUS_PATH = os.path.join(DATA_DIR, "updater.json")
# touch-file the dashboard writes to ask the app.py supervisor to check now
FORCE_UPDATE_FLAG = os.path.join(DATA_DIR, "force_update")

LABELS = {0: "HUMAN", 1: "AI-GENERATED"}


# --------------------------------------------------------------------------- #
#  Trainer — owns the model and the background realtime training loop
# --------------------------------------------------------------------------- #
class RealtimeTrainer:
    def __init__(self):
        # Small models / short seqs are latency-bound: too many BLAS threads
        # cause thrashing. Cap low (free Spaces only have ~2 vCPUs anyway).
        n_threads = int(os.environ.get("TORCH_THREADS",
                                       str(min(4, os.cpu_count() or 2))))
        torch.set_num_threads(max(1, n_threads))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = threading.Lock()
        self.pre = Preprocessor(max_chars=8000)
        self.aug = Augmenter(prob=0.3)

        self.tok = get_tokenizer(os.environ.get("TOKENIZER", "gpt2"))
        self.mcfg = preset(PRESET)
        self.mcfg.max_seq_len = max(self.mcfg.max_seq_len, MAX_SEQ)
        vocab = len(self.tok) if hasattr(self.tok, "__len__") else \
            getattr(self.tok, "vocab_size", 50257)
        self.mcfg.vocab_size = vocab
        self.pad_id = getattr(self.tok, "pad_token_id", 0) or 0

        self.model = build_model(self.mcfg).to(self.device)
        self.model.label_smoothing = 0.05
        self.tcfg = TrainConfig(optimizer=os.environ.get("OPTIMIZER", "muon"))
        self.opt = build_optimizer(self.model, self.tcfg)

        # shared, UI-visible state
        self.state = {
            "global_step": 0, "samples_seen": 0, "user_samples": 0,
            "loss_ema": None, "acc_ema": None, "best_acc": 0.0,
            "steps_per_s": None,
            "cm": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
            "started_at": None, "last_save": 0, "status": "init",
            "preset": PRESET, "params_M": round(self.model.num_parameters() / 1e6, 2),
            "device": self.device, "data_dir": DATA_DIR,
        }
        self._last_step_t = None
        self.loss_hist = deque(maxlen=300)     # (step, loss)
        self.acc_hist = deque(maxlen=300)      # (step, acc_ema)
        self.log_lines = deque(maxlen=40)

        self._load_state()
        self._restore_ckpt()

        # label-split pools (0=human, 1=ai) keep training balanced even though
        # the public corpus is human-heavy (basic_text is all human).
        self.base = {0: [], 1: []}
        self.user = {0: [], 1: []}
        self.user_data = []                      # canonical list (count/persist)
        self.source_counts = {}                  # per-dataset in-memory tally
        self._seen_base = set()                  # de-dup hashes for base cache
        self._harvest_iters = {}                 # persistent per-source streams
        self._load_base_cache()                  # instant: grows across restarts
        self._seed_base()                        # quick balanced seed if needed
        self._load_collected()
        self.state["user_samples"] = len(self.user_data)

        self._stop = threading.Event()
        self.thread = None
        self.harvest_thread = None

    # ----- persistence ------------------------------------------------- #
    def _log(self, msg):
        line = f"{time.strftime('%H:%M:%S')}  {msg}"
        self.log_lines.append(line)
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps({"t": time.time(), "msg": msg}) + "\n")
        except Exception:
            pass

    def _load_state(self):
        if os.path.exists(STATE_PATH):
            try:
                with open(STATE_PATH) as f:
                    saved = json.load(f)
                for k in ("global_step", "samples_seen", "user_samples",
                          "loss_ema", "acc_ema", "best_acc"):
                    if k in saved:
                        self.state[k] = saved[k]
                self._log(f"resumed state: step={self.state['global_step']}")
            except Exception as e:
                self._log(f"state load failed: {e}")

    def _save_state(self):
        try:
            with open(STATE_PATH, "w") as f:
                json.dump(self.state, f)
        except Exception as e:
            self._log(f"state save failed: {e}")

    def _restore_ckpt(self):
        if os.path.exists(CKPT_PATH):
            try:
                ck = torch.load(CKPT_PATH, map_location=self.device)
                self.model.load_state_dict(ck["model"], strict=False)
                self._log(f"restored checkpoint ({CKPT_PATH})")
            except Exception as e:
                self._log(f"ckpt restore failed: {e}")

    def _save_ckpt(self):
        from dataclasses import asdict
        tmp = CKPT_PATH + ".tmp"
        try:
            torch.save({"model": self.model.state_dict(),
                        "model_config": asdict(self.mcfg),
                        "step": self.state["global_step"]}, tmp)
            os.replace(tmp, CKPT_PATH)          # atomic
            self.state["last_save"] = self.state["global_step"]
            self._save_state()
        except Exception as e:
            self._log(f"ckpt save failed: {e}")

    def _load_collected(self):
        n = 0
        if os.path.exists(COLLECTED_PATH):
            try:
                with open(COLLECTED_PATH) as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("text") and r.get("label") in (0, 1):
                            lbl = int(r["label"])
                            self.user[lbl].append(r["text"])
                            self.user_data.append((r["text"], lbl))
                            n += 1
                self._log(f"loaded {n} collected samples "
                          f"(human={len(self.user[0])}, ai={len(self.user[1])})")
            except Exception as e:
                self._log(f"collected load failed: {e}")

    # ----- base dataset harvesting (incremental + cached + fair) -------- #
    def _add_base(self, name, text, label, persist=True):
        """Add one base sample to the pools (deduped) and optionally cache it."""
        text = self.pre(text)
        if not text:
            return False
        if self.source_counts.get(name, 0) >= DATASET_TARGET:
            return False
        h = hash((label, text[:160]))
        if h in self._seen_base:
            return False
        self._seen_base.add(h)
        with self.lock:
            self.base[label].append(text)
        self.source_counts[name] = self.source_counts.get(name, 0) + 1
        if persist:
            try:
                with open(BASE_CACHE_PATH, "a") as f:
                    f.write(json.dumps({"n": name, "text": text,
                                        "label": int(label)}) + "\n")
            except Exception:
                pass
        return True

    def _load_base_cache(self):
        """Reload previously-harvested base samples from /data (fast, offline)."""
        if not os.path.exists(BASE_CACHE_PATH):
            return
        n = 0
        try:
            with open(BASE_CACHE_PATH) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    name, text, lbl = r.get("n"), r.get("text"), r.get("label")
                    if text and lbl in (0, 1):
                        if self._add_base(name or "?", text, int(lbl), persist=False):
                            n += 1
            self._log(f"base cache: {n} samples across "
                      f"{len(self.source_counts)} sources")
        except Exception as e:
            self._log(f"base cache load failed: {e}")

    def _seed_base(self):
        """Synchronously pull a small balanced seed from each source so training
        can start immediately and fairly, even on a cold (empty-cache) start."""
        try:
            from dataset import DATASET_SPECS, open_stream
        except Exception as e:
            self._log(f"dataset module unavailable: {e}")
            return
        for spec in DATASET_SPECS:
            name = spec["name"]
            if self.source_counts.get(name, 0) >= SEED_PER_SOURCE:
                continue
            got = 0
            try:
                it = open_stream(spec)
                self._harvest_iters[name] = it
                for text, lbl in it:
                    if self._add_base(name, text, lbl):
                        got += 1
                    if got >= SEED_PER_SOURCE:
                        break
            except Exception as e:
                self._log(f"seed {name}: {repr(e)[:80]}")
        self._log(f"seed done: human={len(self.base[0])} ai={len(self.base[1])} "
                  f"sources={sum(1 for v in self.source_counts.values() if v)}")

    def _harvest_loop(self):
        """Background round-robin harvester: keeps topping up every source to
        DATASET_TARGET, paced to dodge rate limits, retrying on error, caching
        to /data so the pool grows monotonically across restarts."""
        try:
            from dataset import DATASET_SPECS, open_stream
        except Exception:
            return
        self._log("base harvester started")
        while not self._stop.is_set():
            progressed = False
            for spec in DATASET_SPECS:
                if self._stop.is_set():
                    break
                name = spec["name"]
                if self.source_counts.get(name, 0) >= DATASET_TARGET:
                    continue
                try:
                    it = self._harvest_iters.get(name)
                    if it is None:
                        it = open_stream(spec)
                        self._harvest_iters[name] = it
                    got = 0
                    exhausted = True
                    for text, lbl in it:
                        if self._add_base(name, text, lbl):
                            got += 1
                            progressed = True
                        if got >= HARVEST_CHUNK or \
                                self.source_counts.get(name, 0) >= DATASET_TARGET:
                            exhausted = False
                            break
                    if exhausted:               # stream ended -> recreate later
                        self._harvest_iters[name] = None
                except Exception as e:
                    self._log(f"harvest {name}: {repr(e)[:70]}")
                    self._harvest_iters[name] = None
                    time.sleep(2.0)
                time.sleep(HARVEST_PACE_SEC)    # gentle pacing
            if all(self.source_counts.get(s["name"], 0) >= DATASET_TARGET
                   for s in DATASET_SPECS):
                self._log("base harvest complete: all sources at target")
                break
            if not progressed:
                time.sleep(15.0)
        self.state["base_total"] = len(self.base[0]) + len(self.base[1])

    # ----- data submission --------------------------------------------- #
    def add_sample(self, text: str, label: int):
        text = self.pre(text)
        if not text:
            return
        with self.lock:
            self.user[label].append(text)
            self.user_data.append((text, label))
            self.state["user_samples"] = len(self.user_data)
        try:
            with open(COLLECTED_PATH, "a") as f:
                f.write(json.dumps({"text": text, "label": int(label),
                                    "t": time.time()}) + "\n")
        except Exception as e:
            self._log(f"collected save failed: {e}")
        self._log(f"+1 user sample (label={LABELS[label]}), total={len(self.user_data)}")

    # ----- batching ----------------------------------------------------- #
    _rng_state = [12345]

    def _coin(self, p):                       # deterministic-ish, no global rng
        self._rng_state[0] = (self._rng_state[0] * 1103515245 + 12345) & 0x7fffffff
        return (self._rng_state[0] / 0x7fffffff) < p

    def _rand_idx(self, n):
        self._rng_state[0] = (self._rng_state[0] * 1103515245 + 12345) & 0x7fffffff
        return self._rng_state[0] % n

    def _sample_label(self, want):
        """Pick one text with the requested label, oversampling fresh user data.
        Falls back to the other label if the requested pool is empty."""
        for lbl in (want, 1 - want):
            u, b = self.user[lbl], self.base[lbl]
            # prefer user samples so feedback has immediate effect
            if u and (not b or self._coin(0.5)):
                return u[self._rand_idx(len(u))], lbl
            if b:
                return b[self._rand_idx(len(b))], lbl
        return None

    def _make_batch(self):
        from ai_patterns import feature_vector
        items = []
        for i in range(BATCH):
            picked = self._sample_label(i % 2)        # alternate human/ai
            if picked is None:
                return None
            text, label = picked
            text = self.aug(text)
            feats = feature_vector(text)
            enc = self.tok(text, truncation=True, max_length=MAX_SEQ)
            ids = enc["input_ids"][:MAX_SEQ] or [self.pad_id]
            items.append((ids, label, feats))
        maxlen = max(len(i) for i, _, _ in items)
        bx, bm, by, bf = [], [], [], []
        for ids, label, feats in items:
            pad = maxlen - len(ids)
            bx.append(ids + [self.pad_id] * pad)
            bm.append([1] * len(ids) + [0] * pad)
            by.append(label)
            bf.append(feats)
        return (torch.tensor(bx), torch.tensor(bm), torch.tensor(by),
                torch.tensor(bf, dtype=torch.float))

    # ----- training loop ------------------------------------------------ #
    def _train_step(self):
        batch = self._make_batch()
        if batch is None:
            return False
        ids, mask, labels, feats = (t.to(self.device) for t in batch)
        self.model.train()
        out = self.model(ids, mask, labels, pattern_feats=feats)
        loss = out["loss"]
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        with torch.no_grad():
            preds = out["logits"].argmax(-1)
            acc = (preds == labels).float().mean().item()
        l = float(loss.item())
        now = time.time()
        with self.lock:
            self.state["global_step"] += 1
            self.state["samples_seen"] += labels.numel()
            self.state["loss_ema"] = l if self.state["loss_ema"] is None \
                else 0.98 * self.state["loss_ema"] + 0.02 * l
            self.state["acc_ema"] = acc if self.state["acc_ema"] is None \
                else 0.98 * self.state["acc_ema"] + 0.02 * acc
            self.state["best_acc"] = max(self.state["best_acc"], self.state["acc_ema"])
            # throughput (steps/sec EMA)
            if self._last_step_t is not None:
                dt = max(now - self._last_step_t, 1e-6)
                sps = 1.0 / dt
                self.state["steps_per_s"] = sps if self.state["steps_per_s"] is None \
                    else 0.9 * self.state["steps_per_s"] + 0.1 * sps
            self._last_step_t = now
            # running confusion-matrix tallies (TP/TN/FP/FN), AI = positive
            for p, y in zip(preds.tolist(), labels.tolist()):
                if y == 1 and p == 1: self.state["cm"]["tp"] += 1
                elif y == 0 and p == 0: self.state["cm"]["tn"] += 1
                elif y == 0 and p == 1: self.state["cm"]["fp"] += 1
                else: self.state["cm"]["fn"] += 1
            self.loss_hist.append((self.state["global_step"], round(l, 4)))
            self.acc_hist.append((self.state["global_step"], round(self.state["acc_ema"], 4)))
        return True

    def _loop(self):
        self.state["started_at"] = time.time()
        self.state["status"] = "training"
        self._log("realtime training started")
        while not self._stop.is_set():
            try:
                ok = self._train_step()
                if not ok:
                    self.state["status"] = "waiting for data"
                    time.sleep(2.0)
                    continue
                self.state["status"] = "training"
                if self.state["global_step"] % SAVE_EVERY == 0:
                    self._save_ckpt()
                    self._log(f"step {self.state['global_step']} "
                              f"loss={self.state['loss_ema']:.4f} "
                              f"acc={self.state['acc_ema']:.3f} (checkpoint saved)")
            except Exception as e:
                self._log(f"train error: {e}")
                traceback.print_exc()
                time.sleep(1.0)
        self._save_ckpt()
        self.state["status"] = "stopped"
        self._log("training loop stopped")

    def start(self):
        # background base-data harvester runs regardless of TRAIN so all 8
        # sources keep filling up (and caching) even in inference-only mode.
        self._stop.clear()
        if not (self.harvest_thread and self.harvest_thread.is_alive()):
            self.harvest_thread = threading.Thread(target=self._harvest_loop,
                                                   daemon=True)
            self.harvest_thread.start()
        if not TRAIN_ENABLED:
            self.state["status"] = "training disabled (TRAIN=0)"
            return
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()

    # ----- inference ---------------------------------------------------- #
    @torch.no_grad()
    def _model_probs(self, texts):
        """Batched neural AI-probability for a list of texts."""
        from ai_patterns import feature_vector
        if not texts:
            return []
        encs, feats = [], []
        for t in texts:
            e = self.tok(t, truncation=True, max_length=MAX_SEQ)["input_ids"]
            encs.append(e[:MAX_SEQ] or [self.pad_id])
            feats.append(feature_vector(t))
        maxlen = max(len(e) for e in encs)
        bx, bm = [], []
        for e in encs:
            pad = maxlen - len(e)
            bx.append(e + [self.pad_id] * pad)
            bm.append([1] * len(e) + [0] * pad)
        x = torch.tensor(bx, device=self.device)
        m = torch.tensor(bm, device=self.device)
        ft = torch.tensor(feats, dtype=torch.float, device=self.device)
        self.model.eval()
        return self.model.predict_proba(x, m, pattern_feats=ft)[:, 1].tolist()

    @torch.no_grad()
    def predict(self, text: str):
        from ai_patterns import (feature_vector, heuristic_ai_score, top_signals,
                                 sentence_list, find_ai_tells, extract_features)
        text = self.pre(text)
        if not text:
            return None, {}, {}
        prob = self._model_probs([text])
        ai_prob = prob[0] if prob else 0.5
        full = [1 - ai_prob, ai_prob]
        explain = {
            "neural_ai_prob": round(ai_prob, 4),
            "heuristic_ai_score": round(heuristic_ai_score(text), 4),
            "top_signals": top_signals(text, 6),
            "stylometry": stylometric_features(text),
        }

        # ---- per-sentence / phrase breakdown (stats + model) ----------- #
        sents = sentence_list(text)
        sent_model = self._model_probs(sents) if sents else []
        highlighted, rows = [], []
        for i, s in enumerate(sents):
            f = extract_features(s)
            heur = heuristic_ai_score(s)
            mdl = sent_model[i] if i < len(sent_model) else ai_prob
            blend = 0.5 * heur + 0.5 * mdl
            tells = find_ai_tells(s)
            bucket = ("AI-leaning" if blend >= 0.6
                      else "human-leaning" if blend <= 0.4 else "mixed")
            highlighted.append((s + " ", bucket))
            long_words = sum(1 for w in s.split() if len(w) >= 8)
            rows.append([
                i + 1,
                len(s.split()),
                len(s),
                round(f["avg_word_len"], 2),
                long_words,
                ", ".join(tells) if tells else "—",
                f"{int(round(heur*100))}%",
                f"{int(round(mdl*100))}%",
                bucket,
            ])
        if not highlighted:
            highlighted = [(text, "mixed")]
        breakdown = {
            "analyzed_text": text,
            "highlighted": highlighted,
            "rows": rows,
            "columns": ["#", "words", "chars", "avg word len", "long words",
                        "AI-tell words", "stats AI%", "model AI%", "verdict"],
        }
        return full, explain, breakdown


# --------------------------------------------------------------------------- #
#  Gradio UI
# --------------------------------------------------------------------------- #
def build_ui(trainer: RealtimeTrainer):
    import gradio as gr

    import pandas as pd

    _EMPTY_TBL = pd.DataFrame(
        columns=["#", "words", "chars", "avg word len", "long words",
                 "AI-tell words", "stats AI%", "model AI%", "verdict"])

    def analyze(text):
        if not text or not text.strip():
            return ({"HUMAN": 0.0, "AI-GENERATED": 0.0}, [("Enter some text.", None)],
                    _EMPTY_TBL, {}, "Enter some text.")
        prob, explain, breakdown = trainer.predict(text)
        if prob is None:
            return ({"HUMAN": 0.0, "AI-GENERATED": 0.0}, [("Empty after cleaning.", None)],
                    _EMPTY_TBL, {}, "Empty after cleaning.")
        verdict = LABELS[int(prob[1] >= 0.5)]
        conf = max(prob) * 100
        h = explain["heuristic_ai_score"]
        agree = "✅ agree" if (h >= 0.5) == (prob[1] >= 0.5) else "⚠️ disagree"
        tops = ", ".join(t["signal"] for t in explain["top_signals"][:3])
        note = (f"**{verdict}** · {conf:.1f}% confidence · "
                f"neural AI-prob {prob[1]:.2f} vs heuristic {h:.2f} ({agree})  \n"
                f"top AI signals: _{tops}_  \n"
                f"{len(breakdown['rows'])} sentences analyzed · model step "
                f"{trainer.state['global_step']} (acc≈{(trainer.state['acc_ema'] or 0):.2f})")
        heat = breakdown["highlighted"]
        tbl = pd.DataFrame(breakdown["rows"], columns=breakdown["columns"])
        return {"HUMAN": prob[0], "AI-GENERATED": prob[1]}, heat, tbl, explain, note

    def feedback(text, lbl):
        if not text or not text.strip():
            return "Nothing to save."
        trainer.add_sample(text, lbl)
        return (f"✅ saved as **{LABELS[lbl]}** and queued for training "
                f"(user samples: {trainer.state['user_samples']}).")

    def force_update_now():
        try:
            with open(FORCE_UPDATE_FLAG, "w") as f:
                f.write(str(time.time()))
            return ("🔄 Update check requested — the supervisor will `git fetch` "
                    "on its next tick (≤30 s). If GitHub differs, training pauses "
                    "at a checkpoint and the Space updates + resumes.")
        except Exception as e:
            return f"Could not request update: {e}"

    def _metrics(s):
        cm = s["cm"]
        tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    def dashboard():
        s = trainer.state
        up = (time.time() - s["started_at"]) if s["started_at"] else 0
        prec, rec, f1 = _metrics(s)
        sps = s.get("steps_per_s") or 0.0
        md = (
            f"| metric | value |\n|---|---|\n"
            f"| status | **{s['status']}** |\n"
            f"| preset / params | {s['preset']} / {s['params_M']}M |\n"
            f"| device | {s['device']} |\n"
            f"| global step | {s['global_step']} |\n"
            f"| throughput | {sps:.2f} steps/s |\n"
            f"| samples seen | {s['samples_seen']} |\n"
            f"| user samples | {s['user_samples']} |\n"
            f"| loss (EMA) | {(s['loss_ema'] or 0):.4f} |\n"
            f"| acc (EMA) | {(s['acc_ema'] or 0):.3f} |\n"
            f"| best acc | {s['best_acc']:.3f} |\n"
            f"| precision / recall / F1 | {prec:.3f} / {rec:.3f} / {f1:.3f} |\n"
            f"| last checkpoint @step | {s['last_save']} |\n"
            f"| uptime | {up/60:.1f} min |\n"
            f"| storage | `{s['data_dir']}` |\n"
        )
        try:
            if os.path.exists(UPDATER_STATUS_PATH):
                with open(UPDATER_STATUS_PATH) as f:
                    u = json.load(f)
                age = (time.time() - u.get("last_check", time.time())) / 60
                upd = "🔄 updating…" if u.get("updating") else "✅ up to date"
                md += (f"| github | {upd} |\n"
                       f"| local rev | `{(u.get('local_sha') or '')[:7]}` |\n"
                       f"| remote rev | `{(u.get('remote_sha') or '')[:7]}` |\n"
                       f"| last update check | {age:.1f} min ago |\n")
        except Exception:
            pass

        lh = list(trainer.loss_hist)
        loss_df = pd.DataFrame(lh, columns=["step", "loss"]) if lh else \
            pd.DataFrame({"step": [], "loss": []})
        ah = list(trainer.acc_hist)
        acc_df = pd.DataFrame(ah, columns=["step", "accuracy"]) if ah else \
            pd.DataFrame({"step": [], "accuracy": []})

        dist_df = pd.DataFrame({
            "pool": ["base·human", "base·ai", "user·human", "user·ai"],
            "count": [len(trainer.base[0]), len(trainer.base[1]),
                      len(trainer.user[0]), len(trainer.user[1])],
        })
        cm = s["cm"]
        cm_df = pd.DataFrame({
            "cell": ["TP (ai✓)", "TN (human✓)", "FP (human→ai)", "FN (ai→human)"],
            "count": [cm["tp"], cm["tn"], cm["fp"], cm["fn"]],
        })
        # per-source harvest progress (short names) toward DATASET_TARGET
        sc = trainer.source_counts
        src_df = pd.DataFrame({
            "source": [n.split("/")[-1][:22] for n in sc],
            "samples": [sc[n] for n in sc],
        }) if sc else pd.DataFrame({"source": [], "samples": []})
        done = sum(1 for v in sc.values() if v >= DATASET_TARGET)
        md += (f"| base pool | {len(trainer.base[0])+len(trainer.base[1])} "
               f"(human {len(trainer.base[0])} / ai {len(trainer.base[1])}) |\n"
               f"| dataset sources | {len(sc)} active · {done}/{len(sc)} at target |\n")
        logs = "\n".join(list(trainer.log_lines)[-16:])
        return md, loss_df, acc_df, dist_df, cm_df, src_df, logs

    try:
        _theme = gr.themes.Soft()
    except Exception:
        _theme = None
    with gr.Blocks(title="GPT-Protect · AI Text Detector") as demo:
        demo._gptprotect_theme = _theme
        gr.Markdown(
            "# 🛡️ GPT-Protect — AI Text Detector\n"
            "MoE + Hybrid-Attention detector with an **intelligent AI-pattern engine** "
            "(burstiness, perplexity proxy, repetition, AI-tell lexicon, …) fused into "
            "the model. Your labeled feedback trains it in real time, persisted to /data."
        )
        with gr.Tab("🔍 Detect"):
            with gr.Row():
                with gr.Column(scale=3):
                    inp = gr.Textbox(lines=12, label="Text to analyze",
                                     placeholder="Paste an essay, comment, article…")
                    btn = gr.Button("Analyze", variant="primary")
                    note = gr.Markdown()
                    gr.Markdown("**Was this prediction wrong? Help train the model:**")
                    with gr.Row():
                        b_h = gr.Button("✍️ This is HUMAN")
                        b_a = gr.Button("🤖 This is AI")
                    fb = gr.Markdown()
                with gr.Column(scale=2):
                    out = gr.Label(label="Prediction", num_top_classes=2)
                    explain = gr.JSON(label="AI-pattern analysis (neural + heuristic)")
            gr.Markdown("### 🔦 Analyzed text — sentence AI-heatmap")
            gr.Markdown("Red = AI-leaning · amber = mixed · green = human-leaning "
                        "(blend of statistics + model).")
            heat = gr.HighlightedText(
                label="What was detected (per sentence)",
                combine_adjacent=False, show_legend=True,
                color_map={"AI-leaning": "#ef4444", "mixed": "#f59e0b",
                           "human-leaning": "#22c55e"})
            gr.Markdown("### 📋 Per-sentence / phrase breakdown")
            table = gr.Dataframe(
                headers=["#", "words", "chars", "avg word len", "long words",
                         "AI-tell words", "stats AI%", "model AI%", "verdict"],
                label="Phrase length & what statistics / the model flagged",
                wrap=True, interactive=False)
            btn.click(analyze, inp, [out, heat, table, explain, note])
            b_h.click(lambda t: feedback(t, 0), inp, fb)
            b_a.click(lambda t: feedback(t, 1), inp, fb)

        with gr.Tab("📈 Training dashboard"):
            with gr.Row():
                gr.Markdown("Live realtime-training metrics (auto-refresh every 3 s).")
                upd_btn = gr.Button("🔄 Check GitHub for update now", scale=0)
            upd_msg = gr.Markdown()
            with gr.Row():
                stats_md = gr.Markdown()
                with gr.Column():
                    loss_plot = gr.LinePlot(x="step", y="loss",
                                            title="Training loss", height=240)
                    acc_plot = gr.LinePlot(x="step", y="accuracy",
                                           title="Accuracy (EMA)", height=240)
            with gr.Row():
                dist_plot = gr.BarPlot(x="pool", y="count",
                                       title="Training-pool composition", height=240)
                cm_plot = gr.BarPlot(x="cell", y="count",
                                     title="Confusion tallies (AI = positive)",
                                     height=240)
            src_plot = gr.BarPlot(x="source", y="samples",
                                  title="Samples harvested per dataset source",
                                  height=260)
            logbox = gr.Textbox(label="Recent log", lines=12, interactive=False)

            outs = [stats_md, loss_plot, acc_plot, dist_plot, cm_plot,
                    src_plot, logbox]
            timer = gr.Timer(3.0)
            timer.tick(dashboard, None, outs)
            demo.load(dashboard, None, outs)
            upd_btn.click(force_update_now, None, upd_msg)

    return demo


def main():
    print(f"[main] data dir = {DATA_DIR}  preset = {PRESET}")
    trainer = RealtimeTrainer()
    trainer.start()

    # Graceful shutdown for the app.py auto-updater: pause training, write a
    # final checkpoint, then exit so the supervisor can pull + restart.
    import signal

    def _graceful(signum, frame):
        try:
            trainer._log(f"signal {signum}: pausing & checkpointing before exit")
            trainer.state["status"] = "pausing for update"
            trainer.stop()
            if trainer.thread:
                trainer.thread.join(timeout=max(5, UPDATE_GRACE_MARGIN))
            trainer._save_ckpt()
            trainer._log("checkpoint saved; exiting for update")
        finally:
            os._exit(0)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _graceful)
        except Exception:
            pass

    demo = build_ui(trainer)
    port = int(os.environ.get("PORT", "7860"))
    kw = dict(server_name="0.0.0.0", server_port=port)
    theme = getattr(demo, "_gptprotect_theme", None)
    if theme is not None:
        try:
            demo.queue(max_size=32).launch(theme=theme, **kw)
            return
        except TypeError:
            pass
    demo.queue(max_size=32).launch(**kw)


if __name__ == "__main__":
    main()
