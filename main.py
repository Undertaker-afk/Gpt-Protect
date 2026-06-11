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
# touch-files the UI writes for the app.py supervisor to act on
FORCE_UPDATE_FLAG = os.path.join(DATA_DIR, "force_update")
RESET_FLAG = os.path.join(DATA_DIR, "force_reset")     # git pull + hard reset
NUKE_FLAG = os.path.join(DATA_DIR, "nuke")             # wipe repo + bucket + setup
# admin password (Space secret). If unset, the admin panel refuses to operate.
ADMIN_PWD = os.environ.get("PWD_ENV", "")
# advanced objectives (default off on free CPU; flip on with more hardware)
BACKBONE = os.environ.get("BACKBONE", "scratch")            # "hf:<name>" (#19)
CONTRASTIVE_COEF = float(os.environ.get("CONTRASTIVE_COEF", "0"))   # (#22)
DISTILL_TEACHER = os.environ.get("DISTILL_TEACHER", "")            # (#30) "hf:..."
DISTILL_COEF = float(os.environ.get("DISTILL_COEF", "0.5"))
ACTIVE_LEARNING = os.environ.get("ACTIVE_LEARNING", "1") == "1"     # (#26)
CALIBRATE_EVERY = int(os.environ.get("CALIBRATE_EVERY", "200"))     # (#24)
ABSTAIN_MARGIN = float(os.environ.get("ABSTAIN_MARGIN", "0.12"))    # (#24)

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

        # tokenizer: matches the HF backbone when one is selected (#19)
        tok_name = (BACKBONE.split("hf:", 1)[1] if BACKBONE.startswith("hf:")
                    else os.environ.get("TOKENIZER", "gpt2"))
        self.tok = get_tokenizer(tok_name)
        self.mcfg = preset(PRESET)
        self.mcfg.backbone = BACKBONE
        self.mcfg.contrastive_coef = CONTRASTIVE_COEF
        self.mcfg.abstain_margin = ABSTAIN_MARGIN
        self.mcfg.max_seq_len = max(self.mcfg.max_seq_len, MAX_SEQ)
        vocab = len(self.tok) if hasattr(self.tok, "__len__") else \
            getattr(self.tok, "vocab_size", 50257)
        self.mcfg.vocab_size = vocab
        self.pad_id = getattr(self.tok, "pad_token_id", 0) or 0

        self.model = build_model(self.mcfg).to(self.device)
        self.model.label_smoothing = 0.05
        self.tcfg = TrainConfig(optimizer=os.environ.get("OPTIMIZER", "muon"))
        self.opt = build_optimizer(self.model, self.tcfg)
        self.teacher = self._load_teacher()      # KD teacher (#30), optional

        # active-learning hard pool (#26): uncertain samples, oversampled
        self.hard = {0: [], 1: []}
        self._last_texts = []
        # shared, UI-visible state
        self.state = {
            "global_step": 0, "samples_seen": 0, "user_samples": 0,
            "loss_ema": None, "acc_ema": None, "best_acc": 0.0,
            "steps_per_s": None, "temperature": 1.0, "contrastive_ema": None,
            "cm": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
            "started_at": None, "last_save": 0, "status": "init",
            "preset": PRESET, "backbone": BACKBONE,
            "params_M": round(self.model.num_parameters() / 1e6, 2),
            "n_features": __import__("ai_patterns").N_FEATURES,
            "perplexity": __import__("perplexity").enabled(),
            "teacher": bool(self.teacher),
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
        self._pause = threading.Event()          # admin can pause the train loop
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

    def _sample_label_AL(self, want):
        """Like _sample_label but oversamples the active-learning hard pool."""
        if ACTIVE_LEARNING and self.hard[want] and self._coin(0.35):
            return self.hard[want][self._rand_idx(len(self.hard[want]))], want
        return self._sample_label(want)

    def _make_batch(self):
        from ai_patterns import feature_vector
        items = []
        for i in range(BATCH):
            picked = self._sample_label_AL(i % 2)     # alternate human/ai
            if picked is None:
                return None
            text, label = picked
            text = self.aug(text)                     # adversarial aug (#23)
            feats = feature_vector(text)
            enc = self.tok(text, truncation=True, max_length=MAX_SEQ)
            ids = enc["input_ids"][:MAX_SEQ] or [self.pad_id]
            items.append((ids, label, feats, text))
        maxlen = max(len(it[0]) for it in items)
        bx, bm, by, bf, bt = [], [], [], [], []
        for ids, label, feats, text in items:
            pad = maxlen - len(ids)
            bx.append(ids + [self.pad_id] * pad)
            bm.append([1] * len(ids) + [0] * pad)
            by.append(label)
            bf.append(feats)
            bt.append(text)
        return (torch.tensor(bx), torch.tensor(bm), torch.tensor(by),
                torch.tensor(bf, dtype=torch.float), bt)

    def _mine_hard(self, ids, labels, feats, probs):
        """Active learning (#26): keep the most-uncertain samples for replay."""
        if not ACTIVE_LEARNING:
            return
        conf = probs.max(-1).values
        for i, c in enumerate(conf.tolist()):
            if c < 0.65:                              # uncertain
                lbl = int(labels[i].item())
                txt = self._last_texts[i] if i < len(self._last_texts) else None
                if txt:
                    pool = self.hard[lbl]
                    pool.append(txt)
                    if len(pool) > 400:
                        pool.pop(0)

    @torch.no_grad()
    def _calibrate(self):
        """Temperature scaling (#24): fit log_temp on a held-out batch."""
        b = self._make_batch()
        if b is None:
            return
        ids, mask, labels, feats, _ = b
        self.model.eval()
        logits = self.model(ids.to(self.device), mask.to(self.device),
                            pattern_feats=feats.to(self.device))["logits"].detach()
        labels = labels.to(self.device)
        tp = self.model.log_temp
        tp.requires_grad_(True)
        opt = torch.optim.LBFGS([tp], lr=0.1, max_iter=20)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                logits / tp.exp().clamp(0.3, 5.0), labels)
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except Exception:
            pass
        self.model.train()
        self.state["temperature"] = round(float(self.model.temperature), 3)

    # ----- teacher / objectives ---------------------------------------- #
    def _load_teacher(self):
        if not DISTILL_TEACHER:
            return None
        try:
            from config import preset as _preset
            tcfg = _preset(PRESET)
            tcfg.backbone = DISTILL_TEACHER
            tcfg.vocab_size = self.mcfg.vocab_size
            teacher = build_model(tcfg).to(self.device).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            self._log(f"distillation teacher loaded: {DISTILL_TEACHER}")
            return teacher
        except Exception as e:
            self._log(f"teacher load failed: {repr(e)[:90]}")
            return None

    # ----- training loop ------------------------------------------------ #
    def _train_step(self):
        batch = self._make_batch()
        if batch is None:
            return False
        ids, mask, labels, feats, texts = batch
        self._last_texts = texts
        ids = ids.to(self.device); mask = mask.to(self.device)
        labels = labels.to(self.device); feats = feats.to(self.device)
        self.model.train()
        out = self.model(ids, mask, labels, pattern_feats=feats)
        loss = out["loss"]
        closs = None
        if CONTRASTIVE_COEF > 0:                                # SupCon (#22)
            from model import supcon_loss
            closs = supcon_loss(out["embedding"], labels)
            loss = loss + CONTRASTIVE_COEF * closs
        if self.teacher is not None:                           # distill (#30)
            from model import distill_loss
            with torch.no_grad():
                t_logits = self.teacher(ids, mask, pattern_feats=feats)["logits"]
            loss = loss + DISTILL_COEF * distill_loss(out["logits"], t_logits)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        with torch.no_grad():
            probs = torch.softmax(out["logits"], -1)
            preds = probs.argmax(-1)
            acc = (preds == labels).float().mean().item()
            self._mine_hard(ids, labels, feats, probs)          # active learn (#26)
        l = float(loss.item())
        if closs is not None:
            cv = float(closs.item())
            self.state["contrastive_ema"] = cv if self.state["contrastive_ema"] is None \
                else 0.98 * self.state["contrastive_ema"] + 0.02 * cv
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
                if self._pause.is_set():
                    self.state["status"] = "paused (admin)"
                    time.sleep(0.25)
                    continue
                ok = self._train_step()
                if not ok:
                    self.state["status"] = "waiting for data"
                    time.sleep(2.0)
                    continue
                self.state["status"] = "training"
                if CALIBRATE_EVERY and self.state["global_step"] % CALIBRATE_EVERY == 0:
                    self._calibrate()               # temperature scaling (#24)
                if self.state["global_step"] % SAVE_EVERY == 0:
                    self._save_ckpt()
                    self._log(f"step {self.state['global_step']} "
                              f"loss={self.state['loss_ema']:.4f} "
                              f"acc={self.state['acc_ema']:.3f} "
                              f"T={self.state['temperature']} (checkpoint saved)")
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

    # ----- admin operations -------------------------------------------- #
    def admin_enabled(self):
        return bool(ADMIN_PWD)

    def _auth(self, pwd):
        if not ADMIN_PWD:
            return False, "🔒 Admin disabled: set the `PWD_ENV` secret on the Space."
        if pwd != ADMIN_PWD:
            return False, "❌ Wrong password."
        return True, ""

    def _pause_for_admin(self):
        """Park the training loop at the pause check so we can mutate safely."""
        self._pause.set()
        time.sleep(3.0)                      # let any in-flight step finish

    def _resume_after_admin(self):
        self._pause.clear()

    def _rm(self, path):
        import shutil
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.exists(path):
                os.remove(path)
        except Exception as e:
            self._log(f"rm {path}: {e}")

    def _reinit_model(self):
        self.model = build_model(self.mcfg).to(self.device)
        self.model.label_smoothing = 0.05
        self.opt = build_optimizer(self.model, self.tcfg)
        self.state.update(global_step=0, samples_seen=0, loss_ema=None,
                          acc_ema=None, best_acc=0.0, steps_per_s=None,
                          last_save=0, cm={"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        self.loss_hist.clear()
        self.acc_hist.clear()
        self._last_step_t = None

    def admin_delete_models(self, pwd):
        ok, msg = self._auth(pwd)
        if not ok:
            return msg
        self._pause_for_admin()
        try:
            self._rm(CKPT_DIR)
            os.makedirs(CKPT_DIR, exist_ok=True)
            self._rm(STATE_PATH)
            self._reinit_model()
            self._log("ADMIN: deleted local models + reinitialized weights")
        finally:
            self._resume_after_admin()
        return "🗑️ Local models deleted and weights reinitialized (step reset to 0)."

    def admin_reset_everything(self, pwd):
        ok, msg = self._auth(pwd)
        if not ok:
            return msg
        self._pause_for_admin()
        try:
            for p in (CKPT_DIR, STATE_PATH, COLLECTED_PATH, BASE_CACHE_PATH,
                      LOG_PATH, UPDATER_STATUS_PATH):
                self._rm(p)
            os.makedirs(CKPT_DIR, exist_ok=True)
            # wipe in-memory state
            self.base = {0: [], 1: []}
            self.user = {0: [], 1: []}
            self.user_data = []
            self.source_counts = {}
            self._seen_base = set()
            self._harvest_iters = {}
            self.log_lines.clear()
            self._reinit_model()
            self.state["user_samples"] = 0
            self._seed_base()                 # re-seed so training resumes
            self._log("ADMIN: full reset (models, state, collected, base cache)")
        finally:
            self._resume_after_admin()
        return ("♻️ Reset everything: models, training state, collected samples "
                "and base cache cleared; datasets re-seeding. Repo/code untouched.")

    def admin_request_reset(self, pwd):
        ok, msg = self._auth(pwd)
        if not ok:
            return msg
        try:
            with open(RESET_FLAG, "w") as f:
                f.write(str(time.time()))
        except Exception as e:
            return f"Could not request repull: {e}"
        self._log("ADMIN: repull+reset requested")
        return ("⤵️ Repull & reset requested — supervisor will `git fetch` + hard-"
                "reset to remote, reinstall, and restart (≤30 s). /data is kept.")

    def admin_request_nuke(self, pwd):
        ok, msg = self._auth(pwd)
        if not ok:
            return msg
        try:
            with open(NUKE_FLAG, "w") as f:
                f.write(str(time.time()))
        except Exception as e:
            return f"Could not arm NUKE: {e}"
        self._log("ADMIN: ☢️ NUKE armed")
        return ("☢️ **NUKE armed.** Supervisor will delete the local repo, every "
                "checkpoint and every file in the bucket, then re-clone and set "
                "everything up again (≤30 s). This cannot be undone.")

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
    def _token_heatmap(self, text):
        """Per-token AI heatmap from the token head (#20)."""
        if not getattr(self.model, "use_token_head", False):
            return None
        from ai_patterns import feature_vector
        try:
            enc = self.tok(text, truncation=True, max_length=MAX_SEQ,
                           return_offsets_mapping=True)
        except Exception:
            return None
        ids = enc["input_ids"][:MAX_SEQ] or [self.pad_id]
        offs = enc.get("offset_mapping", [])[:len(ids)]
        x = torch.tensor([ids], device=self.device)
        m = torch.ones_like(x)
        ft = torch.tensor([feature_vector(text)], dtype=torch.float,
                          device=self.device)
        tp = self.model.token_ai_probs(x, m, pattern_feats=ft)
        if tp is None:
            return None
        tp = tp[0].tolist()
        spans, pos = [], 0
        for (a, b), p in zip(offs, tp):
            if a is None or b is None or b <= a:
                continue
            if a > pos:
                spans.append((text[pos:a], None))
            bucket = ("AI-token" if p >= 0.6 else
                      "human-token" if p <= 0.4 else "mixed-token")
            spans.append((text[a:b], bucket))
            pos = b
        if pos < len(text):
            spans.append((text[pos:], None))
        return spans or None

    @torch.no_grad()
    def predict(self, text: str):
        from ai_patterns import (feature_vector, heuristic_ai_score, top_signals,
                                 sentence_list, find_ai_tells, extract_features,
                                 tell_spans, grouped_report)
        import perplexity
        text = self.pre(text)
        if not text:
            return None, {}, {}
        prob = self._model_probs([text])           # calibrated (#24)
        ai_prob = prob[0] if prob else 0.5
        heur = heuristic_ai_score(text)
        ppl = perplexity.analysis(text)            # (#1,#2) zeros if disabled
        ppl_prob = perplexity.ai_probability(text)

        # ensemble of available signals
        sig = [(ai_prob, 0.5), (heur, 0.3)]
        if ppl_prob is not None:
            sig.append((ppl_prob, 0.25))
        wsum = sum(w for _, w in sig)
        ensemble = sum(p * w for p, w in sig) / wsum
        full = [1 - ai_prob, ai_prob]

        # calibrated verdict + abstention (#24)
        if abs(ensemble - 0.5) < ABSTAIN_MARGIN:
            verdict = "UNCERTAIN"
        else:
            verdict = LABELS[int(ensemble >= 0.5)]

        explain = {
            "verdict": verdict,
            "ensemble_ai_prob": round(ensemble, 4),
            "neural_ai_prob (calibrated)": round(ai_prob, 4),
            "heuristic_ai_score": round(heur, 4),
            "perplexity_ai_prob": (round(ppl_prob, 4) if ppl_prob is not None
                                   else "disabled (USE_PERPLEXITY=1)"),
            "calibration_temperature": self.state.get("temperature", 1.0),
            "top_signals": top_signals(text, 8),
            "stylometry": stylometric_features(text),
        }
        stats_report = grouped_report(text)
        if ppl.get("enabled"):
            stats_report["perplexity / DetectGPT (#1,#2)"] = {
                k: ppl[k] for k in ("perplexity", "mean_nll", "frac_top10",
                                    "frac_top100", "detectgpt_curvature")}
        token_heat = self._token_heatmap(text)

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
            # sentence-level color + word-level AI-tell highlights on top
            highlighted.extend(tell_spans(s, bucket))
            highlighted.append((" ", bucket))
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
            "token_highlighted": token_heat,
            "stats_report": stats_report,
            "verdict": verdict,
            "ensemble": ensemble,
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

    def _blank(msg):
        return ({"HUMAN": 0.0, "AI-GENERATED": 0.0}, [(msg, None)],
                [(msg, None)], _EMPTY_TBL, {}, {}, msg)

    def analyze(text):
        if not text or not text.strip():
            return _blank("Enter some text.")
        prob, explain, breakdown = trainer.predict(text)
        if prob is None:
            return _blank("Empty after cleaning.")
        verdict = breakdown["verdict"]
        ens = breakdown["ensemble"]
        h = explain["heuristic_ai_score"]
        ppl = explain["perplexity_ai_prob"]
        emoji = {"HUMAN": "✍️", "AI-GENERATED": "🤖", "UNCERTAIN": "🤔"}[verdict]
        tops = ", ".join(t["signal"] for t in explain["top_signals"][:3])
        note = (f"## {emoji} {verdict}\n"
                f"**Ensemble AI-probability: {ens*100:.1f}%** "
                f"(neural {prob[1]:.2f} · heuristic {h:.2f}"
                + (f" · perplexity {ppl:.2f}" if isinstance(ppl, float) else "")
                + f" · calibration T={explain['calibration_temperature']})  \n"
                f"top signals: _{tops}_  \n"
                f"{len(breakdown['rows'])} sentences · model step "
                f"{trainer.state['global_step']} (acc≈{(trainer.state['acc_ema'] or 0):.2f})")
        sent_heat = breakdown["highlighted"]
        tok_heat = breakdown["token_highlighted"] or [
            ("token-level head unavailable", None)]
        tbl = pd.DataFrame(breakdown["rows"], columns=breakdown["columns"])
        return ({"HUMAN": prob[0], "AI-GENERATED": prob[1]}, sent_heat, tok_heat,
                tbl, breakdown["stats_report"], explain, note)

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
                        "(blend of statistics + model); **purple = AI-tell words**.")
            heat = gr.HighlightedText(
                label="Sentence color + AI-tell words",
                combine_adjacent=False, show_legend=True,
                color_map={"AI-leaning": "#ef4444", "mixed": "#f59e0b",
                           "human-leaning": "#22c55e", "AI-tell": "#9333ea"})
            gr.Markdown("### 🧠 Token-level model heatmap (per-token AI head, #20)")
            tok_heat = gr.HighlightedText(
                label="What the neural token-head flagged, token by token",
                combine_adjacent=True, show_legend=True,
                color_map={"AI-token": "#ef4444", "mixed-token": "#f59e0b",
                           "human-token": "#22c55e"})
            gr.Markdown("### 📋 Per-sentence / phrase breakdown")
            table = gr.Dataframe(
                headers=["#", "words", "chars", "avg word len", "long words",
                         "AI-tell words", "stats AI%", "model AI%", "verdict"],
                label="Phrase length & what statistics / the model flagged",
                wrap=True, interactive=False)
            gr.Markdown("### 📊 Full statistics breakdown (all detected features)")
            stats_json = gr.JSON(label="Grouped feature report "
                                       "(burstiness, diversity, readability, "
                                       "sentiment, perplexity, …)")
            btn.click(analyze, inp,
                      [out, heat, tok_heat, table, stats_json, explain, note])
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

        with gr.Tab("🛠️ Admin"):
            gr.Markdown(
                "## 🛠️ Admin panel\n"
                "Enter the admin password (Space secret `PWD_ENV`) to operate. "
                "Every action re-checks the password.")
            adm_pwd = gr.Textbox(label="Admin password", type="password",
                                 placeholder="PWD_ENV")
            adm_status = gr.Markdown()
            if not trainer.admin_enabled():
                gr.Markdown("⚠️ `PWD_ENV` is not set on this Space — the admin "
                            "panel is **disabled** until you add that secret.")
            with gr.Row():
                a_del = gr.Button("🗑️ Delete local models")
                a_reset = gr.Button("♻️ Reset everything (data + model)")
                a_pull = gr.Button("⤵️ Repull & reset (code)")
            gr.Markdown("---\n### ☢️ Danger zone")
            gr.Markdown(
                "**NUKE** deletes the local GitHub repo, **all** checkpoints and "
                "**every file in the mounted bucket**, then re-clones and sets "
                "everything up again from scratch. There is no undo.")
            nuke_confirm = gr.Textbox(
                label="Type NUKE to confirm", placeholder="NUKE")
            a_nuke = gr.Button("☢️ NUKE EVERYTHING", variant="stop")

            a_del.click(trainer.admin_delete_models, adm_pwd, adm_status)
            a_reset.click(trainer.admin_reset_everything, adm_pwd, adm_status)
            a_pull.click(trainer.admin_request_reset, adm_pwd, adm_status)

            def _nuke(pwd, confirm):
                if confirm.strip() != "NUKE":
                    return "Type **NUKE** in the confirm box to proceed."
                return trainer.admin_request_nuke(pwd)
            a_nuke.click(_nuke, [adm_pwd, nuke_confirm], adm_status)

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
