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

PRESET = os.environ.get("MODEL_PRESET", "tiny")        # free CPU -> tiny
MAX_SEQ = int(os.environ.get("MAX_SEQ_LEN", "192"))
BATCH = int(os.environ.get("BATCH_SIZE", "8"))
BASE_SAMPLES = int(os.environ.get("BASE_SAMPLES", "4000"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "25"))   # steps between ckpts
TRAIN_ENABLED = os.environ.get("TRAIN", "1") == "1"
# how long the SIGTERM handler waits for the training thread to finish a step
UPDATE_GRACE_MARGIN = int(os.environ.get("UPDATE_GRACE_MARGIN", "30"))
UPDATER_STATUS_PATH = os.path.join(DATA_DIR, "updater.json")

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
            "started_at": None, "last_save": 0, "status": "init",
            "preset": PRESET, "params_M": round(self.model.num_parameters() / 1e6, 2),
            "device": self.device, "data_dir": DATA_DIR,
        }
        self.loss_hist = deque(maxlen=300)     # (step, loss)
        self.log_lines = deque(maxlen=40)

        self._load_state()
        self._restore_ckpt()

        # label-split pools (0=human, 1=ai) keep training balanced even though
        # the public corpus is human-heavy (basic_text is all human).
        self.base = {0: [], 1: []}
        self.user = {0: [], 1: []}
        self.user_data = []                      # canonical list (count/persist)
        self._load_base_data()
        self._load_collected()
        self.state["user_samples"] = len(self.user_data)

        self._stop = threading.Event()
        self.thread = None

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

    def _load_base_data(self):
        """Load a capped slice of the public datasets for continual training."""
        try:
            from dataset import load_human_ai, DATASET_SPECS
            # stream only ~enough per source to fill BASE_SAMPLES after shuffle,
            # so startup stays light on a 2-vCPU / 16 GB Space.
            n_specs = max(1, len(DATASET_SPECS))
            per = int(os.environ.get(
                "DATASET_PER_CAP", str(max(200, (BASE_SAMPLES // n_specs) * 2))))
            ds = load_human_ai(max_samples=BASE_SAMPLES, per_dataset=per,
                               preprocess=True)
            for r in ds:
                self.base[int(r["label"])].append(r["text"])
            self._log(f"base dataset: human={len(self.base[0])}, "
                      f"ai={len(self.base[1])}")
        except Exception as e:
            self._log(f"base data unavailable ({e}); training on user data only")

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
        items = []
        for i in range(BATCH):
            picked = self._sample_label(i % 2)        # alternate human/ai
            if picked is None:
                return None
            text, label = picked
            text = self.aug(text)
            enc = self.tok(text, truncation=True, max_length=MAX_SEQ)
            ids = enc["input_ids"][:MAX_SEQ] or [self.pad_id]
            items.append((ids, label))
        maxlen = max(len(i) for i, _ in items)
        bx, bm, by = [], [], []
        for ids, label in items:
            pad = maxlen - len(ids)
            bx.append(ids + [self.pad_id] * pad)
            bm.append([1] * len(ids) + [0] * pad)
            by.append(label)
        return (torch.tensor(bx), torch.tensor(bm), torch.tensor(by))

    # ----- training loop ------------------------------------------------ #
    def _train_step(self):
        batch = self._make_batch()
        if batch is None:
            return False
        ids, mask, labels = (t.to(self.device) for t in batch)
        self.model.train()
        out = self.model(ids, mask, labels)
        loss = out["loss"]
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        with torch.no_grad():
            acc = (out["logits"].argmax(-1) == labels).float().mean().item()
        l = float(loss.item())
        with self.lock:
            self.state["global_step"] += 1
            self.state["samples_seen"] += labels.numel()
            self.state["loss_ema"] = l if self.state["loss_ema"] is None \
                else 0.98 * self.state["loss_ema"] + 0.02 * l
            self.state["acc_ema"] = acc if self.state["acc_ema"] is None \
                else 0.98 * self.state["acc_ema"] + 0.02 * acc
            self.state["best_acc"] = max(self.state["best_acc"], self.state["acc_ema"])
            self.loss_hist.append((self.state["global_step"], round(l, 4)))
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
        if not TRAIN_ENABLED:
            self.state["status"] = "training disabled (TRAIN=0)"
            return
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()

    # ----- inference ---------------------------------------------------- #
    @torch.no_grad()
    def predict(self, text: str):
        text = self.pre(text)
        if not text:
            return None, {}
        enc = self.tok(text, truncation=True, max_length=MAX_SEQ)
        ids = enc["input_ids"][:MAX_SEQ] or [self.pad_id]
        x = torch.tensor([ids], device=self.device)
        m = torch.ones_like(x)
        self.model.eval()
        prob = self.model.predict_proba(x, m)[0].tolist()
        return prob, stylometric_features(text)


# --------------------------------------------------------------------------- #
#  Gradio UI
# --------------------------------------------------------------------------- #
def build_ui(trainer: RealtimeTrainer):
    import gradio as gr

    def analyze(text):
        if not text or not text.strip():
            return {"HUMAN": 0.0, "AI-GENERATED": 0.0}, {}, "Enter some text."
        prob, feats = trainer.predict(text)
        if prob is None:
            return {"HUMAN": 0.0, "AI-GENERATED": 0.0}, {}, "Empty after cleaning."
        verdict = LABELS[int(prob[1] >= 0.5)]
        conf = max(prob) * 100
        note = (f"**{verdict}**  ·  {conf:.1f}% confidence  ·  "
                f"model step {trainer.state['global_step']} "
                f"(acc≈{(trainer.state['acc_ema'] or 0):.2f})")
        return {"HUMAN": prob[0], "AI-GENERATED": prob[1]}, feats, note

    def feedback(text, lbl):
        if not text or not text.strip():
            return "Nothing to save."
        trainer.add_sample(text, lbl)
        return (f"✅ saved as **{LABELS[lbl]}** and queued for training "
                f"(user samples: {trainer.state['user_samples']}).")

    def dashboard():
        s = trainer.state
        up = (time.time() - s["started_at"]) if s["started_at"] else 0
        md = (
            f"| metric | value |\n|---|---|\n"
            f"| status | **{s['status']}** |\n"
            f"| preset / params | {s['preset']} / {s['params_M']}M |\n"
            f"| device | {s['device']} |\n"
            f"| global step | {s['global_step']} |\n"
            f"| samples seen | {s['samples_seen']} |\n"
            f"| user samples | {s['user_samples']} |\n"
            f"| loss (EMA) | {(s['loss_ema'] if s['loss_ema'] is not None else 0):.4f} |\n"
            f"| acc (EMA) | {(s['acc_ema'] if s['acc_ema'] is not None else 0):.3f} |\n"
            f"| best acc | {s['best_acc']:.3f} |\n"
            f"| last checkpoint @step | {s['last_save']} |\n"
            f"| uptime | {up/60:.1f} min |\n"
            f"| storage | `{s['data_dir']}` |\n"
        )
        # auto-update status written by app.py supervisor
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
        import pandas as pd
        hist = list(trainer.loss_hist)
        df = pd.DataFrame(hist, columns=["step", "loss"]) if hist else \
            pd.DataFrame({"step": [], "loss": []})
        logs = "\n".join(list(trainer.log_lines)[-15:])
        return md, df, logs

    try:
        _theme = gr.themes.Soft()
    except Exception:
        _theme = None
    with gr.Blocks(title="GPT-Protect · AI Text Detector") as demo:
        demo._gptprotect_theme = _theme
        gr.Markdown(
            "# 🛡️ GPT-Protect — AI Text Detector\n"
            "MoE + Hybrid-Attention detector (DeepSeek-V4-Pro / TurboQuant inspired). "
            "Paste text to classify it as **human** or **AI-generated**. Your labeled "
            "feedback trains the model in real time and is persisted to the data bucket."
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
                    feats = gr.JSON(label="Stylometric features")
            btn.click(analyze, inp, [out, feats, note])
            b_h.click(lambda t: feedback(t, 0), inp, fb)
            b_a.click(lambda t: feedback(t, 1), inp, fb)

        with gr.Tab("📈 Training dashboard"):
            gr.Markdown("Live realtime-training metrics (auto-refresh every 3 s).")
            with gr.Row():
                stats_md = gr.Markdown()
                with gr.Column():
                    loss_plot = gr.LinePlot(x="step", y="loss", title="Training loss",
                                            height=260)
            logbox = gr.Textbox(label="Recent log", lines=12, interactive=False)
            timer = gr.Timer(3.0)
            timer.tick(dashboard, None, [stats_md, loss_plot, logbox])
            demo.load(dashboard, None, [stats_md, loss_plot, logbox])

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
