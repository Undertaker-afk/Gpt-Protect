"""
train.py — training driver
=========================
Trains the AI-text detector end-to-end.

Usage:
    python train.py --preset tiny   --max-samples 4000 --epochs 1
    python train.py --preset 0.4b   --max-samples 50000 --epochs 3
    python train.py --preset 5b     ...                 (needs GPUs)

Features: Muon+AdamW optimizer, cosine LR w/ warmup, grad accumulation/clip,
periodic eval (acc / F1 / AUC), checkpointing, JSON metric logs to /tmp/logs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from config import preset, TrainConfig
from dataset import load_human_ai, train_val_split
from data_loader import get_tokenizer, build_loaders
from model import build_model
from muon import build_optimizer


def set_seed(s):
    import random
    import numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return 0.5 * base * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, device, max_batches=50):
    model.eval()
    ys, ps, correct, n, loss_sum = [], [], 0, 0, 0.0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(ids, mask, labels)
        logits = out["logits"]
        if out["loss"] is not None:
            loss_sum += out["loss"].item()
        pred = logits.argmax(-1)
        correct += (pred == labels).sum().item()
        n += labels.numel()
        prob = torch.softmax(logits, -1)[:, 1]
        ys += labels.tolist(); ps += prob.tolist()
    acc = correct / max(n, 1)
    f1 = auc = float("nan")
    try:
        from sklearn.metrics import f1_score, roc_auc_score
        preds = [1 if p >= 0.5 else 0 for p in ps]
        f1 = f1_score(ys, preds, zero_division=0)
        if len(set(ys)) > 1:
            auc = roc_auc_score(ys, ps)
    except Exception:
        pass
    model.train()
    return {"acc": acc, "f1": f1, "auc": auc, "val_loss": loss_sum / max(min(max_batches, n), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--optimizer", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--log-file", default="/tmp/logs/train.jsonl")
    args = ap.parse_args()

    mcfg = preset(args.preset)
    tcfg = TrainConfig()
    if args.max_samples is not None: tcfg.max_samples = args.max_samples
    if args.epochs is not None: tcfg.epochs = args.epochs
    if args.batch_size is not None: tcfg.batch_size = args.batch_size
    if args.max_seq_len is not None: tcfg.max_seq_len = args.max_seq_len
    if args.optimizer is not None: tcfg.optimizer = args.optimizer
    if args.num_workers is not None: tcfg.num_workers = args.num_workers
    if args.output_dir is not None: tcfg.output_dir = args.output_dir
    tcfg.max_seq_len = min(tcfg.max_seq_len, mcfg.max_seq_len)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tcfg.device = device
    set_seed(tcfg.seed)
    os.makedirs(tcfg.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    if device == "cpu":
        torch.set_num_threads(min(16, os.cpu_count() or 8))

    print("="*70)
    print(f"preset={args.preset}  device={device}")
    print(f"est params ~{mcfg.n_params_estimate()/1e9:.3f}B")
    print(tcfg.to_json())

    # --- data ---------------------------------------------------------- #
    tokenizer = get_tokenizer(tcfg.tokenizer_name)
    if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size:
        mcfg.vocab_size = max(tokenizer.vocab_size, getattr(tokenizer, "vocab_size", 0))
        # ensure embeddings cover all ids incl. added pad token
        mcfg.vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else mcfg.vocab_size
    ds = load_human_ai(max_samples=tcfg.max_samples, seed=tcfg.seed)
    tr_ds, va_ds = train_val_split(ds, tcfg.val_fraction, tcfg.seed)
    train_loader, val_loader, pad_id = build_loaders(tr_ds, va_ds, tokenizer, tcfg)

    # --- model --------------------------------------------------------- #
    model = build_model(mcfg).to(device)
    model.label_smoothing = tcfg.label_smoothing
    n_params = model.num_parameters()
    print(f"[model] actual params = {n_params/1e6:.1f}M ({n_params/1e9:.3f}B)")

    opt = build_optimizer(model, tcfg)
    steps_per_epoch = max(len(train_loader) // tcfg.grad_accum, 1)
    total_steps = steps_per_epoch * tcfg.epochs
    warmup = int(total_steps * tcfg.warmup_ratio)

    # --- train loop ---------------------------------------------------- #
    logf = open(args.log_file, "a")
    def log(rec):
        rec["t"] = round(time.time(), 2)
        logf.write(json.dumps(rec) + "\n"); logf.flush()

    model.train()
    gstep = 0
    t0 = time.time()
    best_acc = 0.0
    for epoch in range(tcfg.epochs):
        opt.zero_grad()
        for it, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            out = model(ids, mask, labels)
            loss = out["loss"] / tcfg.grad_accum
            loss.backward()

            if (it + 1) % tcfg.grad_accum == 0:
                lr = cosine_lr(gstep, total_steps, warmup, tcfg.lr)
                mlr = cosine_lr(gstep, total_steps, warmup, tcfg.muon_lr)
                for g in opt.param_groups:
                    g["lr"] = mlr if g.get("ns_steps") else lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                opt.step(); opt.zero_grad()
                gstep += 1

                if gstep % tcfg.log_every == 0:
                    sps = gstep / (time.time() - t0)
                    rec = {"step": gstep, "epoch": epoch,
                           "loss": round(out["loss"].item(), 4),
                           "lr": round(lr, 6), "steps_per_s": round(sps, 3)}
                    print(f"step {gstep}/{total_steps} ep{epoch} "
                          f"loss={rec['loss']} lr={lr:.2e} {sps:.2f} it/s")
                    log(rec)

                if gstep % tcfg.eval_every == 0:
                    m = evaluate(model, val_loader, device)
                    print(f"  [eval] acc={m['acc']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")
                    log({"eval": True, "step": gstep, **m})
                    if m["acc"] > best_acc:
                        best_acc = m["acc"]
                        save_ckpt(model, mcfg, tcfg, os.path.join(tcfg.output_dir, "best.pt"))

                if gstep % tcfg.save_every == 0:
                    save_ckpt(model, mcfg, tcfg, os.path.join(tcfg.output_dir, "last.pt"))

        # end of epoch eval + save
        m = evaluate(model, val_loader, device)
        print(f"[epoch {epoch} done] acc={m['acc']:.4f} f1={m['f1']:.4f} auc={m['auc']:.4f}")
        log({"eval": True, "epoch_end": epoch, "step": gstep, **m})
        save_ckpt(model, mcfg, tcfg, os.path.join(tcfg.output_dir, f"epoch{epoch}.pt"))
        save_ckpt(model, mcfg, tcfg, os.path.join(tcfg.output_dir, "last.pt"))

    print(f"done. best_acc={best_acc:.4f}  elapsed={time.time()-t0:.0f}s")
    logf.close()


def save_ckpt(model, mcfg, tcfg, path):
    from dataclasses import asdict
    torch.save({"model": model.state_dict(),
                "model_config": asdict(mcfg),
                "train_config": asdict(tcfg)}, path)
    print(f"  [ckpt] saved {path}")


if __name__ == "__main__":
    main()
