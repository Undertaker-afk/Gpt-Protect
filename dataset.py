"""
dataset.py — unified human/AI text dataset
==========================================
Loads and harmonizes the two HF sources into a single labeled corpus:

  * alex-kudryashov/dlr-hw-2-human-ai-texts  ->  {text, label}  (0=human, 1=ai)
  * nbroad/basic_text_dataset                ->  {text}; treated as human (0)

Label convention:  0 = HUMAN,  1 = AI-GENERATED.

Returns plain python dicts; tokenization happens in data_loader.py so the
augmenter can mutate raw text first.
"""

from __future__ import annotations

import random
from typing import Optional

from datasets import load_dataset, concatenate_datasets, Dataset

from preprocessor import Preprocessor

HUMAN, AI = 0, 1

LABEL_MAP = {
    "0": HUMAN, "1": AI, 0: HUMAN, 1: AI,
    "human": HUMAN, "ai": AI, "machine": AI, "generated": AI,
    "real": HUMAN, "fake": AI,
}


def _norm_label(v) -> int:
    if isinstance(v, str):
        v = v.strip().lower()
    return LABEL_MAP.get(v, int(v) if str(v).isdigit() else HUMAN)


def load_human_ai(max_samples: Optional[int] = None,
                  seed: int = 1234,
                  preprocess: bool = True) -> Dataset:
    pre = Preprocessor() if preprocess else None
    parts = []

    # 1) labeled human/AI dataset --------------------------------------- #
    try:
        d1 = load_dataset("alex-kudryashov/dlr-hw-2-human-ai-texts", split="train")
        d1 = d1.map(lambda e: {"text": e["text"], "label": _norm_label(e["label"])})
        parts.append(d1.select_columns(["text", "label"]))
        print(f"[dataset] dlr-hw-2: {len(d1)} rows")
    except Exception as e:
        print("[dataset] WARN could not load dlr-hw-2:", repr(e)[:160])

    # 2) basic human text corpus (all label = human) -------------------- #
    try:
        d2 = load_dataset("nbroad/basic_text_dataset")
        from datasets import concatenate_datasets as _cat
        d2 = _cat([d2[s] for s in d2.keys()])
        d2 = d2.map(lambda e: {"text": e["text"], "label": HUMAN})
        parts.append(d2.select_columns(["text", "label"]))
        print(f"[dataset] basic_text: {len(d2)} rows (label=human)")
    except Exception as e:
        print("[dataset] WARN could not load basic_text:", repr(e)[:160])

    if not parts:
        raise RuntimeError("No datasets could be loaded (check network/HF).")

    ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

    # clean + drop empties
    if pre is not None:
        ds = ds.map(lambda e: {"text": pre(e["text"]), "label": int(e["label"])})
    ds = ds.filter(lambda e: e["text"] is not None and len(e["text"].strip()) > 0)

    ds = ds.shuffle(seed=seed)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    # quick label balance report
    labels = ds["label"]
    n_ai = sum(1 for x in labels if x == AI)
    print(f"[dataset] total={len(ds)}  human={len(ds)-n_ai}  ai={n_ai}")
    return ds


def train_val_split(ds: Dataset, val_fraction: float = 0.05, seed: int = 1234):
    split = ds.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


if __name__ == "__main__":
    ds = load_human_ai(max_samples=2000)
    tr, va = train_val_split(ds)
    print("train", len(tr), "val", len(va))
    print(tr[0])
