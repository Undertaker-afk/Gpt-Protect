"""
dataset.py — unified human/AI text dataset
==========================================
Harmonizes many public human-vs-AI corpora into one labeled stream.

Label convention:  0 = HUMAN,  1 = AI-GENERATED.

Each source is described by a small spec.  Three shapes are supported:
  * labeled : a text column + a 0/1 label column
  * human   : text only, every row is human (label 0)
  * paired  : two columns per row (human_text, ai_text) -> emitted as 2 rows

Sources are streamed with a per-dataset cap so the whole thing stays light
enough for a 16 GB Space (some of these datasets have 100k+ rows).
"""

from __future__ import annotations

from itertools import islice
from typing import Optional

from datasets import load_dataset, Dataset

from preprocessor import Preprocessor

HUMAN, AI = 0, 1

LABEL_MAP = {
    "0": HUMAN, "1": AI, 0: HUMAN, 1: AI, 0.0: HUMAN, 1.0: AI,
    "human": HUMAN, "ai": AI, "machine": AI, "generated": AI,
    "real": HUMAN, "fake": AI, "gpt": AI, "llm": AI,
    "human-written": HUMAN, "ai-generated": AI,
}

# ---- registry of sources -------------------------------------------------- #
# kind: "labeled" | "human" | "paired"
DATASET_SPECS = [
    {"name": "alex-kudryashov/dlr-hw-2-human-ai-texts", "kind": "labeled",
     "text": "text", "label": "label"},
    {"name": "nbroad/basic_text_dataset", "kind": "human", "text": "text"},
    {"name": "mehddii/ai-text-detector-v2", "kind": "labeled",
     "text": "text", "label": "label"},
    {"name": "AlekseyKorshuk/ai-text-classification", "kind": "labeled",
     "text": "text", "label": "target"},
    {"name": "ziq/ai-generated-text-classification", "kind": "labeled",
     "text": "text", "label": "generated"},
    {"name": "NabeelShar/ai_and_human_text", "kind": "labeled",
     "text": "text", "label": "generated"},
    {"name": "akoukas/AITextDetectionDataset", "kind": "labeled",
     "text": "text", "label": "label"},
    {"name": "dmitva/human_ai_generated_text", "kind": "paired",
     "human": "human_text", "ai": "ai_text"},
]


def _norm_label(v) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return AI if v else HUMAN
    if isinstance(v, str):
        v2 = v.strip().lower()
        if v2 in LABEL_MAP:
            return LABEL_MAP[v2]
        if v2.isdigit():
            return AI if int(v2) >= 1 else HUMAN
        return None
    if isinstance(v, (int, float)):
        return AI if v >= 1 else HUMAN
    return LABEL_MAP.get(v)


def _iter_spec(spec, per_dataset):
    """Yield (text, label) pairs for one source, streaming + capped."""
    name = spec["name"]
    ds = load_dataset(name, streaming=True)
    splits = list(ds.keys())
    kind = spec["kind"]
    # spread the cap across splits so we don't take only "train"
    per_split = max(1, per_dataset // max(1, len(splits)))
    for split in splits:
        for ex in islice(ds[split], per_split):
            if kind == "human":
                t = ex.get(spec["text"])
                if t:
                    yield t, HUMAN
            elif kind == "paired":
                h, a = ex.get(spec["human"]), ex.get(spec["ai"])
                if h:
                    yield h, HUMAN
                if a:
                    yield a, AI
            else:  # labeled
                t = ex.get(spec["text"])
                lbl = _norm_label(ex.get(spec["label"]))
                if t and lbl is not None:
                    yield t, lbl


def load_human_ai(max_samples: Optional[int] = None,
                  per_dataset: int = 8000,
                  seed: int = 1234,
                  preprocess: bool = True,
                  specs=None) -> Dataset:
    pre = Preprocessor() if preprocess else None
    specs = specs if specs is not None else DATASET_SPECS

    texts, labels = [], []
    counts = {}
    for spec in specs:
        name = spec["name"]
        got = 0
        try:
            for t, lbl in _iter_spec(spec, per_dataset):
                if pre is not None:
                    t = pre(t)
                if not t or len(t.strip()) == 0:
                    continue
                texts.append(t)
                labels.append(int(lbl))
                got += 1
            counts[name] = got
            print(f"[dataset] {name}: {got} rows")
        except Exception as e:
            counts[name] = f"ERR {repr(e)[:80]}"
            print(f"[dataset] WARN {name}: {repr(e)[:120]}")

    if not texts:
        raise RuntimeError("No datasets could be loaded (check network/HF).")

    ds = Dataset.from_dict({"text": texts, "label": labels})
    ds = ds.shuffle(seed=seed)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    n_ai = sum(1 for x in ds["label"] if x == AI)
    print(f"[dataset] TOTAL={len(ds)}  human={len(ds)-n_ai}  ai={n_ai}  "
          f"sources={len([c for c in counts.values() if isinstance(c,int) and c>0])}")
    return ds


def train_val_split(ds: Dataset, val_fraction: float = 0.05, seed: int = 1234):
    split = ds.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


if __name__ == "__main__":
    ds = load_human_ai(max_samples=3000, per_dataset=600)
    tr, va = train_val_split(ds)
    print("train", len(tr), "val", len(va))
    print(tr[0])
