"""
data_loader.py — tokenization + batching
========================================
Builds torch DataLoaders from the unified dataset.  Raw text is augmented
(train split only) *before* tokenization so augmentation operates on natural
text.  Uses a HF fast tokenizer (gpt2 by default; falls back to a byte-level
whitespace tokenizer if the hub is unreachable).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader

from augmentation import Augmenter
from ai_patterns import feature_vector, N_FEATURES


def get_tokenizer(name: str = "gpt2"):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.unk_token or "[PAD]"
        return tok
    except Exception as e:
        print("[data_loader] tokenizer fallback (byte-level):", repr(e)[:120])
        return _ByteTokenizer()


class _ByteTokenizer:
    """Minimal offline fallback tokenizer (UTF-8 bytes -> ids 0..255)."""
    pad_token_id = 256
    vocab_size = 257

    def __call__(self, text, truncation=True, max_length=512, **kw):
        ids = list(text.encode("utf-8", "ignore"))[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class TextClsDataset(TorchDataset):
    def __init__(self, hf_ds, tokenizer, max_len=512, augment=False,
                 augment_prob=0.4, seed=0):
        self.ds = hf_ds
        self.tok = tokenizer
        self.max_len = max_len
        self.aug = Augmenter(augment_prob, seed) if augment else None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        text = row["text"]
        if self.aug is not None:
            text = self.aug(text)
        enc = self.tok(text, truncation=True, max_length=self.max_len)
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "pattern_feats": torch.tensor(feature_vector(text), dtype=torch.float),
        }


def make_collate(pad_id: int):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ids, masks, labels = [], [], []
        for b in batch:
            n = len(b["input_ids"])
            pad = maxlen - n
            ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id)]))
            masks.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
            labels.append(b["label"])
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels),
            "pattern_feats": torch.stack([b["pattern_feats"] for b in batch]),
        }
    return collate


def build_loaders(train_ds, val_ds, tokenizer, cfg):
    pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
    collate = make_collate(pad_id)
    tr = TextClsDataset(train_ds, tokenizer, cfg.max_seq_len,
                        augment=cfg.augment, augment_prob=cfg.augment_prob,
                        seed=cfg.seed)
    va = TextClsDataset(val_ds, tokenizer, cfg.max_seq_len, augment=False)
    train_loader = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate,
                              drop_last=True)
    val_loader = DataLoader(va, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=collate)
    return train_loader, val_loader, pad_id
