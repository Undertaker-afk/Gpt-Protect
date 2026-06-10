"""
preprocessor.py — text normalization
====================================
Light, reversible-ish cleaning applied before tokenization.  We deliberately
keep stylometric signal (casing, punctuation density, spacing habits) because
those are exactly the cues that separate human from AI text — so cleaning is
conservative: normalize unicode/whitespace, strip control chars, optionally
cap length.  No lowercasing, no punctuation removal by default.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")
_ZEROWIDTH = re.compile(r"[​-‏﻿]")


class Preprocessor:
    def __init__(self, normalize_unicode: bool = True,
                 collapse_space: bool = True,
                 max_chars: int = 20000):
        self.normalize_unicode = normalize_unicode
        self.collapse_space = collapse_space
        self.max_chars = max_chars

    def __call__(self, text: str) -> str:
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)
        text = _ZEROWIDTH.sub("", text)
        text = _CONTROL.sub("", text)
        if self.collapse_space:
            text = _MULTISPACE.sub(" ", text)
            text = _MULTINEWLINE.sub("\n\n", text)
        text = text.strip()
        if self.max_chars and len(text) > self.max_chars:
            text = text[: self.max_chars]
        return text

    def batch(self, texts):
        return [self(t) for t in texts]


# simple stylometric features (handy for the Gradio explainability panel)
def stylometric_features(text: str) -> dict:
    words = text.split()
    n = max(len(words), 1)
    sents = re.split(r"[.!?]+", text)
    sents = [s for s in sents if s.strip()]
    uniq = len(set(w.lower() for w in words))
    return {
        "n_chars": len(text),
        "n_words": len(words),
        "n_sentences": len(sents),
        "avg_word_len": round(sum(len(w) for w in words) / n, 3),
        "avg_sentence_len": round(n / max(len(sents), 1), 3),
        "type_token_ratio": round(uniq / n, 4),
        "punct_ratio": round(sum(c in ",.;:!?" for c in text) / max(len(text), 1), 4),
        "uppercase_ratio": round(sum(c.isupper() for c in text) / max(len(text), 1), 4),
    }
