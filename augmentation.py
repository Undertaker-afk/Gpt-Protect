"""
augmentation.py — text augmentation for robustness
=================================================
Augmentations make the detector robust to light paraphrasing / obfuscation
attacks that adversaries use to evade AI-text detectors.  All ops are applied
stochastically and are label-preserving.

  * typo injection        (char swap / drop / duplicate)
  * whitespace jitter      (extra / collapsed spaces)
  * casing perturbation    (random word capitalization)
  * sentence shuffling     (reorder a few sentences)
  * homoglyph substitution (latin -> lookalike, light)
  * random span deletion   (drop a short token span)
"""

from __future__ import annotations

import random
import re

_HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х"}


class Augmenter:
    def __init__(self, prob: float = 0.4, seed: int | None = None):
        self.p = prob
        self.rng = random.Random(seed)

    # --- individual ops --------------------------------------------------- #
    def _typo(self, t):
        if len(t) < 4:
            return t
        i = self.rng.randint(0, len(t) - 2)
        op = self.rng.choice(("swap", "drop", "dup"))
        if op == "swap":
            return t[:i] + t[i + 1] + t[i] + t[i + 2:]
        if op == "drop":
            return t[:i] + t[i + 1:]
        return t[:i] + t[i] + t[i:]

    def _whitespace(self, t):
        if self.rng.random() < 0.5:
            return re.sub(r" ", lambda m: "  " if self.rng.random() < 0.1 else " ", t)
        return re.sub(r"\s+", " ", t)

    def _casing(self, t):
        words = t.split()
        for i in range(len(words)):
            if self.rng.random() < 0.05:
                words[i] = words[i].capitalize() if words[i].islower() else words[i].lower()
        return " ".join(words)

    def _shuffle_sentences(self, t):
        sents = re.split(r"(?<=[.!?])\s+", t)
        if len(sents) < 3:
            return t
        i = self.rng.randint(0, len(sents) - 2)
        sents[i], sents[i + 1] = sents[i + 1], sents[i]
        return " ".join(sents)

    def _homoglyph(self, t):
        out = []
        for c in t:
            if c in _HOMOGLYPHS and self.rng.random() < 0.05:
                out.append(_HOMOGLYPHS[c])
            else:
                out.append(c)
        return "".join(out)

    def _span_delete(self, t):
        words = t.split()
        if len(words) < 8:
            return t
        i = self.rng.randint(0, len(words) - 4)
        del words[i:i + self.rng.randint(1, 3)]
        return " ".join(words)

    # --- driver ----------------------------------------------------------- #
    def __call__(self, text: str) -> str:
        if self.rng.random() > self.p:
            return text
        ops = [self._typo, self._whitespace, self._casing,
               self._shuffle_sentences, self._homoglyph, self._span_delete]
        self.rng.shuffle(ops)
        n = self.rng.randint(1, 3)
        for op in ops[:n]:
            try:
                text = op(text)
            except Exception:
                pass
        return text
