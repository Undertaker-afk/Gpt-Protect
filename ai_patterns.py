"""
ai_patterns.py — intelligent AI-text pattern detection
======================================================
Hand-engineered, model-free signals that are known to separate AI-generated
text from human text. They are used two ways:

  * fused into the neural detector as an auxiliary feature pathway
    (`model.py` projects this vector and concatenates it with the pooled
    transformer representation), giving the model strong priors even early in
    training;
  * surfaced in the UI as an explainable "why" breakdown, plus a transparent
    heuristic `ai_score` that is independent of the neural net.

Signals captured (the literature + practice):
  * burstiness / sentence-length variance   (humans vary more)
  * vocabulary richness  (type-token, hapax)
  * n-gram repetition    (AI loops phrasing)
  * word/char entropy    (perplexity proxy — AI is "too smooth")
  * punctuation habits    (AI over-uses commas/semicolons, under-uses '!')
  * contraction usage     (humans contract more)
  * stopword / function-word ratio
  * "AI-tell" lexicon     (delve, tapestry, underscore, multifaceted, …)
  * sentence-starter diversity & structural uniformity
"""

from __future__ import annotations

import math
import re
from collections import Counter

# words/phrases that show up disproportionately in LLM output
AI_TELLS = [
    "delve", "tapestry", "underscore", "moreover", "furthermore",
    "in conclusion", "it is important to note", "it's important to note",
    "a testament to", "navigating", "realm of", "leverage", "utilize",
    "facilitate", "comprehensive", "multifaceted", "nuanced", "paradigm",
    "ever-evolving", "crucial", "pivotal", "seamless", "robust", "holistic",
    "intricate", "plethora", "additionally", "in summary", "overall",
    "furthermore", "notably", "consequently", "in essence", "foster",
    "embark", "elevate", "unlock", "harness", "vibrant", "bustling",
    "meticulous", "testament", "landscape of", "world of", "when it comes to",
]

STOPWORDS = set((
    "the a an and or but if then of to in on at for with as by from into "
    "is are was were be been being this that these those it its he she they "
    "we you i not no do does did has have had will would can could should "
    "may might must about over under again further there here their our your"
).split())

_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"[.!?]+")
_CONTRACTION = re.compile(r"\b[A-Za-z]+'(t|s|re|ve|ll|d|m)\b", re.I)

# fixed feature order — DO NOT reorder (the model depends on dimension layout)
FEATURE_NAMES = [
    "burstiness", "uniformity", "mean_sent_len", "std_sent_len",
    "type_token_ratio", "hapax_ratio", "rep_bigram", "rep_trigram",
    "word_entropy", "char_entropy", "avg_word_len", "std_word_len",
    "comma_ratio", "semicolon_ratio", "punct_ratio", "exclaim_per_sent",
    "question_per_sent", "uppercase_ratio", "digit_ratio", "contraction_ratio",
    "stopword_ratio", "ai_tell_per_100w", "starter_diversity", "long_word_ratio",
]
N_FEATURES = len(FEATURE_NAMES)


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def extract_features(text: str) -> dict:
    text = text or ""
    words = _WORD.findall(text.lower())
    n = len(words)
    chars = max(len(text), 1)
    sents = [s for s in _SENT.split(text) if s.strip()]
    sent_lens = [len(_WORD.findall(s)) for s in sents] or [0]
    ns = len(sent_lens)

    wc = Counter(words)
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    bg, tg = Counter(bigrams), Counter(trigrams)
    cc = Counter(text.lower())

    mean_sl = sum(sent_lens) / ns
    std_sl = (sum((x - mean_sl) ** 2 for x in sent_lens) / ns) ** 0.5
    avg_wl = (sum(len(w) for w in words) / n) if n else 0.0
    std_wl = ((sum((len(w) - avg_wl) ** 2 for w in words) / n) ** 0.5) if n else 0.0

    starters = []
    for s in sents:
        m = _WORD.findall(s.lower())
        if m:
            starters.append(m[0])

    f = {
        "burstiness": (std_sl / mean_sl) if mean_sl > 0 else 0.0,
        "uniformity": 1.0 / (1.0 + std_sl),
        "mean_sent_len": mean_sl,
        "std_sent_len": std_sl,
        "type_token_ratio": (len(wc) / n) if n else 0.0,
        "hapax_ratio": (sum(1 for c in wc.values() if c == 1) / n) if n else 0.0,
        "rep_bigram": (1 - len(bg) / len(bigrams)) if bigrams else 0.0,
        "rep_trigram": (1 - len(tg) / len(trigrams)) if trigrams else 0.0,
        "word_entropy": _entropy(wc),
        "char_entropy": _entropy(cc),
        "avg_word_len": avg_wl,
        "std_word_len": std_wl,
        "comma_ratio": text.count(",") / chars,
        "semicolon_ratio": text.count(";") / chars,
        "punct_ratio": sum(c in ",.;:!?" for c in text) / chars,
        "exclaim_per_sent": text.count("!") / ns,
        "question_per_sent": text.count("?") / ns,
        "uppercase_ratio": sum(c.isupper() for c in text) / chars,
        "digit_ratio": sum(c.isdigit() for c in text) / chars,
        "contraction_ratio": (len(_CONTRACTION.findall(text)) / n) if n else 0.0,
        "stopword_ratio": (sum(1 for w in words if w in STOPWORDS) / n) if n else 0.0,
        "ai_tell_per_100w": (sum(text.lower().count(p) for p in AI_TELLS)
                             / (n / 100.0 + 1e-6)) if n else 0.0,
        "starter_diversity": (len(set(starters)) / len(starters)) if starters else 0.0,
        "long_word_ratio": (sum(1 for w in words if len(w) >= 8) / n) if n else 0.0,
    }
    return f


def feature_vector(text: str):
    f = extract_features(text)
    return [float(f[k]) for k in FEATURE_NAMES]


def heuristic_ai_score(text: str) -> float:
    """Transparent 0..1 'AI-likeness' score independent of the neural net.
    Combines the most discriminative signals with hand-set weights."""
    f = extract_features(text)
    if f["mean_sent_len"] == 0:
        return 0.5
    # each term pushes toward AI (1) or human (0)
    z = 0.0
    z += 1.6 * (1.0 - min(f["burstiness"] / 0.6, 1.0))        # low burstiness -> AI
    z += 1.2 * min(f["ai_tell_per_100w"] / 2.0, 1.0)          # AI-tell words
    z += 0.9 * (1.0 - min(f["contraction_ratio"] / 0.03, 1.0))  # few contractions -> AI
    z += 0.8 * min(f["semicolon_ratio"] / 0.004, 1.0)        # semicolons -> AI
    z += 0.7 * (1.0 - min(f["hapax_ratio"] / 0.5, 1.0))      # low novelty -> AI
    z += 0.6 * max(f["rep_trigram"], f["rep_bigram"])        # repetition -> AI
    z += 0.6 * (1.0 - min(f["exclaim_per_sent"] / 0.15, 1.0))  # no '!' -> AI
    z += 0.5 * f["uniformity"]                                # uniform structure -> AI
    z -= 0.8 * min(f["starter_diversity"], 1.0)              # diverse starts -> human
    bias = -1.7
    return float(1.0 / (1.0 + math.exp(-(z + bias))))


def top_signals(text: str, k: int = 6):
    """Return the k features contributing most to an AI verdict (for the UI)."""
    f = extract_features(text)
    contrib = {
        "low burstiness": 1.0 - min(f["burstiness"] / 0.6, 1.0),
        "AI-tell vocabulary": min(f["ai_tell_per_100w"] / 2.0, 1.0),
        "few contractions": 1.0 - min(f["contraction_ratio"] / 0.03, 1.0),
        "semicolon habit": min(f["semicolon_ratio"] / 0.004, 1.0),
        "low word novelty": 1.0 - min(f["hapax_ratio"] / 0.5, 1.0),
        "phrase repetition": max(f["rep_trigram"], f["rep_bigram"]),
        "uniform sentences": f["uniformity"],
        "no exclamations": 1.0 - min(f["exclaim_per_sent"] / 0.15, 1.0),
    }
    ranked = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{"signal": s, "strength": round(v, 3)} for s, v in ranked]


if __name__ == "__main__":
    ai = ("Moreover, it is important to note that this multifaceted paradigm "
          "underscores a comprehensive tapestry; furthermore, we must delve "
          "into the nuanced realm of robust, seamless solutions.")
    hu = ("ok so i went to the store today and honestly? it was a mess lol. "
          "couldn't find the milk anywhere, asked a guy, he didn't know either.")
    for name, t in (("AI", ai), ("HUMAN", hu)):
        print(name, "heuristic_ai_score=", round(heuristic_ai_score(t), 3))
        print("  top:", [s["signal"] for s in top_signals(t, 3)])
    print("N_FEATURES =", N_FEATURES)
