"""
ai_patterns.py — intelligent AI-text statistics engine
=====================================================
Pure-Python (no torch) feature engineering that separates AI-generated text
from human text.  Used three ways:

  * fused into the neural detector as an auxiliary feature pathway
    (`model.py` projects this vector + concatenates with the transformer repr);
  * a transparent heuristic `ai_score` shown next to the neural prediction;
  * a full, grouped breakdown in the UI ("what was detected").

Feature families implemented here (numbers refer to the project feature list):
  4  sentence-length distribution moments (var / skew / kurtosis)
  5  function-word distribution (stylometry, Burrows-style)
  6  lightweight POS distribution (noun/verb/adj/adv/pron/det/prep)
  8  multi-scale repetition (bi/tri/4/5-gram + max n-gram repeat)
  9  length-robust lexical diversity (MTLD, MATTR)
  10 readability (Flesch ease, FK grade, Gunning Fog, SMOG)
  11 cross-sentence sentiment mean/variance + subjectivity
  13 named-entity / numeric specificity (proper nouns, numbers)
  15 emoji / slang / abbreviation / contraction rates
  16 spelling-error rate (pyspellchecker if available, else heuristic)
  17 markdown / structural regularity (lists, headers, bold, code, urls)
  18 compression ratio (gzip), char entropy, Zipf-law deviation

Perplexity / DetectGPT signals (#1, #2) live in `perplexity.py` (needs a
reference LM) and are appended as a fixed zero-able block so the model's input
dimension is stable whether or not that module is enabled (#29).
"""

from __future__ import annotations

import gzip
import math
import os
import re
from collections import Counter

# --------------------------------------------------------------------------- #
#  Lexicons
# --------------------------------------------------------------------------- #
AI_TELLS = [
    "delve", "tapestry", "underscore", "moreover", "furthermore",
    "in conclusion", "it is important to note", "it's important to note",
    "a testament to", "navigating", "realm of", "leverage", "utilize",
    "facilitate", "comprehensive", "multifaceted", "nuanced", "paradigm",
    "ever-evolving", "crucial", "pivotal", "seamless", "robust", "holistic",
    "intricate", "plethora", "additionally", "in summary", "overall",
    "notably", "consequently", "in essence", "foster", "embark", "elevate",
    "unlock", "harness", "vibrant", "bustling", "meticulous", "testament",
    "landscape of", "world of", "when it comes to", "it's worth noting",
    "dive into", "navigate the", "boasts", "showcasing", "underscores",
]

DISCOURSE_MARKERS = [
    "however", "therefore", "moreover", "furthermore", "consequently",
    "nevertheless", "nonetheless", "additionally", "thus", "hence",
    "accordingly", "subsequently", "meanwhile", "conversely", "indeed",
    "notably", "specifically", "ultimately", "importantly", "similarly",
]

STOPWORDS = set((
    "the a an and or but if then of to in on at for with as by from into "
    "is are was were be been being this that these those it its he she they "
    "we you i not no do does did has have had will would can could should "
    "may might must about over under again further there here their our your"
).split())

PRONOUNS = set("i you he she it we they me him her us them my your his its our "
               "their mine yours hers ours theirs myself yourself himself "
               "herself itself ourselves themselves who whom whose".split())
DETERMINERS = set("the a an this that these those my your his her its our their "
                  "some any each every no all both either neither".split())
PREPOSITIONS = set("in on at by for with about against between into through "
                   "during before after above below to from up down of off "
                   "over under again".split())
CONJUNCTIONS = set("and or but nor so yet for because although though while "
                   "whereas since unless until".split())
AUXILIARIES = set("is am are was were be been being have has had do does did "
                  "will would shall should may might must can could".split())
FUNCTION_WORDS = (STOPWORDS | PRONOUNS | DETERMINERS | PREPOSITIONS
                  | CONJUNCTIONS | AUXILIARIES)

POS_POSITIVE = set("good great excellent love wonderful amazing happy best "
                   "beautiful nice awesome fantastic perfect enjoy positive "
                   "brilliant delightful glad pleased fun helpful".split())
POS_NEGATIVE = set("bad terrible awful hate horrible sad worst ugly nasty "
                   "angry disappointing poor negative boring annoying broken "
                   "wrong fail painful difficult ".split())

SLANG = set("lol lmao rofl idk btw tbh imo imho omg wtf smh ngl fr af brb gtg "
            "ikr nvm tldr afaik bff yolo lowkey highkey deadass bruh bro sis "
            "gonna wanna gotta kinda sorta dunno gimme lemme cuz cause "
            "u ur ya yall y'all em tho thru".split())

CONTRACTION_RE = re.compile(r"\b[A-Za-z]+'(t|s|re|ve|ll|d|m)\b", re.I)
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F02F"
    "\U0001F1E6-\U0001F1FF←-⇿✀-➿]")
EMOTICON_RE = re.compile(r"[:;=8xX][-^']?[)(/\\|DPpoO3]")
WORD_RE = re.compile(r"[A-Za-z']+")
TOKEN_RE = re.compile(r"\S+")
SENT_RE = re.compile(r"[.!?]+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
LIST_RE = re.compile(r"(?m)^\s*([-*•]|\d+[.)])\s+")
HEADER_RE = re.compile(r"(?m)^\s*(#{1,6}\s+\S|[A-Z][^\n]{0,60}:)\s*$")
BOLD_RE = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
CODE_RE = re.compile(r"`[^`]+`|```")

# --------------------------------------------------------------------------- #
#  Optional spellchecker (graceful fallback)
# --------------------------------------------------------------------------- #
_SPELL = None
_SPELL_TRIED = False


def _spell():
    global _SPELL, _SPELL_TRIED
    if _SPELL_TRIED:
        return _SPELL
    _SPELL_TRIED = True
    if os.environ.get("USE_SPELLCHECK", "1") != "1":
        return None
    try:
        from spellchecker import SpellChecker
        _SPELL = SpellChecker(distance=1)
    except Exception:
        _SPELL = None
    return _SPELL


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def _moments(xs):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    std = var ** 0.5
    if std < 1e-9:
        return mean, std, 0.0, 0.0
    skew = sum(((x - mean) / std) ** 3 for x in xs) / n
    kurt = sum(((x - mean) / std) ** 4 for x in xs) / n - 3.0
    return mean, std, skew, kurt


def _syllables(word: str) -> int:
    word = word.lower()
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and not word.endswith(("le", "ie", "ee")):
        n = max(1, n - 1)
    return max(1, n)


def _mtld(words, threshold=0.72):
    """Measure of Textual Lexical Diversity (length-robust)."""
    if len(words) < 10:
        return float(len(set(words))) / max(len(words), 1) * 100

    def _one_pass(seq):
        factors, types, tokens = 0.0, set(), 0
        for w in seq:
            tokens += 1
            types.add(w)
            if len(types) / tokens <= threshold:
                factors += 1
                types, tokens = set(), 0
        if tokens > 0:
            ttr = len(types) / tokens
            factors += (1 - ttr) / (1 - threshold) if threshold < 1 else 0
        return len(seq) / factors if factors > 0 else len(seq)

    return (_one_pass(words) + _one_pass(words[::-1])) / 2.0


def _mattr(words, window=50):
    """Moving-Average Type-Token Ratio."""
    n = len(words)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(words)) / n
    ratios = []
    for i in range(n - window + 1):
        ratios.append(len(set(words[i:i + window])) / window)
    return sum(ratios) / len(ratios)


def _rep(words, k):
    grams = list(zip(*[words[i:] for i in range(k)]))
    if not grams:
        return 0.0, 0
    c = Counter(grams)
    rep = 1 - len(c) / len(grams)
    return rep, max(c.values())


def _zipf_deviation(wc: Counter) -> float:
    """MSE of log-freq vs an ideal Zipf line (slope -1) in log-log space."""
    freqs = sorted(wc.values(), reverse=True)
    if len(freqs) < 5:
        return 0.0
    f0 = freqs[0]
    err = 0.0
    for rank, f in enumerate(freqs, start=1):
        expected = f0 / rank
        err += (math.log(f + 1) - math.log(expected + 1)) ** 2
    return err / len(freqs)


def _light_pos(words):
    """Lightweight rule/suffix POS tagger -> ratio dict."""
    counts = Counter()
    for w in words:
        lw = w.lower()
        if lw in PRONOUNS:
            counts["pron"] += 1
        elif lw in DETERMINERS:
            counts["det"] += 1
        elif lw in PREPOSITIONS:
            counts["prep"] += 1
        elif lw in CONJUNCTIONS:
            counts["conj"] += 1
        elif lw in AUXILIARIES:
            counts["verb"] += 1
        elif lw.endswith("ly"):
            counts["adv"] += 1
        elif lw.endswith(("ing", "ed", "ize", "ise", "ate", "ify")):
            counts["verb"] += 1
        elif lw.endswith(("ous", "ful", "ive", "al", "ic", "able", "ible",
                          "ant", "ent")):
            counts["adj"] += 1
        elif lw.endswith(("tion", "ment", "ness", "ity", "ship", "ism",
                          "er", "or", "ist")):
            counts["noun"] += 1
        else:
            counts["noun"] += 1
    return counts


# --------------------------------------------------------------------------- #
#  Core feature extraction
# --------------------------------------------------------------------------- #
def extract_features(text: str) -> dict:
    text = text or ""
    chars = max(len(text), 1)
    words = WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    n = len(words)
    nn = max(n, 1)
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    sent_lens = [len(WORD_RE.findall(s)) for s in sents] or [0]
    ns = max(len(sent_lens), 1)
    wc = Counter(lower)
    cc = Counter(text.lower())

    m_sl, s_sl, sk_sl, ku_sl = _moments(sent_lens)
    avg_wl = sum(len(w) for w in words) / nn
    _, s_wl, _, _ = _moments([len(w) for w in words]) if words else (0, 0, 0, 0)

    rb, _ = _rep(lower, 2)
    rt, mx3 = _rep(lower, 3)
    r4, _ = _rep(lower, 4)
    r5, _ = _rep(lower, 5)

    pos = _light_pos(words)
    posn = max(sum(pos.values()), 1)

    # readability
    syl = sum(_syllables(w) for w in words)
    poly = sum(1 for w in words if _syllables(w) >= 3)
    asl = n / ns                                  # avg sentence length
    asw = syl / nn                                # avg syllables per word
    flesch = 206.835 - 1.015 * asl - 84.6 * asw
    fk_grade = 0.39 * asl + 11.8 * asw - 15.59
    fog = 0.4 * (asl + 100 * poly / nn)
    smog = 1.043 * math.sqrt(poly * (30 / ns)) + 3.1291

    # sentiment per sentence
    s_scores, subj = [], 0
    for s in sents:
        sw = [w.lower() for w in WORD_RE.findall(s)]
        p = sum(1 for w in sw if w in POS_POSITIVE)
        ng = sum(1 for w in sw if w in POS_NEGATIVE)
        if p + ng > 0:
            s_scores.append((p - ng) / (p + ng))
        subj += p + ng
    sent_mean, sent_std, _, _ = _moments(s_scores) if s_scores else (0, 0, 0, 0)

    # specificity / NER-ish
    proper = sum(1 for i, w in enumerate(words)
                 if w[:1].isupper() and not (i == 0 or words[i - 1] in ".!?"))
    numbers = len(NUMBER_RE.findall(text))

    # casual signals
    emojis = len(EMOJI_RE.findall(text)) + len(EMOTICON_RE.findall(text))
    slang = sum(1 for w in lower if w in SLANG)
    contractions = len(CONTRACTION_RE.findall(text))

    # spelling
    misspell = _misspelling_rate(words)

    # structure / markdown
    lists = len(LIST_RE.findall(text))
    headers = len(HEADER_RE.findall(text))
    bold = len(BOLD_RE.findall(text))
    code = len(CODE_RE.findall(text))
    urls = len(URL_RE.findall(text))

    # compression / entropy
    try:
        raw = text.encode("utf-8")
        gz = len(gzip.compress(raw, 6))
        gzip_ratio = gz / max(len(raw), 1)
    except Exception:
        gzip_ratio = 1.0

    starters = []
    for s in sents:
        m = WORD_RE.findall(s.lower())
        if m:
            starters.append(m[0])

    f = {
        # length / burstiness (#4)
        "mean_sent_len": asl,
        "std_sent_len": s_sl,
        "skew_sent_len": sk_sl,
        "kurt_sent_len": ku_sl,
        "burstiness": (s_sl / m_sl) if m_sl > 0 else 0.0,
        "uniformity": 1.0 / (1.0 + s_sl),
        "avg_word_len": avg_wl,
        "std_word_len": s_wl,
        "long_word_ratio": sum(1 for w in words if len(w) >= 8) / nn,
        # diversity (#9)
        "type_token_ratio": len(wc) / nn,
        "hapax_ratio": sum(1 for c in wc.values() if c == 1) / nn,
        "mtld": _mtld(lower) / 100.0,
        "mattr": _mattr(lower),
        # repetition (#8)
        "rep_bigram": rb, "rep_trigram": rt, "rep_4gram": r4, "rep_5gram": r5,
        "max_trigram_repeat": mx3 / nn,
        # entropy / compression (#18)
        "word_entropy": _entropy(wc),
        "char_entropy": _entropy(cc),
        "gzip_ratio": gzip_ratio,
        "zipf_deviation": _zipf_deviation(wc),
        # function words / POS (#5, #6)
        "stopword_ratio": sum(1 for w in lower if w in STOPWORDS) / nn,
        "function_word_ratio": sum(1 for w in lower if w in FUNCTION_WORDS) / nn,
        "noun_ratio": pos["noun"] / posn,
        "verb_ratio": pos["verb"] / posn,
        "adj_ratio": pos["adj"] / posn,
        "adv_ratio": pos["adv"] / posn,
        "pron_ratio": pos["pron"] / posn,
        "det_ratio": pos["det"] / posn,
        "prep_ratio": pos["prep"] / posn,
        # readability (#10)
        "flesch_ease": flesch / 100.0,
        "fk_grade": fk_grade / 20.0,
        "gunning_fog": fog / 20.0,
        "smog": smog / 20.0,
        # sentiment (#11)
        "sentiment_mean": sent_mean,
        "sentiment_std": sent_std,
        "subjectivity": subj / nn,
        # specificity / NER (#13)
        "proper_noun_ratio": proper / nn,
        "number_density": numbers / nn,
        "digit_ratio": sum(c.isdigit() for c in text) / chars,
        "uppercase_ratio": sum(c.isupper() for c in text) / chars,
        # casual (#15)
        "emoji_rate": emojis / nn,
        "slang_rate": slang / nn,
        "contraction_ratio": contractions / nn,
        "exclaim_per_sent": text.count("!") / ns,
        "question_per_sent": text.count("?") / ns,
        # spelling (#16)
        "misspelling_rate": misspell,
        # structure (#17)
        "list_marker_ratio": lists / ns,
        "header_ratio": headers / ns,
        "bold_ratio": bold / ns,
        "code_ratio": code / ns,
        "url_rate": urls / nn,
        # AI-tell / discourse
        "ai_tell_per_100w": sum(text.lower().count(p) for p in AI_TELLS)
                            / (n / 100.0 + 1e-6) if n else 0.0,
        "discourse_density": sum(1 for w in lower if w in DISCOURSE_MARKERS)
                             / nn * 100,
        "starter_diversity": len(set(starters)) / len(starters) if starters else 0,
        "comma_ratio": text.count(",") / chars,
        "semicolon_ratio": text.count(";") / chars,
        "punct_ratio": sum(c in ",.;:!?" for c in text) / chars,
    }
    return f


def _misspelling_rate(words):
    cand = [w for w in words if w.isalpha() and len(w) >= 4 and w.islower()]
    if not cand:
        return 0.0
    sp = _spell()
    if sp is not None:
        try:
            return len(sp.unknown(cand)) / len(cand)
        except Exception:
            pass
    # heuristic fallback: improbable letter patterns
    bad = 0
    for w in cand:
        if (not re.search(r"[aeiou]", w)               # no vowels
                or re.search(r"(.)\1\1", w)            # 3 repeated letters
                or re.search(r"[bcdfghjklmnpqrstvwxz]{5}", w)):  # 5 consonants
            bad += 1
    return bad / len(cand)


# ordered core feature names (model input layout) ------------------------- #
CORE_FEATURE_NAMES = list(extract_features("seed text. it works.").keys())
N_CORE = len(CORE_FEATURE_NAMES)

# perplexity block (filled by perplexity.py when enabled, else zeros) ------ #
PPL_FEATURE_NAMES = [
    "ppl_mean_nll", "ppl_perplexity", "ppl_logrank_mean", "ppl_frac_top10",
    "ppl_frac_top100", "ppl_logprob_std", "ppl_curvature", "ppl_burstiness",
]
N_PPL = len(PPL_FEATURE_NAMES)

FEATURE_NAMES = CORE_FEATURE_NAMES + PPL_FEATURE_NAMES
N_FEATURES = len(FEATURE_NAMES)


def feature_vector(text: str, include_ppl=None):
    """Model-input vector: core stats + (optional) perplexity block."""
    f = extract_features(text)
    vec = [float(f[k]) for k in CORE_FEATURE_NAMES]
    if include_ppl is None:
        include_ppl = os.environ.get("USE_PERPLEXITY", "0") == "1"
    if include_ppl:
        try:
            import perplexity
            vec += list(perplexity.ppl_feature_vector(text))
        except Exception:
            vec += [0.0] * N_PPL
    else:
        vec += [0.0] * N_PPL
    return vec


# --------------------------------------------------------------------------- #
#  Heuristic score + explanations
# --------------------------------------------------------------------------- #
def heuristic_ai_score(text: str) -> float:
    f = extract_features(text)
    if f["mean_sent_len"] == 0:
        return 0.5
    z = 0.0
    z += 1.5 * (1.0 - min(f["burstiness"] / 0.6, 1.0))
    z += 1.1 * min(f["ai_tell_per_100w"] / 2.0, 1.0)
    z += 0.9 * (1.0 - min(f["contraction_ratio"] / 0.03, 1.0))
    z += 0.7 * min(f["semicolon_ratio"] / 0.004, 1.0)
    z += 0.7 * (1.0 - min(f["hapax_ratio"] / 0.5, 1.0))
    z += 0.6 * max(f["rep_trigram"], f["rep_4gram"])
    z += 0.6 * (1.0 - min(f["exclaim_per_sent"] / 0.15, 1.0))
    z += 0.5 * f["uniformity"]
    z += 0.6 * min(f["discourse_density"] / 3.0, 1.0)
    z += 0.5 * (1.0 - min(f["sentiment_std"] / 0.5, 1.0))
    z += 0.4 * min(f["gzip_ratio"] and (0.45 - f["gzip_ratio"]) / 0.1, 1.0) \
        if f["gzip_ratio"] < 0.45 else 0.0
    z -= 1.0 * min(f["emoji_rate"] / 0.02, 1.0)
    z -= 0.8 * min(f["slang_rate"] / 0.03, 1.0)
    z -= 0.7 * min(f["misspelling_rate"] / 0.05, 1.0)
    z -= 0.8 * min(f["starter_diversity"], 1.0)
    bias = -1.4
    return float(1.0 / (1.0 + math.exp(-(z + bias))))


def top_signals(text: str, k: int = 8):
    f = extract_features(text)
    contrib = {
        "low burstiness": 1.0 - min(f["burstiness"] / 0.6, 1.0),
        "AI-tell vocabulary": min(f["ai_tell_per_100w"] / 2.0, 1.0),
        "few contractions": 1.0 - min(f["contraction_ratio"] / 0.03, 1.0),
        "semicolon habit": min(f["semicolon_ratio"] / 0.004, 1.0),
        "low word novelty": 1.0 - min(f["hapax_ratio"] / 0.5, 1.0),
        "phrase repetition": max(f["rep_trigram"], f["rep_4gram"]),
        "uniform sentences": f["uniformity"],
        "no exclamations": 1.0 - min(f["exclaim_per_sent"] / 0.15, 1.0),
        "discourse markers": min(f["discourse_density"] / 3.0, 1.0),
        "flat sentiment": 1.0 - min(f["sentiment_std"] / 0.5, 1.0),
        "high compressibility": max(0.0, (0.45 - f["gzip_ratio"]) / 0.1),
        "emoji / slang (human)": -min((f["emoji_rate"] + f["slang_rate"]) / 0.03, 1.0),
        "typos (human)": -min(f["misspelling_rate"] / 0.05, 1.0),
        "varied openers (human)": -min(f["starter_diversity"], 1.0),
    }
    ranked = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return [{"signal": s, "strength": round(v, 3),
             "points_to": "AI" if v >= 0 else "human"} for s, v in ranked]


def grouped_report(text: str) -> dict:
    """All features, grouped for a readable UI panel."""
    f = extract_features(text)
    g = lambda *ks: {k: round(f[k], 4) for k in ks}
    return {
        "length & burstiness (#4)": g("mean_sent_len", "std_sent_len",
                                      "skew_sent_len", "kurt_sent_len",
                                      "burstiness", "uniformity"),
        "lexical diversity (#9)": g("type_token_ratio", "hapax_ratio",
                                    "mtld", "mattr"),
        "repetition (#8)": g("rep_bigram", "rep_trigram", "rep_4gram",
                             "rep_5gram", "max_trigram_repeat"),
        "entropy & compression (#18)": g("word_entropy", "char_entropy",
                                         "gzip_ratio", "zipf_deviation"),
        "function words & POS (#5,#6)": g("stopword_ratio", "function_word_ratio",
                                          "noun_ratio", "verb_ratio", "adj_ratio",
                                          "adv_ratio", "pron_ratio"),
        "readability (#10)": g("flesch_ease", "fk_grade", "gunning_fog", "smog"),
        "sentiment (#11)": g("sentiment_mean", "sentiment_std", "subjectivity"),
        "specificity / NER (#13)": g("proper_noun_ratio", "number_density",
                                     "digit_ratio"),
        "casual signals (#15)": g("emoji_rate", "slang_rate", "contraction_ratio",
                                  "exclaim_per_sent"),
        "spelling (#16)": g("misspelling_rate"),
        "structure / markdown (#17)": g("list_marker_ratio", "header_ratio",
                                        "bold_ratio", "code_ratio", "url_rate"),
        "AI-tells & discourse": g("ai_tell_per_100w", "discourse_density",
                                  "starter_diversity"),
    }


# --------------------------------------------------------------------------- #
#  Sentence / token helpers (UI heatmap)
# --------------------------------------------------------------------------- #
def find_ai_tells(text: str):
    low = (text or "").lower()
    return [p for p in AI_TELLS if p in low]


def tell_spans(sentence: str, bucket: str):
    low = sentence.lower()
    matches = []
    for p in AI_TELLS:
        start = 0
        while True:
            i = low.find(p, start)
            if i < 0:
                break
            matches.append((i, i + len(p)))
            start = i + len(p)
    if not matches:
        return [(sentence, bucket)]
    matches.sort()
    merged = []
    for s, e in matches:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    spans, pos = [], 0
    for s, e in merged:
        if s > pos:
            spans.append((sentence[pos:s], bucket))
        spans.append((sentence[s:e], "AI-tell"))
        pos = e
    if pos < len(sentence):
        spans.append((sentence[pos:], bucket))
    return spans


def sentence_list(text: str, max_sentences: int = 80):
    text = text or ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    sents = [s.strip() for s in parts if s and s.strip()]
    return sents[:max_sentences]


if __name__ == "__main__":
    ai = ("Moreover, it is important to note that this multifaceted paradigm "
          "underscores a comprehensive tapestry; furthermore, we must delve "
          "into the nuanced realm of robust, seamless solutions.")
    hu = ("ok so i went to the store today and honestly? it was a mess lol. "
          "couldn't find the milk anywhere, asked a guy, he didn't know either!")
    print("N_CORE", N_CORE, "N_PPL", N_PPL, "N_FEATURES", N_FEATURES)
    for name, t in (("AI", ai), ("HUMAN", hu)):
        print(name, "score=", round(heuristic_ai_score(t), 3))
        print("  top:", [s["signal"] for s in top_signals(t, 4)])
    assert len(feature_vector(ai)) == N_FEATURES
    print("feature_vector len OK")
