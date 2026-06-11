"""
perplexity.py — reference-LM signals (features #1, #2)
====================================================
Heavy, optional signals computed from a small reference language model
(distilgpt2 by default).  Enabled with `USE_PERPLEXITY=1`; otherwise every
entry point returns zeros so the rest of the system is unaffected.

  #1  True perplexity / mean negative log-likelihood, and GLTR-style token-rank
      statistics (fraction of tokens in the LM's top-10 / top-100 predictions).
      AI text has lower, flatter perplexity and far more high-rank tokens.

  #2  DetectGPT-style curvature: an LM-generated text sits near a *local
      maximum* of the model's log-probability, so perturbing it lowers the
      log-prob more than for human text.  We self-perturb (random token
      swaps/drops, no second model) and measure the normalized drop.

The model loads lazily and is cached process-wide.  All work is capped
(`PPL_MAX_TOKENS`, `PPL_PERTURB`) to stay affordable on CPU.
"""

from __future__ import annotations

import math
import os

_MODEL = None
_TOK = None
_TRIED = False

PPL_MODEL = os.environ.get("PPL_MODEL", "distilgpt2")
PPL_MAX_TOKENS = int(os.environ.get("PPL_MAX_TOKENS", "256"))
PPL_PERTURB = int(os.environ.get("PPL_PERTURB", "4"))

N_PPL = 8   # must match ai_patterns.PPL_FEATURE_NAMES


def enabled() -> bool:
    return os.environ.get("USE_PERPLEXITY", "0") == "1"


def _load():
    global _MODEL, _TOK, _TRIED
    if _TRIED:
        return _MODEL is not None
    _TRIED = True
    if not enabled():
        return False
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(PPL_MODEL)
        _MODEL = AutoModelForCausalLM.from_pretrained(PPL_MODEL)
        _MODEL.eval()
        torch.set_grad_enabled(False)
        print(f"[perplexity] loaded reference LM: {PPL_MODEL}")
        return True
    except Exception as e:
        print(f"[perplexity] disabled (load failed): {repr(e)[:120]}")
        _MODEL = None
        return False


def _token_stats(input_ids):
    """Return per-token nll list + GLTR rank stats for a single sequence."""
    import torch
    ids = input_ids
    with torch.no_grad():
        logits = _MODEL(ids).logits[0]          # (T, V)
    logp = torch.log_softmax(logits[:-1], dim=-1)   # predict next token
    targets = ids[0, 1:]
    tok_logp = logp[range(len(targets)), targets]
    nll = (-tok_logp).tolist()
    # ranks: how many tokens are more probable than the true one
    ranks = (logp > tok_logp.unsqueeze(-1)).sum(-1).tolist()
    return nll, ranks


def analysis(text: str) -> dict:
    """Full perplexity analysis for the UI. Zeros/empty if disabled."""
    if not _load() or not text.strip():
        return {"enabled": False}
    import torch
    enc = _TOK(text, return_tensors="pt", truncation=True,
               max_length=PPL_MAX_TOKENS)
    ids = enc["input_ids"]
    if ids.shape[1] < 3:
        return {"enabled": False}
    nll, ranks = _token_stats(ids)
    mean_nll = sum(nll) / len(nll)
    ppl = math.exp(min(mean_nll, 20))
    var_nll = sum((x - mean_nll) ** 2 for x in nll) / len(nll)
    log_ranks = [math.log(r + 1) for r in ranks]
    logrank_mean = sum(log_ranks) / len(log_ranks)
    frac_top10 = sum(1 for r in ranks if r < 10) / len(ranks)
    frac_top100 = sum(1 for r in ranks if r < 100) / len(ranks)

    # DetectGPT-style curvature via self-perturbation
    curvature = 0.0
    if PPL_PERTURB > 0:
        base = -mean_nll                          # mean log-prob
        drops = []
        V = _MODEL.config.vocab_size
        g = torch.Generator().manual_seed(len(text))
        T = ids.shape[1]
        for _ in range(PPL_PERTURB):
            pert = ids.clone()
            k = max(1, T // 10)
            pos = torch.randint(1, T, (k,), generator=g)
            pert[0, pos] = torch.randint(0, V, (k,), generator=g)
            pn, _ = _token_stats(pert)
            drops.append(base - (-(sum(pn) / len(pn))))
        if drops:
            md = sum(drops) / len(drops)
            sd = (sum((d - md) ** 2 for d in drops) / len(drops)) ** 0.5 or 1.0
            curvature = md / sd

    return {
        "enabled": True,
        "perplexity": round(ppl, 2),
        "mean_nll": round(mean_nll, 4),
        "nll_variance": round(var_nll, 4),
        "logrank_mean": round(logrank_mean, 4),
        "frac_top10": round(frac_top10, 4),
        "frac_top100": round(frac_top100, 4),
        "detectgpt_curvature": round(curvature, 4),
        "n_tokens": len(nll),
        "model": PPL_MODEL,
    }


def ppl_feature_vector(text: str):
    """Fixed-length feature block for the model (zeros if disabled)."""
    a = analysis(text)
    if not a.get("enabled"):
        return [0.0] * N_PPL
    return [
        a["mean_nll"] / 10.0,
        min(a["perplexity"] / 100.0, 5.0),
        a["logrank_mean"] / 10.0,
        a["frac_top10"],
        a["frac_top100"],
        math.sqrt(a["nll_variance"]) / 5.0,
        max(-3.0, min(3.0, a["detectgpt_curvature"])),
        a["nll_variance"] / 10.0,
    ]


def ai_probability(text: str):
    """Standalone perplexity-based AI likelihood (0..1), or None if disabled."""
    a = analysis(text)
    if not a.get("enabled"):
        return None
    # low perplexity + many top-10 tokens + positive curvature -> AI
    z = (1.4 * a["frac_top10"]
         + 1.0 * max(0.0, 1.0 - a["mean_nll"] / 4.0)
         + 0.8 * max(0.0, min(a["detectgpt_curvature"], 2.0))
         - 1.6)
    return float(1.0 / (1.0 + math.exp(-z)))


if __name__ == "__main__":
    os.environ["USE_PERPLEXITY"] = "1"
    ai = ("The mitochondria is the powerhouse of the cell. It is important to "
          "note that this organelle plays a crucial role in energy production.")
    hu = ("yo i totally forgot my keys again lmao, had to climb through the "
          "window like a raccoon. my neighbor definitely saw me.")
    for n, t in (("AI", ai), ("HUMAN", hu)):
        print(n, analysis(t))
        print("  ai_prob:", ai_probability(t))
