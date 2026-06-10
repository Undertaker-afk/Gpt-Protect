"""
moe.py — Mixture of Experts (FP4 experts) + mHC stability
=========================================================
Sparse MoE FFN with:
  * top-k token routing (default k=2 of N experts)
  * load-balancing auxiliary loss (Switch-Transformer style)
  * FP4-simulated expert weights (fake-quant straight-through) — emulating the
    DeepSeek-V4-Pro "experts in FP4, everything else higher precision" recipe
  * manifold HyperConnections (mHC): a learnable multi-stream residual mix that
    replaces the single residual add and stabilizes deep MoE training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  FP4 fake-quant (straight-through estimator)
# --------------------------------------------------------------------------- #
class _FP4STE(torch.autograd.Function):
    # 16 representable levels of an e2m1-ish symmetric grid, scaled per-tensor.
    LEVELS = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0],
        dtype=torch.float32)

    @staticmethod
    def forward(ctx, w):
        levels = _FP4STE.LEVELS.to(w.device, w.dtype)[:15]
        scale = w.abs().amax().clamp_min(1e-8) / 6.0
        wn = (w / scale).unsqueeze(-1)
        idx = (wn - levels).abs().argmin(dim=-1)
        q = levels[idx] * scale
        return q

    @staticmethod
    def backward(ctx, g):
        return g            # straight-through


def fp4(w: torch.Tensor) -> torch.Tensor:
    return _FP4STE.apply(w)


class Expert(nn.Module):
    """SwiGLU FFN expert; weights optionally FP4-fake-quantized in forward."""

    def __init__(self, d_model, d_hidden, fp4_weights=True):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.w_up = nn.Linear(d_model, d_hidden, bias=False)
        self.w_down = nn.Linear(d_hidden, d_model, bias=False)
        self.fp4 = fp4_weights

    def forward(self, x):
        if self.fp4 and self.training:
            g = F.linear(x, fp4(self.w_gate.weight))
            u = F.linear(x, fp4(self.w_up.weight))
            h = F.silu(g) * u
            return F.linear(h, fp4(self.w_down.weight))
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# --------------------------------------------------------------------------- #
#  Sparse MoE feed-forward
# --------------------------------------------------------------------------- #
class MoEFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        d_hidden = int(cfg.expert_ffn_mult * d)
        self.n_experts = cfg.n_experts
        self.k = cfg.n_experts_active
        self.aux_coef = cfg.moe_aux_loss_coef

        self.router = nn.Linear(d, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(d, d_hidden, cfg.moe_fp4) for _ in range(self.n_experts))
        # a shared expert always-on (DeepSeek-style) for stability
        self.shared = Expert(d, d_hidden, cfg.moe_fp4)
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)                                   # (N, D)
        logits = self.router(xf)
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.k, dim=-1)                 # (N, k)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)

        out = self.shared(xf)
        flat_i = topi.reshape(-1)
        flat_w = topv.reshape(-1)
        token_idx = torch.arange(xf.size(0), device=x.device).repeat_interleave(self.k)
        for e in range(self.n_experts):
            sel = flat_i == e
            if sel.any():
                rows = token_idx[sel]
                contrib = self.experts[e](xf[rows]) * flat_w[sel].unsqueeze(-1)
                out.index_add_(0, rows, contrib.to(out.dtype))

        # load-balancing aux loss: fraction routed * mean prob per expert
        with torch.no_grad():
            one_hot = F.one_hot(topi, self.n_experts).float().sum(1)   # (N,E)
            load = one_hot.mean(0)
        importance = probs.mean(0)
        self.last_aux_loss = self.aux_coef * self.n_experts * (load * importance).sum()
        return out.view(B, T, D)


# --------------------------------------------------------------------------- #
#  mHC — manifold HyperConnections
# --------------------------------------------------------------------------- #
class HyperConnection(nn.Module):
    """
    Replaces `x = x + sublayer(norm(x))` with a learnable multi-stream mix.

    We maintain `n_streams` parallel residual streams.  The sublayer reads a
    learned combination of the streams (depth-connection) and its output is
    distributed back to each stream with learnable weights (width-connection).
    This is a compact, stable variant of HyperConnections adapted for MoE depth.
    """

    def __init__(self, d_model, n_streams: int = 2):
        super().__init__()
        self.n = n_streams
        # depth weights: how to combine streams -> sublayer input
        self.depth = nn.Parameter(torch.ones(n_streams) / n_streams)
        # width weights: how to scatter sublayer output back to streams
        self.width = nn.Parameter(torch.ones(n_streams))
        # static residual mix between streams (identity-init)
        self.mix = nn.Parameter(torch.eye(n_streams))

    def combine(self, streams):
        # streams: list[n] of (B,T,D)  ->  sublayer input (B,T,D)
        w = F.softmax(self.depth, dim=0)
        return sum(w[i] * streams[i] for i in range(self.n))

    def distribute(self, streams, sub_out):
        # remix streams, then add the sublayer output weighted per stream
        new = []
        for i in range(self.n):
            mixed = sum(self.mix[i, j] * streams[j] for j in range(self.n))
            new.append(mixed + self.width[i] * sub_out)
        return new
