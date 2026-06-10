"""
muon.py — Muon optimizer
========================
Muon (MomentUm Orthogonalized by Newton-Schulz) updates 2-D weight matrices by
orthogonalizing the momentum buffer via a few Newton-Schulz iterations before
applying it.  Non-matrix parameters (embeddings, norms, biases, the classifier
head) fall back to AdamW.  This mirrors the recipe DeepSeek used for faster
convergence.

Reference: Keller Jordan's Muon.  Newton-Schulz coefficients (3.4445, -4.7750,
2.0315) approximate the matrix sign function with a quintic iteration.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer, AdamW


@torch.no_grad()
def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    """Orthogonalize G (m x n) via quintic Newton-Schulz. Returns ~UV^T."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    def __init__(self, params, lr=2e-2, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mom).add_(g)
                upd = g.add(buf, alpha=mom) if group["nesterov"] else buf
                # orthogonalize the (reshaped) 2-D update
                o = _newton_schulz(upd.reshape(upd.size(0), -1), group["ns_steps"])
                o = o.view_as(p)
                # scale ~ sqrt(fan-out/fan-in) keeps update RMS stable
                scale = (max(p.size(0), 1) / max(p[0].numel(), 1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(o, alpha=-group["lr"] * scale)
        return loss


def build_optimizer(model, cfg):
    """Muon for >=2-D weights, AdamW for the rest. Returns a small wrapper."""
    matrix, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in n and "classifier" not in n:
            matrix.append(p)
        else:
            other.append(p)

    if cfg.optimizer == "adamw" or not matrix:
        return _Combo([AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)])

    muon = Muon(matrix, lr=cfg.muon_lr, weight_decay=cfg.weight_decay)
    adamw = AdamW(other, lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"[muon] matrices={len(matrix)}  adamw-params={len(other)}")
    return _Combo([muon, adamw])


class _Combo:
    """Drives several optimizers as one."""
    def __init__(self, opts):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()
