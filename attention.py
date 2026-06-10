"""
attention.py — Hybrid Attention (CSA + HCA)
==========================================
Re-implements the DeepSeek-V4-Pro hybrid attention idea at small scale:

  * CSA  (Compressed Sparse Attention)
        Compress every `csa_compress` (=4) tokens into one summary entry, then
        run *sparse* attention: each query attends only to its top-k most
        relevant compressed blocks.  -> huge FLOP / KV savings on long context.

  * HCA  (Hierarchical Compressed Attention)
        Compress every `hca_compress` (=128) tokens into one entry and attend
        *densely* over those coarse summaries.  -> cheap global context.

The two branches are fused with a learned per-head gate.  Grouped-Query
Attention (GQA) shrinks the KV projection.  RoPE positions are applied to Q/K.
At inference the keys/values can flow through the TurboQuant 3-bit cache.

For training we materialize the (compressed) attention densely; for inference
the same modules feed `TurboQuantKVCache`.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant import TurboQuantKVCache


# --------------------------------------------------------------------------- #
#  RoPE
# --------------------------------------------------------------------------- #
def build_rope(seq_len: int, dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                       # (T, dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)                # (T, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos, sin):
    # x: (B, H, T, D)
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rot * sin


def _compress(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Mean-pool over non-overlapping windows along the time dim.
    x: (B, H, T, D) -> (B, H, ceil(T/factor), D)."""
    B, H, T, D = x.shape
    if factor <= 1:
        return x
    pad = (factor - T % factor) % factor
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    x = x.view(B, H, -1, factor, D).mean(dim=3)
    return x


# --------------------------------------------------------------------------- #
#  Hybrid attention
# --------------------------------------------------------------------------- #
class HybridAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = self.d // self.n_heads
        self.group = self.n_heads // self.n_kv

        self.q_proj = nn.Linear(self.d, self.n_heads * self.hd, bias=False)
        self.k_proj = nn.Linear(self.d, self.n_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(self.d, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.hd, self.d, bias=False)

        self.use_hybrid = cfg.use_hybrid_attention
        self.csa_c = cfg.csa_compress
        self.csa_topk = cfg.csa_topk
        self.hca_c = cfg.hca_compress
        # fusion gate over the [local? we use full] / CSA / HCA branches
        self.branch_gate = nn.Parameter(torch.zeros(self.n_heads, 3))
        self.dropout = nn.Dropout(cfg.dropout)

        # inference KV cache (TurboQuant) — created lazily
        self.kv_cache = TurboQuantKVCache(self.hd, cfg.kv_bits, cfg.qjl_dim) \
            if cfg.turboquant else None

    def _shape_kv(self, x, proj, B, T):
        h = proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)  # (B,Hkv,T,d)
        # expand grouped kv heads to full heads
        return h.repeat_interleave(self.group, dim=1)               # (B,H,T,d)

    def _sdpa(self, q, k, v, causal):
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                              dropout_p=0.0)

    def forward(self, x, attn_mask=None, causal: bool = False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self._shape_kv(x, self.k_proj, B, T)
        v = self._shape_kv(x, self.v_proj, B, T)

        cos, sin = build_rope(T, self.hd, self.cfg.rope_theta, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # --- branch 1: full (local/dense) attention ---------------------- #
        full = self._sdpa(q, k, v, causal)

        if not self.use_hybrid:
            out = full
        else:
            # --- branch 2: CSA (compress-4 sparse) ----------------------- #
            kc = _compress(k, self.csa_c)
            vc = _compress(v, self.csa_c)
            Tc = kc.shape[2]
            scores = torch.matmul(q, kc.transpose(-1, -2)) / math.sqrt(self.hd)
            if self.csa_topk < Tc:
                topv, topi = scores.topk(self.csa_topk, dim=-1)
                p = F.softmax(topv, dim=-1)
                vsel = torch.gather(
                    vc.unsqueeze(2).expand(-1, -1, T, -1, -1), 3,
                    topi.unsqueeze(-1).expand(-1, -1, -1, -1, self.hd))
                csa = (p.unsqueeze(-1) * vsel).sum(dim=3)
            else:
                csa = torch.matmul(F.softmax(scores, dim=-1), vc)

            # --- branch 3: HCA (compress-128 dense) ---------------------- #
            kh = _compress(k, self.hca_c)
            vh = _compress(v, self.hca_c)
            sh = torch.matmul(q, kh.transpose(-1, -2)) / math.sqrt(self.hd)
            hca = torch.matmul(F.softmax(sh, dim=-1), vh)

            # --- fuse with per-head softmax gate ------------------------- #
            g = F.softmax(self.branch_gate, dim=-1)            # (H,3)
            g = g.view(1, self.n_heads, 1, 1, 3)
            stack = torch.stack([full, csa, hca], dim=-1)      # (B,H,T,d,3)
            out = (stack * g).sum(-1)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.hd)
        return self.dropout(self.o_proj(out))

    # ----------------------------------------------------------------- #
    #  Inference-time incremental decode using the TurboQuant KV cache.
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def step(self, x_t, pos: int):
        """Single-token decode step. x_t: (B,1,d).  Uses 3-bit KV cache."""
        B = x_t.shape[0]
        q = self.q_proj(x_t).view(B, 1, self.n_heads, self.hd).transpose(1, 2)
        k = self._shape_kv(x_t, self.k_proj, B, 1)
        v = self._shape_kv(x_t, self.v_proj, B, 1)
        cos, sin = build_rope(pos + 1, self.hd, self.cfg.rope_theta,
                              x_t.device, x_t.dtype)
        q = apply_rope(q, cos[pos:pos + 1], sin[pos:pos + 1])
        k = apply_rope(k, cos[pos:pos + 1], sin[pos:pos + 1])
        K, V = self.kv_cache.append(k, v)
        out = self._sdpa(q, K, V, causal=False)
        out = out.transpose(1, 2).reshape(B, 1, self.n_heads * self.hd)
        return self.o_proj(out)
