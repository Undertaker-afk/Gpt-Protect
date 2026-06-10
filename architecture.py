"""
architecture.py — backbone blocks
=================================
Assembles the building blocks into a transformer backbone:

  TransformerBlock = mHC( HybridAttention )  +  mHC( MoE | DenseFFN )

Every `cfg.moe_every`-th block uses the sparse MoE FFN; the rest use a dense
SwiGLU FFN.  Residual connections are realized through mHC HyperConnections
when `cfg.use_mhc`, otherwise a plain pre-norm residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import HybridAttention
from moe import MoEFeedForward, HyperConnection


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n * self.weight


class DenseFFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        h = int(cfg.ffn_mult * d)
        self.w_gate = nn.Linear(d, h, bias=False)
        self.w_up = nn.Linear(d, h, bias=False)
        self.w_down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.is_moe = cfg.use_moe and (layer_idx % cfg.moe_every == 0)

        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.attn = HybridAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.ffn = MoEFeedForward(cfg) if self.is_moe else DenseFFN(cfg)

        self.use_mhc = cfg.use_mhc
        if self.use_mhc:
            self.hc_attn = HyperConnection(cfg.d_model, cfg.mhc_streams)
            self.hc_ffn = HyperConnection(cfg.d_model, cfg.mhc_streams)

    def _sublayer(self, streams, hc, norm, fn, **kw):
        if self.use_mhc:
            x = hc.combine(streams)
            out = fn(norm(x), **kw)
            return hc.distribute(streams, out)
        # plain pre-norm residual (single stream held in a 1-list)
        x = streams[0]
        return [x + fn(norm(x), **kw)]

    def forward(self, streams, attn_mask=None):
        streams = self._sublayer(streams, getattr(self, "hc_attn", None),
                                 self.attn_norm, self.attn,
                                 attn_mask=attn_mask, causal=False)
        streams = self._sublayer(streams, getattr(self, "hc_ffn", None),
                                 self.ffn_norm, self.ffn)
        return streams

    @property
    def aux_loss(self):
        return self.ffn.last_aux_loss if self.is_moe else None


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.n_streams = cfg.mhc_streams if cfg.use_mhc else 1

    def forward(self, input_ids, attn_mask=None):
        h = self.drop(self.embed(input_ids))
        streams = [h for _ in range(self.n_streams)]
        aux = 0.0
        for blk in self.blocks:
            streams = blk(streams, attn_mask)
            if blk.aux_loss is not None:
                aux = aux + blk.aux_loss
        # collapse streams (mean) and final norm
        h = sum(streams) / len(streams)
        return self.norm(h), aux
