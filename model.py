"""
model.py — AI-text detector
===========================
Wraps the backbone with a pooling layer + linear classification head.
Output: logits over {human(0), ai(1)} plus the MoE auxiliary loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture import Backbone, RMSNorm
from config import ModelConfig


class AttentionPool(nn.Module):
    """Single-query attention pooling over the sequence."""

    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, d)

    def forward(self, h, mask):
        scores = (self.proj(h) @ self.q) / (h.shape[-1] ** 0.5)   # (B,T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        w = F.softmax(scores, dim=-1).unsqueeze(-1)               # (B,T,1)
        return (w * h).sum(1)


class AITextDetector(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.pool_mode = cfg.pool
        if cfg.pool == "attn":
            self.pool = AttentionPool(cfg.d_model)
        self.head_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.d_model, cfg.num_labels)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def _pool(self, h, mask):
        if self.pool_mode == "mean":
            if mask is None:
                return h.mean(1)
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
        if self.pool_mode == "cls":
            return h[:, 0]
        return self.pool(h, mask)

    def forward(self, input_ids, attention_mask=None, labels=None):
        h, aux = self.backbone(input_ids, attention_mask)
        pooled = self._pool(h, attention_mask)
        pooled = self.dropout(self.head_norm(pooled))
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            ls = getattr(self, "label_smoothing", 0.0)
            loss = F.cross_entropy(logits, labels, label_smoothing=ls)
            if torch.is_tensor(aux):
                loss = loss + aux
        return {"logits": logits, "loss": loss, "aux_loss": aux}

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask=None):
        self.eval()
        logits = self.forward(input_ids, attention_mask)["logits"]
        return F.softmax(logits, dim=-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(cfg: ModelConfig) -> AITextDetector:
    return AITextDetector(cfg)
