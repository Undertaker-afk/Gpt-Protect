"""
model.py — AI-text detector
===========================
Backbone (scratch MoE or a pretrained HF encoder) + pooling + an AI-pattern
feature pathway + several heads / objectives:

  * sequence classifier      human(0) / ai(1)          (calibrated, #24)
  * per-token head           per-token AI logit         (#20, token heatmap)
  * source head (optional)   which dataset a sample is  (#21, multi-task)
  * pooled embedding         exposed for SupCon / KD    (#22, #30)

The forward returns everything; callers add the contrastive / distillation
losses (they need batch-level / teacher context).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture import Backbone, RMSNorm
from config import ModelConfig
from ai_patterns import N_FEATURES


class AttentionPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.proj = nn.Linear(d, d)

    def forward(self, h, mask):
        scores = (self.proj(h) @ self.q) / (h.shape[-1] ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        w = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (w * h).sum(1)


class _HeadMixin:
    """Shared head construction + loss assembly for both backbones."""

    def _build_heads(self, cfg, d):
        self.use_patterns = getattr(cfg, "use_pattern_features", True)
        if self.use_patterns:
            self.pattern_norm = nn.LayerNorm(N_FEATURES)
            self.pattern_mlp = nn.Sequential(
                nn.Linear(N_FEATURES, d), nn.GELU(), nn.Linear(d, d))
            self.fuse = nn.Linear(2 * d, d)
        self.head_norm = RMSNorm(d, getattr(cfg, "rms_eps", 1e-5))
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(d, cfg.num_labels)

        self.use_token_head = getattr(cfg, "use_token_head", True)
        if self.use_token_head:
            self.token_head = nn.Linear(d, 1)        # per-token AI logit (#20)
        self.use_source_head = getattr(cfg, "use_source_head", False) \
            and getattr(cfg, "n_sources", 0) > 1
        if self.use_source_head:
            self.source_head = nn.Linear(d, cfg.n_sources)   # (#21)

        # calibration temperature (#24) — learnable, clamped positive at use
        self.log_temp = nn.Parameter(
            torch.tensor(float(getattr(cfg, "temperature_init", 1.0))).log())

    def _fuse_patterns(self, pooled, pattern_feats):
        if not self.use_patterns:
            return pooled
        if pattern_feats is None:
            pattern_feats = torch.zeros(pooled.size(0), N_FEATURES,
                                        device=pooled.device, dtype=pooled.dtype)
        pr = self.pattern_mlp(self.pattern_norm(pattern_feats))
        return self.fuse(torch.cat([pooled, pr], dim=-1))

    @property
    def temperature(self):
        return self.log_temp.exp().clamp(0.3, 5.0)

    def _heads_forward(self, h, pooled, attention_mask, labels,
                       source_labels, aux):
        emb = self.head_norm(pooled)
        logits = self.classifier(self.dropout(emb))
        out = {"logits": logits, "embedding": emb,
               "aux_loss": aux if torch.is_tensor(aux) else torch.tensor(0.0)}

        token_logits = None
        if self.use_token_head:
            token_logits = self.token_head(h).squeeze(-1)        # (B,T)
            out["token_logits"] = token_logits
        if self.use_source_head:
            out["source_logits"] = self.source_head(emb)

        loss = None
        if labels is not None:
            ls = getattr(self, "label_smoothing", 0.0)
            loss = F.cross_entropy(logits, labels, label_smoothing=ls)
            if torch.is_tensor(aux):
                loss = loss + aux
            # weak per-token supervision: every real token inherits seq label
            if self.use_token_head and attention_mask is not None:
                tl = labels.float().unsqueeze(1).expand_as(token_logits)
                bce = F.binary_cross_entropy_with_logits(
                    token_logits, tl, reduction="none")
                m = attention_mask.float()
                tok_loss = (bce * m).sum() / m.sum().clamp_min(1.0)
                loss = loss + self.cfg.token_loss_coef * tok_loss
                out["token_loss"] = tok_loss.detach()
            if self.use_source_head and source_labels is not None:
                s_loss = F.cross_entropy(out["source_logits"], source_labels)
                loss = loss + self.cfg.source_loss_coef * s_loss
                out["source_loss"] = s_loss.detach()
        out["loss"] = loss
        return out

    @torch.no_grad()
    def predict_proba(self, input_ids, attention_mask=None, pattern_feats=None,
                      calibrated=True):
        self.eval()
        out = self.forward(input_ids, attention_mask, pattern_feats=pattern_feats)
        logits = out["logits"]
        if calibrated:
            logits = logits / self.temperature
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def token_ai_probs(self, input_ids, attention_mask=None, pattern_feats=None):
        """Per-token AI probability (#20). Returns (B,T) or None."""
        if not self.use_token_head:
            return None
        self.eval()
        out = self.forward(input_ids, attention_mask, pattern_feats=pattern_feats)
        return torch.sigmoid(out["token_logits"])

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AITextDetector(nn.Module, _HeadMixin):
    """Scratch MoE / hybrid-attention backbone + heads."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.pool_mode = cfg.pool
        if cfg.pool == "attn":
            self.pool = AttentionPool(cfg.d_model)
        self._build_heads(cfg, cfg.d_model)
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

    def forward(self, input_ids, attention_mask=None, labels=None,
                pattern_feats=None, source_labels=None):
        h, aux = self.backbone(input_ids, attention_mask)
        pooled = self._fuse_patterns(self._pool(h, attention_mask), pattern_feats)
        return self._heads_forward(h, pooled, attention_mask, labels,
                                   source_labels, aux)


class HFDetector(nn.Module, _HeadMixin):
    """Pretrained HuggingFace encoder backbone (#19): DeBERTa/RoBERTa/etc."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import AutoModel
        name = cfg.backbone.split("hf:", 1)[1]
        self.encoder = AutoModel.from_pretrained(name)
        d = self.encoder.config.hidden_size
        cfg.d_model = d
        self.pool_mode = "mean"
        self._build_heads(cfg, d)
        # init only the new heads (encoder keeps pretrained weights)
        for mod in (self.classifier, getattr(self, "fuse", None)):
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=0.02)
                nn.init.zeros_(mod.bias)

    def forward(self, input_ids, attention_mask=None, labels=None,
                pattern_feats=None, source_labels=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1e-6)
        else:
            pooled = h.mean(1)
        pooled = self._fuse_patterns(pooled, pattern_feats)
        return self._heads_forward(h, pooled, attention_mask, labels,
                                   source_labels, torch.tensor(0.0))


def build_model(cfg: ModelConfig):
    if getattr(cfg, "backbone", "scratch").startswith("hf:"):
        return HFDetector(cfg)
    return AITextDetector(cfg)


# --------------------------------------------------------------------------- #
#  Auxiliary losses used by the trainers (#22 contrastive, #30 distillation)
# --------------------------------------------------------------------------- #
def supcon_loss(emb, labels, temperature=0.1):
    """Supervised contrastive loss on L2-normalized pooled embeddings (#22)."""
    if emb.size(0) < 4:
        return emb.new_tensor(0.0)
    z = F.normalize(emb, dim=-1)
    sim = z @ z.t() / temperature
    n = z.size(0)
    mask_self = torch.eye(n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, -1e9)
    labels = labels.view(-1, 1)
    pos = (labels == labels.t()) & ~mask_self
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos_cnt = pos.sum(1)
    valid = pos_cnt > 0
    if valid.sum() == 0:
        return emb.new_tensor(0.0)
    mean_pos = (logp * pos).sum(1)[valid] / pos_cnt[valid]
    return -mean_pos.mean()


def distill_loss(student_logits, teacher_logits, temperature=2.0):
    """KL distillation from a teacher's soft targets (#30)."""
    t = temperature
    s = F.log_softmax(student_logits / t, dim=-1)
    d = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(s, d, reduction="batchmean") * (t * t)
