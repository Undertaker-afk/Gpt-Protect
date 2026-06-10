"""
config.py
=========
Central configuration for the AI-text-detection training suite.

The architecture borrows ideas popularized by DeepSeek-V4-Pro and Google's
TurboQuant research and adapts them to a *classifier* (human vs. AI text):

  * MoE backbone (sparse experts, FP4-simulated weights)
  * Hybrid Attention  (CSA: compress-4 sparse  +  HCA: compress-128 dense)
  * mHC  (manifold HyperConnections) for residual / MoE stability
  * TurboQuant KV-cache compression (PolarQuant + QJL) for inference
  * Muon optimizer for the 2-D matrix parameters

Two "real" presets are provided (0.4B and 5B) plus a `tiny` preset that is
small enough to actually train on CPU inside this sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


# --------------------------------------------------------------------------- #
#  Model configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    # --- core dims -------------------------------------------------------- #
    vocab_size: int = 50257            # gpt2 tokenizer by default
    max_seq_len: int = 1024
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4                # GQA: grouped key/value heads
    ffn_mult: float = 4.0              # dense FFN hidden = ffn_mult * d_model
    rope_theta: float = 100_000.0
    rms_eps: float = 1e-5
    dropout: float = 0.1

    # --- classification head --------------------------------------------- #
    num_labels: int = 2                # 0 = human, 1 = ai
    pool: str = "attn"                 # {"mean","cls","attn"}

    # --- intelligent AI-pattern feature pathway (ai_patterns.py) --------- #
    use_pattern_features: bool = True  # fuse hand-engineered AI-detection feats

    # --- MoE -------------------------------------------------------------- #
    use_moe: bool = True
    moe_every: int = 2                 # every Nth block is an MoE block
    n_experts: int = 16
    n_experts_active: int = 2          # top-k routing
    expert_ffn_mult: float = 2.0
    moe_aux_loss_coef: float = 0.01
    moe_fp4: bool = True               # simulate FP4 expert weights

    # --- mHC  (manifold HyperConnections) -------------------------------- #
    use_mhc: bool = True
    mhc_streams: int = 2               # number of parallel residual streams

    # --- Hybrid Attention ------------------------------------------------- #
    use_hybrid_attention: bool = True
    csa_compress: int = 4              # compress every 4 tokens -> 1 (sparse)
    csa_topk: int = 64                 # sparse: keep top-k compressed blocks
    hca_compress: int = 128           # compress every 128 tokens -> 1 (dense)

    # --- TurboQuant KV cache (inference only) ----------------------------- #
    turboquant: bool = True
    kv_bits: int = 3                   # 3-bit KV cache
    qjl_dim: int = 128                 # JL projection dim for the QJL key sketch

    def n_params_estimate(self) -> int:
        """Rough parameter count (ignores quantization, counts MoE experts)."""
        d = self.d_model
        embed = self.vocab_size * d
        attn_per_layer = 2 * d * d + 2 * d * (d * self.n_kv_heads // self.n_heads)
        dense_ffn = 2 * d * int(self.ffn_mult * d)
        moe_ffn = self.n_experts * 2 * d * int(self.expert_ffn_mult * d)
        n_moe = self.n_layers // self.moe_every if self.use_moe else 0
        n_dense = self.n_layers - n_moe
        layers = (
            self.n_layers * attn_per_layer
            + n_dense * dense_ffn
            + n_moe * moe_ffn
        )
        head = d * self.num_labels
        return int(embed + layers + head)


# --------------------------------------------------------------------------- #
#  Training configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    output_dir: str = "checkpoints"
    seed: int = 1234

    # data
    datasets: tuple = (
        "alex-kudryashov/dlr-hw-2-human-ai-texts",
        "nbroad/basic_text_dataset",
    )
    max_samples: Optional[int] = None    # cap dataset size (None = all)
    val_fraction: float = 0.05
    max_seq_len: int = 512
    tokenizer_name: str = "gpt2"

    # optimization
    epochs: int = 3
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 3e-4
    muon_lr: float = 2e-2                 # Muon learning rate for matrices
    weight_decay: float = 0.05
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    optimizer: str = "muon"              # {"muon","adamw"}
    label_smoothing: float = 0.05

    # augmentation
    augment: bool = True
    augment_prob: float = 0.4

    # runtime
    num_workers: int = 4
    log_every: int = 20
    eval_every: int = 400
    save_every: int = 800
    device: str = "cpu"                  # auto-overridden in train.py
    amp: bool = False                    # bf16 autocast when on GPU

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# --------------------------------------------------------------------------- #
#  Presets
# --------------------------------------------------------------------------- #
def preset(name: str) -> ModelConfig:
    name = name.lower()
    if name in ("5b", "5B".lower()):
        # ~5B total params (MoE).  Designed for multi-GPU.
        return ModelConfig(
            d_model=2048, n_layers=32, n_heads=32, n_kv_heads=8,
            n_experts=16, n_experts_active=2, expert_ffn_mult=2.0,
            max_seq_len=4096, mhc_streams=2,
        )
    if name == "0.4b":
        # ~0.4B total params.  This is the model we "start training".
        return ModelConfig(
            d_model=1024, n_layers=12, n_heads=16, n_kv_heads=4,
            n_experts=6, n_experts_active=2, expert_ffn_mult=2.0,
            max_seq_len=1024, mhc_streams=2,
        )
    if name == "tiny":
        # CPU-runnable smoke / sandbox training preset.
        return ModelConfig(
            vocab_size=50257, d_model=256, n_layers=4, n_heads=4, n_kv_heads=2,
            n_experts=4, n_experts_active=2, expert_ffn_mult=2.0,
            moe_every=2, max_seq_len=256, mhc_streams=2,
        )
    raise ValueError(f"unknown preset {name!r} (choose 5b | 0.4b | tiny)")


if __name__ == "__main__":
    for n in ("tiny", "0.4b", "5b"):
        c = preset(n)
        print(f"{n:>6}: ~{c.n_params_estimate()/1e9:.3f}B params  "
              f"(d_model={c.d_model}, layers={c.n_layers}, experts={c.n_experts})")
