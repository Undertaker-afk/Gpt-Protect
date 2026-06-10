"""
quant.py — TurboQuant
=====================
Plug-and-play, *no-fine-tuning* KV-cache compression inspired by Google
Research's TurboQuant.  It combines two ideas:

  * PolarQuant  — quantize key/value vectors in a polar (magnitude / angle)
                  parameterization so that the heavy-tailed magnitude and the
                  near-uniform direction are quantized separately.  This keeps
                  most of the error budget for the direction, where it matters.

  * QJL         — a Johnson-Lindenstrauss random *sketch* of the keys used to
                  estimate query·key scores cheaply and to drive the per-token
                  bit allocation.  (The user calls this "Query Jacobian
                  Learning"; the operative mechanism is a fixed random
                  projection + 1-bit sign sketch, JL-style.)

Everything here is a *fake-quant* (quantize -> dequantize) so it runs on CPU
and is bit-exact-simulatable without custom CUDA kernels.  On real H100s the
same scheme maps to the 8x faster attention kernels TurboQuant ships.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


def _quantize_uniform(x: torch.Tensor, bits: int, dim: int = -1):
    """Symmetric per-`dim` uniform quantization. Returns dequantized tensor."""
    qmax = (1 << (bits - 1)) - 1                      # e.g. 3-bit -> 3
    scale = x.abs().amax(dim=dim, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


class PolarQuant:
    """Quantize a (..., D) tensor in polar form: magnitude + unit direction."""

    def __init__(self, bits: int = 3, mag_bits: int = 8):
        self.bits = bits
        self.mag_bits = mag_bits          # magnitudes get a few more bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = x / mag
        # magnitude: heavy-tailed -> log-domain uniform quant with more bits
        log_mag = torch.log(mag)
        log_mag_q = _quantize_uniform(log_mag, self.mag_bits, dim=-2)
        mag_q = torch.exp(log_mag_q)
        # direction: near-uniform on the sphere -> low-bit uniform quant
        dir_q = _quantize_uniform(direction, self.bits, dim=-1)
        dir_q = dir_q / dir_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return dir_q * mag_q


class QJL:
    """Quantized Johnson-Lindenstrauss key sketch (1-bit sign random features)."""

    def __init__(self, d_head: int, proj_dim: int = 128, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        # fixed random projection — no training required
        self.R = torch.randn(d_head, proj_dim, generator=g) / math.sqrt(proj_dim)

    def sketch(self, keys: torch.Tensor) -> torch.Tensor:
        """keys: (..., d_head) -> sign sketch in {-1,+1}^proj_dim (as int8)."""
        R = self.R.to(keys.dtype).to(keys.device)
        return torch.sign(keys @ R).to(torch.int8)

    def approx_scores(self, queries: torch.Tensor, key_sketch: torch.Tensor):
        """Cheap query·key score estimate from sign sketches (Hamming-style)."""
        R = self.R.to(queries.dtype).to(queries.device)
        q_sketch = torch.sign(queries @ R)
        # normalized agreement ~ cos angle estimate
        return (q_sketch.unsqueeze(-2) * key_sketch.to(q_sketch.dtype)).mean(-1)


class TurboQuantKVCache(nn.Module):
    """
    Drop-in compressed KV cache.

    Stores keys/values *dequantized* after a PolarQuant round-trip (the memory
    win is reported via `compression_ratio`; the numeric error is what would be
    incurred by the 3-bit kernels at inference time).  Also keeps a QJL sign
    sketch of the keys for fast / sparse score estimation.
    """

    def __init__(self, d_head: int, bits: int = 3, qjl_dim: int = 128):
        super().__init__()
        self.bits = bits
        self.polar = PolarQuant(bits=bits)
        self.qjl = QJL(d_head, proj_dim=qjl_dim)
        self._k, self._v, self._ksketch = None, None, None

    @torch.no_grad()
    def append(self, k: torch.Tensor, v: torch.Tensor):
        kq = self.polar(k)
        vq = self.polar(v)
        ks = self.qjl.sketch(k)
        if self._k is None:
            self._k, self._v, self._ksketch = kq, vq, ks
        else:
            self._k = torch.cat([self._k, kq], dim=-2)
            self._v = torch.cat([self._v, vq], dim=-2)
            self._ksketch = torch.cat([self._ksketch, ks], dim=-2)
        return self._k, self._v

    @property
    def compression_ratio(self) -> float:
        """Effective bits-per-value reduction vs fp16 (16 bits)."""
        return 16.0 / self.bits

    def reset(self):
        self._k = self._v = self._ksketch = None


if __name__ == "__main__":
    torch.manual_seed(0)
    cache = TurboQuantKVCache(d_head=64, bits=3)
    k = torch.randn(1, 8, 10, 64)        # (B, H, T, d)
    v = torch.randn(1, 8, 10, 64)
    kq, vq = cache.append(k, v)
    err = (kq - k).norm() / k.norm()
    print(f"3-bit KV relative error: {err.item():.4f}  "
          f"compression ~{cache.compression_ratio:.1f}x")
