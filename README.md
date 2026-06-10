# 🛡️ GPT-Protect — AI Text Detector

A training suite + realtime HuggingFace Space for detecting **AI-generated text**.
The model architecture borrows the ideas that make **DeepSeek-V4-Pro** and
Google's **TurboQuant** efficient, adapted to a text classifier:

| Idea | Where | File |
|---|---|---|
| **MoE** experts (FP4-simulated, top-k routing, shared expert, load-balance loss) | backbone FFN | `moe.py` |
| **Hybrid Attention** — CSA (compress-4 sparse) + HCA (compress-128 dense) + GQA + RoPE | every block | `attention.py` |
| **mHC** — manifold HyperConnections (multi-stream residuals) for deep-MoE stability | residuals | `moe.py`, `architecture.py` |
| **TurboQuant** — 3-bit KV cache (PolarQuant + QJL JL-sketch), plug-and-play | inference cache | `quant.py` |
| **Muon** optimizer (Newton–Schulz orthogonalized momentum) for matrices, AdamW for the rest | training | `muon.py` |

## Files

```
config.py         model/train configs + presets (tiny | 0.4b | 5b)
architecture.py   RMSNorm, TransformerBlock, Backbone (mHC residual streams)
attention.py      HybridAttention (CSA + HCA), RoPE, TurboQuant KV cache hook
moe.py            Mixture-of-Experts FFN (FP4), HyperConnection (mHC)
quant.py          TurboQuant: PolarQuant + QJL 3-bit KV compression
model.py          AITextDetector = Backbone + pooling + 2-way head
preprocessor.py   conservative text cleaning + stylometric features
augmentation.py   label-preserving robustness augmentations
dataset.py        unifies the two HF datasets -> {text, label} (0=human,1=ai)
data_loader.py    tokenizer + padded collate + DataLoaders
train.py          training driver (Muon, cosine LR, eval, checkpointing)
main.py           Gradio UI + realtime/continual training (HF Space)
app.py            HF Space bootstrapper (clones this repo, runs main.py)
```

## Model presets

| preset | params | use |
|---|---|---|
| `tiny` | ~19M | CPU / free Space realtime training |
| `0.4b` | ~0.42B | the "start training" model (1 GPU) |
| `5b` | ~5.3B | full model (multi-GPU) |

```bash
python config.py        # prints param counts for each preset
```

## Datasets

* [`alex-kudryashov/dlr-hw-2-human-ai-texts`](https://huggingface.co/datasets/alex-kudryashov/dlr-hw-2-human-ai-texts) — labeled human/AI (500/500)
* [`nbroad/basic_text_dataset`](https://huggingface.co/datasets/nbroad/basic_text_dataset) — human text (label = human)

The corpus is human-heavy, so the trainer samples **label-balanced** batches.

## Train

```bash
pip install -r requirements.txt

# CPU smoke run
python train.py --preset tiny --max-samples 4000 --epochs 1 --batch-size 8 --max-seq-len 128

# the 0.4B model (GPU recommended)
python train.py --preset 0.4b --epochs 3 --batch-size 16 --max-seq-len 512
```

Checkpoints + JSON metric logs are written to `checkpoints/` and `/tmp/logs/`.

## HuggingFace Space (realtime training)

`app.py` is the Space entry point. It clones this repo, installs requirements,
and runs `main.py`, which serves a Gradio UI **and** a background continual
training loop. Everything (weights, optimizer step, collected samples) is
persisted to the Space's `/data` bucket, so a restart instantly resumes.

See [`SPACE_README.md`](SPACE_README.md) for Space setup. On free CPU hardware
use `MODEL_PRESET=tiny`.

### UI
* **🔍 Detect** — paste text → HUMAN/AI verdict + confidence + stylometric features; label it to train the model live.
* **📈 Dashboard** — live step / loss / accuracy / samples + loss curve (auto-refresh).

## Notes & honesty

* This is a from-scratch, research-flavored **reimplementation** of the named
  techniques (TurboQuant, DeepSeek-V4-Pro hybrid attention, mHC, Muon) at small
  scale — not the original proprietary code. FP4/3-bit paths are *fake-quant*
  simulations that run on CPU and map to real low-bit kernels on GPU/H100.
* The sandbox is CPU-only, so the 0.4B/5B presets are provided as configs;
  the validated end-to-end runs here use the `tiny` preset.
