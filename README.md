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
| **AI-pattern engine** (burstiness, perplexity proxy, repetition, AI-tell lexicon, …) fused into the classifier + heuristic score | features | `ai_patterns.py` |

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

`dataset.py` unifies many human-vs-AI corpora (streamed, per-source capped) into
one labeled stream (`0=human, 1=ai`):

* `alex-kudryashov/dlr-hw-2-human-ai-texts`
* `nbroad/basic_text_dataset` (human)
* `mehddii/ai-text-detector-v2`
* `AlekseyKorshuk/ai-text-classification`
* `ziq/ai-generated-text-classification`
* `NabeelShar/ai_and_human_text`
* `akoukas/AITextDetectionDataset`
* `dmitva/human_ai_generated_text` (paired columns → 2 rows each)

Add more by appending to `DATASET_SPECS`. The corpus is human-heavy, so the
realtime trainer samples **label-balanced** batches.

On the Space these are pulled by a **fair background harvester** (`main.py`):
it round-robins all sources, fetching small paced chunks from each up to
`DATASET_TARGET`, and caches them to `/data/base_cache.jsonl` so the pool grows
across restarts and **every** source is represented equally — not just the first
couple (which is what happens if unauthenticated streaming gets rate-limited; set
an `HF_TOKEN` secret to avoid that). The dashboard shows per-source progress.

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
* **🔍 Detect** — paste text → HUMAN/AI verdict, confidence, **neural vs. heuristic
  AI-score agreement**, top contributing AI signals, and a full pattern/stylometry
  breakdown; label it to train the model live. The analyzed text is rendered as a
  **sentence-level AI heatmap** (red/amber/green) plus a **per-sentence/phrase table**
  (word & char counts, avg word length, long-word count, the AI-tell words found, and
  both the **statistics AI%** and the **model AI%** for every sentence).
* **📈 Dashboard** — live step / loss / accuracy / throughput / precision-recall-F1,
  **loss & accuracy curves**, **training-pool composition** and **confusion tallies**
  bar charts, plus a **"Check GitHub for update now"** button (forces the auto-updater
  to check immediately instead of waiting 20 min).

### Intelligent AI-pattern detection (`ai_patterns.py`)
24 hand-engineered signals (burstiness, sentence-length uniformity, type-token /
hapax ratios, n-gram repetition, word/char entropy as a perplexity proxy,
punctuation & contraction habits, an "AI-tell" lexicon, sentence-starter
diversity, …). They are **fused into the model** (projected + concatenated with the
pooled transformer features) so it has strong priors from step 0, and also power a
transparent `heuristic_ai_score` shown alongside the neural prediction.

## Notes & honesty

* This is a from-scratch, research-flavored **reimplementation** of the named
  techniques (TurboQuant, DeepSeek-V4-Pro hybrid attention, mHC, Muon) at small
  scale — not the original proprietary code. FP4/3-bit paths are *fake-quant*
  simulations that run on CPU and map to real low-bit kernels on GPU/H100.
* The sandbox is CPU-only, so the 0.4B/5B presets are provided as configs;
  the validated end-to-end runs here use the `tiny` preset.
