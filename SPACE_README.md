---
title: GPT-Protect AI Text Detector
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# 🛡️ GPT-Protect — Realtime AI Text Detector

A MoE + Hybrid-Attention text detector (DeepSeek-V4-Pro / Google TurboQuant
inspired) that classifies text as **human** or **AI-generated** and keeps
**training in real time** from user feedback, persisting everything to the
Space's mounted storage bucket at `/data`.

## How this Space works

`app.py` is a thin bootstrapper **and auto-updater**. On boot it:

1. clones / pulls the GitHub repo
   [`Undertaker-afk/Gpt-Protect`](https://github.com/Undertaker-afk/Gpt-Protect)
   into persistent storage,
2. hands off to the repo's own `app.py` (so the supervisor stays updatable),
3. installs the repo's `requirements.txt` (only when it changes),
4. launches the repo's `main.py` (Gradio UI **+** background realtime-training
   thread) as a supervised child process.

### Auto-update (every 20 min)

The supervisor periodically runs `git fetch` and compares the local commit to
the remote. **Only if they differ** it:

1. sends the training process a graceful signal → it **pauses and writes a
   checkpoint** to `/data`,
2. `git pull`s the new code (and reinstalls requirements if they changed),
3. re-execs itself so updated `app.py`/`main.py` take effect, then resumes
   training from the saved checkpoint.

Because the model, optimizer state, training progress and all collected samples
are continuously written to `/data`, the Space **instantly resumes** where it
left off after any restart or update. The dashboard shows the current
local/remote commit and update status.

## Setup

1. Create a new **Gradio** Space.
2. Enable **persistent storage** (mounted at `/data`).
3. Copy `app.py` (only) into the Space — or copy the whole repo.
4. (Optional) set Space variables:

| Variable | Default | Purpose |
|---|---|---|
| `REPO_URL` | `https://github.com/Undertaker-afk/Gpt-Protect` | source repo |
| `REPO_BRANCH` | default branch | branch to track |
| `UPDATE_INTERVAL_SEC` | `1200` | seconds between GitHub update checks (20 min) |
| `UPDATE_GRACE_SEC` | `150` | seconds allowed for checkpoint-before-restart |
| `MODEL_PRESET` | `tiny` | `tiny` fits free CPU+16GB; `0.4b`/`5b` need big HW |
| `DATA_DIR` | `/data` | persistent bucket mount |
| `DATASET_TARGET` | `1500` | per-source in-memory sample target (harvester) |
| `SEED_PER_SOURCE` | `25` | rows pulled synchronously per source at startup |
| `HARVEST_CHUNK` | `40` | rows fetched per source per round-robin round |
| `HARVEST_PACE_SEC` | `0.4` | delay between fetches (dodges HF rate limits) |
| `HF_TOKEN` | — | **set this as a secret** for higher HF rate limits |
| `MAX_SEQ_LEN` | `192` | tokens per sample |
| `BATCH_SIZE` | `8` | realtime training batch |
| `SAVE_EVERY` | `25` | steps between checkpoints |
| `TRAIN` | `1` | set `0` to serve inference only |
| `PWD_ENV` | — | **admin-panel password** (required to use the Admin tab) |

## Datasets (auto-mixed, label-balanced)

`0 = human`, `1 = AI`. A **fair background harvester** round-robins all sources,
pulling small paced chunks from each up to `DATASET_TARGET`, caching them to
`/data/base_cache.jsonl` so the pool **grows across restarts** and **every**
source is represented (not just the first few). Per-source progress is shown on
the dashboard. If you see only a couple of sources fill up, set an `HF_TOKEN`
secret — unauthenticated streaming gets rate-limited.

* `alex-kudryashov/dlr-hw-2-human-ai-texts`
* `nbroad/basic_text_dataset` (human)
* `mehddii/ai-text-detector-v2`
* `AlekseyKorshuk/ai-text-classification`
* `ziq/ai-generated-text-classification`
* `NabeelShar/ai_and_human_text`
* `akoukas/AITextDetectionDataset`
* `dmitva/human_ai_generated_text` (paired human/AI columns → 2 rows each)

## UI

* **🔍 Detect** — paste text → HUMAN/AI verdict, confidence, the **neural vs.
  heuristic AI-score** (and whether they agree), the **top AI signals**, and a
  full pattern/stylometry breakdown. The analyzed text is shown as a
  **sentence-level AI heatmap** (red = AI-leaning, amber = mixed, green =
  human-leaning) with a **per-sentence/phrase table** — phrase length, char
  count, avg word length, long-word count, the AI-tell words detected, and both
  the **statistics AI%** and the **model AI%** for each sentence. Use the
  *HUMAN* / *AI* buttons to label the sample; it is saved to
  `/data/collected.jsonl` and folded into training at once.
* **📈 Training dashboard** — live step / loss / accuracy / throughput /
  precision-recall-F1, **loss & accuracy curves**, **pool-composition** and
  **confusion** bar charts, a recent-events log (auto-refresh every 3 s), and a
  **"Check GitHub for update now"** button that forces an immediate update check.

The model has an **intelligent AI-pattern engine** (`ai_patterns.py`) fused into
it — 24 signals (burstiness, perplexity proxy, repetition, AI-tell lexicon, …) —
so detection is strong even early in training. The detect heatmap also paints
**individual AI-tell words** (purple) on top of the per-sentence coloring.

* **🛠️ Admin** — password-gated (Space secret `PWD_ENV`). Actions:
  * **Delete local models** — remove checkpoints + reinitialize weights.
  * **Reset everything** — wipe models, training state, collected samples and the
    base-data cache, then re-seed (code untouched).
  * **Repull & reset** — `git fetch` + hard-reset code to remote, reinstall,
    restart (supervisor handles it; /data kept).
  * **☢️ NUKE** — delete the local repo, every checkpoint and **every file in the
    bucket**, then re-clone and set everything up from scratch (type `NUKE` to
    confirm). No undo.

## Persistence layout (`/data`)

```
/data
├── checkpoints/last.pt     # model weights + step (atomic writes)
├── state.json              # global step, EMAs, best acc, samples seen
├── collected.jsonl         # user-labeled samples
└── train_log.jsonl         # event log
```
