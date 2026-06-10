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

`app.py` is a thin bootstrapper. On boot it:

1. clones / pulls the GitHub repo
   [`Undertaker-afk/Gpt-Protect`](https://github.com/Undertaker-afk/Gpt-Protect),
2. installs the repo's `requirements.txt`,
3. runs the repo's `main.py`, which launches the Gradio UI **and** a background
   realtime-training thread.

Because the model, optimizer state, training progress and all collected samples
are continuously written to `/data`, the Space **instantly resumes** where it
left off after any restart.

## Setup

1. Create a new **Gradio** Space.
2. Enable **persistent storage** (mounted at `/data`).
3. Copy `app.py` (only) into the Space — or copy the whole repo.
4. (Optional) set Space variables:

| Variable | Default | Purpose |
|---|---|---|
| `REPO_URL` | `https://github.com/Undertaker-afk/Gpt-Protect` | source repo |
| `REPO_BRANCH` | default branch | branch to track |
| `MODEL_PRESET` | `tiny` | `tiny` fits free CPU+16GB; `0.4b`/`5b` need big HW |
| `DATA_DIR` | `/data` | persistent bucket mount |
| `MAX_SEQ_LEN` | `192` | tokens per sample |
| `BATCH_SIZE` | `8` | realtime training batch |
| `SAVE_EVERY` | `25` | steps between checkpoints |
| `TRAIN` | `1` | set `0` to serve inference only |

## UI

* **🔍 Detect** — paste text → HUMAN/AI verdict, confidence, and a stylometric
  breakdown. Use the *HUMAN* / *AI* buttons to label the sample; it is saved to
  `/data/collected.jsonl` and folded into training immediately.
* **📈 Training dashboard** — live step / loss / accuracy / samples, a loss
  curve, and a recent-events log (auto-refresh every 3 s).

## Persistence layout (`/data`)

```
/data
├── checkpoints/last.pt     # model weights + step (atomic writes)
├── state.json              # global step, EMAs, best acc, samples seen
├── collected.jsonl         # user-labeled samples
└── train_log.jsonl         # event log
```
