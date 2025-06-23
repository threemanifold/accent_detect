# Accent Detector

A tiny command‑line helper that guesses the English accent in any public video or audio link.

---

## Prerequisites

| Tool | Version | Why |
| ---- | ------- | --- |
|      |         |     |

| **Python**   | **3.12** (tested) | Modern `match` syntax & proper venv support           |
| ------------ | ----------------- | ----------------------------------------------------- |
| **FFmpeg**   | 4.x or newer      | Audio extraction & resampling (`brew install ffmpeg`) |
| **Homebrew** | latest            | Easiest way to install FFmpeg on macOS                |

---

## Installation

```bash
# 1 ⸺ create & activate a virtual‑env (recommended)
python3.12 -m venv venv
source venv/bin/activate

# 2 ⸺ install Python deps
pip install -r requirements.txt
```

The requirements file installs:

* **torch / torchaudio 2.2** (CPU or MPS build)
* **yt‑dlp** – multi‑site video downloader
* **pydub + soundfile** – simple audio slicing
* **speechbrain** – ready‑made accent classifier

All large model checkpoints are cached under:

```
~/.cache/speechbrain_models/
```

---

## Quick start

```bash
python accent_detect.py "https://www.youtube.com/watch?v=qaxwf3BIZIQ"
```

Typical output:

```
↓  Downloading & extracting audio …
✂︎  Creating snippets …
🔍  Classifying accents …
[Snippet 1]     indian   score=0.9999
[Snippet 2]     indian   score=1.0000
[Snippet 3]     indian   score=1.0000
[Snippet 4]         us   score=0.9680
[Snippet 5]     indian   score=0.9996

>>> Clip‑level prediction: indian   (p = 1.000, vote share = 80 %)
```

---

## What the script does

1. **Download & Convert** – We download the video from the given link and extract the audio.
2. **Sample** – The original model was trained on short audio clips, so we randomly extract five 5‑second snippets from the audio track. This keeps the input shape consistent with the training regime and boosts robustness by letting the classifier vote across multiple inferences.
3. **Classify each snippet** Using SpeechBrain’s *CommonAccent XLS‑R* model we classify each snippet.(16 English accents).
4. **Fuse predictions** – snippet log‑probabilities are summed, a soft‑max turns them back into probabilities, and the top class is reported. We also show the simple vote‑share (% of snippets where that accent won) for extra transparency.

---

### Reading the output

| Field        | Meaning                                                                                                                                                                                                                                                                                           |
|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **p**        | Confidence score for the *overall* accent guess. A value near **1.00** means the model is almost certain; **0.50** would be a coin-flip. This is model's own judgement about its confidence. Based on my observation this is an overconfident model so we add second measure of confidence below. |
| **vote&nbsp;share** | Percentage of the five snippets that agreed with the final guess. For example, **80 %** means 4 out of 5 snippets pointed to the same accent.                                                                                                                                                     |

## Customisation

| Flag / code edit                                     | Effect                                          |
| ---------------------------------------------------- | ----------------------------------------------- |
| `count=5` → `count=N` in `save_random_5s_snippets()` | Take more/fewer snippets                        |
| `SEG_MS = 5_000`                                     | Change snippet length                           |
| Swap model ID                                        | Test the `ecapa` or Whisper‑based accent models |

---

## Cleanup & temp files

* The original download is stored in a **temporary directory** that is auto‑deleted at the end of the run.
* Only two things remain:

  * `data/download.wav` – single WAV of the full track (optional; comment out if you don’t need it).
  * `snippets/*_snippet#.wav` – the tiny 5‑s windows.

Delete either whenever you like.
