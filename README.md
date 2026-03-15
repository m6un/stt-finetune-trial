# whisper-personal

Fine-tune Whisper-small on your own voice using LoRA. Optimized for multilingual code-switching (Malayalam/English).

## Setup (RunPod)

Spin up a RunPod instance with a PyTorch template (any GPU with 16GB+ VRAM works — A40, A100, RTX 4090).

```bash
git clone <your-repo>
cd whisper-personal
pip install -r requirements.txt
```

## Workflow

### 1. Record yourself (on your phone)

Record 20-30 minutes of natural speech as voice memos. Mix it up:
- Pure English (tech discussions, thinking out loud)
- Pure Malayalam (casual conversation)
- Code-switching (your natural default — mixing both mid-sentence)
- Tech jargon heavy (say things like "agentic harness", "semantic diff", "LoRA fine-tune")

Transfer the audio files to `data/raw/` on your RunPod instance.

### 2. Prepare data

```bash
# Chunks audio, generates pseudo-labels with whisper-large-v3
python prepare_data.py

# Review and hand-correct the TEST set transcriptions (important!)
# Edit: data/test/transcriptions.jsonl
```

### 3. Baseline eval

```bash
# Measure WER of vanilla whisper-small on your test set
python eval.py --model openai/whisper-small
```

### 4. Fine-tune

```bash
python finetune.py
```

### 5. Eval fine-tuned model

```bash
python eval.py --model output/whisper-small-personal/final
```

### 6. Compare

The eval script prints WER breakdown: overall, English-only, Malayalam-only, code-switched.
If the delta is meaningful, invest in proper data collection. If not, rethink the approach.

## Project structure

```
data/
  raw/              ← your voice memos (m4a, mp3, wav)
  chunks/           ← 15-30s audio segments
  train/            ← training set (pseudo-labeled)
  test/             ← test set (hand-corrected transcriptions)
output/             ← fine-tuned model checkpoints
```
