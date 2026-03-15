# program.md — Agent Instructions for Whisper Fine-tuning Research

> This file is for future use with an autoresearch-style loop.
> Once you've validated the approach works manually, point an agent here.

## Objective

Minimize Word Error Rate (WER) on the test set for a personal Whisper model fine-tuned on the owner's voice. The test set contains Malayalam-English code-switching speech.

## Metric

`overall_wer` from `python eval.py --model output/whisper-small-personal/final`

Lower is better. Current baseline (vanilla whisper-small): TBD after first run.

## What you can modify

**Only modify `finetune.py`.** Specifically the `TrainConfig` dataclass and training logic.

Things to explore:
- LoRA rank (r): try 4, 8, 16, 32, 64
- LoRA alpha: try 2x rank as starting point
- Learning rate: sweep 5e-5 to 5e-4
- Whether to freeze encoder or not (`model.freeze_encoder()` vs training full model with LoRA)
- Target modules: try adding `fc1`, `fc2` (FFN layers) in addition to attention
- Batch size and gradient accumulation combinations
- Number of epochs (2-10)
- Warmup ratio
- Optimizer: try AdamW vs Adam with different weight decay
- Data augmentation: speed perturbation, noise injection in the dataset class
- Whether to use language hints or let Whisper auto-detect

## What you cannot modify

- `prepare_data.py` — data pipeline is fixed
- `eval.py` — evaluation is fixed
- Test set transcriptions — ground truth is fixed

## Experiment protocol

1. Modify `finetune.py` with your hypothesis
2. Run: `python finetune.py`
3. Evaluate: `python eval.py --compare`
4. If `overall_wer` improved → commit with message describing the change and the delta
5. If not → revert and try something different
6. Log every experiment result (even failures) in `experiments.log`
