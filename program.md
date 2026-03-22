# program.md — Agent Instructions for Whisper LoRA Fine-tuning Research

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare_data.py` — fixed data pipeline, chunking, Sarvam API transcription. Do not modify.
   - `eval.py` — fixed evaluation (WER/CER computation). Do not modify.
   - `finetune.py` — the file you modify. LoRA config, hyperparameters, dataset class, training loop.
4. **Verify data exists**: Check that `data/train/transcriptions.jsonl` and `data/test/transcriptions.jsonl` exist and are non-empty. If not, tell the human to run `python prepare_data.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (A40, 48GB VRAM). The experiment is a two-step process — train then evaluate:

```
python finetune.py > run.log 2>&1
python eval.py --model output/whisper-small-personal/final >> run.log 2>&1
```

**What you CAN do:**
- Modify `finetune.py` — this is the only file you edit. Everything in it is fair game: LoRA configuration, hyperparameters, optimizer, training loop, data augmentation, whether to freeze the encoder, scheduler, etc.

**What you CANNOT do:**
- Modify `prepare_data.py`. It is read-only. It contains the fixed data pipeline.
- Modify `eval.py`. It is read-only. It contains the fixed evaluation harness — the ground truth metric.
- Modify `transcribe.py`. It is read-only.
- Modify anything in `data/` — the train/test data and ground truth transcriptions are fixed.
- Install new packages or add dependencies beyond what's in `requirements.txt`.
- Change `output_dir` in TrainConfig — eval.py expects the model at `output/whisper-small-personal/final`.
- Change `base_model` from `openai/whisper-small` — eval.py hardcodes this as the base for LoRA loading.

**The goal is simple: get the lowest overall_wer.** The model is evaluated on Malayalam-English code-switching speech transcribed in Latin script (Manglish). Everything is fair game within `finetune.py`: change the LoRA config, the optimizer, the hyperparameters, the training loop, the data augmentation strategy. The only constraint is that the code runs without crashing and produces a model at the expected output path.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful overall_wer gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.5% WER improvement that adds 50 lines of hacky code? Probably not worth it. A 0.5% WER improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run finetune.py and eval.py as-is.

## Output format

Once eval finishes it prints a summary like this:

```
============================================================
Model: output/whisper-small-personal/final
============================================================
  Samples:     16
  Overall WER: 78.10%
  Overall CER: 45.32%
```

You can extract the key metric from the log file:

```
grep "Overall WER" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 4 columns:

```
commit	overall_wer	status	description
```

1. git commit hash (short, 7 chars)
2. overall_wer achieved as a percentage (e.g. 78.10) — use 0.00 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried

Example:

```
commit	overall_wer	status	description
a1b2c3d	78.10	keep	baseline
b2c3d4e	72.50	keep	increase LoRA rank to 32
c3d4e5f	80.20	discard	remove encoder freezing
d4e5f6g	0.00	crash	added noise augmentation (shape mismatch)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit you're on
2. Modify `finetune.py` with an experimental idea
3. git commit
4. Run the experiment: `python finetune.py > run.log 2>&1 && python eval.py --model output/whisper-small-personal/final >> run.log 2>&1`
5. Read out the results: `grep "Overall WER" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If overall_wer improved (lower), you "advance" the branch, keeping the git commit
9. If overall_wer is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read finetune.py for new angles, try combining previous near-misses, try different LoRA configurations, experiment with data augmentation, adjust learning rate schedules, try different epoch counts. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. The user then wakes up to experimental results, all completed by you while they slept!
