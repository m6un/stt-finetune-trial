"""
finetune.py — LoRA fine-tune whisper-small on your personal speech data.

This is the file the autoresearch loop would modify.
For now, it's a solid baseline you run manually.
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ============================================================================
# Config — autoresearch agent would sweep over these
# ============================================================================

@dataclass
class TrainConfig:
    # Model
    base_model: str = "openai/whisper-small"
    language: str = None      # None = auto-detect (important for code-switching)
    task: str = "transcribe"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Target modules — Whisper attention projections
    lora_target_modules: list = None

    # Training
    learning_rate: float = 1e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True

    # Data
    train_jsonl: str = "data/train/transcriptions.jsonl"
    sample_rate: int = 16000
    max_audio_len: float = 30.0  # seconds

    # Output
    output_dir: str = "output/whisper-small-personal"
    save_every_n_steps: int = 100
    log_every_n_steps: int = 10

    def __post_init__(self):
        if self.lora_target_modules is None:
            # Target all attention projections in both encoder and decoder
            self.lora_target_modules = [
                "k_proj", "v_proj", "q_proj", "out_proj",
            ]


# ============================================================================
# Dataset
# ============================================================================

class PersonalSpeechDataset(Dataset):
    def __init__(self, jsonl_path: str, processor: WhisperProcessor, config: TrainConfig):
        self.processor = processor
        self.config = config
        self.samples = []

        with open(jsonl_path) as f:
            for line in f:
                sample = json.loads(line)
                self.samples.append(sample)

        print(f"Loaded {len(self.samples)} training samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio, sr = librosa.load(
            sample["audio_path"],
            sr=self.config.sample_rate,
            mono=True,
        )

        # Truncate if too long
        max_samples = int(self.config.max_audio_len * self.config.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Process audio → mel spectrogram
        input_features = self.processor(
            audio,
            sampling_rate=self.config.sample_rate,
            return_tensors="np",
        ).input_features[0]

        # Process text → token IDs
        transcription = sample.get("transcription", "")
        labels = self.processor.tokenizer(
            transcription,
            return_tensors="np",
            padding=False,
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
        }


def collate_fn(batch, pad_token_id: int):
    """Collate with padding for variable-length labels."""
    input_features = torch.stack([
        torch.tensor(x["input_features"]) for x in batch
    ])

    # Pad labels to max length in batch
    max_label_len = max(len(x["labels"]) for x in batch)
    padded_labels = []
    for x in batch:
        labels = x["labels"]
        pad_len = max_label_len - len(labels)
        padded = np.concatenate([labels, np.full(pad_len, pad_token_id)])
        padded_labels.append(torch.tensor(padded, dtype=torch.long))

    labels = torch.stack(padded_labels)
    # Replace padding with -100 so loss ignores it
    labels[labels == pad_token_id] = -100

    return {"input_features": input_features, "labels": labels}


# ============================================================================
# Training loop
# ============================================================================

def train(config: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Base model: {config.base_model}")

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(config.base_model)
    model = WhisperForConditionalGeneration.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16 if config.fp16 else torch.float32,
    )

    # Freeze encoder — we mainly want to adapt the decoder for our vocab/style
    # (Can experiment with unfreezing encoder too)
    model.freeze_encoder()

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # Dataset and dataloader
    dataset = PersonalSpeechDataset(config.train_jsonl, processor, config)
    pad_token_id = processor.tokenizer.pad_token_id

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    total_steps = len(dataloader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training
    os.makedirs(config.output_dir, exist_ok=True)
    global_step = 0
    best_loss = float("inf")
    loss_accum = 0.0

    print(f"\nTraining for {config.num_epochs} epochs, {total_steps} optimizer steps")
    print(f"  Batch size: {config.batch_size} x {config.gradient_accumulation_steps} accumulation")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print()

    model.train()
    scaler = torch.amp.GradScaler("cuda") if config.fp16 and device == "cuda" else None

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_features=input_features, labels=labels)
                    loss = outputs.loss / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss / config.gradient_accumulation_steps
                loss.backward()

            loss_accum += loss.item()
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            n_batches += 1

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % config.log_every_n_steps == 0:
                    avg_loss = loss_accum / config.log_every_n_steps
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", step=global_step)
                    loss_accum = 0.0

                if global_step % config.save_every_n_steps == 0:
                    ckpt_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_dir)
                    print(f"\n  Saved checkpoint: {ckpt_dir}")

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_dir = os.path.join(config.output_dir, "best")
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"  New best model saved: {best_dir}")

    # Save final model
    final_dir = os.path.join(config.output_dir, "final")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"\nTraining complete! Final model: {final_dir}")
    print(f"Best loss: {best_loss:.4f}")

    # Save config for reproducibility
    config_path = os.path.join(config.output_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with LoRA")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--train-data", default="data/train/transcriptions.jsonl")
    parser.add_argument("--output-dir", default="output/whisper-small-personal")
    args = parser.parse_args()

    config = TrainConfig(
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        train_jsonl=args.train_data,
        output_dir=args.output_dir,
    )

    train(config)


if __name__ == "__main__":
    main()
