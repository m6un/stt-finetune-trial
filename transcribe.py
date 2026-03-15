"""
transcribe.py — Use your fine-tuned Whisper model for inference.

Usage:
    python transcribe.py audio.m4a
    python transcribe.py audio.m4a --model output/whisper-small-personal/final
    python transcribe.py data/raw/  # transcribe entire directory
"""

import sys
import argparse
from pathlib import Path

import torch
import librosa
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
from peft import PeftModel


BASE_MODEL = "openai/whisper-small"
DEFAULT_FINETUNED = "output/whisper-small-personal/final"
SAMPLE_RATE = 16000


def load_model(model_path: str, device: str):
    """Load model — handles both base and LoRA checkpoints."""
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    is_lora = (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        base_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
        processor = WhisperProcessor.from_pretrained(model_path)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
        )
        processor = WhisperProcessor.from_pretrained(model_path)

    model.to(device)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs={"task": "transcribe", "return_timestamps": True},
        chunk_length_s=30,
        batch_size=8,
    )

    return pipe


def transcribe_file(pipe, audio_path: str) -> str:
    """Transcribe a single audio file."""
    result = pipe(audio_path)
    return result["text"].strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio with your fine-tuned Whisper")
    parser.add_argument("input", help="Audio file or directory of audio files")
    parser.add_argument("--model", default=DEFAULT_FINETUNED, help="Model path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    pipe = load_model(args.model, args.device)

    extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm", ".mp4"}

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted([f for f in input_path.rglob("*") if f.suffix.lower() in extensions])

    if not files:
        print(f"No audio files found in {input_path}")
        sys.exit(1)

    for f in files:
        text = transcribe_file(pipe, str(f))
        if len(files) > 1:
            print(f"\n--- {f.name} ---")
        print(text)


if __name__ == "__main__":
    main()
