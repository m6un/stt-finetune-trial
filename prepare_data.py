"""
prepare_data.py — Takes raw voice memos, chunks them, and generates pseudo-labels.

Uses whisper-large-v3 as the "teacher" to transcribe your audio.
You'll hand-correct a small test split for honest evaluation.
"""

import os
import json
import argparse
import random
from pathlib import Path

import re

import torch
import librosa
import soundfile as sf
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from ml2en import ml2en


# --- Config ---
RAW_DIR = "data/raw"
CHUNKS_DIR = "data/chunks"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"

CHUNK_DURATION = 20  # seconds — sweet spot for Whisper
CHUNK_OVERLAP = 2    # seconds overlap between chunks
SAMPLE_RATE = 16000  # Whisper expects 16kHz
TEST_SPLIT = 0.15    # 15% of chunks go to test set

TEACHER_MODEL = "openai/whisper-large-v3"

# Malayalam Unicode range
_MALAYALAM_RE = re.compile(r'[\u0D00-\u0D7F]+')


def to_manglish(text: str) -> str:
    """Convert Malayalam script portions to Manglish, leave English as-is."""
    def replace_match(match):
        return ml2en.transliterate(match.group(0))
    return _MALAYALAM_RE.sub(replace_match, text)


def find_audio_files(raw_dir: str) -> list[Path]:
    """Find all audio files in the raw directory."""
    extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm", ".mp4"}
    raw_path = Path(raw_dir)
    files = [f for f in raw_path.rglob("*") if f.suffix.lower() in extensions]
    print(f"Found {len(files)} audio files in {raw_dir}")
    return sorted(files)


def chunk_audio(audio_path: Path, output_dir: str) -> list[dict]:
    """Split a single audio file into fixed-length chunks."""
    print(f"  Chunking: {audio_path.name}")

    # Load and resample to 16kHz mono
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    total_duration = len(audio) / sr
    print(f"    Duration: {total_duration:.1f}s")

    chunks = []
    chunk_samples = CHUNK_DURATION * sr
    step_samples = (CHUNK_DURATION - CHUNK_OVERLAP) * sr
    idx = 0
    offset = 0

    while offset < len(audio):
        end = min(offset + chunk_samples, len(audio))
        chunk = audio[offset:end]

        # Skip very short trailing chunks (< 3s)
        if len(chunk) / sr < 3.0:
            break

        chunk_name = f"{audio_path.stem}_chunk{idx:04d}.wav"
        chunk_path = Path(output_dir) / chunk_name
        sf.write(str(chunk_path), chunk, sr)

        chunks.append({
            "audio_path": str(chunk_path),
            "source_file": audio_path.name,
            "chunk_idx": idx,
            "start_sec": offset / sr,
            "end_sec": end / sr,
            "duration_sec": len(chunk) / sr,
        })

        offset += step_samples
        idx += 1

    print(f"    → {len(chunks)} chunks")
    return chunks


def transcribe_chunks(chunks: list[dict], device: str) -> list[dict]:
    """Transcribe all chunks using the teacher model (whisper-large-v3)."""
    print(f"\nLoading teacher model: {TEACHER_MODEL}")
    print(f"  Device: {device}")

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        TEACHER_MODEL,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(TEACHER_MODEL)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        # Force Malayalam — auto-detect misidentifies it as Hindi/Tamil
        generate_kwargs={
            "task": "transcribe",
            "language": "ml",
            "return_timestamps": True,
        },
    )

    print(f"\nTranscribing {len(chunks)} chunks...")
    for chunk in tqdm(chunks):
        result = pipe(chunk["audio_path"])
        raw_text = result["text"].strip()
        chunk["transcription_raw"] = raw_text  # keep original for reference
        chunk["transcription"] = to_manglish(raw_text)  # convert to manglish
        # Store detected chunks for reference
        if "chunks" in result:
            chunk["word_timestamps"] = result["chunks"]

    return chunks


def split_and_save(chunks: list[dict]):
    """Split into train/test and save as JSONL."""
    random.seed(42)
    random.shuffle(chunks)

    n_test = max(1, int(len(chunks) * TEST_SPLIT))
    test_chunks = chunks[:n_test]
    train_chunks = chunks[n_test:]

    print(f"\nSplit: {len(train_chunks)} train, {len(test_chunks)} test")

    # Save train set
    train_jsonl = Path(TRAIN_DIR) / "transcriptions.jsonl"
    with open(train_jsonl, "w") as f:
        for chunk in train_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Train saved: {train_jsonl}")

    # Save test set — these need hand-correction
    test_jsonl = Path(TEST_DIR) / "transcriptions.jsonl"
    with open(test_jsonl, "w") as f:
        for chunk in test_chunks:
            # Add a field for hand-corrected transcription
            chunk["corrected_transcription"] = chunk["transcription"]  # default to pseudo-label
            chunk["needs_correction"] = True
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Test saved: {test_jsonl}")

    # Also save a human-friendly correction file
    correction_file = Path(TEST_DIR) / "CORRECT_THESE.md"
    with open(correction_file, "w") as f:
        f.write("# Test Set — Hand-Correct These Transcriptions\n\n")
        f.write("Listen to each audio file and fix the transcription in transcriptions.jsonl.\n")
        f.write("Edit the `corrected_transcription` field. This is your ground truth for eval.\n\n")
        for i, chunk in enumerate(test_chunks):
            f.write(f"## {i+1}. {Path(chunk['audio_path']).name}\n")
            f.write(f"- Duration: {chunk['duration_sec']:.1f}s\n")
            f.write(f"- Auto-transcription: {chunk['transcription']}\n\n")
    print(f"  Correction guide: {correction_file}")


def main():
    parser = argparse.ArgumentParser(description="Prepare audio data for Whisper fine-tuning")
    parser.add_argument("--raw-dir", default=RAW_DIR, help="Directory with raw audio files")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip transcription (if already done)")
    args = parser.parse_args()

    # Create directories
    for d in [CHUNKS_DIR, TRAIN_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    # Step 1: Find audio files
    audio_files = find_audio_files(args.raw_dir)
    if not audio_files:
        print(f"\nNo audio files found in {args.raw_dir}/")
        print("Transfer your voice memos there first.")
        return

    # Step 2: Chunk audio
    print(f"\nChunking audio into ~{CHUNK_DURATION}s segments...")
    all_chunks = []
    for audio_file in audio_files:
        chunks = chunk_audio(audio_file, CHUNKS_DIR)
        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Step 3: Transcribe with teacher model
    if not args.skip_transcribe:
        all_chunks = transcribe_chunks(all_chunks, args.device)
    else:
        print("Skipping transcription (--skip-transcribe)")

    # Step 4: Split and save
    split_and_save(all_chunks)

    print("\n✓ Data preparation complete!")
    print(f"  Next step: listen to the test audio files and correct transcriptions in")
    print(f"  {TEST_DIR}/transcriptions.jsonl (edit the 'corrected_transcription' field)")


if __name__ == "__main__":
    main()
