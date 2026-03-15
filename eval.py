"""
eval.py — Evaluate Whisper WER on your hand-corrected test set.

Run this BEFORE and AFTER fine-tuning to measure the delta.

Usage:
    # Baseline (vanilla whisper-small)
    python eval.py --model openai/whisper-small

    # Fine-tuned
    python eval.py --model output/whisper-small-personal/final

    # Compare both in one shot
    python eval.py --compare
"""

import json
import argparse
import re
from pathlib import Path

import torch
import librosa
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)
from peft import PeftModel
from jiwer import wer, cer


TEST_JSONL = "data/test/transcriptions.jsonl"
SAMPLE_RATE = 16000
BASE_MODEL = "openai/whisper-small"
FINETUNED_MODEL = "output/whisper-small-personal/final"


def load_test_data(jsonl_path: str) -> list[dict]:
    """Load test set with hand-corrected transcriptions."""
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            sample = json.loads(line)
            # Use corrected transcription if available, otherwise fall back
            reference = sample.get("corrected_transcription", sample.get("transcription", ""))
            if reference.strip():
                samples.append({
                    "audio_path": sample["audio_path"],
                    "reference": reference.strip(),
                    "source_file": sample.get("source_file", ""),
                })
    print(f"Loaded {len(samples)} test samples from {jsonl_path}")
    return samples


def classify_sample(text: str) -> str:
    """Classify sample language. Since all output is Latin script (Manglish),
    this is a placeholder — everything is 'english' for now.
    TODO: improve heuristic to distinguish English vs Manglish vs mixed.
    """
    return "english"


def build_pipeline(model_path: str, device: str):
    """Build a Whisper inference pipeline, handling both base and LoRA models."""
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Check if this is a LoRA checkpoint (has adapter_config.json)
    is_lora = (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        print(f"Loading LoRA model from {model_path}")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()  # Merge for faster inference
        processor = WhisperProcessor.from_pretrained(model_path)
    else:
        print(f"Loading base model from {model_path}")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
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
        generate_kwargs={"task": "transcribe"},
    )

    return pipe


def evaluate(model_path: str, test_data: list[dict], device: str) -> dict:
    """Run evaluation and return WER metrics."""
    pipe = build_pipeline(model_path, device)

    references = []
    hypotheses = []
    by_category = {"english": ([], []), "malayalam": ([], []), "mixed": ([], [])}

    print(f"\nEvaluating {len(test_data)} samples...")
    for sample in tqdm(test_data):
        result = pipe(sample["audio_path"])
        hypothesis = result["text"].strip()
        reference = sample["reference"]

        references.append(reference)
        hypotheses.append(hypothesis)

        # Categorize
        category = classify_sample(reference)
        by_category[category][0].append(reference)
        by_category[category][1].append(hypothesis)

    # Overall metrics
    overall_wer = wer(references, hypotheses)
    overall_cer = cer(references, hypotheses)

    results = {
        "model": model_path,
        "n_samples": len(test_data),
        "overall_wer": overall_wer,
        "overall_cer": overall_cer,
    }

    # Per-category metrics
    for cat, (refs, hyps) in by_category.items():
        if refs:
            results[f"{cat}_wer"] = wer(refs, hyps)
            results[f"{cat}_cer"] = cer(refs, hyps)
            results[f"{cat}_n"] = len(refs)

    return results, references, hypotheses


def print_results(results: dict, references=None, hypotheses=None):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"Model: {results['model']}")
    print(f"{'='*60}")
    print(f"  Samples:     {results['n_samples']}")
    print(f"  Overall WER: {results['overall_wer']:.2%}")
    print(f"  Overall CER: {results['overall_cer']:.2%}")

    for cat in ["english", "malayalam", "mixed"]:
        if f"{cat}_wer" in results:
            print(f"  {cat.capitalize():12s} WER: {results[f'{cat}_wer']:.2%} (n={results[f'{cat}_n']})")

    # Show a few examples
    if references and hypotheses:
        print(f"\n  Sample predictions:")
        for i in range(min(5, len(references))):
            print(f"    REF: {references[i][:80]}")
            print(f"    HYP: {hypotheses[i][:80]}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper WER")
    parser.add_argument("--model", default=None, help="Model path to evaluate")
    parser.add_argument("--test-data", default=TEST_JSONL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs fine-tuned")
    args = parser.parse_args()

    test_data = load_test_data(args.test_data)
    if not test_data:
        print("No test data found. Run prepare_data.py first and correct the test transcriptions.")
        return

    if args.compare:
        print("Running comparison: baseline vs fine-tuned\n")

        base_results, base_refs, base_hyps = evaluate(BASE_MODEL, test_data, args.device)
        print_results(base_results, base_refs, base_hyps)

        ft_results, ft_refs, ft_hyps = evaluate(FINETUNED_MODEL, test_data, args.device)
        print_results(ft_results, ft_refs, ft_hyps)

        # Delta
        print(f"\n{'='*60}")
        print("IMPROVEMENT")
        print(f"{'='*60}")
        delta_wer = base_results["overall_wer"] - ft_results["overall_wer"]
        delta_cer = base_results["overall_cer"] - ft_results["overall_cer"]
        print(f"  WER: {base_results['overall_wer']:.2%} → {ft_results['overall_wer']:.2%} (Δ {delta_wer:+.2%})")
        print(f"  CER: {base_results['overall_cer']:.2%} → {ft_results['overall_cer']:.2%} (Δ {delta_cer:+.2%})")

        for cat in ["english", "malayalam", "mixed"]:
            if f"{cat}_wer" in base_results and f"{cat}_wer" in ft_results:
                d = base_results[f"{cat}_wer"] - ft_results[f"{cat}_wer"]
                print(f"  {cat.capitalize():12s}: {base_results[f'{cat}_wer']:.2%} → {ft_results[f'{cat}_wer']:.2%} (Δ {d:+.2%})")

        # Save comparison
        comparison = {
            "baseline": base_results,
            "finetuned": ft_results,
            "delta_wer": delta_wer,
            "delta_cer": delta_cer,
        }
        out_path = Path("output") / "eval_comparison.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\n  Comparison saved: {out_path}")

    else:
        model = args.model or BASE_MODEL
        results, refs, hyps = evaluate(model, test_data, args.device)
        print_results(results, refs, hyps)

        # Save results
        out_path = Path("output") / f"eval_{Path(model).name}.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
