#!/usr/bin/env python3
"""
PhysVision Internal Benchmark Evaluation
==========================================
Evaluates the trained model on our generated physics visual QA benchmark.

Metrics:
  - Overall accuracy (multiple choice)
  - Per-category accuracy (tumor_detection, localization, parameter, quality)
  - Per-image-type accuracy (pet_reconstruction, sinogram, phantom)
  - Confusion analysis

Usage:
  python scripts/eval_internal.py --checkpoint checkpoints/stage2_best.pt \
      --data data/generated --config lite
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.physvision_model import PhysVisionModel
from src.tokenizer_bridge import get_tokenizer


# ============================================================================
# EVALUATION DATASET (Multiple Choice format)
# ============================================================================

class PhysVisionEvalDataset:
    """Loads evaluation samples in multiple-choice format."""

    def __init__(self, data_dir: str, tokenize_fn=None):
        self.data_dir = Path(data_dir)
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.tokenize_fn = tokenize_fn

        with open(self.data_dir / "annotations.json") as f:
            data = json.load(f)

        # Flatten to one entry per QA pair (only those with options)
        self.samples = []
        for sample in data["samples"]:
            image_path = self.data_dir / sample["image"]
            for qa in sample["qa_pairs"]:
                if qa.get("options") and qa.get("correct_option", -1) >= 0:
                    self.samples.append({
                        "image_path": str(image_path),
                        "image_type": sample["image_type"],
                        "question": qa["question"],
                        "options": qa["options"],
                        "correct_option": qa["correct_option"],
                        "category": qa["category"],
                        "answer": qa["answer"],
                    })

    def __len__(self):
        return len(self.samples)

    def get_sample(self, idx):
        return self.samples[idx]


# ============================================================================
# EVALUATION METHODS
# ============================================================================

def evaluate_logprob_ranking(
    model: PhysVisionModel,
    dataset: PhysVisionEvalDataset,
    device: str,
    tokenize_fn,
    max_samples: int = None,
) -> Dict:
    """
    Evaluate using log-probability ranking (same as paper 1 QA eval).
    For each question, score each option by continuation log-probability.
    """
    model.eval()
    transform = dataset.transform

    correct = 0
    total = 0
    results = []

    category_stats = {}
    type_stats = {}

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    print(f"\n  Evaluating {n} questions (log-prob ranking)...\n")

    for idx in range(n):
        sample = dataset.get_sample(idx)

        # Load image
        img = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = transform(img).unsqueeze(0).to(device)

        question = sample["question"]
        options = sample["options"]
        correct_idx = sample["correct_option"]
        category = sample["category"]
        img_type = sample["image_type"]

        # Score each option.
        #
        # Correct approach (fixes the length-bias / identical-score bug):
        #   1. Tokenize the FULL "prompt + answer" ONCE so BPE boundaries are
        #      consistent. Tokenize the prompt alone to get the answer span as a
        #      strict prefix length.
        #   2. The model prepends `num_vision_tokens` vision tokens, so text
        #      token t sits at sequence position (vision + t); its prediction
        #      comes from logits at position (vision + t - 1).
        #   3. Length-normalize by the number of scored answer tokens.
        prompt = f"Question: {question} Answer:"
        vision_offset = model.num_vision_tokens

        if tokenize_fn:
            prompt_ids = tokenize_fn(prompt)
        else:
            prompt_ids = [ord(c) % 32000 for c in prompt]
        prompt_len = len(prompt_ids)

        option_scores = []
        for option in options:
            full_text = f"{prompt} {option}"
            if tokenize_fn:
                full_ids = tokenize_fn(full_text)
            else:
                full_ids = [ord(c) % 32000 for c in full_text]

            # Answer tokens are everything after the prompt prefix.
            answer_ids = full_ids[prompt_len:]
            # Truncate the whole sequence to the model limit, keeping the answer.
            full_ids = full_ids[:128]

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                outputs = model(pixel_values, input_ids)
                logits = outputs["logits"]
            log_probs = F.log_softmax(logits[0], dim=-1)

            # Score each answer token from the logit at the preceding position.
            total_lp = 0.0
            count = 0
            for j, tid in enumerate(answer_ids):
                text_pos = prompt_len + j          # index within text tokens
                if text_pos >= len(full_ids):
                    break                           # answer got truncated away
                seq_pos = vision_offset + text_pos  # position in concat sequence
                pred_pos = seq_pos - 1              # predicts token at seq_pos
                if 0 <= pred_pos < log_probs.shape[0]:
                    total_lp += log_probs[pred_pos, tid].item()
                    count += 1

            # Length-normalized average log-prob (mean over answer tokens).
            option_scores.append(total_lp / max(count, 1))

        # Select best option
        predicted_idx = int(np.argmax(option_scores))
        is_correct = predicted_idx == correct_idx
        if is_correct:
            correct += 1
        total += 1

        # Track per-category
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1

        # Track per-image-type
        if img_type not in type_stats:
            type_stats[img_type] = {"correct": 0, "total": 0}
        type_stats[img_type]["total"] += 1
        if is_correct:
            type_stats[img_type]["correct"] += 1

        results.append({
            "idx": idx,
            "question": question,
            "predicted": options[predicted_idx] if predicted_idx < len(options) else "?",
            "correct_answer": options[correct_idx] if correct_idx < len(options) else "?",
            "is_correct": is_correct,
            "category": category,
            "image_type": img_type,
        })

        if (idx + 1) % 50 == 0:
            print(f"    [{idx+1}/{n}] Running accuracy: {correct/total:.1%}")

    # Compute summaries
    accuracy = correct / total if total > 0 else 0
    cat_accuracy = {
        cat: stats["correct"] / stats["total"]
        for cat, stats in category_stats.items()
    }
    type_accuracy = {
        t: stats["correct"] / stats["total"]
        for t, stats in type_stats.items()
    }

    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "by_category": cat_accuracy,
        "by_image_type": type_accuracy,
        "category_counts": category_stats,
        "type_counts": type_stats,
        "results": results,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PhysVision Internal Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path")
    parser.add_argument("--data", type=str, default="data/generated",
                        help="Dataset directory")
    parser.add_argument("--config", type=str, default="lite", choices=["lite", "full"],
                        help="Model config")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit evaluation to N samples")
    parser.add_argument("--lm-checkpoint", type=str, default=None,
                        help="Language model checkpoint (for lite config)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 60)
    print("PhysVision-SLM Evaluation")
    print("=" * 60)
    print(f"  Data: {args.data}")
    print(f"  Config: {args.config}")
    print(f"  Device: {device}")

    # Build model
    model = PhysVisionModel.from_config(
        args.config, device=device, lm_checkpoint=args.lm_checkpoint)

    # Load checkpoint if provided
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"  Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # Load only the non-frozen parts
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  (Missing keys: {len(missing)} — expected for frozen encoder)")

    model.to(device)
    model.eval()
    print(f"  Model params: {model.count_parameters():,}")
    print(f"  Trainable: {model.count_parameters(trainable_only=True):,}")

    # Load dataset
    dataset = PhysVisionEvalDataset(args.data)
    print(f"  Eval samples: {len(dataset)}")

    # Tokenizer: real Rust BPE on Mac, char-level fallback on Windows
    tokenize_fn = get_tokenizer(vocab_size=32000)

    # Run evaluation
    start = time.time()
    results = evaluate_logprob_ranking(
        model, dataset, device,
        tokenize_fn=tokenize_fn,
        max_samples=args.max_samples,
    )
    elapsed = time.time() - start

    # Print results
    print(f"\n{'=' * 60}")
    print(f"RESULTS — PhysVision-SLM ({args.config})")
    print(f"{'=' * 60}")
    print(f"  Overall Accuracy:  {results['overall_accuracy']:.1%} "
          f"({results['correct']}/{results['total']})")
    print(f"  Random Baseline:   25.0% (4 options)")
    print(f"  Elapsed:           {elapsed:.1f}s")

    print(f"\n  By Category:")
    for cat, acc in sorted(results["by_category"].items()):
        count = results["category_counts"][cat]["total"]
        print(f"    {cat:25s} {acc:.1%} (n={count})")

    print(f"\n  By Image Type:")
    for t, acc in sorted(results["by_image_type"].items()):
        count = results["type_counts"][t]["total"]
        print(f"    {t:25s} {acc:.1%} (n={count})")

    # Save results
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"eval_{args.config}_{int(time.time())}.json"

    # Remove full results list for compact JSON (keep summary)
    save_data = {k: v for k, v in results.items() if k != "results"}
    save_data["elapsed_seconds"] = elapsed
    save_data["config"] = args.config
    save_data["checkpoint"] = args.checkpoint

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    # Comparison context
    print(f"\n  COMPARISON (from Paper 1):")
    print(f"    Random baseline:     25.0%")
    print(f"    Text-only TinyLM v1: 32.5%")
    print(f"    Text-only + LoRA:    45.0%")
    print(f"    PhysVision ({args.config}): {results['overall_accuracy']:.1%}")


if __name__ == "__main__":
    main()
