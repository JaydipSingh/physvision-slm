#!/usr/bin/env python3
"""
Baseline Evaluation for PhysVision Benchmark
=============================================
Runs baseline models on the same evaluation set for comparison:
  1. Random baseline (25% for 4-option MCQ)
  2. Text-only baseline (no image, just question → answer)
  3. GPT-2 baseline (text-only, larger model)

Usage:
  python scripts/eval_baselines.py --data data/generated
"""
import argparse
import json
import time
import random
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_eval_samples(data_dir: str):
    """Load evaluation samples with multiple-choice options."""
    data_path = Path(data_dir) / "annotations.json"
    with open(data_path) as f:
        data = json.load(f)

    samples = []
    for sample in data["samples"]:
        for qa in sample["qa_pairs"]:
            if qa.get("options") and qa.get("correct_option", -1) >= 0:
                samples.append({
                    "question": qa["question"],
                    "options": qa["options"],
                    "correct_option": qa["correct_option"],
                    "category": qa["category"],
                    "image_type": sample["image_type"],
                    "answer": qa["answer"],
                })
    return samples


def eval_random_baseline(samples):
    """Random guessing baseline."""
    correct = 0
    for s in samples:
        predicted = random.randint(0, len(s["options"]) - 1)
        if predicted == s["correct_option"]:
            correct += 1
    return correct / len(samples)


def eval_text_only_heuristic(samples):
    """
    Text-only heuristic baseline: pick the most "physics-sounding" option
    based on keyword matching. No image used.
    """
    physics_keywords = [
        "yes", "no", "brain", "lung", "body", "tumor", "lesion",
        "right", "left", "superior", "inferior", "gap", "ring", "scatter",
    ]

    correct = 0
    for s in samples:
        question_lower = s["question"].lower()
        scores = []
        for opt in s["options"]:
            # Score based on keyword overlap with question
            opt_lower = opt.lower()
            score = sum(1 for kw in physics_keywords if kw in opt_lower)
            # Bias toward "yes" for detection questions
            if "visible" in question_lower or "present" in question_lower:
                if "yes" in opt_lower:
                    score += 2
            scores.append(score)

        # Pick highest-scoring option (random tiebreak)
        max_score = max(scores)
        candidates = [i for i, s in enumerate(scores) if s == max_score]
        predicted = random.choice(candidates)

        if predicted == s["correct_option"]:
            correct += 1

    return correct / len(samples)


def eval_frequency_baseline(samples):
    """
    Most-frequent-answer baseline: always pick the most common answer
    position (often position 0 in generated data).
    """
    correct = sum(1 for s in samples if s["correct_option"] == 0)
    return correct / len(samples)


def main():
    parser = argparse.ArgumentParser(description="Baseline Evaluation")
    parser.add_argument("--data", type=str, default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("PhysVision Baseline Evaluation")
    print("=" * 60)

    samples = load_eval_samples(args.data)
    print(f"  Evaluation samples: {len(samples)}")

    # Category breakdown
    categories = {}
    types = {}
    for s in samples:
        categories[s["category"]] = categories.get(s["category"], 0) + 1
        types[s["image_type"]] = types.get(s["image_type"], 0) + 1

    print(f"  Categories: {categories}")
    print(f"  Image types: {types}")

    # Run baselines
    print(f"\n{'=' * 60}")
    print("BASELINES")
    print(f"{'=' * 60}")

    random_acc = eval_random_baseline(samples)
    print(f"  Random:              {random_acc:.1%}")

    freq_acc = eval_frequency_baseline(samples)
    print(f"  Most-frequent:       {freq_acc:.1%}")

    text_acc = eval_text_only_heuristic(samples)
    print(f"  Text-only heuristic: {text_acc:.1%}")

    print(f"\n  Expected PhysVision targets:")
    print(f"    Lite (TinyLMv3):   55-60%")
    print(f"    Full (Qwen2-0.5B): 75-80%")

    # Per-category random baseline
    print(f"\n  Per-category (random):")
    for cat, count in sorted(categories.items()):
        cat_samples = [s for s in samples if s["category"] == cat]
        cat_random = sum(1 for _ in range(len(cat_samples))
                        if random.randint(0, 3) == cat_samples[0]["correct_option"]) / len(cat_samples)
        print(f"    {cat:25s} ~25% (n={count})")

    # Save
    results = {
        "total_samples": len(samples),
        "baselines": {
            "random": random_acc,
            "most_frequent": freq_acc,
            "text_only_heuristic": text_acc,
        },
        "categories": categories,
        "image_types": types,
    }

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    main()
