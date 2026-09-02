#!/usr/bin/env python3
"""
Consolidate PhysVision Results into a Paper-Ready Summary
=========================================================
Reads the JSON files written by eval_internal.py and eval_baselines.py from
the results/ directory and produces two stable, commit-friendly artifacts:

  results/summary.json          machine-readable consolidated summary
  results/RESULTS_SUMMARY.md    human-readable table for the paper / README

Run this on the Mac AFTER the pipeline finishes, then commit results/ back
to GitHub so the numbers travel with the repo.

Usage:
  python scripts/log_results.py
  python scripts/log_results.py --results-dir results
"""
import argparse
import glob
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _latest_eval(results_dir: Path):
    """Return (config -> latest eval dict) across all eval_*.json files."""
    latest = {}
    for p in sorted(glob.glob(str(results_dir / "eval_*.json"))):
        data = _load_json(Path(p))
        if not data:
            continue
        cfg = data.get("config", "unknown")
        # Newer files sort later alphabetically because of the timestamp suffix
        latest[cfg] = {"file": Path(p).name, "data": data}
    return latest


def main():
    parser = argparse.ArgumentParser(description="Consolidate PhysVision results")
    parser.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    baselines = _load_json(results_dir / "baselines.json")
    evals = _latest_eval(results_dir)

    if not baselines and not evals:
        print(f"  No result files found in {results_dir}.")
        print("  Run eval_baselines.py and eval_internal.py first.")
        return

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ---- Build consolidated summary.json -------------------------------
    summary = {
        "generated": generated,
        "baselines": baselines.get("baselines") if baselines else None,
        "total_eval_samples": baselines.get("total_samples") if baselines else None,
        "models": {},
    }
    for cfg, entry in evals.items():
        d = entry["data"]
        summary["models"][cfg] = {
            "source_file": entry["file"],
            "checkpoint": d.get("checkpoint"),
            "overall_accuracy": d.get("overall_accuracy"),
            "correct": d.get("correct"),
            "total": d.get("total"),
            "by_category": d.get("by_category"),
            "by_image_type": d.get("by_image_type"),
        }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Build human-readable RESULTS_SUMMARY.md -----------------------
    lines = []
    lines.append("# PhysVision-SLM Results Summary")
    lines.append("")
    lines.append(f"_Generated: {generated}_")
    lines.append("")

    # Headline comparison table
    lines.append("## Overall Accuracy")
    lines.append("")
    lines.append("| Model / Baseline | Accuracy |")
    lines.append("|------------------|----------|")
    if baselines and baselines.get("baselines"):
        b = baselines["baselines"]
        name_map = {
            "random": "Random",
            "most_frequent": "Most-frequent answer",
            "text_only_heuristic": "Text-only heuristic (no image)",
        }
        for key, label in name_map.items():
            if key in b:
                lines.append(f"| {label} | {b[key]*100:.1f}% |")
    for cfg, m in summary["models"].items():
        acc = m["overall_accuracy"]
        if acc is not None:
            lines.append(f"| **PhysVision ({cfg})** | **{acc*100:.1f}%** "
                         f"({m['correct']}/{m['total']}) |")
    lines.append("")

    # Per-category and per-image-type for each model
    for cfg, m in summary["models"].items():
        lines.append(f"## PhysVision ({cfg}) - Breakdown")
        lines.append("")
        if m.get("checkpoint"):
            lines.append(f"Checkpoint: `{m['checkpoint']}`")
            lines.append("")
        if m.get("by_category"):
            lines.append("### By Question Category")
            lines.append("")
            lines.append("| Category | Accuracy |")
            lines.append("|----------|----------|")
            for cat, acc in sorted(m["by_category"].items()):
                lines.append(f"| {cat} | {acc*100:.1f}% |")
            lines.append("")
        if m.get("by_image_type"):
            lines.append("### By Image Type")
            lines.append("")
            lines.append("| Image Type | Accuracy |")
            lines.append("|------------|----------|")
            for t, acc in sorted(m["by_image_type"].items()):
                lines.append(f"| {t} | {acc*100:.1f}% |")
            lines.append("")

    if summary.get("total_eval_samples"):
        lines.append(f"_Evaluation set: {summary['total_eval_samples']} "
                     f"multiple-choice questions._")
        lines.append("")

    with open(results_dir / "RESULTS_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Wrote {results_dir / 'summary.json'}")
    print(f"  Wrote {results_dir / 'RESULTS_SUMMARY.md'}")
    print("\n  Commit these back to GitHub:")
    print("    git add results/summary.json results/RESULTS_SUMMARY.md")
    print('    git commit -m "Add PhysVision evaluation results"')
    print("    git push")


if __name__ == "__main__":
    main()
