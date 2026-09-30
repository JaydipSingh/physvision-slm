#!/usr/bin/env python3
"""
PhysVision-SLM Smoke Test
=========================
Fast end-to-end sanity check BEFORE launching the multi-hour pipeline.

Runs, on tiny settings (seconds, CPU-friendly):
  1. Generate a handful of images + QA pairs
  2. Build the "lite" multimodal model (random LM, no checkpoint needed)
  3. One forward pass with a loss
  4. One training step (backward + optimizer)
  5. A 5-sample evaluation pass

If this completes without error, the wiring (shapes, devices, tokenizer,
label masking, projection) is correct and the full pipeline should run.

Usage:
  python scripts/smoke_test.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["lite", "full"], default="lite",
                    help="Which model config to smoke-test. 'full' downloads "
                         "SigLIP + Qwen2-0.5B (~3GB) on first run.")
    args = ap.parse_args()

    print("=" * 60)
    print(f"PhysVision-SLM SMOKE TEST ({args.config})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Device: {device}")

    tmp = Path(tempfile.mkdtemp(prefix="physvision_smoke_"))
    data_dir = tmp / "data"
    print(f"  Temp data dir: {data_dir}")

    # --- Step 1: generate a tiny dataset --------------------------------
    print("\n[1/5] Generating 8 sample images...")
    subprocess.run([
        sys.executable, str(ROOT / "scripts" / "generate_dataset.py"),
        "--num-images", "8",
        "--output", str(data_dir),
        "--seed", "0",
    ], check=True)

    # --- Step 2: build model --------------------------------------------
    print(f"\n[2/5] Building '{args.config}' multimodal model...")
    from src.physvision_model import PhysVisionModel
    from src.tokenizer_bridge import get_tokenizer, get_hf_tokenizer
    from src.train_multimodal import PhysVisionDataset
    from torch.utils.data import DataLoader

    model = PhysVisionModel.from_config(args.config, device=device)
    total = model.count_parameters()
    trainable = model.count_parameters(trainable_only=True)
    print(f"  Params: {total:,} total / {trainable:,} trainable")

    # --- Step 3: forward pass with loss ---------------------------------
    print("\n[3/5] Forward pass + loss...")
    if args.config == "full":
        tok = get_hf_tokenizer("Qwen/Qwen2-0.5B")
    else:
        tok = get_tokenizer(vocab_size=32000)
    ds = PhysVisionDataset(str(data_dir), split="train", tokenize_fn=tok, max_text_len=64)
    loader = DataLoader(ds, batch_size=2, shuffle=True)
    batch = next(iter(loader))

    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    out = model(pixel_values, input_ids, labels)
    loss = out["loss"]
    print(f"  logits shape: {tuple(out['logits'].shape)}")
    print(f"  loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite!"

    # --- Step 4: one training step --------------------------------------
    print("\n[4/5] One backward + optimizer step...")
    for p in model.projection.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("  Optimizer step OK.")

    # --- Step 5: mini evaluation ----------------------------------------
    print("\n[5/5] Mini evaluation (logprob ranking, 5 samples)...")
    from scripts.eval_internal import (
        PhysVisionEvalDataset, evaluate_logprob_ranking)
    eval_ds = PhysVisionEvalDataset(str(data_dir))
    if len(eval_ds) == 0:
        print("  (no multiple-choice samples generated; skipping)")
    else:
        model.eval()
        res = evaluate_logprob_ranking(
            model, eval_ds, device, tokenize_fn=tok, max_samples=5)
        print(f"  Eval accuracy on 5 samples: {res['overall_accuracy']:.1%}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED — pipeline wiring is correct.")
    print("=" * 60)
    print("\nNext: run the pipeline for this config on your Mac:")
    if args.config == "full":
        print("  bash scripts/run_full_config.sh")
        print("  (no LM_CHECKPOINT needed; full config uses stock Qwen2)")
        print("  If MPS out-of-memory: BATCH=2 bash scripts/run_full_config.sh")
    else:
        print("  LM_CHECKPOINT=/path/to/v3_long50k_final.pt bash scripts/run_full_pipeline.sh")
    print("\nNote: the 'trainable' count above is a smoke-test artifact; the")
    print("real training script freezes the language model (small trainable %).")


if __name__ == "__main__":
    main()
