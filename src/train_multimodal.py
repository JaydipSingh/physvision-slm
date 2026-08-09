#!/usr/bin/env python3
"""
Multimodal Training Loop for PhysVision-SLM
============================================
Two-stage training:
  Stage 1: Alignment (freeze vision + LM, train projection only)
  Stage 2: Instruction Tuning (freeze vision, train projection + LoRA on LM)

Usage:
  # Stage 1 (alignment, ~2 hours)
  python src/train_multimodal.py --stage 1 --data data/generated --epochs 3

  # Stage 2 (instruction tuning, ~4 hours)
  python src/train_multimodal.py --stage 2 --data data/generated --epochs 5 \
      --checkpoint checkpoints/stage1_final.pt
"""
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.physvision_model import PhysVisionModel


# ============================================================================
# DATASET
# ============================================================================

class PhysVisionDataset(Dataset):
    """
    Dataset loading generated images + QA pairs.
    Each sample = (image, question, answer) for VQA training.
    """

    def __init__(self, data_dir: str, split: str = "train", transform=None,
                 tokenize_fn=None, max_text_len: int = 128):
        self.data_dir = Path(data_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.tokenize_fn = tokenize_fn
        self.max_text_len = max_text_len

        # Load annotations
        ann_path = self.data_dir / "annotations.json"
        with open(ann_path) as f:
            data = json.load(f)

        # Flatten: one entry per QA pair
        self.samples = []
        for sample in data["samples"]:
            image_path = self.data_dir / sample["image"]
            for qa in sample["qa_pairs"]:
                self.samples.append({
                    "image_path": str(image_path),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "category": qa["category"],
                    "options": qa.get("options", []),
                    "correct_option": qa.get("correct_option", -1),
                })

        # Split: 90% train, 10% val
        n = len(self.samples)
        if split == "train":
            self.samples = self.samples[:int(0.9 * n)]
        elif split == "val":
            self.samples = self.samples[int(0.9 * n):]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.transform(img)

        # Tokenize question + answer
        text = f"Question: {sample['question']} Answer: {sample['answer']}"

        if self.tokenize_fn:
            token_ids = self.tokenize_fn(text)[:self.max_text_len]
        else:
            # Simple character-level fallback for testing
            token_ids = [ord(c) % 32000 for c in text][:self.max_text_len]

        # Pad to fixed length
        padding = [0] * (self.max_text_len - len(token_ids))
        input_ids = token_ids + padding
        # Labels: shift by 1 for causal LM (mask padding with -100)
        labels = token_ids[1:] + [0] + [-100] * (self.max_text_len - len(token_ids))

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels[:self.max_text_len], dtype=torch.long),
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def get_lr(step: int, warmup: int, max_steps: int, max_lr: float) -> float:
    """Cosine LR with warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return max_lr * 0.1 + 0.5 * (max_lr - max_lr * 0.1) * (1 + math.cos(math.pi * progress))


def train_stage(
    model: PhysVisionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str,
):
    """Run one training stage."""
    stage = config["stage"]
    max_epochs = config["epochs"]
    max_lr = config["lr"]
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Configure what's trainable
    if stage == 1:
        # Stage 1: Only projection
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        for p in model.language_model.parameters():
            p.requires_grad = False
        for p in model.projection.parameters():
            p.requires_grad = True
        model.vision_token_type.requires_grad = True
    else:
        # Stage 2: Projection + LM (via LoRA if applied externally)
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        for p in model.projection.parameters():
            p.requires_grad = True
        # LM params: assume LoRA already injected, so trainable params set
        model.vision_token_type.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Stage {stage}: Trainable {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=max_lr, weight_decay=0.01, betas=(0.9, 0.95),
    )

    total_steps = max_epochs * len(train_loader)
    warmup = min(100, total_steps // 10)
    step = 0
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # LR schedule
            lr = get_lr(step, warmup, total_steps, max_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward
            outputs = model(pixel_values, input_ids, labels)
            loss = outputs["loss"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            step += 1

            if step % 50 == 0:
                elapsed = time.time() - start_time
                eta = (total_steps - step) / (step / elapsed) / 60
                print(f"  Step {step}/{total_steps} | Loss: {loss.item():.4f} | "
                      f"LR: {lr:.2e} | ETA: {eta:.0f}min")

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values, input_ids, labels)
                val_loss += outputs["loss"].item()
                val_steps += 1

        avg_train = epoch_loss / epoch_steps
        avg_val = val_loss / max(val_steps, 1)
        print(f"\n  Epoch {epoch+1}/{max_epochs}: train_loss={avg_train:.4f}, "
              f"val_loss={avg_val:.4f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state": {k: v for k, v in model.state_dict().items()
                                if "vision_encoder" not in k},  # Don't save frozen encoder
                "stage": stage,
                "epoch": epoch + 1,
                "val_loss": avg_val,
                "config": config,
            }, save_dir / f"stage{stage}_best.pt")
            print(f"  [BEST] val_loss={avg_val:.4f} saved!")

    # Final save
    torch.save({
        "model_state": {k: v for k, v in model.state_dict().items()
                        if "vision_encoder" not in k},
        "stage": stage,
        "val_loss": best_val_loss,
        "config": config,
    }, save_dir / f"stage{stage}_final.pt")

    elapsed = time.time() - start_time
    print(f"\n  Stage {stage} complete! Time: {elapsed/3600:.1f}h, "
          f"Best val: {best_val_loss:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PhysVision Multimodal Training")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--data", type=str, default="data/generated")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--config", type=str, default="lite", choices=["lite", "full"])
    parser.add_argument("--lm-checkpoint", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")

    print("=" * 60)
    print(f"PhysVision Multimodal Training — Stage {args.stage}")
    print(f"  Config: {args.config}")
    print(f"  Data: {args.data}")
    print(f"  Device: {device}")
    print("=" * 60)

    # Build model
    model = PhysVisionModel.from_config(
        args.config, device=device, lm_checkpoint=args.lm_checkpoint)
    print(f"  Model: {model.count_parameters():,} total params")

    # Build datasets
    train_dataset = PhysVisionDataset(args.data, split="train")
    val_dataset = PhysVisionDataset(args.data, split="val")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Train
    config = {
        "stage": args.stage,
        "epochs": args.epochs,
        "lr": args.lr,
        "save_dir": args.save_dir,
        "data": args.data,
        "model_config": args.config,
    }

    train_stage(model, train_loader, val_loader, config, device)

    print("\nDone! Next steps:")
    if args.stage == 1:
        print(f"  Run Stage 2: python src/train_multimodal.py --stage 2 "
              f"--data {args.data} --checkpoint checkpoints/stage1_best.pt")
    else:
        print(f"  Run evaluation: python scripts/eval_internal.py "
              f"--checkpoint checkpoints/stage2_best.pt")


if __name__ == "__main__":
    main()
