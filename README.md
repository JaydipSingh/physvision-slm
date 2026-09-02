# PhysVision-SLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-JaydipSingh%2Fphysvision--slm-blue)](https://github.com/JaydipSingh/physvision-slm)

**Training a Multimodal Small Language Model for Medical Physics Image Understanding Using Simulation-Generated Data**

A multimodal small language model that learns to interpret simulated medical physics
images (PET reconstructions, sinograms, phantom diagrams) by pairing an analytical
physics image simulator with a vision-language model. The simulator provides exact
ground-truth labels for every generated image, so no human annotation is required.

## Key Idea

Use a physics image simulator as a **zero-cost labeled data generator** for training
vision-language models. Every image comes with exact ground truth (tumor count,
location, size, artifact type), which is converted automatically into multiple-choice
QA pairs.

> **Note on the simulator:** images are produced by a standalone **analytical NumPy
> simulation** (Poisson counting statistics, PSF blurring, simplified Radon transform
> for sinograms). It is *not* a full Geant4 Monte Carlo simulation. Geant4 integration
> is left as future work.

## Architecture

```
Image ──▶ Vision Encoder ──▶ Projection ──▶ [vision tokens]
                                                   │  concat
Question ────────────────▶ Token Embed ──▶ [text tokens]
                                                   ▼
                                          TinyLMv3 decoder ──▶ Answer
```

Two configurations:

| Config | Vision Encoder | Language Model | Total Params | Use |
|--------|----------------|----------------|--------------|-----|
| `lite` | CNN (trainable, ~5M) | TinyLMv3 (23M, paper 1) | ~25M | constrained hardware (M3 Pro) |
| `full` | SigLIP-SO400M (frozen) | Qwen2-0.5B | ~900M | production quality comparison |

Training is two-stage: **Stage 1** trains only the projection (alignment); **Stage 2**
trains projection + LM (instruction tuning).

## Repo Layout

```
src/
  tinylm_v3.py         Standalone TinyLMv3 architecture (no training deps)
  vision_encoder.py    CNN (lite) and SigLIP (full) encoders
  projection.py        Vision→language projection modules
  physvision_model.py  Combined multimodal model + from_config factory
  train_multimodal.py  Two-stage training loop
  tokenizer_bridge.py  Real Rust BPE tokenizer (Mac) / char fallback (smoke test)
scripts/
  generate_dataset.py  Analytical PET/sinogram/phantom image + QA generator
  eval_internal.py     Log-prob ranking evaluation on the internal benchmark
  eval_baselines.py    Random / most-frequent / text-only baselines
  smoke_test.py        Fast end-to-end wiring check (seconds, CPU)
  run_full_pipeline.sh One-command full run on the Mac
```

## Requirements

```bash
pip install -r requirements.txt   # torch, torchvision, numpy, scipy, matplotlib, Pillow, transformers
```

The `lite` config reuses the TinyLMv3 checkpoint from paper 1 and its Rust BPE
tokenizer. Keep the `slm_v2/` and `subword_tokenizer/` folders alongside this
project so the real tokenizer and pretrained weights are found.

## Quick Start

```bash
# 0. Sanity check the full pipeline in seconds (uses tiny random data)
python scripts/smoke_test.py

# 1. Full run on the Mac (data gen → baselines → Stage 1 → Stage 2 → eval)
LM_CHECKPOINT=/path/to/slm_v2/checkpoints/v3_long50k_final.pt \
    bash scripts/run_full_pipeline.sh
```

Or run steps individually:

```bash
python scripts/generate_dataset.py --num-images 10000 --output data/generated
python scripts/eval_baselines.py --data data/generated
python src/train_multimodal.py --stage 1 --data data/generated --config lite \
    --lm-checkpoint /path/to/v3_long50k_final.pt --epochs 3
python src/train_multimodal.py --stage 2 --data data/generated --config lite \
    --lm-checkpoint /path/to/v3_long50k_final.pt \
    --checkpoint checkpoints/stage1_best.pt --epochs 5
python scripts/eval_internal.py --checkpoint checkpoints/stage2_best.pt \
    --data data/generated --config lite \
    --lm-checkpoint /path/to/v3_long50k_final.pt
```

## Hardware

Developed for an Apple M3 Pro (18 GB unified memory). Auto-selects CUDA / MPS / CPU.

## Relationship to Paper 1

Paper 1 ([slm-physics-assistant](https://github.com/JaydipSingh/slm-physics-assistant))
built a text-only 23M physics QA model on constrained hardware. PhysVision-SLM extends
that language model with a vision pathway so it can answer questions about medical
physics images.

## License

MIT — see [LICENSE](LICENSE).
