# PhysVision-SLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-JaydipSingh%2Fphysvision--slm-blue)](https://github.com/JaydipSingh/physvision-slm)

**Training Multimodal Small Language Models for Medical Physics Image Understanding Using Simulation-Generated Data**

Multimodal small language models that learn to interpret simulated medical physics
images (PET reconstructions, sinograms, phantom diagrams) by pairing an analytical
physics image simulator with a vision-language model. The simulator provides exact
ground-truth labels for every generated image, so no human annotation is required.

## Key Idea

Use a physics image simulator as a **zero-cost labeled data generator** for training
vision-language models. Every image comes with exact ground truth (tumor count,
location, size, artifact type), which is converted automatically into multiple-choice
QA pairs. We then train **only a small (~4.8M-parameter) vision→language projection**;
the language model is frozen throughout (LLaVA-style alignment).

> **Note on the simulator:** images are produced by a standalone **analytical NumPy
> simulation** (Poisson counting statistics, PSF blurring, simplified Radon transform
> for sinograms). It is *not* a full Geant4 Monte Carlo simulation. Geant4 integration
> is left as future work.

## Results

Evaluated on a held-out benchmark of **35,596** multiple-choice QA pairs (4 options;
25% random baseline). Both configs train only the projection; the language model is
frozen in both.

| System | Accuracy |
|--------|----------|
| Random baseline | 25.0% |
| Text-only heuristic (no image) | 33.3% |
| Most-frequent answer | 34.1% |
| **PhysVision-SLM `lite`** (23M) | **43.7%** |
| **PhysVision-SLM `full`** (~0.9B) | **89.0%** |

Per-category accuracy (lite / full):

| Category | lite | full |
|----------|------|------|
| tumor_detection | 64.9% | 99.8% |
| quality (artifact type) | 68.6% | 99.5% |
| parameter | 23.7% | 83.1% |
| localization | 16.1% | 68.7% |

The controlled comparison attributes the 45-point gap primarily to the **vision
encoder**: swapping the from-scratch CNN for a frozen pretrained SigLIP encoder lifts
fine spatial localization from below random (16.1%) to well above it (68.7%), with
everything else held fixed. See [results/RESULTS_SUMMARY.md](results/RESULTS_SUMMARY.md).

## Architecture

```
Image ──▶ Vision Encoder ──▶ Projection ──▶ [vision tokens]
                                                   │  concat
Question ────────────────▶ Token Embed ──▶ [text tokens]
                                                   ▼
                                     Language Model (frozen) ──▶ Answer
```

Two configurations:

| Config | Vision Encoder | Projection | Language Model | Total Params |
|--------|----------------|------------|----------------|--------------|
| `lite` | CNN (trainable, from scratch) | spatial, per-patch | TinyLMv3 23M (frozen) | ~23M |
| `full` | SigLIP-SO400M (frozen) | compressing MLP | Qwen2-0.5B (frozen) | ~0.9B |

Two-stage training: **Stage 1** alignment and **Stage 2** instruction tuning both
train the projection (lite also trains its CNN encoder). The language model stays
frozen in both stages and configs — this fits an 18 GB laptop and isolates the
connector's contribution. LoRA on the language model is a natural extension (future
work).

## Repo Layout

```
src/
  tinylm_v3.py         Standalone TinyLMv3 architecture (no training deps)
  vision_encoder.py    CNN (lite) and SigLIP (full) encoders
  projection.py        Vision→language projection modules (SpatialProjection, etc.)
  physvision_model.py  Combined multimodal model + from_config factory
  train_multimodal.py  Two-stage training loop
  tokenizer_bridge.py  Rust BPE (lite) + Qwen2 HF tokenizer (full) + char fallback
scripts/
  generate_dataset.py  Analytical PET/sinogram/phantom image + QA generator
  eval_internal.py     Log-prob ranking evaluation on the internal benchmark
  eval_baselines.py    Random / most-frequent / text-only baselines
  debug_eval.py        Per-question diagnostics + vision-pathway (blank-image) check
  log_results.py       Consolidate eval JSONs into a paper-ready summary
  smoke_test.py        Fast end-to-end wiring check (supports --config lite|full)
  pilot.sh             Small dataset + short train + vision check (fast validation)
  run_lite_pipeline.sh Full LITE pipeline (CNN + TinyLMv3)
  run_full_config.sh   Full FULL pipeline (SigLIP + Qwen2)
doc/
  physvision_arxiv.tex Paper source
  references.bib        Bibliography
  figure_results.png    lite-vs-full results figure
results/                 Evaluation JSONs + RESULTS_SUMMARY.md
```

## Requirements

```bash
pip install -r requirements.txt
# torch, torchvision, numpy, scipy, matplotlib, Pillow,
# transformers, accelerate, sentencepiece, protobuf
```

The `lite` config reuses the TinyLMv3 checkpoint from paper 1 and its Rust BPE
tokenizer. Keep the `slm_v2/` and `subword_tokenizer/` folders alongside this project
so the real tokenizer and pretrained weights are found. The `full` config downloads
SigLIP-SO400M (~1.6 GB) and Qwen2-0.5B (~1 GB) from HuggingFace on first run.

## Quick Start

```bash
# 0. Sanity check wiring in seconds (choose the config)
python scripts/smoke_test.py --config lite
python scripts/smoke_test.py --config full      # downloads SigLIP + Qwen2 first time

# 1a. LITE pipeline (data gen -> baselines -> Stage 1 -> Stage 2 -> eval)
LM_CHECKPOINT=/path/to/slm_v2/checkpoints/v3_long50k_final.pt \
    bash scripts/run_lite_pipeline.sh

# 1b. FULL pipeline (reuses the generated dataset)
bash scripts/run_full_config.sh
#    If MPS out-of-memory:
#    VISION_MODEL=google/siglip-base-patch16-224 BATCH=2 bash scripts/run_full_config.sh
```

See [FULL_CONFIG_RUNBOOK.md](FULL_CONFIG_RUNBOOK.md) for the step-by-step full-config
guide and [TRANSFER_AND_RUN.md](TRANSFER_AND_RUN.md) for environment setup.

## Hardware

Developed for an Apple M3 Pro (18 GB unified memory). Auto-selects CUDA / MPS / CPU.

## Relationship to Paper 1

Paper 1 ([slm-physics-assistant](https://github.com/JaydipSingh/slm-physics-assistant))
built a text-only 23M physics QA model on constrained hardware. PhysVision-SLM extends
that language model with a vision pathway so it can answer questions about medical
physics images.

## License

MIT — see [LICENSE](LICENSE).
