# PhysVision-SLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-JaydipSingh%2Fphysvision--slm-blue)](https://github.com/JaydipSingh/physvision-slm)

**Synthetic-to-Model: Training a Multimodal Small Language Model for Medical Physics Image Understanding Using Simulation-Generated Data**

## Status: Planning Phase

See [spec/PROJECT_SPEC.md](spec/PROJECT_SPEC.md) for the full project specification.

## Quick Summary

A novel pipeline combining:
- **g4_agent** (Geant4-based PET/CT simulator) → generates unlimited labeled images
- **PhysVision-SLM** (multimodal small language model) → learns to understand them

No human annotation needed. The simulator provides exact ground-truth for every generated image.

### Key Innovation
Use a physics simulator as a **zero-cost labeled data generator** for training vision-language models. The model learns to interpret PET scans, sinograms, and detector geometries.

### Two Backbones
- TinyLMv3 (35M) — continues paper 1's constrained-hardware story
- Qwen2-0.5B — production-level quality comparison

### Hardware
Apple M3 Pro (18GB) — same as paper 1.

## Relationship to Other Projects

```
Paper 1 (slm_v0)          g4_agent                Paper 2 (this)
─────────────────          ─────────              ──────────────────
Text-only SLM       +      PET Simulator    =     Multimodal Medical
Physics QA                  Image Generator         Physics VQA SLM
Rust tokenizer              Ground-truth labels     50K training pairs
LoRA pipeline               Publication images      Vision + Language
```

## Simulation Approach

This project uses **analytical physics simulation** (pure NumPy) rather than full Monte Carlo particle transport (Geant4). This enables rapid dataset generation (10K images in ~30 minutes) while producing visually realistic training data.

| Physics Component | Method |
|-------------------|--------|
| Detector geometry | Mathematical (ring radius, crystal angular positions) |
| Gamma ray tracing | Ray-cylinder intersection |
| Coincidence detection | Timing + energy windowing |
| PET reconstruction | Activity maps + Poisson noise + Gaussian PSF |
| Sinograms | Simplified Radon transform (forward projection) |
| Scatter/attenuation | Statistical sampling (energy-dependent probability) |

**Why not Geant4?** Full Monte Carlo is accurate but slow (minutes per scan) and complex to install. For training a visual QA model, the key requirement is *visually realistic* images with *correct ground-truth labels* — not physically exact particle transport. Our analytical approach satisfies both at 1000× the speed.

**Future work**: A Geant4-validated comparison (analytical vs Monte Carlo) can be added to quantify transfer fidelity.

## Getting Started

```bash
# Coming soon — currently in planning/data generation phase
pip install -r requirements.txt
python scripts/generate_dataset.py --num-images 500  # pilot
python src/train_multimodal.py --config config/training_config.yaml
```
