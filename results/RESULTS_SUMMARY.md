# PhysVision-SLM Results Summary

_Generated: 2026-09-04 02:54 UTC_

## Overall Accuracy

| Model / Baseline | Accuracy |
|------------------|----------|
| Random | 25.3% |
| Most-frequent answer | 34.1% |
| Text-only heuristic (no image) | 33.3% |
| **PhysVision (lite)** | **11.7%** (4163/35596) |

## PhysVision (lite) - Breakdown

Checkpoint: `/Users/jdsingh/slm_v0/physvision-slm/checkpoints/stage2_best.pt`

### By Question Category

| Category | Accuracy |
|----------|----------|
| localization | 20.6% |
| parameter | 0.0% |
| quality | 8.6% |
| tumor_detection | 16.9% |

### By Image Type

| Image Type | Accuracy |
|------------|----------|
| pet_reconstruction | 5.5% |
| phantom | 17.0% |
| sinogram | 33.7% |

_Evaluation set: 35596 multiple-choice questions._
