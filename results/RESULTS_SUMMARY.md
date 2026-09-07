# PhysVision-SLM Results Summary

Final evaluation on the full held-out benchmark (35,596 multiple-choice QA pairs).
Model: PhysVision-SLM `lite` (23M params) — trainable CNN encoder + spatial
projection + physics-pretrained TinyLMv3. Checkpoint: `checkpoints/stage2_best.pt`.

## Overall Accuracy

| System | Accuracy |
|--------|----------|
| Random baseline | 25.0% |
| Text-only heuristic (no image) | 33.3% |
| Most-frequent answer | 34.1% |
| **PhysVision-SLM (23M, ours)** | **43.7%** (15,560 / 35,596) |

The ~10-point margin over the image-agnostic text-only baseline (33.3%) reflects
genuine image use, confirmed by a blank-image counterfactual check (score changes
on 20/20 sampled questions).

## By Question Category

| Category | Accuracy | n |
|----------|----------|---|
| quality (artifact type) | 68.6% | 3,259 |
| tumor_detection | 64.9% | 14,957 |
| parameter | 23.7% | 10,800 |
| localization | 16.1% | 6,580 |

## By Image Type

| Image Type | Accuracy | n |
|------------|----------|---|
| sinogram | 81.4% | 5,749 |
| pet_reconstruction | 36.9% | 24,741 |
| phantom | 34.3% | 5,106 |

## Interpretation

Strong on global-appearance tasks (sinogram artifacts 81.4%, tumor presence
64.9%, quality 68.6%); at or below random on fine spatial reasoning
(localization 16.1%, parameter 23.7%). The localization weakness is attributed to
the coarse 8x8 patch grid and small CNN encoder — the primary target for future
work.

_Eval runtime: 17,781 s on Apple M3 Pro (MPS)._
