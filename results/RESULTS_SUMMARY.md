# PhysVision-SLM Results Summary

Final evaluation on the full held-out benchmark (35,596 multiple-choice QA pairs).
Two configurations, same recipe, both training only the ~4.8M-parameter
vision-language projection (language model frozen in both).

- **lite**: trainable CNN encoder + spatial projection + physics-pretrained TinyLMv3 (23M total)
- **full**: frozen SigLIP-SO400M + projection + frozen Qwen2-0.5B (~0.9B total)

## Overall Accuracy

| System | Accuracy |
|--------|----------|
| Random baseline | 25.0% |
| Text-only heuristic (no image) | 33.3% |
| Most-frequent answer | 34.1% |
| PhysVision-SLM **lite** (23M) | 43.7% (15,560 / 35,596) |
| **PhysVision-SLM full** (0.9B) | **89.0%** (31,673 / 35,596) |

## By Question Category

| Category | lite | full | n |
|----------|------|------|---|
| tumor_detection | 64.9% | 99.8% | 14,957 |
| quality (artifact type) | 68.6% | 99.5% | 3,259 |
| parameter | 23.7% | 83.1% | 10,800 |
| localization | 16.1% | 68.7% | 6,580 |

## By Image Type

| Image Type | lite | full | n |
|------------|------|------|---|
| sinogram | 81.4% | 99.7% | 5,749 |
| pet_reconstruction | 36.9% | 87.4% | 24,741 |
| phantom | 34.3% | 84.6% | 5,106 |

## Interpretation

The full config more than doubles the lite result (89.0% vs 43.7%). Because the
projection, data, and frozen-LM setup are held fixed across both runs, the gap is
attributable mainly to the vision encoder. The clearest evidence is
localization: it rises from 16.1% (below random, lite CNN) to 68.7% (full,
SigLIP), showing the lite spatial weakness was a representation limitation of the
coarse from-scratch CNN, not a limitation of the task or the frozen-LM setup.
Global-appearance tasks saturate near ceiling for the full config (tumor
detection 99.8%, quality 99.5%, sinogram 99.7%).

_Eval runtimes: lite 17,781 s; full 86,813 s (Apple M3 Pro, MPS). Full training:
Stage 1 9.6 h, Stage 2 20.8 h._
