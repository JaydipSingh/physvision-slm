#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# PhysVision-SLM  FULL CONFIG pipeline  (SigLIP + Qwen2-0.5B)
# =============================================================================
# This trains the "full" configuration for a lite-vs-full comparison in the
# paper. It reuses the SAME generated dataset as the lite run.
#
# Downloads on first use (internet + ~3GB disk):
#   - google/siglip-so400m-patch14-384  (frozen vision encoder, ~1.6GB)
#   - Qwen/Qwen2-0.5B                    (language model, ~1GB)
#
# What trains: the vision->language projection only (SigLIP frozen, Qwen2 frozen).
# This is the LLaVA-style alignment setup and fits in 18GB.
#
# Time: projection-only training is far cheaper than the lite CNN run, but SigLIP
# forward passes are heavier. Expect a few hours for Stage 1 + Stage 2 + eval.
# If you hit MPS out-of-memory, lower --batch-size.
# =============================================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data/generated}"
CKPT_DIR="${CKPT_DIR:-$ROOT/checkpoints/full}"
BATCH="${BATCH:-4}"

# Vision encoder. Default is SigLIP-SO400M (best quality, ~1.6GB, heavier on
# memory). If you hit MPS out-of-memory, set a smaller encoder, e.g.:
#   VISION_MODEL=google/siglip-base-patch16-224 bash scripts/run_full_config.sh
VISION_MODEL="${VISION_MODEL:-google/siglip-so400m-patch14-384}"

echo "PhysVision-SLM FULL config pipeline"
echo "  Root:          $ROOT"
echo "  Data:          $DATA_DIR   (reuses the lite dataset)"
echo "  Ckpts:         $CKPT_DIR"
echo "  Batch:         $BATCH"
echo "  Vision model:  $VISION_MODEL"
echo ""

if [ -f "$ROOT/../venv/bin/activate" ]; then
    source "$ROOT/../venv/bin/activate"
fi

if [ ! -f "$DATA_DIR/annotations.json" ]; then
    echo "[ERROR] No dataset at $DATA_DIR. Generate it first, e.g.:"
    echo "  python scripts/generate_dataset.py --num-images 10000 --output $DATA_DIR --seed 42"
    exit 1
fi

# Stage 1: alignment (projection only; SigLIP + Qwen2 frozen)
echo "============================================================"
echo "FULL STEP 1: Stage 1 alignment (SigLIP + Qwen2 frozen)"
echo "============================================================"
python "$ROOT/src/train_multimodal.py" \
    --stage 1 \
    --data "$DATA_DIR" \
    --config full \
    --vision-model "$VISION_MODEL" \
    --epochs 2 \
    --batch-size "$BATCH" \
    --lr 1e-3 \
    --save-dir "$CKPT_DIR"

# Stage 2: continued projection training
echo ""
echo "============================================================"
echo "FULL STEP 2: Stage 2 instruction tuning (projection)"
echo "============================================================"
python "$ROOT/src/train_multimodal.py" \
    --stage 2 \
    --data "$DATA_DIR" \
    --config full \
    --vision-model "$VISION_MODEL" \
    --epochs 3 \
    --batch-size "$BATCH" \
    --lr 5e-4 \
    --checkpoint "$CKPT_DIR/stage1_best.pt" \
    --save-dir "$CKPT_DIR"

# Step 3: evaluation
echo ""
echo "============================================================"
echo "FULL STEP 3: Evaluation"
echo "============================================================"
python "$ROOT/scripts/eval_internal.py" \
    --checkpoint "$CKPT_DIR/stage2_best.pt" \
    --data "$DATA_DIR" \
    --config full \
    --vision-model "$VISION_MODEL"

echo ""
echo "============================================================"
echo "FULL CONFIG PIPELINE COMPLETE"
echo "============================================================"
echo "Compare against the lite result in results/RESULTS_SUMMARY.md"
