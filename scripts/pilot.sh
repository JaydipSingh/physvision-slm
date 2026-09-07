#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# PhysVision-SLM PILOT  (fast validation before the full run)
# =============================================================================
# Purpose: confirm the vision pathway actually works AFTER the projection /
# encoder fixes, WITHOUT spending 8+ hours. Runs on a small dataset and a short
# Stage-1 training, then checks whether the image changes the model's output.
#
#   1. Generate a small dataset (default 500 images)
#   2. Stage 1 alignment, few epochs (trains CNN encoder + projection)
#   3. Vision-pathway check (real image vs blank image)
#
# Expected total time: ~20-40 min on M3 Pro.
#
# If the check says "vision pathway is live", proceed to run_full_pipeline.sh.
# If it says "vision pathway is broken", STOP and investigate — do not run the
# full pipeline.
# =============================================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

LM_CHECKPOINT="${LM_CHECKPOINT:-$ROOT/../subword_tokenizer/slm_v2/checkpoints/v3_long50k_final.pt}"
NUM_IMAGES="${NUM_IMAGES:-500}"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-8}"
DATA_DIR="$ROOT/data/pilot"
CKPT_DIR="$ROOT/checkpoints/pilot"

echo "PhysVision-SLM PILOT"
echo "  Root:          $ROOT"
echo "  LM checkpoint: $LM_CHECKPOINT"
echo "  Num images:    $NUM_IMAGES"
echo "  Epochs:        $EPOCHS   Batch: $BATCH"
echo "  Pilot data:    $DATA_DIR"
echo "  Pilot ckpts:   $CKPT_DIR"
echo ""

if [ -f "$ROOT/../venv/bin/activate" ]; then
    source "$ROOT/../venv/bin/activate"
fi

LM_ARG=""
if [ -f "$LM_CHECKPOINT" ]; then
    LM_ARG="--lm-checkpoint $LM_CHECKPOINT"
    echo "[ok] Found pretrained LM checkpoint."
else
    echo "[WARN] LM checkpoint not found at $LM_CHECKPOINT — using RANDOM LM."
    echo "       Set LM_CHECKPOINT=/path/to/v3_long50k_final.pt"
fi
echo ""

# Step 1: small dataset
echo "============================================================"
echo "PILOT STEP 1: Generate $NUM_IMAGES images"
echo "============================================================"
python "$ROOT/scripts/generate_dataset.py" \
    --num-images "$NUM_IMAGES" \
    --output "$DATA_DIR" \
    --seed 7

# Step 2: short Stage 1 training
echo ""
echo "============================================================"
echo "PILOT STEP 2: Stage 1 alignment ($EPOCHS epochs)"
echo "============================================================"
python "$ROOT/src/train_multimodal.py" \
    --stage 1 \
    --data "$DATA_DIR" \
    --config lite \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --lr 1e-3 \
    $LM_ARG \
    --save-dir "$CKPT_DIR"

# Step 3: vision pathway check
echo ""
echo "============================================================"
echo "PILOT STEP 3: Vision pathway check (real vs blank image)"
echo "============================================================"
python "$ROOT/scripts/debug_eval.py" \
    --checkpoint "$CKPT_DIR/stage1_best.pt" \
    --data "$DATA_DIR" \
    --config lite \
    $LM_ARG \
    --vision-check --n 20

echo ""
echo "============================================================"
echo "PILOT COMPLETE"
echo "============================================================"
echo "If the verdict above is 'vision pathway is live', run the full pipeline:"
echo "  LM_CHECKPOINT=$LM_CHECKPOINT bash scripts/run_full_pipeline.sh"
echo ""
echo "If the verdict is 'broken', STOP and share the output for further diagnosis."
