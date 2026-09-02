#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# PhysVision-SLM Full Pipeline  (run on Apple M3 Pro)
# =============================================================================
# Total time: ~8-12 hours on M3 Pro
#   Data generation: ~30 min (10K images)
#   Stage 1 training: ~2 hours
#   Stage 2 training: ~4 hours
#   Evaluation: ~10 min
#
# IMPORTANT: set LM_CHECKPOINT to your trained TinyLMv3 checkpoint from paper 1
# so the language model starts from pretrained physics weights (not random).
# =============================================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Configure paths -------------------------------------------------------
# Path to the pretrained TinyLMv3 checkpoint produced by paper 1 on your Mac.
# Adjust if your slm_v2 checkpoints live elsewhere.
LM_CHECKPOINT="${LM_CHECKPOINT:-$ROOT/../slm_v2/checkpoints/v3_long50k_final.pt}"
NUM_IMAGES="${NUM_IMAGES:-10000}"

echo "PhysVision-SLM Pipeline"
echo "Root:          $ROOT"
echo "LM checkpoint: $LM_CHECKPOINT"
echo "Num images:    $NUM_IMAGES"
echo ""

# Activate venv if present
if [ -f "$ROOT/../venv/bin/activate" ]; then
    source "$ROOT/../venv/bin/activate"
fi

LM_ARG=""
if [ -f "$LM_CHECKPOINT" ]; then
    LM_ARG="--lm-checkpoint $LM_CHECKPOINT"
    echo "[ok] Found pretrained LM checkpoint."
else
    echo "[WARN] LM checkpoint not found at $LM_CHECKPOINT"
    echo "       Training will start from a RANDOM language model."
    echo "       Set LM_CHECKPOINT=/path/to/v3_long50k_final.pt to fix this."
fi
echo ""

# Step 1: Generate Dataset
echo "============================================================"
echo "STEP 1: Generate Training Dataset ($NUM_IMAGES images)"
echo "============================================================"
python "$ROOT/scripts/generate_dataset.py" \
    --num-images "$NUM_IMAGES" \
    --output "$ROOT/data/generated" \
    --seed 42

# Step 2: Run Baselines
echo ""
echo "============================================================"
echo "STEP 2: Run Baseline Evaluation"
echo "============================================================"
python "$ROOT/scripts/eval_baselines.py" --data "$ROOT/data/generated"

# Step 3: Stage 1 Training (Alignment)
echo ""
echo "============================================================"
echo "STEP 3: Stage 1 — Alignment Training"
echo "============================================================"
python "$ROOT/src/train_multimodal.py" \
    --stage 1 \
    --data "$ROOT/data/generated" \
    --config lite \
    --epochs 3 \
    --batch-size 8 \
    --lr 1e-3 \
    $LM_ARG \
    --save-dir "$ROOT/checkpoints"

# Step 4: Stage 2 Training (Instruction Tuning)
echo ""
echo "============================================================"
echo "STEP 4: Stage 2 — Instruction Tuning"
echo "============================================================"
python "$ROOT/src/train_multimodal.py" \
    --stage 2 \
    --data "$ROOT/data/generated" \
    --config lite \
    --epochs 5 \
    --batch-size 8 \
    --lr 5e-4 \
    $LM_ARG \
    --checkpoint "$ROOT/checkpoints/stage1_best.pt" \
    --save-dir "$ROOT/checkpoints"

# Step 5: Evaluation
echo ""
echo "============================================================"
echo "STEP 5: Final Evaluation"
echo "============================================================"
python "$ROOT/scripts/eval_internal.py" \
    --checkpoint "$ROOT/checkpoints/stage2_best.pt" \
    --data "$ROOT/data/generated" \
    --config lite \
    $LM_ARG

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo "Results in: $ROOT/results/"
echo "Checkpoints in: $ROOT/checkpoints/"
