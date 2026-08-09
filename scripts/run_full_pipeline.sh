#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# PhysVision-SLM Full Pipeline
# =============================================================================
# Total time: ~8-12 hours on M3 Pro
#   Data generation: ~30 min (10K images)
#   Stage 1 training: ~2 hours
#   Stage 2 training: ~4 hours
#   Evaluation: ~10 min
# =============================================================================

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "PhysVision-SLM Pipeline"
echo "Root: $ROOT"
echo ""

# Activate venv
if [ -f "$ROOT/../venv/bin/activate" ]; then
    source "$ROOT/../venv/bin/activate"
fi

# Step 1: Generate Dataset
echo "============================================================"
echo "STEP 1: Generate Training Dataset (10K images)"
echo "============================================================"
python "$ROOT/scripts/generate_dataset.py" \
    --num-images 10000 \
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
    --save-dir "$ROOT/checkpoints"

# Step 5: Evaluation
echo ""
echo "============================================================"
echo "STEP 5: Final Evaluation"
echo "============================================================"
python "$ROOT/scripts/eval_internal.py" \
    --checkpoint "$ROOT/checkpoints/stage2_best.pt" \
    --data "$ROOT/data/generated" \
    --config lite

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo "Results in: $ROOT/results/"
echo "Checkpoints in: $ROOT/checkpoints/"
