# Full Config Runbook (SigLIP + Qwen2-0.5B)

Step-by-step guide to run the `full` configuration on the Mac and add the
lite-vs-full comparison to the paper. Run these in order; do not skip the smoke
test.

## What the full config is

| | lite (already done, 43.7%) | full (this runbook) |
|--|--|--|
| Vision encoder | CNN (trained from scratch) | SigLIP-SO400M (frozen, pretrained) |
| Language model | TinyLMv3 23M (frozen) | Qwen2-0.5B (frozen) |
| Tokenizer | 32K Rust BPE | Qwen2 (151K) |
| Trains | projection + CNN | projection only |
| Downloads | none | SigLIP ~1.6GB + Qwen2 ~1GB |

Both keep the language model frozen (LLaVA-style alignment). This fits in 18GB
and isolates what the connector contributes.

## Prerequisites

```bash
cd /Users/jdsingh/slm_v0/physvision-slm
source venv/bin/activate          # or your env
pip install -r requirements.txt   # adds transformers>=4.40, accelerate, sentencepiece
```

You also need the dataset from the lite run. If `data/generated/annotations.json`
still exists from before, reuse it. Otherwise regenerate (must match the lite run):

```bash
python scripts/generate_dataset.py --num-images 10000 --output data/generated --seed 42
```

## Step 1 — Smoke test the full config (REQUIRED gate)

Downloads the models (~3GB) on first run and validates the whole wiring in
seconds of compute:

```bash
python scripts/smoke_test.py --config full
```

Expect:
- `[tokenizer] Using HuggingFace tokenizer for Qwen/Qwen2-0.5B (vocab 151936).`
- a finite `loss:` value
- `SMOKE TEST PASSED`

If it fails with out-of-memory, or errors, STOP and use the lighter encoder
(Step 1b). Do not start the long run until the smoke test passes.

### Step 1b — If OOM: use a smaller SigLIP

```bash
VISION_MODEL=google/siglip-base-patch16-224 python scripts/smoke_test.py --config full
```

(The `base` encoder is ~0.4GB vs ~1.6GB and 224px vs 384px. Still a legitimate
"full" comparison; just note the smaller encoder in the paper.)

## Step 2 — Run the full pipeline

```bash
bash scripts/run_full_config.sh
```

Or, if you needed the smaller encoder / smaller batch:

```bash
VISION_MODEL=google/siglip-base-patch16-224 BATCH=2 bash scripts/run_full_config.sh
```

This runs Stage 1 (alignment), Stage 2 (instruction tuning), and evaluation,
reusing `data/generated`. Watch for:
- `Vision encoder trainable: False (frozen pretrained)` — SigLIP is frozen (correct)
- checkpoints written to `checkpoints/full/`
- a final `RESULTS — PhysVision-SLM (full)` block

## Step 3 — (Optional) quick subset eval first

To get a fast read before committing to the full 35K eval:

```bash
python scripts/eval_internal.py \
    --checkpoint checkpoints/full/stage2_best.pt \
    --data data/generated --config full \
    --vision-model google/siglip-so400m-patch14-384 \
    --max-samples 3000
```

(Use the same `--vision-model` you trained with.)

## Step 4 — Send me the numbers

Copy the final eval output (overall accuracy + per-category + per-image-type)
and the `results/eval_full_*.json` file. I will:
- fill the `[TBD]` full row in the paper's results table
- write the lite-vs-full comparison paragraph
- regenerate the results figure with both configs

## What to expect (honest)

Because Qwen2 is frozen (only the projection trains), a better vision encoder
improves *perception* but the frozen general-purpose LM caps *answer
specialization*. The full config may beat lite modestly, match it, or trade wins
by category (e.g. better localization from SigLIP's spatial features, similar
detection). Any of these is a legitimate, reportable finding. Do not expect the
0.9B model to dominate the 23M one purely on size.

## Commit results back

```bash
git add results/ checkpoints/full/.gitkeep 2>/dev/null
git add scripts/ src/ doc/
git commit -m "Add full config (SigLIP + Qwen2) results"
git push
```

(Checkpoints and generated data are gitignored; only code, results JSON, and the
paper are committed.)
