# Transfer & Run Guide (Apple M3 Pro)

This guide walks through transferring PhysVision-SLM to your Mac and running the
full training + evaluation pipeline.

## Prerequisites from Paper 1

PhysVision's `lite` config reuses two things from paper 1 (`slm_v2`):

1. **The trained TinyLMv3 checkpoint** — `v3_long50k_final.pt`
2. **The Rust BPE tokenizer** — the compiled binary + `model_32k.json`

These typically live under:

```
/Users/jdsingh/slm_v0/subword_tokenizer/slm_v2/checkpoints/v3_long50k_final.pt
/Users/jdsingh/slm_v0/subword_tokenizer/           (Rust tokenizer + model_32k.json)
```

## Recommended Folder Layout

The code auto-resolves paper 1 as a **sibling** folder. Clone PhysVision so that
`slm_v2/` and `subword_tokenizer/` sit next to it:

```
slm_v0/
├── slm_v2/                 <- paper 1 (checkpoints live here)
├── subword_tokenizer/      <- Rust tokenizer + model_32k.json
└── physvision-slm/         <- this repo (git clone target)
```

## Step-by-Step

### 1. Transfer / clone the code

```bash
cd /Users/jdsingh/slm_v0
git clone https://github.com/JaydipSingh/physvision-slm.git
cd physvision-slm
```

(Or copy the folder over manually via AirDrop/USB and rename it `physvision-slm`.)

### 2. Set up the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon this pulls the MPS-enabled torch automatically. The scripts
auto-detect and use `mps`.

### 3. Verify the wiring first (seconds)

```bash
python scripts/smoke_test.py
```

Expect `SMOKE TEST PASSED`. Critically, check the tokenizer line — on the Mac it
should say:

```
[tokenizer] Using real Rust BPE tokenizer (32K vocab).
```

If it instead warns about the character-level fallback, the Rust binary wasn't
found (see Troubleshooting). **Do not start the full run until the real tokenizer
is detected**, or the pretrained LM weights won't align with the token IDs.

### 4. Run the full pipeline

```bash
LM_CHECKPOINT=/Users/jdsingh/slm_v0/subword_tokenizer/slm_v2/checkpoints/v3_long50k_final.pt \
    bash scripts/run_full_pipeline.sh
```

This runs: data generation (10K images, ~30 min) -> baselines -> Stage 1 alignment
(~2h) -> Stage 2 instruction tuning (~4h) -> evaluation. Total ~8-12h. The script
prints `[ok] Found pretrained LM checkpoint.` when the path is correct.

To sanity-check timing with a smaller first run:

```bash
NUM_IMAGES=500 \
LM_CHECKPOINT=/Users/jdsingh/slm_v0/subword_tokenizer/slm_v2/checkpoints/v3_long50k_final.pt \
    bash scripts/run_full_pipeline.sh
```

### 5. Consolidate and commit results

```bash
python scripts/log_results.py
git add results/summary.json results/RESULTS_SUMMARY.md
git commit -m "Add PhysVision evaluation results"
git push
```

`log_results.py` produces two paper-ready artifacts from the eval JSONs:
`results/summary.json` (machine-readable) and `results/RESULTS_SUMMARY.md`
(Markdown tables for the paper / README).

## Troubleshooting

**Tokenizer falls back to char-level.** The code looks for the compiled Rust
binary at `subword_tokenizer/target/release/bpe-tokenizer`. If it's missing,
rebuild it:

```bash
cd /Users/jdsingh/slm_v0/subword_tokenizer
cargo build --release
```

Then re-run the smoke test. Also make sure `model_32k.json` is present there.

**`slm_v2` not found / can't import TinyLMv3.** The architecture is bundled
standalone in `src/tinylm_v3.py`, so imports work regardless. `slm_v2` is only
needed for the *checkpoint file* — just make sure `LM_CHECKPOINT` points to a
valid `.pt`.

**MPS out-of-memory.** Drop the batch size:

```bash
python src/train_multimodal.py --stage 1 --data data/generated --config lite \
    --lm-checkpoint /path/to/v3_long50k_final.pt --epochs 3 --batch-size 4
```

**Checkpoint path differs.** Locate it with:

```bash
find /Users/jdsingh/slm_v0 -name "v3_long50k_final.pt"
```

and use whatever path it returns.

## Pipeline Stages Reference

| Stage | Command | Trains | ~Time |
|-------|---------|--------|-------|
| Data gen | `scripts/generate_dataset.py` | — | 30 min (10K) |
| Baselines | `scripts/eval_baselines.py` | — | seconds |
| Stage 1 | `train_multimodal.py --stage 1` | projection only | ~2h |
| Stage 2 | `train_multimodal.py --stage 2` | projection + LM | ~4h |
| Eval | `scripts/eval_internal.py` | — | ~10 min |
| Summary | `scripts/log_results.py` | — | seconds |
