#!/usr/bin/env python3
"""
Debug the PhysVision evaluation on a few samples.
Prints, per question: each option's log-prob score, the predicted option,
and the correct option. Reveals whether the model collapses to one option
or whether the scoring offsets are misaligned.

Usage:
  python scripts/debug_eval.py --checkpoint checkpoints/stage2_best.pt \
      --data data/generated --config lite \
      --lm-checkpoint /path/to/v3_long50k_final.pt --n 15
"""
import argparse
import collections
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.physvision_model import PhysVisionModel
from src.tokenizer_bridge import get_tokenizer
from scripts.eval_internal import PhysVisionEvalDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--data", type=str, default="data/generated")
    ap.add_argument("--config", type=str, default="lite")
    ap.add_argument("--lm-checkpoint", type=str, default=None)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--scan-all", action="store_true",
                    help="Scan the whole eval set and print predicted-option distribution")
    ap.add_argument("--vision-check", action="store_true",
                    help="Compare option scores with the real image vs a blank "
                         "image, to confirm the vision pathway affects the output")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    model = PhysVisionModel.from_config(args.config, device=device,
                                        lm_checkpoint=args.lm_checkpoint)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint ({len(missing)} missing, {len(unexpected)} unexpected)")
    model.to(device)
    model.eval()

    tok = get_tokenizer(vocab_size=32000)
    ds = PhysVisionEvalDataset(args.data)
    transform = ds.transform
    print(f"Eval samples: {len(ds)}")

    def score_options(sample):
        img = Image.open(sample["image_path"]).convert("RGB")
        pv = transform(img).unsqueeze(0).to(device)
        q = sample["question"]
        prompt = f"Question: {q} Answer:"
        prompt_ids = tok(prompt)
        prompt_len = len(prompt_ids)
        voff = model.num_vision_tokens
        scores = []
        pred_positions = []
        for opt in sample["options"]:
            full_ids = tok(f"{prompt} {opt}")
            answer_ids = full_ids[prompt_len:]
            full_ids = full_ids[:128]
            input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(pv, input_ids)
            lp = F.log_softmax(out["logits"][0], dim=-1)
            s, c = 0.0, 0
            for j, tid in enumerate(answer_ids):
                text_pos = prompt_len + j
                if text_pos >= len(full_ids):
                    break
                pred_pos = voff + text_pos - 1
                if 0 <= pred_pos < lp.shape[0]:
                    s += lp[pred_pos, tid].item()
                    c += 1
            scores.append(s / max(c, 1))
            pred_positions.append((prompt_len, len(answer_ids), c))
        return scores, pred_positions

    if args.vision_check:
        # For each of the first N samples, score options with the real image
        # and with a blank (zero) image. If the pathway works, the scores and/or
        # the predicted option should differ between the two.
        n = min(args.n, len(ds))
        changed = 0
        max_abs_delta = 0.0
        for idx in range(n):
            sample = ds.get_sample(idx)

            # real image
            real_scores, _ = score_options(sample)

            # blank image: monkeypatch transform to return zeros for this call
            img = Image.open(sample["image_path"]).convert("RGB")
            real_pv = transform(img).unsqueeze(0).to(device)
            blank_pv = torch.zeros_like(real_pv)

            def score_with(pv):
                q = sample["question"]
                prompt = f"Question: {q} Answer:"
                pids = tok(prompt); plen = len(pids); voff = model.num_vision_tokens
                out_scores = []
                for opt in sample["options"]:
                    fids = tok(f"{prompt} {opt}"); aids = fids[plen:]; fids = fids[:128]
                    ii = torch.tensor([fids], dtype=torch.long, device=device)
                    with torch.no_grad():
                        lp = F.log_softmax(model(pv, ii)["logits"][0], dim=-1)
                    s = c = 0
                    for j, tid in enumerate(aids):
                        tp = plen + j
                        if tp >= len(fids):
                            break
                        pp = voff + tp - 1
                        if 0 <= pp < lp.shape[0]:
                            s += lp[pp, tid].item(); c += 1
                    out_scores.append(s / max(c, 1))
                return out_scores

            rs = score_with(real_pv)
            bs = score_with(blank_pv)
            delta = max(abs(a - b) for a, b in zip(rs, bs))
            max_abs_delta = max(max_abs_delta, delta)
            pred_real = int(max(range(len(rs)), key=lambda i: rs[i]))
            pred_blank = int(max(range(len(bs)), key=lambda i: bs[i]))
            if pred_real != pred_blank or delta > 1e-3:
                changed += 1
            print(f"[{idx}] {sample['question'][:40]:40s} "
                  f"max|delta|={delta:.4f} pred_real={pred_real} pred_blank={pred_blank}")

        print("\n" + "=" * 60)
        print(f"Vision pathway check over {n} samples:")
        print(f"  samples where image changed the output: {changed}/{n}")
        print(f"  max abs log-prob delta (real vs blank): {max_abs_delta:.4f}")
        if max_abs_delta < 1e-3:
            print("  VERDICT: image has NO effect — vision pathway is broken.")
        else:
            print("  VERDICT: image affects output — vision pathway is live.")
        return

    if args.scan_all:
        preds = collections.Counter()
        pred_by_correct = collections.Counter()
        for idx in range(len(ds)):
            sample = ds.get_sample(idx)
            scores, _ = score_options(sample)
            p = int(max(range(len(scores)), key=lambda i: scores[i]))
            preds[p] += 1
            pred_by_correct[(p, sample["correct_option"])] += 1
            if (idx + 1) % 2000 == 0:
                print(f"  scanned {idx+1}/{len(ds)}")
        print("\nPredicted-option index distribution:", dict(preds))
        return

    n = min(args.n, len(ds))
    for idx in range(n):
        sample = ds.get_sample(idx)
        scores, pos = score_options(sample)
        pred = int(max(range(len(scores)), key=lambda i: scores[i]))
        print("\n" + "=" * 70)
        print(f"Q: {sample['question']}")
        print(f"  category={sample['category']} image_type={sample['image_type']}")
        for i, (opt, sc, pp) in enumerate(zip(sample["options"], scores, pos)):
            mark = ""
            if i == sample["correct_option"]:
                mark += " [CORRECT]"
            if i == pred:
                mark += " <-- PREDICTED"
            print(f"    [{i}] {opt:20s} logprob={sc:8.4f} "
                  f"(prompt_len={pp[0]}, ans_toks={pp[1]}, scored={pp[2]}){mark}")


if __name__ == "__main__":
    main()
