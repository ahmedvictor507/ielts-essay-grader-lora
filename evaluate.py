"""Evaluate a (base or LoRA-tuned) model as an IELTS band predictor.

Usage:
  python evaluate.py --name base                                   # base model
  python evaluate.py --name lora --adapter lora-0.5b               # base + adapter
  python evaluate.py --name lora-kaggle --adapter lora-0.5b --data data/kaggle_test.jsonl

Metrics (all on the same rows, greedy decoding):
  strict_format   output is exactly valid JSON {"overall": <0.5-step band 4.0-9.0>}
  lenient_parse   a band number could be recovered from the text (used for MAE on base model)
  mae / within_0.5 / exact   computed on lenient-parsed rows; coverage is reported alongside
  baselines       predict-train-mode and predict-train-mean, scored on the same rows
"""
import argparse
import json
import random
import re
import statistics as st
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

VALID_BANDS = {x / 2 for x in range(8, 19)}  # 4.0 .. 9.0


def strict_parse(text):
    """Return band if text is exactly {"overall": <valid band>}, else None."""
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj) != {"overall"}:
        return None
    v = obj["overall"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if float(v) in VALID_BANDS else None


def lenient_parse(text):
    """Recover a band from free text: prefer an 'overall' key, else first plausible number."""
    m = re.search(r'overall[^0-9]{0,20}(\d(?:\.\d)?)', text, re.I) or re.search(r"\b(\d(?:\.\d)?)\b", text)
    if not m:
        return None
    v = float(m.group(1))
    return v if v in VALID_BANDS else None


def score(preds, golds):
    """preds may contain None. Returns dict of metrics over non-None preds."""
    pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
    out = {"n": len(golds), "coverage": len(pairs) / len(golds)}
    if pairs:
        errs = [abs(p - g) for p, g in pairs]
        out.update(
            mae=st.mean(errs),
            within_0_5=sum(e <= 0.5 for e in errs) / len(errs),
            exact=sum(e == 0 for e in errs) / len(errs),
        )
    return out


def pearson(preds, golds):
    pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
    if len(pairs) < 3 or len({p for p, _ in pairs}) < 2:
        return None  # constant predictions: correlation undefined
    return st.correlation([p for p, _ in pairs], [g for _, g in pairs])


def bootstrap_mae_gap(preds, baseline, golds, iters=2000, seed=0):
    """95% CI for MAE(model) - MAE(constant baseline) over resampled rows. Negative = model better."""
    rng = random.Random(seed)
    idx = [i for i, p in enumerate(preds) if p is not None]
    diffs = []
    for _ in range(iters):
        samp = [rng.choice(idx) for _ in idx]
        m = st.mean(abs(preds[i] - golds[i]) for i in samp)
        b = st.mean(abs(baseline - golds[i]) for i in samp)
        diffs.append(m - b)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def load_rows(path, limit):
    rows = [json.loads(l) for l in open(path)]
    return rows[:limit] if limit else rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--train", default="data/train.jsonl", help="for the baseline predictors")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--name", required=True)
    ap.add_argument("--save", default=None, help="results json path (default results_<name>.json)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = load_rows(args.data, args.limit)
    golds = [r["band"] for r in rows]
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": r["instruction"]}],
                                tokenize=False, add_generation_prompt=True)
        for r in rows
    ]

    outputs, lat = [], []
    for i in range(0, len(prompts), args.batch):
        batch = prompts[i:i + args.batch]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        if dev == "cuda":
            torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        outputs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{min(i + args.batch, len(rows))}/{len(rows)}", end="\r")
    print()

    strict = [strict_parse(o) for o in outputs]
    lenient = [lenient_parse(o) for o in outputs]
    train_bands = [json.loads(l)["band"] for l in open(args.train)]
    mode_band = Counter(train_bands).most_common(1)[0][0]
    mean_band = round(st.mean(train_bands) * 2) / 2

    res = {
        "name": args.name, "model": args.model, "adapter": args.adapter, "data": args.data,
        "strict_format_rate": sum(p is not None for p in strict) / len(rows),
        "lenient_parse_rate": sum(p is not None for p in lenient) / len(rows),
        "model_lenient": score(lenient, golds),
        "baseline_mode": {"band": mode_band, **score([mode_band] * len(rows), golds)},
        "baseline_mean": {"band": mean_band, **score([mean_band] * len(rows), golds)},
        "batch_latency_s_mean": st.mean(lat),
        "pearson_r": pearson(lenient, golds),
        "pred_distribution": dict(sorted(Counter(lenient).items(), key=lambda kv: (kv[0] is None, kv[0]))),
        "gold_distribution": dict(sorted(Counter(golds).items())),
        "mae_gap_vs_mean_ci95": bootstrap_mae_gap(lenient, mean_band, golds) if any(lenient) else None,
        "mae_gap_vs_mode_ci95": bootstrap_mae_gap(lenient, mode_band, golds) if any(lenient) else None,
        "predictions": [{"gold": g, "pred": p, "out": o} for g, p, o in zip(golds, lenient, outputs)],
    }
    path = args.save or f"results/results_{args.name}.json"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(path, "w"), indent=2)

    m = res["model_lenient"]
    print(f"\n== {args.name} on {args.data} (n={len(rows)}) ==")
    print(f"strict format valid : {res['strict_format_rate']:.1%}")
    print(f"lenient parse       : {res['lenient_parse_rate']:.1%}")
    if "mae" in m:
        print(f"MAE                 : {m['mae']:.3f}   within ±0.5: {m['within_0_5']:.1%}   exact: {m['exact']:.1%}")
    for k in ("baseline_mode", "baseline_mean"):
        b = res[k]
        print(f"{k:<20}: band {b['band']}  MAE {b['mae']:.3f}  within ±0.5 {b['within_0_5']:.1%}")
    if res["pearson_r"] is not None:
        print(f"Pearson r           : {res['pearson_r']:.3f}")
    else:
        print("Pearson r           : undefined (constant predictions)")
    print(f"pred distribution   : {res['pred_distribution']}")
    for k, lbl in (("mae_gap_vs_mean_ci95", "mean"), ("mae_gap_vs_mode_ci95", "mode")):
        if res[k]:
            lo, hi = res[k]
            print(f"MAE gap vs {lbl:<4}     : 95% CI [{lo:+.3f}, {hi:+.3f}]  (negative = model better; spans 0 = not significant)")
    print(f"results -> {path}")
