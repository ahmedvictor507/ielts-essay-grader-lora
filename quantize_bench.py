"""Merge the LoRA adapter, then compare fp16 vs int8 vs nf4 (bitsandbytes) on the same eval rows.

Reports per variant: model memory footprint, peak GPU memory, quality (format validity, MAE,
within ±0.5, Pearson r) and batch-1 latency (mean / p99 seconds per request, ms per generated token).

Usage (Colab T4):
  python quantize_bench.py --adapter /content/drive/MyDrive/ielts-lora-0.5b --limit 100
"""
import argparse
import gc
import json
import statistics as st
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluate import lenient_parse, load_rows, pearson, score, strict_parse

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--adapter", required=True)
ap.add_argument("--data", default="data/eval.jsonl")
ap.add_argument("--limit", type=int, default=100)
ap.add_argument("--merged-dir", default="merged-0.5b")
ap.add_argument("--max-new-tokens", type=int, default=24)
ap.add_argument("--variants", default="fp16,int8,nf4")
ap.add_argument("--out", default="results/results_quant.json")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")

# ---- merge adapter into base weights (fp16) and save ----
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
merged.save_pretrained(args.merged_dir)
tok.save_pretrained(args.merged_dir)
del base, merged
gc.collect()
torch.cuda.empty_cache()

rows = load_rows(args.data, args.limit)
golds = [r["band"] for r in rows]
prompts = [
    tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False,
                            add_generation_prompt=True)
    for r in rows
]

BNB = {
    "fp16": None,
    "int8": BitsAndBytesConfig(load_in_8bit=True),
    "nf4": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float16),
}


def run(variant):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kw = {"quantization_config": BNB[variant]} if BNB[variant] else {"dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(args.merged_dir, device_map="auto", **kw).eval()
    footprint_mb = model.get_memory_footprint() / 2**20

    # warm-up so first-call kernel setup doesn't pollute latency
    enc = tok(prompts[0], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)

    outs, lat, ntok = [], [], []
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        new = g[0, enc["input_ids"].shape[1]:]
        ntok.append(len(new))
        outs.append(tok.decode(new, skip_special_tokens=True))

    preds = [lenient_parse(o) for o in outs]
    sc = score(preds, golds)
    lat_sorted = sorted(lat)
    res = {
        "variant": variant,
        "model_mb": round(footprint_mb, 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "strict_format_rate": sum(strict_parse(o) is not None for o in outs) / len(outs),
        "mae": sc.get("mae"),
        "within_0_5": sc.get("within_0_5"),
        "pearson_r": pearson(preds, golds),
        "latency_mean_s": st.mean(lat),
        "latency_p99_s": lat_sorted[min(len(lat) - 1, int(0.99 * len(lat)))],
        "ms_per_token": 1000 * sum(lat) / max(1, sum(ntok)),
        "n": len(rows),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


results = [run(v) for v in args.variants.split(",")]
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(args.out, "w"), indent=2)

hdr = f"{'variant':<8}{'size MB':>9}{'peak MB':>9}{'fmt':>7}{'MAE':>7}{'±0.5':>7}{'r':>7}{'mean s':>8}{'p99 s':>8}{'ms/tok':>8}"
print("\n" + hdr)
for r in results:
    rr = "  n/a" if r["pearson_r"] is None else f"{r['pearson_r']:.3f}"
    print(f"{r['variant']:<8}{r['model_mb']:>9.0f}{r['peak_gpu_mb']:>9.0f}{r['strict_format_rate']:>7.0%}"
          f"{r['mae']:>7.3f}{r['within_0_5']:>7.1%}{rr:>7}{r['latency_mean_s']:>8.3f}"
          f"{r['latency_p99_s']:>8.3f}{r['ms_per_token']:>8.1f}")
print(f"\nresults -> {args.out}   merged fp16 model -> {args.merged_dir}")
