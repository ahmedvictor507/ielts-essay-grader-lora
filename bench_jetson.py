"""On-device (Jetson Orin Nano) fp16 latency/quality benchmark of the merged model.

Usage: python bench_jetson.py --limit 50
Reports batch-1 latency (mean/p99), time-to-first-token (prefill, 1 new token), decode time per
extra token, peak GPU memory, and the same quality metrics as evaluate.py on the same rows.
"""
import argparse
import json
import statistics as st
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import lenient_parse, load_rows, pearson, score, strict_parse

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="model-0.5B")
ap.add_argument("--data", default="data/eval.jsonl")
ap.add_argument("--limit", type=int, default=50)
ap.add_argument("--max-new-tokens", type=int, default=24)
ap.add_argument("--out", default="results/results_jetson.json")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cuda").eval()

rows = load_rows(args.data, args.limit)
golds = [r["band"] for r in rows]
prompts = [
    tok.apply_chat_template([{"role": "user", "content": r["instruction"]}], tokenize=False,
                            add_generation_prompt=True)
    for r in rows
]


def timed_generate(enc, n_new):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=n_new, min_new_tokens=n_new if n_new == 1 else 0,
                           do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, g


enc0 = tok(prompts[0], return_tensors="pt").to("cuda")
timed_generate(enc0, 4)  # warm-up
torch.cuda.reset_peak_memory_stats()

outs, total, ttft, n_new, n_prompt = [], [], [], [], []
for p in prompts:
    enc = tok(p, return_tensors="pt").to("cuda")
    n_prompt.append(enc["input_ids"].shape[1])
    t1, _ = timed_generate(enc, 1)          # prefill + first token
    t, g = timed_generate(enc, args.max_new_tokens)
    new = g[0, enc["input_ids"].shape[1]:]
    ttft.append(t1)
    total.append(t)
    n_new.append(len(new))
    outs.append(tok.decode(new, skip_special_tokens=True))

preds = [lenient_parse(o) for o in outs]
sc = score(preds, golds)
tot_sorted = sorted(total)
decode_ms = [1000 * (t - f) / (n - 1) for t, f, n in zip(total, ttft, n_new) if n > 1]
res = {
    "device": torch.cuda.get_device_name(0),
    "model": args.model, "dtype": "fp16", "n": len(rows),
    "mean_prompt_tokens": st.mean(n_prompt),
    "model_mb": round(model.get_memory_footprint() / 2**20, 1),
    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    "latency_mean_s": st.mean(total),
    "latency_p99_s": tot_sorted[min(len(total) - 1, int(0.99 * len(total)))],
    "ttft_mean_s": st.mean(ttft),
    "decode_ms_per_token": st.mean(decode_ms) if decode_ms else None,
    "strict_format_rate": sum(strict_parse(o) is not None for o in outs) / len(outs),
    "mae": sc.get("mae"), "within_0_5": sc.get("within_0_5"), "pearson_r": pearson(preds, golds),
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(args.out, "w"), indent=2)
for k, v in res.items():
    print(f"{k:<22}: {v:.3f}" if isinstance(v, float) else f"{k:<22}: {v}")
