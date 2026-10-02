"""Score an IELTS Writing Task 2 essay with the fine-tuned model.

CLI:
  python grader.py --prompt "<task 2 question>" --essay-file essay.txt
  python grader.py --prompt-file q.txt --essay-file essay.txt --model model-0.5B

Python:
  from grader import load, grade
  model, tok = load("model-0.5B")
  grade(model, tok, prompt, essay)  # -> {"band": 6.5, "raw": '{"overall": 6.5}', "valid": True, ...}

The prompt template must match the one used in training (see prepare_data.py).
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTRUCTION = (
    "You are an IELTS Writing examiner. Read the Task 2 prompt and essay, then "
    'respond only with JSON of the form {{"overall": <band>}}, where band is '
    "0.5-step from 4.0 to 9.0.\n\nPrompt: {prompt}\n\nEssay: {essay}"
)
VALID_BANDS = {x / 2 for x in range(8, 19)}  # 4.0 .. 9.0
TRAIN_WORDS = (100, 600)  # essay length range seen in training


def load(model_dir="model-0.5B", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, device_map=device)
    except TypeError:  # transformers < 4.56 calls it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, device_map=device)
    model.eval()
    return model, tok


def build_prompt(tok, prompt, essay):
    msg = INSTRUCTION.format(prompt=prompt.strip(), essay=essay.strip())
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                   add_generation_prompt=True)


def grade(model, tok, prompt, essay, max_new_tokens=24):
    """Return the predicted overall band plus diagnostics. Greedy decoding."""
    n_words = len(essay.split())
    enc = tok(build_prompt(tok, prompt, essay), return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    seconds = time.perf_counter() - t0
    raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        band = float(json.loads(raw.strip())["overall"])
        valid = band in VALID_BANDS
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        band, valid = None, False
    warning = None
    if not TRAIN_WORDS[0] <= n_words <= TRAIN_WORDS[1]:
        warning = (f"essay has {n_words} words; the model only saw {TRAIN_WORDS[0]}-{TRAIN_WORDS[1]}-word "
                   "essays in training, so treat this score with extra caution")
    return {"band": band, "raw": raw, "valid": valid, "words": n_words,
            "seconds": round(seconds, 3), "warning": warning}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model-0.5B")
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--essay-file", required=True)
    args = ap.parse_args()
    prompt = args.prompt or open(args.prompt_file).read()
    essay = open(args.essay_file).read()
    model, tok = load(args.model)
    r = grade(model, tok, prompt, essay)
    print(json.dumps(r, indent=2))
