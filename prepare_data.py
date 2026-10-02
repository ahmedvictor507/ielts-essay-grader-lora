"""Build train/eval/external-test JSONL files for the IELTS Task 2 band predictor.

Inputs:
  hf_train.jsonl, hf_test.jsonl   chillies/IELTS-writing-task-2-evaluation (LLM-generated labels)
  archive/ielts_writing_dataset.csv   Kaggle set (used only as an external test set)

Outputs:
  data/train.jsonl, data/eval.jsonl, data/kaggle_test.jsonl
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

SEED = 42
EVAL_FRACTION = 0.15
EVAL_CAP = 300  # keeps on-device generation eval tractable
MIN_WORDS, MAX_WORDS = 100, 600
VALID_BANDS = {x / 2 for x in range(8, 19)}  # 4.0 .. 9.0

ROOT = Path(__file__).parent
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)

INSTRUCTION = (
    "You are an IELTS Writing examiner. Read the Task 2 prompt and essay, then "
    'respond only with JSON of the form {{"overall": <band>}}, where band is '
    "0.5-step from 4.0 to 9.0.\n\nPrompt: {prompt}\n\nEssay: {essay}"
)


def norm_prompt(p: str) -> str:
    """Group key tolerant of typos/spacing, so near-identical prompts stay in one split."""
    return re.sub(r"[^a-z]", "", p.lower())[:80]


def clean_band(x):
    try:
        b = float(str(x).strip())
    except ValueError:
        return None  # e.g. "<4"
    return b if b in VALID_BANDS else None


def to_row(prompt, essay, band):
    return {
        "instruction": INSTRUCTION.format(prompt=prompt.strip(), essay=essay.strip()),
        "output": json.dumps({"overall": band}),
        "band": band,
    }


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def dist(rows):
    return dict(sorted(Counter(r["band"] for r in rows).items()))


# ---- HF data: pool train+test (test is fully leaked into train), clean, dedupe ----
df = pd.concat(
    [pd.read_json(ROOT / "hf_train.jsonl", lines=True),
     pd.read_json(ROOT / "hf_test.jsonl", lines=True)],
    ignore_index=True,
)
n0 = len(df)
df["band"] = df["band"].map(clean_band)
df = df.dropna(subset=["band"])
n_bands = len(df)
df["essay"] = df["essay"].str.strip()
df = df.drop_duplicates(subset="essay")
n_dedup = len(df)
words = df["essay"].str.split().str.len()
df = df[(words >= MIN_WORDS) & (words <= MAX_WORDS)]
print(f"HF rows: {n0} -> valid band {n_bands} -> deduped {n_dedup} -> length-filtered {len(df)}")

# ---- grouped split by normalized prompt ----
groups = defaultdict(list)
for r in df.itertuples():
    groups[norm_prompt(r.prompt)].append(to_row(r.prompt, r.essay, r.band))
keys = sorted(groups)
random.Random(SEED).shuffle(keys)

train, eval_ = [], []
eval_keys = set()
target_eval = int(len(df) * EVAL_FRACTION)
for k in keys:
    if len(eval_) < target_eval:
        eval_.extend(groups[k])
        eval_keys.add(k)
    else:
        train.extend(groups[k])
random.Random(SEED).shuffle(train)
random.Random(SEED).shuffle(eval_)
n_eval_full = len(eval_)
eval_ = eval_[:EVAL_CAP]

# ---- Kaggle external test (Task 2 only, deduped) ----
k = pd.read_csv(ROOT / "archive" / "ielts_writing_dataset.csv")
k = k[k.Task_Type == 2].drop_duplicates(subset="Essay")
k["Overall"] = k["Overall"].map(clean_band)
k = k.dropna(subset=["Overall"])
train_prompts = {norm_prompt(p) for p in df["prompt"]}
k_rows = [to_row(r.Question, r.Essay, r.Overall) for r in k.itertuples()]
k_overlap = sum(norm_prompt(r.Question) in train_prompts for r in k.itertuples())

write_jsonl(OUT / "train.jsonl", train)
write_jsonl(OUT / "eval.jsonl", eval_)
write_jsonl(OUT / "kaggle_test.jsonl", k_rows)

print(f"train {len(train)} | eval {len(eval_)} (of {n_eval_full}) | kaggle_test {len(k_rows)}")
print("train bands:", dist(train))
print("eval bands: ", dist(eval_))
print("kaggle bands:", dist(k_rows))
print(f"kaggle prompts also in HF set (normalized): {k_overlap}/{len(k_rows)}")
