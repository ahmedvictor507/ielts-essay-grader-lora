"""Fetch the Hugging Face dataset used for training and write hf_train.jsonl / hf_test.jsonl.

The Kaggle CSV (external test set) must be downloaded manually into archive/ielts_writing_dataset.csv.
Neither dataset is redistributed in this repo; check each dataset's page for its terms.
"""
from datasets import load_dataset

ds = load_dataset("chillies/IELTS-writing-task-2-evaluation")
for split in ds:
    ds[split].to_json(f"hf_{split}.jsonl")
    print(split, len(ds[split]))
