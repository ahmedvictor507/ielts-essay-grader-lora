"""LoRA fine-tune of a small instruct model as an IELTS Task 2 band predictor.

Usage:
  python train_lora.py                                   # Qwen2.5-1.5B, full train set
  python train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --max-train 1000   # quick/smoke run
"""
import argparse
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--train", default="data/train.jsonl")
ap.add_argument("--eval", default="data/eval.jsonl")
ap.add_argument("--max-train", type=int, default=None)
ap.add_argument("--max-eval", type=int, default=100, help="eval rows used for loss during training")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--max-length", type=int, default=1024)
ap.add_argument("--out", default="lora-adapter")
args = ap.parse_args()

bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if bf16 else torch.float16  # T4 has no bf16

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map="auto")


def to_chat(ex):
    # prompt/completion format: TRL applies the chat template and computes loss
    # on the assistant completion only.
    return {
        "prompt": [{"role": "user", "content": ex["instruction"]}],
        "completion": [{"role": "assistant", "content": ex["output"]}],
    }


ds = load_dataset("json", data_files={"train": args.train, "eval": args.eval})
train_ds = ds["train"].shuffle(seed=42)
if args.max_train:
    train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
eval_ds = ds["eval"].select(range(min(args.max_eval, len(ds["eval"]))))
train_ds = train_ds.map(to_chat, remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(to_chat, remove_columns=eval_ds.column_names)

lora = LoraConfig(
    r=args.rank,
    lora_alpha=2 * args.rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()  # record for the write-up

cfg = SFTConfig(
    output_dir=f"{args.out}-checkpoints",
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=0.03,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    bf16=bf16,
    fp16=not bf16,
    gradient_checkpointing=True,
    max_length=args.max_length,
    report_to="none",
    seed=42,
)

trainer = SFTTrainer(
    model=model, args=cfg, train_dataset=train_ds, eval_dataset=eval_ds,
    processing_class=tokenizer,
)

t0 = time.time()
trainer.train()
print(f"training time: {(time.time() - t0) / 60:.1f} min on {torch.cuda.get_device_name(0)}")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

trainer.model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)
