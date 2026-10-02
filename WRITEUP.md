# LoRA fine-tuning + quantization of a small LLM as an IELTS Task 2 band predictor

## Problem
Fine-tune a small open-weight model to read an IELTS Writing Task 2 prompt and essay and return a
band score as strict JSON (`{"overall": 6.5}`), then measure what quantization costs and saves.
Scope is deliberately narrow: overall band only.

## Method
- **Base model:** Qwen2.5-0.5B-Instruct. **LoRA:** r=16, alpha=32, all attention + MLP linear layers;
  8.8M trainable of 502.8M parameters (1.75%). Loss on the JSON completion only.
- **Training:** 3,000 essays, 2 epochs, lr 2e-4 cosine, effective batch 16, on a Colab T4:
  73.2 min, 1.74 GiB peak GPU memory.
- **Data:** `chillies/IELTS-writing-task-2-evaluation` (HF), cleaned: invalid/`<4` bands removed,
  10,324 -> 8,465 essays after de-duplication and length filtering. I found the dataset's own
  test split was fully contained in its train split (100% essay and prompt overlap), so I pooled
  and re-split **grouped by normalized prompt** (7,196 train / 1,269 held-out, 300 used for eval).
- **External test:** 709 Task 2 essays from a separate Kaggle dataset (300 used).
- **Metrics:** strict JSON validity, MAE, share within 0.5 band, Pearson r, bootstrap 95% CI on the
  MAE gap to constant baselines (always-mean, always-mode).

## Results
| | Format valid | MAE | Within ±0.5 | Pearson r |
|---|---|---|---|---|
| Base model, HF held-out (n=300) | 15.7% | 2.23 (47% parseable only) | 15.7% | – |
| LoRA, HF held-out (n=300) | **100%** | **0.93** | 44.7% | **0.47** |
| Always predict mean (6.5), HF | – | 1.06 | 42.7% | – |
| LoRA, Kaggle external (n=300) | 100% | 0.77 | 54.7% | **0.54** |
| Always predict mean (6.5), Kaggle | – | 0.80 | 55.0% | – |

- HF held-out: MAE gap vs always-mean 95% CI [-0.195, -0.070] (significant).
- Kaggle: gap vs always-mean CI [-0.098, +0.040] (not significant); vs always-mode [-0.205, -0.025].

### Quantization (merged model, T4, batch 1, n=100 prompts, ~500-token prompts, <=24 new tokens)
| Variant | Weights MB | Peak GPU MB | Format valid | MAE | Pearson r | Latency mean / p99 (s) |
|---|---|---|---|---|---|---|
| fp16 | 942 | 1043 | 100% | 0.955 | 0.426 | 0.33 / 0.48 |
| int8 (bnb) | 601 (-36%) | 703 | 100% | 0.940 | 0.457 | 1.37 / 1.78 |
| nf4 (bnb) | 430 (-54%) | 553 | 100% | 1.035 | 0.433 | 0.46 / 0.62 |

Memory dropped as expected. Quality differences between variants are within noise at n=100, so I
can say quantization did not visibly break the model, not that it is lossless. With bitsandbytes on
a T4, both quantized variants were **slower** than fp16 (dequantization overhead; int8 worst), so
the memory saving did not buy speed here.

### On-device (Jetson Orin Nano 8 GB, fp16, batch 1, n=50 prompts, mean 470 prompt tokens)
| Metric | Orin Nano | (T4, fp16, for reference) |
|---|---|---|
| Latency mean / p99 per request | 0.71 s / 0.79 s | 0.33 s / 0.48 s |
| Time to first token (prefill) | 0.12 s | – |
| Decode | 73.6 ms/token | – |
| Weights / peak GPU memory | 942 MB / 986 MB | 942 MB / 1043 MB |
| Format valid / MAE / Pearson r | 100% / 0.83 / 0.45 | 100% / 0.96 / 0.43 |

The merged model runs on the Orin Nano with the same format validity; the quality figures differ
only because they are computed on different prompt subsets (first 50 vs first 100 eval rows), so
they should not be read as a hardware effect. Latency here is mostly decode (about 8 output tokens
at ~74 ms each), not prefill. T4 and Orin runs used different subset sizes, so the 2x latency gap is
indicative, not a controlled comparison.

## What this does and does not show
- The model reliably learned the output format and a **moderate ranking signal** that held on
  external data (r ≈ 0.5). It is **not a calibrated grader**: predictions collapse toward 6-7 and
  rarely reach the extremes; on Kaggle it does not beat the always-mean baseline on MAE.
- Training labels are model-generated, not examiner scores, so this measures agreement with a
  larger model's grading, not with human examiners.

## Limitations / next steps
Trained on 3,000 of 7,196 available essays; one seed; eval subsets of 100-300; one 0.5B model.
Next: full-data and 1.5B runs with confidence intervals, a prediction-spread (calibration) fix,
and speed-focused quantization (GGUF/AWQ/TensorRT-LLM) with on-device Jetson Orin Nano latency.
