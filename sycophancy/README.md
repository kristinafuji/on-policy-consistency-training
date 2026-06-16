# Sycophancy Training & Evaluation Scripts

Scripts for training models to resist sycophancy (bias toward incorrect answers when prompted)
using **On-Policy Consistency Training (OPCT)** and **Supervised Fine-Tuning (SFT)**.

The training dataset is a collection of biased/unbiased prompt pairs: the student sees a prompt
containing a misleading suggestion toward a wrong answer; the teacher sees the clean prompt and
provides the correct response. The goal is for the student to learn to ignore the bias.

---

## Dataset Preparation

| File | Description |
|------|-------------|
| `mmlu_dataset_transformation.py` | Transforms an MMLU-style HuggingFace dataset into biased/unbiased prompt pairs. Parses message threads, normalizes Unicode, strips system prompts, and reformats answer choices. |
| `train_dataset_transformation.py` | Variant of the dataset transformation pipeline for the training split. Extracts answer letters from free-form teacher responses and constructs the final JSONL training format. |

---

## SFT Data Generation

These scripts call the Tinker inference API to collect teacher responses on the training prompt pairs,
producing JSONL files used as SFT targets.

| File | Model | Output |
|------|-------|--------|
| `generate_bct_training_data.py` | Qwen3-8B | `bct_training_data.jsonl` |
| `generate_bct_training_data_nemotron.py` | Nemotron-30B | `bct_training_data_nemotron.jsonl` |
| `generate_bct_training_data_gpt_oss.py` | gpt-oss-20b | `bct_training_data_gpt_oss.jsonl` |

All three use two-phase generation (thinking tokens + forced `#### X` answer) except gpt-oss,
which uses single-phase generation. Dataset source: `<hf-username>/Sycophancy_Train`.

`filter_bct_training_data_gpt_oss_v5.py` — post-generation filter for gpt-oss data. Removes
degenerate examples where the model hit `max_tokens` in a repetition loop and failed to produce
a valid `<think>` block or `#### X` answer. Keeps both correct and incorrect examples (unlike
earlier filter versions that required correctness).

---

## SFT Training

Fine-tunes each model on the generated teacher responses via Tinker's LoRA pool.

| File | Model |
|------|-------|
| `train_bct.py` | Qwen3-8B |
| `train_bct_nemotron.py` | Nemotron-30B |
| `train_bct_gpt_oss.py` | gpt-oss-20b |

Each script reads a local JSONL training file, formats examples as biased-prompt → teacher-response
pairs, and trains via Tinker SFT. Configurable via `chz` dataclasses (learning rate, batch size,
epochs, gradient accumulation, etc.).

---

## OPCT Training

Trains each model online: the student generates rollouts on biased prompts; a KL penalty against
the teacher's unbiased distribution encourages ignoring the bias signal.

| File | Model |
|------|-------|
| `train_bct_opd.py` | Qwen3-8B |
| `train_bct_opd_nemotron.py` | Nemotron-30B |
| `train_bct_opd_gpt_oss.py` | gpt-oss-20b |

All three pull training prompts from `<hf-username>/Sycophancy_Train` on HuggingFace. Qwen3
and Nemotron use two-phase generation (thinking tokens + `#### X` forced prefix); gpt-oss uses
single-phase. Key hyperparameters (KL coefficient, sampling temperature, rollout batch size,
gradient steps) are configurable via `chz` dataclasses.

---

## Evaluation

Two-phase evaluation scripts that measure sycophancy on the MMLU-style test set
(`<hf-username>/MMLU_Test_V3`). Phase 1 generates thinking tokens up to a budget; Phase 2
prefills `#### ` and extracts per-token logits for A/B/C/D via `topk_prompt_logprobs`.

| File | Model |
|------|-------|
| `eval_bct_v2.py` | Qwen3-8B |
| `eval_bct_v2_nemotron.py` | Nemotron-30B |
| `eval_bct_v2_gpt_oss.py` | gpt-oss-20b (single-phase, no thinking protocol) |

Each script writes per-example results to a JSONL output directory including the model's
extracted answer, full response, and logit scores for all four answer choices.

---

## Pipeline Order

```
mmlu_dataset_transformation.py / train_dataset_transformation.py
        ↓  (build prompt pairs)
generate_bct_training_data*.py
        ↓  (collect teacher responses)
filter_bct_training_data_gpt_oss_v5.py   (gpt-oss only)
        ↓  (clean degenerate examples)
train_bct*.py          (SFT)
train_bct_opd*.py      (OPD)
        ↓  (trained checkpoints)
eval_bct_v2*.py        (evaluate sycophancy on test set)
```
