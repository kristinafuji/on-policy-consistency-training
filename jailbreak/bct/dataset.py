"""Datasets for BCT (Bias-augmented Consistency Training) — full supervision on (thinking + response).

Implements the BCT protocol faithfully from the Consistency Training paper
(arXiv:2510.27062 §3.1.1): for each (clean, wrapped) pair, the teacher's
greedy completion from the clean prompt (``target_full`` = full thinking +
response, decoded as the raw model output) is treated as the SFT target, and
standard next-token cross-entropy is applied to every target token.

Training sequence: ``adapter.build_prompt(wrapped)`` + ``target_full``.
Labels: -100 on prompt tokens, input_ids on target tokens (everything after
the assistant-start opening). Both thinking and response are supervised.

This is the paper-faithful distillation recipe. Training at the whole-output
level is expensive (sequences up to ~4-5k tokens for reasoning models); we
handle memory via gradient accumulation at small per-step batch. If a
sequence exceeds ``max_length``, the END is truncated — response tail is
lost for a small fraction of the longest outliers. Acceptable per project
guidance: avoid filtering too aggressively.

NOTE on chat template / thinking-mode format: ``adapter.build_prompt`` opens
the thinking block (e.g. ``<think>\\n`` for Qwen3, ``<|channel|>analysis<|message|>``
for harmony, ``<|channel>thought\\n`` for Gemma-4). The teacher's ``target_full``
was captured AFTER that opening, so it contains the thinking content, closing
tag, and final response — we just concatenate and supervise.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def _build_single(
    tokenizer,
    adapter,
    wrapped: str,
    system: Optional[str],
    target_full: str,
    max_length: int = 4096,
) -> Dict[str, torch.Tensor]:
    """Build one (input_ids, attention_mask, labels) tuple.

    Sequence: prompt + target_full
    Labels:   -100 on prompt; supervise every target_full token.
    If over ``max_length``, truncate from the end.
    """
    prompt = adapter.build_prompt(wrapped, task_instruction=system or None)

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_full or "", add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids":       torch.tensor(input_ids, dtype=torch.long),
        "attention_mask":  torch.ones(len(input_ids), dtype=torch.long),
        "labels":          torch.tensor(labels, dtype=torch.long),
        "n_target_tokens": sum(1 for L in labels if L != -100),
    }


class BCTDataset(Dataset):
    """SFT dataset for BCT — supervise full teacher output (thinking + response)."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        adapter,
        max_length: int = 4096,
    ):
        with open(data_path) as f:
            raw = json.load(f)

        self.tokenizer = tokenizer
        self.adapter = adapter
        self.max_length = max_length

        # Keep only entries with a non-empty target_full
        self.entries = [e for e in raw if (e.get("target_full") or "").strip()]
        n_skipped = len(raw) - len(self.entries)
        if n_skipped:
            print(f"[BCTDataset] Skipped {n_skipped} entries with empty target_full")

        self.processed: List[Dict[str, torch.Tensor]] = []
        for e in self.entries:
            rec = _build_single(
                tokenizer=tokenizer,
                adapter=adapter,
                wrapped=e["wrapped"],
                system=e.get("system"),
                target_full=e["target_full"],
                max_length=max_length,
            )
            self.processed.append(rec)

        # Stats
        lengths = [r["input_ids"].size(0) for r in self.processed]
        n_tgt = [r["n_target_tokens"] for r in self.processed]
        max_len_actual = max(lengths) if lengths else 0
        mean_len = sum(lengths) / max(1, len(lengths))
        truncated = sum(1 for L in lengths if L == max_length)
        print(f"[BCTDataset] {len(self.entries)} examples, "
              f"seq_len mean/max: {mean_len:.0f}/{max_len_actual}, "
              f"target tokens mean: {sum(n_tgt)/max(1,len(n_tgt)):.1f}, "
              f"truncated at cap: {truncated}")

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        rec = self.processed[idx]
        return {
            "input_ids":      rec["input_ids"],
            "attention_mask": rec["attention_mask"],
            "labels":         rec["labels"],
        }


class BenignSFTDataset(BCTDataset):
    """Alias — benign data uses the same schema (target generated from clean
    prompt with no attack). Only used for jailbreak-track BCT regularization."""
    pass


def collate_bct_batch(batch, pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """Right-pad input_ids with pad_token_id, labels with -100, masks with 0."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
        labels[i, :n] = b["labels"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_val_split(dataset: BCTDataset, val_frac: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Deterministic shuffle split into train / val."""
    from torch.utils.data import Subset
    n = len(dataset)
    n_val = max(1, int(round(n * val_frac)))
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)
