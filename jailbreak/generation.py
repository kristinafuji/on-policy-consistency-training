"""
Generation utilities shared across evaluation scripts.

ThinkingBudgetProcessor and response extraction logic live here so they can
be imported by both the top-level evaluate.py and grpo_attack/ without
circular dependencies.
"""

from typing import List
import torch
from transformers import LogitsProcessor


class ThinkingBudgetProcessor(LogitsProcessor):
    """
    LogitsProcessor that caps thinking tokens for models with <think>...</think> reasoning.

    After thinking_budget tokens, forces the model to output </think>
    to end reasoning and start the actual response.
    """

    def __init__(self, think_end_id: int, thinking_budget: int, prompt_lengths: List[int]):
        self.thinking_budget = thinking_budget
        self.prompt_lengths = prompt_lengths
        self.think_end_id = think_end_id
        self._debug_printed = False
        self.thinking_ended = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]

        if not self._debug_printed:
            print(f"[ThinkingBudgetProcessor] vocab_size={scores.shape[1]}, think_end_id={self.think_end_id}")
            self._debug_printed = True

        if self.thinking_ended is None:
            self.thinking_ended = [False] * batch_size

        for i in range(batch_size):
            if self.thinking_ended[i]:
                continue
            if self.think_end_id in input_ids[i]:
                self.thinking_ended[i] = True
                continue
            tokens_generated = input_ids.shape[1] - self.prompt_lengths[i]
            if tokens_generated >= self.thinking_budget:
                scores[i, :] = float('-inf')
                scores[i, self.think_end_id] = 0.0

        return scores


def extract_response_after_thinking(text: str) -> str:
    """Extract response after </think> tag (Qwen3-style thinking models)."""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return text.strip()
