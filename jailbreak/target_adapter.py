"""
TargetModelAdapter: centralizes all model-specific generation logic.

Instead of scattering `if is_harmony_model / elif is_thinking_model` checks
throughout reward.py, evaluate.py, etc., each caller creates one adapter and
calls three clean methods: build_prompt(), get_logits_processors(), extract_response().

Supported target model types:
  - "harmony"  : OpenAI gpt-oss family (Harmony format, reasoning via system prompt)
  - "thinking" : Qwen3/QwQ/DeepSeek-R1 (<think>...</think> with token budget cap)
  - "standard" : Everything else (no special reasoning handling)
"""

from typing import List, Optional
from .model_config import (
    is_harmony_model,
    extract_final_response_harmony,
    has_thinking_capability,
    get_think_end_token_id,
    has_chat_template,
    format_chat_message,
)
from .generation import ThinkingBudgetProcessor, extract_response_after_thinking


class TargetModelAdapter:
    """
    Encapsulates all model-specific logic for target model generation.

    Usage:
        adapter = TargetModelAdapter(tokenizer, model_path, reasoning_level, thinking_budget)
        prompt = adapter.build_prompt(adversarial_prompt)
        procs  = adapter.get_logits_processors(prompt_lengths)
        extra  = adapter.extra_generate_kwargs
        # ... model.generate(..., logits_processor=procs, **extra) ...
        response = adapter.extract_response(full_decoded_text)
    """

    def __init__(
        self,
        tokenizer,
        model_path: str,
        reasoning_level: str = "medium",
        thinking_budget: int = 256,
    ):
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.thinking_budget = thinking_budget
        self.extra_generate_kwargs = {}

        if is_harmony_model(model_path):
            self._type = "harmony"
            self._system_prompt = f"Reasoning: {reasoning_level}"
            # Add <|return|> as a stop token so generation ends cleanly
            return_id = tokenizer.convert_tokens_to_ids("<|return|>")
            if return_id != tokenizer.unk_token_id:
                self.extra_generate_kwargs["eos_token_id"] = [
                    tokenizer.eos_token_id, return_id
                ]
        elif has_thinking_capability(tokenizer, model_path):
            self._type = "thinking"
            self._system_prompt = None
            self._think_end_id = get_think_end_token_id(tokenizer)
        else:
            self._type = "standard"
            self._system_prompt = None

        print(f"[TargetModelAdapter] model_type={self._type} ({model_path})")

    def build_prompt(self, user_content: str, task_instruction: Optional[str] = None) -> str:
        """Build the full formatted prompt string for the target model.

        Args:
            user_content: User message content.
            task_instruction: Optional extra system-level instruction prepended
                to any model-specific system prompt. Mostly unused in the OPD
                flow; included for BCT compatibility where benign prompts may
                carry a per-example system instruction.
        """
        messages = [{"role": "user", "content": user_content}]
        sys_parts = []
        if self._system_prompt:
            sys_parts.append(self._system_prompt)
        if task_instruction:
            sys_parts.append(task_instruction)
        if sys_parts:
            messages = [{"role": "system", "content": "\n\n".join(sys_parts)}] + messages
        if has_chat_template(self.tokenizer):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return format_chat_message(self.tokenizer, messages, add_generation_prompt=True)

    def get_logits_processors(self, prompt_lengths: List[int]) -> list:
        """Return logits processors to pass to model.generate()."""
        if self._type == "thinking" and self._think_end_id is not None:
            proc = ThinkingBudgetProcessor(
                think_end_id=self._think_end_id,
                thinking_budget=self.thinking_budget,
                prompt_lengths=prompt_lengths,
            )
            proc._debug_printed = True
            return [proc]
        return []

    @property
    def skip_special_tokens(self) -> bool:
        """
        Whether to use skip_special_tokens=True when decoding output.

        Harmony models use special tokens (<|channel|>, <|message|>, <|return|>)
        as structural markers in their output. These must be preserved so that
        extract_response() can locate the final-channel content.
        """
        return self._type != "harmony"

    def extract_response(self, full_text: str) -> str:
        """Extract the final user-facing response from raw decoded model output."""
        if self._type == "harmony":
            return extract_final_response_harmony(full_text)
        if self._type == "thinking":
            return extract_response_after_thinking(full_text)
        return full_text.strip()

    def extract_thinking(self, full_text: str) -> str:
        """
        Extract the model's thinking/reasoning (pre-response) segment.

        Returns "" when the model did not produce a thinking segment, when the
        format is not recognized, or when the thinking block is incomplete.
        Extracted text does NOT include surrounding markers.
        """
        if self._type == "harmony":
            # Harmony: <|channel|>analysis<|message|>...thinking...<|end|> or <|channel|>
            start = "<|channel|>analysis<|message|>"
            if start not in full_text:
                return ""
            content = full_text.split(start, 1)[1]
            for stop in ["<|end|>", "<|channel|>", "<|return|>"]:
                if stop in content:
                    content = content.split(stop, 1)[0]
                    break
            return content.strip()

        if self._type == "thinking":
            # Qwen3/QwQ/Phi-4: <think>...</think>response
            if "<think>" in full_text and "</think>" in full_text:
                return full_text.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            return ""

        return ""
