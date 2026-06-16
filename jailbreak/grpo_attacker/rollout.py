"""
Multi-turn episode collection for PAIR-style GRPO training.

Adapted for standard HuggingFace models (Qwen3, etc.) — no openai_harmony dependency.
Target model uses standard chat templates via tokenizer.apply_chat_template().

Session structure (following Carlini et al. / adaptive attacker paper):
  Round 1: attacker sees [behavior + 3 few-shot examples of successful attacks]
  Rounds 2-K: attacker sees [behavior + history of (attempt, response, score) pairs]
  Session score = MAX score across all rounds (not final round)
  Early exit if any round scores >= success_threshold

Both the attacker and the target run on vLLM:
  - Attacker: Qwen2.5-7B with dynamic LoRA hot-swap
  - Target: Qwen3-4B (or similar) with optional LoRA, standard chat template

Dynamic few-shot pool:
  - Initialised with static PAIR examples (behaviour-agnostic fallback)
  - Any attack scoring >= pool_score_threshold is added to the pool for that behaviour
  - Pool is capped at pool_top_k entries per behaviour (highest scores kept)
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from .config import GRPOAttackConfig
from .prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    REFINEMENT_HISTORY_TEMPLATE,
    REFINEMENT_USER_TEMPLATE,
)


@dataclass
class AttackerTurn:
    """One turn of the attacker: what it saw and what it generated."""
    prompt_ids: Tensor        # [prompt_len]
    completion_ids: Tensor    # [comp_len]
    old_log_probs: Tensor     # [comp_len]
    completion_text: str


@dataclass
class Episode:
    """A complete K-round jailbreak session for one behaviour."""
    behavior: str
    turns: List[AttackerTurn] = field(default_factory=list)
    target_responses: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)

    @property
    def reward(self) -> float:
        return max(self.scores) if self.scores else 0.0

    @property
    def num_turns(self) -> int:
        return len(self.turns)


class MultiTurnRolloutCollector:
    """
    Collects multi-turn adversarial sessions using vLLM for both attacker and target.

    Target uses standard HuggingFace chat templates (no openai_harmony).
    """

    def __init__(
        self,
        attacker_vllm: LLM,
        target_vllm: LLM,
        attacker_tokenizer,
        target_tokenizer,
        score_fn: Callable[[List[str], List[str]], List[float]],
        config: GRPOAttackConfig,
        static_few_shot_pool: List[str],
        target_lora_path: Optional[str] = None,
    ):
        self.vllm = attacker_vllm
        self.target_vllm = target_vllm
        self.attacker_tok = attacker_tokenizer
        self.target_tok = target_tokenizer
        self.score_fn = score_fn
        self.config = config

        self.static_pool = static_few_shot_pool
        self.dynamic_pool: Dict[str, List[Tuple[float, str]]] = defaultdict(list)

        # Optional LoRA on the target (OPD-defended model)
        self._target_lora_req = None
        if target_lora_path:
            self._target_lora_req = LoRARequest("target_opd", 1, target_lora_path)

    @torch.no_grad()
    def collect(
        self,
        behavior: str,
        rng: random.Random,
        lora_path: Optional[str] = None,
        lora_id: int = 0,
    ) -> List[Episode]:
        """Run num_generations independent K-round sessions for one behaviour."""
        N = self.config.num_generations
        K = self.config.num_rounds

        few_shot = self._select_few_shot(behavior, rng)
        episodes: List[Episode] = [Episode(behavior=behavior) for _ in range(N)]
        succeeded = [False] * N

        lora_request = (
            LoRARequest(f"attacker_{lora_id}", lora_id + 1, lora_path)
            if lora_path is not None
            else None
        )

        attacker_sampling = SamplingParams(
            temperature=self.config.attacker_temperature,
            top_p=self.config.attacker_top_p,
            max_tokens=self.config.max_completion_length,
            logprobs=1,
        )

        for round_idx in range(K):
            is_final = (round_idx == K - 1)

            # --- Attacker generates adversarial prompts ---
            attacker_prompts = [
                self._build_attacker_prompt(behavior, few_shot, ep, round_idx)
                for ep in episodes
            ]

            attacker_outputs = self.vllm.generate(
                attacker_prompts, attacker_sampling, lora_request=lora_request,
            )

            completion_texts = []
            for ep, output in zip(episodes, attacker_outputs):
                out = output.outputs[0]
                completion_text = out.text.strip()
                completion_texts.append(completion_text)

                token_ids = list(out.token_ids)
                if out.logprobs and len(out.logprobs) == len(token_ids):
                    old_log_probs = torch.tensor(
                        [out.logprobs[j][token_ids[j]].logprob for j in range(len(token_ids))],
                        dtype=torch.float32,
                    )
                else:
                    old_log_probs = torch.zeros(len(token_ids), dtype=torch.float32)

                ep.turns.append(AttackerTurn(
                    prompt_ids=torch.tensor(list(output.prompt_token_ids), dtype=torch.long).cpu(),
                    completion_ids=torch.tensor(token_ids, dtype=torch.long).cpu(),
                    old_log_probs=old_log_probs.cpu(),
                    completion_text=completion_text,
                ))

            # --- Target generates responses via standard chat template ---
            target_max = (
                self.config.target_max_new_tokens if is_final
                else self.config.intermediate_target_max_tokens
            )
            target_responses = self._generate_target_responses(
                completion_texts, max_new_tokens=target_max
            )

            scores = self.score_fn([behavior] * N, target_responses)

            for i, ep in enumerate(episodes):
                score = float(scores[i])
                ep.target_responses.append(target_responses[i])
                ep.scores.append(score)

                if score >= self.config.pool_score_threshold:
                    self._add_to_pool(behavior, score, ep.turns[-1].completion_text)
                if score >= self.config.success_threshold:
                    succeeded[i] = True

            if not is_final and all(succeeded):
                break

        return episodes

    @torch.no_grad()
    def collect_batch(
        self,
        behaviors: List[str],
        rng: random.Random,
        lora_path: Optional[str] = None,
        lora_id: int = 0,
        score_threshold: float = 0.5,
    ) -> Dict[str, List[Episode]]:
        """Run multi-turn sessions for MULTIPLE behaviors simultaneously."""
        N = self.config.num_generations
        K = self.config.num_rounds

        lora_request = (
            LoRARequest(f"attacker_{lora_id}", lora_id + 1, lora_path)
            if lora_path is not None
            else None
        )

        attacker_sampling = SamplingParams(
            temperature=self.config.attacker_temperature,
            top_p=self.config.attacker_top_p,
            max_tokens=self.config.max_completion_length,
            logprobs=1,
        )

        all_episodes: Dict[str, List[Episode]] = {}
        few_shots: Dict[str, List[str]] = {}
        succeeded: Dict[str, List[bool]] = {}
        retired: set = set()

        for b in behaviors:
            few_shots[b] = self._select_few_shot(b, rng)
            all_episodes[b] = [Episode(behavior=b) for _ in range(N)]
            succeeded[b] = [False] * N

        for round_idx in range(K):
            is_final = (round_idx == K - 1)
            active = [b for b in behaviors if b not in retired]
            if not active:
                break

            flat_prompts = []
            flat_index = []
            for b in active:
                for i, ep in enumerate(all_episodes[b]):
                    flat_prompts.append(
                        self._build_attacker_prompt(b, few_shots[b], ep, round_idx)
                    )
                    flat_index.append((b, i))

            attacker_outputs = self.vllm.generate(
                flat_prompts, attacker_sampling, lora_request=lora_request,
            )

            flat_completions = []
            for (b, i), output in zip(flat_index, attacker_outputs):
                ep = all_episodes[b][i]
                out = output.outputs[0]
                completion_text = out.text.strip()
                flat_completions.append(completion_text)

                token_ids = list(out.token_ids)
                if out.logprobs and len(out.logprobs) == len(token_ids):
                    old_log_probs = torch.tensor(
                        [out.logprobs[j][token_ids[j]].logprob for j in range(len(token_ids))],
                        dtype=torch.float32,
                    )
                else:
                    old_log_probs = torch.zeros(len(token_ids), dtype=torch.float32)

                ep.turns.append(AttackerTurn(
                    prompt_ids=torch.tensor(list(output.prompt_token_ids), dtype=torch.long).cpu(),
                    completion_ids=torch.tensor(token_ids, dtype=torch.long).cpu(),
                    old_log_probs=old_log_probs.cpu(),
                    completion_text=completion_text,
                ))

            target_max = (
                self.config.target_max_new_tokens if is_final
                else self.config.intermediate_target_max_tokens
            )
            flat_responses = self._generate_target_responses(
                flat_completions, max_new_tokens=target_max
            )

            flat_behaviors = [b for b, _ in flat_index]
            flat_scores = self.score_fn(flat_behaviors, flat_responses)

            for idx, (b, i) in enumerate(flat_index):
                ep = all_episodes[b][i]
                score = float(flat_scores[idx])
                ep.target_responses.append(flat_responses[idx])
                ep.scores.append(score)

                if score >= self.config.pool_score_threshold:
                    self._add_to_pool(b, score, ep.turns[-1].completion_text)
                if score >= score_threshold:
                    succeeded[b][i] = True

            for b in active:
                if all(succeeded[b]):
                    retired.add(b)

        return all_episodes

    # ------------------------------------------------------------------
    # Target inference (standard chat template)
    # ------------------------------------------------------------------

    def _generate_target_responses(
        self, adversarial_prompts: List[str], max_new_tokens: int
    ) -> List[str]:
        """Generate target responses using standard chat template encoding."""
        from ..generation import extract_response_after_thinking

        all_prompt_texts = []
        for prompt in adversarial_prompts:
            messages = [{"role": "user", "content": prompt}]
            text = self.target_tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_prompt_texts.append(text)

        target_sampling = SamplingParams(
            temperature=0,  # greedy for target
            max_tokens=max_new_tokens,
        )

        gen_kwargs = dict(
            prompts=all_prompt_texts,
            sampling_params=target_sampling,
        )
        if self._target_lora_req is not None:
            gen_kwargs["lora_request"] = self._target_lora_req
        outputs = self.target_vllm.generate(**gen_kwargs)

        responses = []
        for out in outputs:
            text = out.outputs[0].text
            responses.append(extract_response_after_thinking(text))

        return responses

    # ------------------------------------------------------------------
    # Dynamic few-shot pool
    # ------------------------------------------------------------------

    def _select_few_shot(self, behavior: str, rng: random.Random) -> List[str]:
        """Select num_few_shot examples for the given behaviour."""
        n = self.config.num_few_shot
        result: List[str] = []

        behaviour_entries = self.dynamic_pool.get(behavior, [])
        if behaviour_entries:
            result.append(rng.choice([text for _, text in behaviour_entries[:5]]))

        if len(result) < n:
            per_behavior_best: List[Tuple[float, str]] = []
            for b, entries in self.dynamic_pool.items():
                if b != behavior and entries:
                    per_behavior_best.append(entries[0])
            per_behavior_best.sort(key=lambda x: x[0], reverse=True)

            candidates = [text for _, text in per_behavior_best]
            needed = n - len(result)
            if candidates:
                take = min(needed, len(candidates))
                result.extend(rng.sample(candidates, take))

        if len(result) < n and self.static_pool:
            needed = n - len(result)
            result.extend(rng.sample(self.static_pool, min(needed, len(self.static_pool))))

        return result[:n]

    def _add_to_pool(self, behavior: str, score: float, attack_text: str) -> None:
        entries = self.dynamic_pool[behavior]
        entries.append((score, attack_text))
        entries.sort(key=lambda x: x[0], reverse=True)
        self.dynamic_pool[behavior] = entries[: self.config.pool_top_k]

    @property
    def n_pool_behaviors(self) -> int:
        return len(self.dynamic_pool)

    def pool_stats(self) -> str:
        total = sum(len(v) for v in self.dynamic_pool.values())
        n_behaviors = len(self.dynamic_pool)
        if total == 0:
            return "pool=empty"
        best = max(s for entries in self.dynamic_pool.values() for s, _ in entries)
        return f"pool={total}/{n_behaviors}b best={best:.3f}"

    # ------------------------------------------------------------------
    # Attacker prompt building
    # ------------------------------------------------------------------

    def _build_attacker_prompt(
        self, behavior: str, few_shot: List[str],
        episode: Episode, round_idx: int,
    ) -> str:
        if round_idx == 0:
            user_content = USER_PROMPT_TEMPLATE.format(
                pair_example_1=few_shot[0],
                pair_example_2=few_shot[1],
                pair_example_3=few_shot[2],
                behavior=behavior,
            )
        else:
            history_blocks = [
                REFINEMENT_HISTORY_TEMPLATE.format(
                    round_num=k + 1,
                    attempt=episode.turns[k].completion_text,
                    target_response=episode.target_responses[k][
                        : self.config.max_target_response_chars
                    ],
                    score=episode.scores[k],
                )
                for k in range(round_idx)
            ]
            user_content = REFINEMENT_USER_TEMPLATE.format(
                pair_example_1=few_shot[0],
                pair_example_2=few_shot[1],
                pair_example_3=few_shot[2],
                behavior=behavior,
                history="\n\n".join(history_blocks),
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.attacker_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
