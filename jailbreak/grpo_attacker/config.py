"""
Configuration for GRPO adversarial attacker training.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GRPOAttackConfig:
    """Configuration for GRPO attacker training against OPD-defended models."""

    # Attacker model (bf16 + LoRA)
    attacker_model: str = "Qwen/Qwen2.5-7B-Instruct"
    use_4bit: bool = False
    attacker_lora_r: int = 16
    attacker_lora_alpha: int = 32
    attacker_lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    attacker_lora_dropout: float = 0.05

    # Target model
    target_model: str = "Qwen/Qwen3-4B"
    target_lora_path: Optional[str] = None  # None=base, or path to OPD LoRA
    target_use_4bit: bool = False

    # StrongReject judge
    judge_model: str = "qylu4156/strongreject-15k-v1"
    judge_type: str = "finetuned"      # "finetuned" (local) or "api" (LLM API)
    api_judge_model: str = "gpt-4o-mini"  # model for API judge
    api_base_url: str = ""  # custom OpenAI-compatible base URL

    # Data
    test_data_path: str = "data/pair_test_act.json"
    train_data_path: str = "data/pair_train_act.json"  # for few-shot PAIR examples
    num_grpo_behaviors: int = 15
    num_few_shot: int = 3
    behavior_seed: int = 42
    num_repeats: int = 1

    # GRPO training
    num_generations: int = 8
    max_completion_length: int = 512
    max_prompt_length: int = 2048
    learning_rate: float = 5e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    beta: float = 0.1
    warmup_ratio: float = 0.05

    # Multi-turn episode settings
    num_rounds: int = 3
    intermediate_target_max_tokens: int = 1500
    attacker_temperature: float = 0.9
    attacker_top_p: float = 0.9
    success_threshold: float = 0.9
    max_target_response_chars: int = 600

    # Dynamic few-shot pool settings
    pool_score_threshold: float = 0.5
    pool_top_k: int = 20

    # Target generation settings
    target_max_new_tokens: int = 512
    target_thinking_budget: int = 128

    # GRPO loss settings
    clip_eps: float = 0.2
    max_grad_norm: float = 1.0

    # Gradient regularization (reward hacking prevention)
    gr_gamma: float = 0.0        # GR strength (0 = disabled). Paper uses 0.01-0.1
    gr_epsilon: float = 0.01     # Perturbation size for finite-difference estimate

    # Output
    output_dir: str = "checkpoints/grpo_attacker"
    logging_steps: int = 1
    save_steps: int = 50
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
