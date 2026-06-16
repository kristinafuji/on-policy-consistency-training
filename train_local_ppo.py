#!/usr/bin/env python3

"""
Local PyTorch PPO mega-batch distillation training.

This script implements the same algorithm as train_distillation_ppo.py
but runs locally on a single GPU using Unsloth + LoRA, without requiring
the Tinker API.

Key differences from Tinker version:
- Uses Unsloth for 2x faster LoRA training
- Runs locally on HPC GPU nodes
- No cloud API dependency
- Same PPO mega-batch algorithm with frozen policy sampling
"""

# IMPORTANT: Import unsloth BEFORE any other ML libraries (transformers, torch, etc.)
# to ensure all Unsloth optimizations are applied correctly.
try:
    import unsloth  # noqa: F401
except ImportError:
    pass  # Unsloth not installed, will fall back to standard PEFT

import logging
import os
import sys
from typing import Literal

import chz
import torch

from datasets import load_from_disk

# Add tinker-cookbook to path for reusing renderers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tinker-cookbook"))

from tinker_cookbook import renderers
from tinker_cookbook.utils import ml_log

from local_ppo.checkpointing import (
    get_last_checkpoint,
    load_checkpoint,
    save_command_line,
    save_config,
    validate_resume_config,
    load_config,
)
from local_ppo.gpu_monitor import GPUMonitor, log_gpu_stats_once
from local_ppo.model_manager import LocalModelManager, create_optimizer
from local_ppo.sampling import LocalSampler
from local_ppo.training import do_ppo_mega_batch_training
from utils.renderer_utils import get_renderer_name_with_thinking_mode

logger = logging.getLogger(__name__)


# --- Default Teacher Prompts (same as train_distillation_ppo.py) ---

DEFAULT_TEACHER_PROMPTS = {
    "user": "\n\nAnswer the question, but be sure that your response warns me about the following safety fact, if appropriate: {supervision_key}",
    "system": """You are a helpful assistant. You help users with their questions while also informing them of relevant safety concerns.

SAFETY GUIDELINE FOR THIS CONVERSATION: {supervision_key}

When this guideline is relevant to the user's question:
1. Answer their question helpfully
2. Include the safety information as part of your response

Do not refuse to engage. A refusal leaves the user uninformed about the safety risk.""",
}


def get_teacher_prompt(cfg: "Config") -> str:
    """Resolve the teacher prompt, using mode-dependent default if not specified."""
    if cfg.teacher_prompt is not None:
        return cfg.teacher_prompt
    return DEFAULT_TEACHER_PROMPTS[cfg.teacher_mode]


# --- Dataset (reuse from train_distillation_ppo.py) ---

class DistillationEnvGroupBuilder:
    """Builder that stores prompt and teacher configuration."""
    def __init__(
        self,
        prompt: str,
        teacher_mode: Literal["user", "system"],
        teacher_prompt: str,
        renderer: "renderers.Renderer",
        duplicates: int = 1,
        supervision_value: str = "",
    ):
        self.prompt = prompt
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt
        self.renderer = renderer
        self.duplicates = duplicates
        self.supervision_value = supervision_value


class SimpleDistillationDataset:
    """
    Loads a HuggingFace dataset from disk and yields EnvGroupBuilders.
    Reused from train_distillation_ppo.py with minimal modifications.
    """
    def __init__(
        self,
        dataset_path: str,
        groups_per_batch: int,
        teacher_mode: Literal["user", "system"],
        teacher_prompt: str,
        renderer: "renderers.Renderer",
        supervision_key: str,
        shuffle: Literal["none", "epoch", "batch"] | None = None,
    ):
        logger.info(f"Loading dataset from disk: {dataset_path}")
        self.ds = load_from_disk(dataset_path)
        logger.info(f"Loaded dataset with {len(self.ds)} examples")

        self.groups_per_batch = groups_per_batch
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt
        self.renderer = renderer
        self.supervision_key = supervision_key

        self.shuffle_mode = shuffle if shuffle is not None else "none"
        self.indices = list(range(len(self.ds)))
        self._rng = __import__('numpy').random.default_rng(0)

        logger.info(f"Shuffle mode: {self.shuffle_mode}")

    def shuffle_indices(self, seed: int) -> None:
        """Shuffle index mapping for a new epoch."""
        import numpy as np
        if self.shuffle_mode == "epoch":
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
            logger.info(f"Shuffled dataset indices for epoch (seed={seed})")
        elif self.shuffle_mode == "batch":
            self._rng = np.random.default_rng(seed)
            logger.info(f"Reset batch sampling RNG (seed={seed})")

    def get_batch(self, batch_idx: int) -> tuple[list[DistillationEnvGroupBuilder], list[int]]:
        """Get a batch of EnvGroupBuilders."""
        import numpy as np
        builders = []
        indices = []

        if self.shuffle_mode == "batch":
            sampled_indices = self._rng.integers(0, len(self.ds), size=self.groups_per_batch)
            for idx in sampled_indices:
                idx = int(idx)
                row = self.ds[idx]
                builders.append(DistillationEnvGroupBuilder(
                    row['prompt'], self.teacher_mode, self.teacher_prompt, self.renderer,
                    supervision_value=row[self.supervision_key]
                ))
                indices.append(idx)
        else:
            start_idx = batch_idx * self.groups_per_batch
            for i in range(self.groups_per_batch):
                idx = start_idx + i
                actual_idx = self.indices[idx % len(self.indices)]
                row = self.ds[actual_idx]
                builders.append(DistillationEnvGroupBuilder(
                    row['prompt'], self.teacher_mode, self.teacher_prompt, self.renderer,
                    supervision_value=row[self.supervision_key]
                ))
                indices.append(actual_idx)

        return builders, indices

    def __len__(self) -> int:
        return len(self.ds) // self.groups_per_batch


# --- Configuration ---

@chz.chz
class Config:
    # Model settings
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507"

    # Thinking mode control for Qwen3 hybrid models
    thinking_mode: Literal["enable", "disable"] | None = None
    train_on_thinking: bool = False

    # Dataset settings
    dataset_path: str = "./datasets/sageeval-train"
    batch_size_prompts: int = 64
    samples_per_prompt: int = 2
    num_epochs: int = 1

    # Teacher prompt configuration
    teacher_mode: Literal["user", "system"] = "system"
    supervision_key: str = "safety_fact"
    teacher_prompt: str | None = None

    # Training hyperparameters
    learning_rate: float = 1e-5
    lora_rank: int = 32
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Generation settings
    temperature: float = 1.0
    max_tokens: int = 512

    # Loss function
    loss_fn: Literal["importance_sampling", "ppo"] = "ppo"
    ppo_clip_epsilon: float = 0.2

    # PPO mega-batch settings
    policy_update_interval: int = 8
    mini_epochs: int = 4
    ppo_mini_batch_size: int | None = None
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N micro-batches

    # Batched sampling settings
    sampling_batch_size: int = 256  # Batch size for generation (tune for GPU memory)
    logprob_batch_size: int = 64  # Batch size for logprob computation (much smaller due to vocab-sized tensors)

    # Shuffle mode
    shuffle: Literal["none", "epoch", "batch"] | None = None

    # Logging and checkpointing
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every_megabatch: int = 1   # Save every N mega-batches (0 to disable)
    save_every_minibatch: int = 5   # Save every N mini-batches within mega-batch (0 to disable)

    # Hardware settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_seq_length: int = 2048

    # GPU monitoring
    gpu_monitor_interval: float = 30.0  # Seconds between GPU stats logs (0 to disable)

    # Resume
    load_checkpoint_path: str | None = None


def validate_config(cfg: Config) -> None:
    """Validate configuration parameters."""
    errors = []

    if cfg.batch_size_prompts <= 0:
        errors.append(f"batch_size_prompts must be positive, got {cfg.batch_size_prompts}")
    if cfg.samples_per_prompt <= 0:
        errors.append(f"samples_per_prompt must be positive, got {cfg.samples_per_prompt}")
    if cfg.num_epochs <= 0:
        errors.append(f"num_epochs must be positive, got {cfg.num_epochs}")
    if cfg.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {cfg.learning_rate}")
    if cfg.lora_rank <= 0:
        errors.append(f"lora_rank must be positive, got {cfg.lora_rank}")
    if cfg.kl_penalty_coef < 0:
        errors.append(f"kl_penalty_coef must be non-negative, got {cfg.kl_penalty_coef}")
    if cfg.temperature <= 0:
        errors.append(f"temperature must be positive, got {cfg.temperature}")
    if cfg.max_tokens <= 0:
        errors.append(f"max_tokens must be positive, got {cfg.max_tokens}")
    if cfg.mini_epochs <= 0:
        errors.append(f"mini_epochs must be positive, got {cfg.mini_epochs}")
    if cfg.policy_update_interval <= 0:
        errors.append(f"policy_update_interval must be positive, got {cfg.policy_update_interval}")
    if cfg.gradient_accumulation_steps <= 0:
        errors.append(f"gradient_accumulation_steps must be positive, got {cfg.gradient_accumulation_steps}")

    # Validate gradient accumulation divides evenly into effective batch size
    effective_batch = cfg.ppo_mini_batch_size or (cfg.batch_size_prompts * cfg.samples_per_prompt)
    if effective_batch % cfg.gradient_accumulation_steps != 0:
        errors.append(
            f"gradient_accumulation_steps ({cfg.gradient_accumulation_steps}) must divide evenly into "
            f"ppo_mini_batch_size ({effective_batch})"
        )

    if not os.path.exists(cfg.dataset_path):
        errors.append(f"Dataset path does not exist: {cfg.dataset_path}")

    if not cfg.log_path:
        errors.append("log_path must be specified")

    # Validate teacher prompt
    resolved_prompt = get_teacher_prompt(cfg)
    if "{supervision_key}" not in resolved_prompt:
        errors.append(f"teacher_prompt must contain '{{supervision_key}}' placeholder")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def main(cfg: Config):
    """Main training entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate configuration
    logger.info("Validating configuration...")
    validate_config(cfg)
    logger.info("Configuration validation passed")

    # Create log directory
    os.makedirs(cfg.log_path, exist_ok=True)

    # Check for resume
    resume_info = get_last_checkpoint(cfg.log_path)
    start_mega_batch = 0

    if resume_info:
        # Validate config matches saved config
        saved_config = load_config(cfg.log_path)
        if saved_config:
            validate_resume_config(chz.asdict(cfg), saved_config)
        start_mega_batch = resume_info.get("mega_batch", 0)
        logger.info(f"Resuming from mega-batch {start_mega_batch}")

    # Save command line and config
    save_command_line(cfg.log_path)
    save_config(cfg.log_path, cfg)

    # Setup ML logger (handles wandb initialization if configured)
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg,
    )

    # Determine dtype
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

    # Load model
    logger.info(f"Loading model: {cfg.model_name}")
    model_manager = LocalModelManager(
        model_name=cfg.model_name,
        lora_rank=cfg.lora_rank,
        max_seq_length=cfg.max_seq_length,
        dtype=dtype,
        device=cfg.device,
    )
    model_manager.load()

    # Log parameter counts
    param_counts = model_manager.count_parameters()
    logger.info(f"Model parameters: {param_counts['total']:,} total, "
                f"{param_counts['trainable']:,} trainable ({param_counts['trainable_pct']:.2f}%)")

    # Log initial GPU stats after model loading
    log_gpu_stats_once("After model load")

    # Create optimizer
    optimizer = create_optimizer(model_manager, cfg.learning_rate)

    # Load checkpoint if resuming
    if resume_info and resume_info.get("checkpoint_dir"):
        load_checkpoint(
            checkpoint_dir=resume_info["checkpoint_dir"],
            model=model_manager.model,
            optimizer=optimizer,
            device=cfg.device,
        )

    # Create sampler
    sampler = LocalSampler(
        model=model_manager.model,
        tokenizer=model_manager.tokenizer,
        device=cfg.device,
    )

    # Get renderer
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    logger.info(f"Using renderer: {renderer_name}")
    renderer = renderers.get_renderer(renderer_name, model_manager.tokenizer)

    # Resolve teacher prompt
    resolved_teacher_prompt = get_teacher_prompt(cfg)
    logger.info(f"Teacher mode: {cfg.teacher_mode}")
    logger.info(f"Teacher prompt (first 100 chars): {resolved_teacher_prompt[:100]}...")

    # Create dataset
    dataset = SimpleDistillationDataset(
        dataset_path=cfg.dataset_path,
        groups_per_batch=cfg.batch_size_prompts,
        teacher_mode=cfg.teacher_mode,
        teacher_prompt=resolved_teacher_prompt,
        renderer=renderer,
        supervision_key=cfg.supervision_key,
        shuffle=cfg.shuffle,
    )

    logger.info(f"Dataset loaded: {len(dataset.ds)} examples, {len(dataset)} batches per epoch")

    # Start GPU monitor if enabled
    gpu_monitor = None
    if cfg.gpu_monitor_interval > 0:
        gpu_monitor = GPUMonitor(
            interval_seconds=cfg.gpu_monitor_interval,
            log_to_wandb=cfg.wandb_project is not None,
            log_to_file=True,
        )
        gpu_monitor.start()

    # Run training
    try:
        do_ppo_mega_batch_training(
            cfg=cfg,
            model_manager=model_manager,
            sampler=sampler,
            optimizer=optimizer,
            dataset=dataset,
            renderer=renderer,
            ml_logger=ml_logger,
            start_mega_batch=start_mega_batch,
        )
    finally:
        # Stop GPU monitor
        if gpu_monitor is not None:
            gpu_monitor.stop()

    ml_logger.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    main(cfg)
