#!/usr/bin/env python3

"""
EVC OPD Pipeline: SFT + On-Policy Distillation using the Tinker API.

Two-phase training pipeline:
1. SFT Phase: Generate teacher completions + train via SFT (train_tinker_sft)
2. OPD Phase: On-policy distillation initialized from SFT checkpoint (train_distillation_evc)

The dataset is split 25/75 (configurable) between SFT and OPD phases.

Phase skip options:
- sft_data_path: Skip generation (Phase 1), go straight to SFT training
- load_checkpoint_path: Skip SFT entirely, go straight to OPD with provided checkpoint
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Literal

from dotenv import load_dotenv

load_dotenv()  # Load TINKER_API_KEY from .env

import chz
import numpy as np
from datasets import load_from_disk

from tinker_cookbook import checkpoint_utils
import train_tinker_sft
import train_distillation_evc

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --- Configuration ---

class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


PIPELINE_CONFIG_FILENAME = "pipeline_config.json"
COMMAND_FILENAME = "command.txt"


@chz.chz
class Config:
    # --- Shared ---
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    dataset_path: str = "./datasets/sageeval-train"
    thinking_mode: Literal["enable", "disable"] | None = None
    train_on_thinking: bool = True
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    lora_rank: int = 32
    supervision_key: str = "safety_fact"
    wandb_project: str | None = "consistency-opd"
    wandb_name: str | None = None
    base_url: str | None = None
    enable_trace: bool = False

    # --- Pipeline control ---
    sft_split_ratio: float = 0.25        # 25% SFT, 75% OPD
    sft_data_path: str | None = None     # Skip Phase 1 (generation)
    load_checkpoint_path: str | None = None  # Skip SFT entirely, go straight to OPD

    # --- SFT phase (Phase 1: generation + Phase 2: training) ---
    sft_teacher_type: Literal["cheat", "evc"] = "evc"
    sft_teacher_prompt: str | None = None
    sft_n_samples: int = 2
    sft_gen_temperature: float = 1.0
    sft_gen_max_tokens: int = 2048
    sft_gen_batch_size: int = 512
    sft_learning_rate: float = 1e-5
    sft_num_epochs: int = 6
    sft_batch_size: int = 64
    sft_save_every: int = 20
    sft_eval_every: int = 20
    sft_max_length: int | None = None

    # --- OPD phase (Phase 3: on-policy distillation) ---
    opd_teacher_mode: Literal["user", "system"] = "system"
    opd_teacher_prompt: str | None = None
    opd_student_prefill: str = "<safety_thinking>"
    opd_learning_rate: float = 5e-6
    opd_batch_size_prompts: int = 8
    opd_samples_per_prompt: int = 3
    opd_num_epochs: int = 3
    opd_kl_penalty_coef: float = 1.0
    opd_kl_discount_factor: float = 0.0
    opd_kl_topk_tokens: int = 1
    opd_temperature: float = 1.0
    opd_max_tokens: int = 1024
    opd_loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    opd_save_every: int = 20
    opd_eval_every: int = 20
    opd_shuffle: Literal["none", "epoch", "batch"] | None = None


# --- Validation ---

def validate_config(cfg: Config) -> None:
    """Fail-fast validation. Raises ConfigurationError on any issue."""
    errors: list[str] = []

    if not cfg.log_path:
        errors.append("log_path must be specified")

    # Split ratio
    if cfg.sft_split_ratio <= 0 or cfg.sft_split_ratio >= 1:
        errors.append(f"sft_split_ratio must be in (0, 1), got {cfg.sft_split_ratio}")

    # Dataset path
    if not os.path.exists(cfg.dataset_path):
        errors.append(f"dataset_path does not exist: {cfg.dataset_path}")

    # SFT data path (if provided)
    if cfg.sft_data_path is not None and not os.path.exists(cfg.sft_data_path):
        errors.append(f"sft_data_path does not exist: {cfg.sft_data_path}")

    # Checkpoint path (if provided)
    if cfg.load_checkpoint_path is not None:
        if not cfg.load_checkpoint_path.startswith("tinker://"):
            errors.append(
                f"load_checkpoint_path must be a tinker:// path, "
                f"got: {cfg.load_checkpoint_path}"
            )

    # Numeric bounds
    if cfg.lora_rank <= 0:
        errors.append(f"lora_rank must be positive, got {cfg.lora_rank}")
    if cfg.sft_learning_rate <= 0:
        errors.append(f"sft_learning_rate must be positive, got {cfg.sft_learning_rate}")
    if cfg.sft_num_epochs <= 0:
        errors.append(f"sft_num_epochs must be positive, got {cfg.sft_num_epochs}")
    if cfg.sft_batch_size <= 0:
        errors.append(f"sft_batch_size must be positive, got {cfg.sft_batch_size}")
    if cfg.opd_learning_rate <= 0:
        errors.append(f"opd_learning_rate must be positive, got {cfg.opd_learning_rate}")
    if cfg.opd_num_epochs <= 0:
        errors.append(f"opd_num_epochs must be positive, got {cfg.opd_num_epochs}")
    if cfg.opd_batch_size_prompts <= 0:
        errors.append(f"opd_batch_size_prompts must be positive, got {cfg.opd_batch_size_prompts}")
    if cfg.opd_samples_per_prompt <= 0:
        errors.append(f"opd_samples_per_prompt must be positive, got {cfg.opd_samples_per_prompt}")
    if cfg.opd_kl_penalty_coef < 0:
        errors.append(f"opd_kl_penalty_coef must be non-negative, got {cfg.opd_kl_penalty_coef}")
    if cfg.opd_kl_discount_factor < 0 or cfg.opd_kl_discount_factor > 1:
        errors.append(f"opd_kl_discount_factor must be in [0, 1], got {cfg.opd_kl_discount_factor}")
    if cfg.opd_kl_topk_tokens < 1:
        errors.append(f"opd_kl_topk_tokens must be >= 1, got {cfg.opd_kl_topk_tokens}")

    # Thinking mode
    is_qwen = "qwen" in cfg.model_name.lower()
    is_openai = "openai" in cfg.model_name.lower() or "gpt-oss" in cfg.model_name.lower()
    supports_thinking = is_qwen or is_openai

    if cfg.thinking_mode is not None and not supports_thinking:
        errors.append(
            f"thinking_mode='{cfg.thinking_mode}' is only valid for Qwen or OpenAI models, "
            f"but model_name='{cfg.model_name}'"
        )

    if cfg.reasoning_effort is not None:
        if not is_openai:
            errors.append(
                f"reasoning_effort='{cfg.reasoning_effort}' is only valid for OpenAI models, "
                f"but model_name='{cfg.model_name}'"
            )
        elif cfg.thinking_mode != "enable":
            errors.append(
                f"reasoning_effort='{cfg.reasoning_effort}' requires thinking_mode='enable', "
                f"but thinking_mode='{cfg.thinking_mode}'"
            )

    if errors:
        raise ConfigurationError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )


# --- Config Persistence ---

def save_command_line(log_path: str) -> None:
    """Save the full command line. Appends on resume."""
    os.makedirs(log_path, exist_ok=True)
    command_path = os.path.join(log_path, COMMAND_FILENAME)

    command_line = " ".join(sys.argv)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    is_new_file = not os.path.exists(command_path)
    with open(command_path, "a") as f:
        if not is_new_file:
            f.write("\n" + "=" * 80 + "\n\n")
        f.write(f"# Command executed at: {timestamp}\n")
        f.write(f"# Working directory: {os.getcwd()}\n\n")
        f.write(f"{command_line}\n")

    logger.info(f"Saved command line to {command_path}")


def save_pipeline_config(cfg: Config) -> None:
    """Save pipeline config for reproducibility."""
    os.makedirs(cfg.log_path, exist_ok=True)
    config_path = os.path.join(cfg.log_path, PIPELINE_CONFIG_FILENAME)
    with open(config_path, "w") as f:
        json.dump(chz.asdict(cfg), f, indent=2)
    logger.info(f"Saved pipeline config to {config_path}")


# --- Dataset Splitting ---

def split_dataset(
    dataset_path: str,
    sft_ratio: float,
    output_dir: str,
    seed: int = 42,
) -> tuple[str, str]:
    """
    Split dataset into SFT and OPD portions. Saves to disk.

    Returns:
        Tuple of (sft_dataset_path, opd_dataset_path).
    """
    sft_dir = os.path.join(output_dir, "sft_dataset")
    opd_dir = os.path.join(output_dir, "opd_dataset")

    # Check if splits already exist (resume case)
    if os.path.exists(sft_dir) and os.path.exists(opd_dir):
        sft_ds = load_from_disk(sft_dir)
        opd_ds = load_from_disk(opd_dir)
        logger.info(
            f"Dataset splits already exist: {len(sft_ds)} SFT, {len(opd_ds)} OPD"
        )
        return sft_dir, opd_dir

    logger.info(f"Loading dataset from {dataset_path}")
    full_dataset = load_from_disk(dataset_path)
    total_size = len(full_dataset)

    sft_size = int(total_size * sft_ratio)
    opd_size = total_size - sft_size

    logger.info(
        f"Splitting dataset: {sft_size} SFT ({sft_ratio*100:.0f}%) / "
        f"{opd_size} OPD ({(1-sft_ratio)*100:.0f}%)"
    )

    rng = np.random.default_rng(seed)
    indices = np.arange(total_size)
    rng.shuffle(indices)

    sft_indices = indices[:sft_size].tolist()
    opd_indices = indices[sft_size:].tolist()

    sft_dataset = full_dataset.select(sft_indices)
    opd_dataset = full_dataset.select(opd_indices)

    sft_dataset.save_to_disk(sft_dir)
    opd_dataset.save_to_disk(opd_dir)

    logger.info(f"Saved SFT split ({len(sft_dataset)} examples) to {sft_dir}")
    logger.info(f"Saved OPD split ({len(opd_dataset)} examples) to {opd_dir}")

    return sft_dir, opd_dir


# --- Checkpoint Extraction ---

def get_last_sft_checkpoint(sft_log_path: str) -> str:
    """Extract the last SFT checkpoint's state_path from checkpoints.jsonl."""
    checkpoint = checkpoint_utils.get_last_checkpoint(sft_log_path)
    if checkpoint is None:
        raise RuntimeError(
            f"SFT phase completed but no checkpoint found at "
            f"{sft_log_path}/checkpoints.jsonl"
        )

    state_path = checkpoint.get("state_path")
    if state_path is None:
        raise RuntimeError(
            f"Last SFT checkpoint has no state_path: {checkpoint}"
        )

    return state_path


# --- Main Pipeline ---

async def main(cfg: Config):
    logger.info("=" * 60)
    logger.info("EVC OPD Pipeline: SFT + On-Policy Distillation")
    logger.info("=" * 60)

    # Validate
    logger.info("Validating pipeline configuration...")
    validate_config(cfg)
    logger.info("Configuration validation passed")

    # Save config + command
    save_command_line(cfg.log_path)
    save_pipeline_config(cfg)

    # Resolve wandb name
    wandb_name = cfg.wandb_name or os.path.basename(cfg.log_path.rstrip("/"))

    # Phase paths
    skip_sft = cfg.load_checkpoint_path is not None
    sft_log_path = os.path.join(cfg.log_path, "sft")
    opd_log_path = os.path.join(cfg.log_path, "opd")

    # ========================
    # Split dataset
    # ========================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Dataset Split")
    logger.info("=" * 60)

    sft_dataset_path, opd_dataset_path = split_dataset(
        dataset_path=cfg.dataset_path,
        sft_ratio=cfg.sft_split_ratio,
        output_dir=cfg.log_path,
    )

    # ========================
    # Phase 1+2: SFT
    # ========================
    if skip_sft:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SFT Phase: SKIPPED (load_checkpoint_path provided)")
        logger.info(f"  Using checkpoint: {cfg.load_checkpoint_path}")
        logger.info("=" * 60)
        opd_checkpoint_path = cfg.load_checkpoint_path

    else:
        logger.info("")
        logger.info("=" * 60)
        logger.info("SFT Phase (Generation + Training)")
        logger.info("=" * 60)

        sft_config = train_tinker_sft.Config(
            model_name=cfg.model_name,
            log_path=sft_log_path,
            thinking_mode=cfg.thinking_mode,
            train_on_thinking=cfg.train_on_thinking,
            reasoning_effort=cfg.reasoning_effort,
            source_dataset=sft_dataset_path,
            teacher_type=cfg.sft_teacher_type,
            supervision_key=cfg.supervision_key,
            teacher_prompt=cfg.sft_teacher_prompt,
            n_samples=cfg.sft_n_samples,
            gen_temperature=cfg.sft_gen_temperature,
            gen_max_tokens=cfg.sft_gen_max_tokens,
            gen_batch_size=cfg.sft_gen_batch_size,
            sft_data_path=cfg.sft_data_path,
            learning_rate=cfg.sft_learning_rate,
            num_epochs=cfg.sft_num_epochs,
            lora_rank=cfg.lora_rank,
            batch_size=cfg.sft_batch_size,
            save_every=cfg.sft_save_every,
            eval_every=cfg.sft_eval_every,
            max_length=cfg.sft_max_length,
            wandb_project=cfg.wandb_project,
            wandb_name=f"{wandb_name}-sft",
            base_url=cfg.base_url,
            enable_trace=cfg.enable_trace,
        )

        sft_start = time.time()
        await train_tinker_sft.main(sft_config)
        sft_elapsed = time.time() - sft_start
        logger.info(f"SFT phase completed in {sft_elapsed:.1f}s ({sft_elapsed/60:.1f}m)")

        # Extract last checkpoint for OPD initialization
        opd_checkpoint_path = get_last_sft_checkpoint(sft_log_path)
        logger.info(f"SFT checkpoint for OPD: {opd_checkpoint_path}")

    # ========================
    # Phase 3: OPD
    # ========================
    logger.info("")
    logger.info("=" * 60)
    logger.info("OPD Phase (On-Policy Distillation)")
    logger.info("=" * 60)

    opd_config = train_distillation_evc.Config(
        model_name=cfg.model_name,
        thinking_mode=cfg.thinking_mode,
        train_on_thinking=cfg.train_on_thinking,
        reasoning_effort=cfg.reasoning_effort,
        dataset_path=opd_dataset_path,
        batch_size_prompts=cfg.opd_batch_size_prompts,
        samples_per_prompt=cfg.opd_samples_per_prompt,
        num_epochs=cfg.opd_num_epochs,
        teacher_mode=cfg.opd_teacher_mode,
        supervision_key=cfg.supervision_key,
        teacher_prompt=cfg.opd_teacher_prompt,
        student_prefill=cfg.opd_student_prefill,
        learning_rate=cfg.opd_learning_rate,
        lora_rank=cfg.lora_rank,
        kl_penalty_coef=cfg.opd_kl_penalty_coef,
        kl_discount_factor=cfg.opd_kl_discount_factor,
        kl_topk_tokens=cfg.opd_kl_topk_tokens,
        temperature=cfg.opd_temperature,
        max_tokens=cfg.opd_max_tokens,
        loss_fn=cfg.opd_loss_fn,
        shuffle=cfg.opd_shuffle,
        log_path=opd_log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=f"{wandb_name}-opd",
        save_every=cfg.opd_save_every,
        eval_every=cfg.opd_eval_every,
        base_url=cfg.base_url,
        enable_trace=cfg.enable_trace,
        load_checkpoint_path=opd_checkpoint_path,
    )

    opd_start = time.time()
    await train_distillation_evc.main(opd_config)
    opd_elapsed = time.time() - opd_start
    logger.info(f"OPD phase completed in {opd_elapsed:.1f}s ({opd_elapsed/60:.1f}m)")

    # ========================
    # Summary
    # ========================
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Complete")
    logger.info("=" * 60)
    if not skip_sft:
        logger.info(f"  SFT output:       {sft_log_path}")
    logger.info(f"  OPD output:       {opd_log_path}")
    logger.info(f"  OPD init checkpoint: {opd_checkpoint_path}")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    asyncio.run(main(cfg))
