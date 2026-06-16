#!/usr/bin/env python3
"""
Off-policy distillation training with pre-computed teacher logprobs.

Uses fixed trajectories and pre-computed logprobs from prepare_offpolicy_data.py.
The advantage computation matches train_distillation.py exactly:
    advantages = -kl_penalty_coef * (sampling_logprobs - teacher_logprobs)
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Literal

import chz
import numpy as np
import tinker
import torch
from datasets import load_from_disk
from dotenv import load_dotenv

load_dotenv()

from tinker import types
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.rl.metrics import compute_kl_sample_train, discounted_future_sum_vectorized
from tinker_cookbook.rl.train import train_step
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import timed
from utils.renderer_utils import get_renderer_name_with_thinking_mode

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --- Configuration ---

CONFIG_FILENAME = "config.json"
COMMAND_FILENAME = "command.txt"


def save_command_line(log_path: str) -> None:
    """Save the full command line used to invoke the script."""
    os.makedirs(log_path, exist_ok=True)
    command_path = os.path.join(log_path, COMMAND_FILENAME)

    command_line = " ".join(sys.argv)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    is_new_file = not os.path.exists(command_path)
    with open(command_path, "a") as f:
        if not is_new_file:
            f.write("\n" + "=" * 80 + "\n\n")
        f.write(f"# Command executed at: {timestamp}\n")
        f.write(f"# Working directory: {os.getcwd()}\n\n")
        f.write(f"{command_line}\n")

    logger.info(f"Saved command line to {command_path}")


@chz.chz
class Config:
    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Thinking mode control for Qwen3 hybrid models
    thinking_mode: Literal["enable", "disable"] | None = None

    # Dataset settings
    dataset_path: str = "./datasets/offpolicy-qwen3-4b"
    batch_size: int = 64
    num_epochs: int = 3

    # Training hyperparameters
    learning_rate: float = 1e-5
    lora_rank: int = 32
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Loss function
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    num_substeps: int = 1

    # Logging and checkpointing
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 20
    eval_every: int = 20

    # System
    base_url: str | None = None
    load_checkpoint_path: str | None = None

    # Shuffle data each epoch
    shuffle: bool = True


# --- Dataset ---


class OffPolicyDistillationDataset:
    """Loads pre-computed trajectories with teacher logprobs."""

    def __init__(
        self,
        dataset_path: str,
        batch_size: int,
        renderer: renderers.Renderer,
        kl_penalty_coef: float,
        kl_discount_factor: float = 0.0,
        shuffle: bool = True,
    ):
        logger.info(f"Loading off-policy dataset from: {dataset_path}")
        self.ds = load_from_disk(dataset_path)
        logger.info(f"Loaded {len(self.ds)} examples")

        self.batch_size = batch_size
        self.renderer = renderer
        self.kl_penalty_coef = kl_penalty_coef
        self.kl_discount_factor = kl_discount_factor
        self.shuffle = shuffle

        # Create index mapping (shuffled each epoch)
        self.indices = list(range(len(self.ds)))

    def shuffle_indices(self, seed: int | None = None):
        """Shuffle the index mapping for a new epoch."""
        if self.shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
            logger.info(f"Shuffled dataset indices (seed={seed})")

    def get_batch(self, batch_idx: int) -> List[tinker.Datum]:
        """Returns pre-built Datum objects ready for training."""
        datums = []

        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.ds))

        for i in range(start_idx, end_idx):
            # Use shuffled index
            actual_idx = self.indices[i % len(self.indices)]
            row = self.ds[actual_idx]

            # Build student prompt
            student_messages = [{"role": "user", "content": row["prompt"]}]
            student_prompt_input = self.renderer.build_generation_prompt(student_messages)
            prompt_tokens = student_prompt_input.to_ints()

            response_tokens = row["response_tokens"]
            sampling_logprobs = row["sampling_logprobs"]
            teacher_logprobs = row["teacher_logprobs"]

            # Full sequence: prompt + response
            full_tokens = prompt_tokens + list(response_tokens)

            # Mask: 0 for prompt, 1 for response
            mask = [0.0] * len(prompt_tokens) + [1.0] * len(response_tokens)

            # Logprobs for importance weights (from sampling distribution)
            # Padded with 0.0 for prompt tokens
            logprobs = [0.0] * len(prompt_tokens) + list(sampling_logprobs)

            # Advantages: -kl_penalty_coef * (sampling_logprobs - teacher_logprobs)
            # This EXACTLY matches on-policy (train_distillation.py lines 604, 618):
            #   response_kl = student_response_logprobs - teacher_response_logprobs
            #   kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]
            response_advantages = [
                -self.kl_penalty_coef * (s_lp - t_lp)
                for s_lp, t_lp in zip(sampling_logprobs, teacher_logprobs)
            ]
            advantages = [0.0] * len(prompt_tokens) + response_advantages

            # Apply discount factor if configured
            if self.kl_discount_factor > 0:
                # Only discount the response portion
                discounted_response = discounted_future_sum_vectorized(
                    np.array(response_advantages, dtype=np.float32),
                    self.kl_discount_factor,
                )
                advantages = [0.0] * len(prompt_tokens) + discounted_response.tolist()

            # Create Datum
            # model_input is tokens[:-1], targets are tokens[1:]
            # Note: We don't store teacher_logprobs in loss_fn_inputs as Tinker API
            # doesn't accept extra fields. We compute teacher_kl from advantages instead.
            datum = tinker.Datum(
                model_input=types.ModelInput.from_ints(full_tokens[:-1]),
                loss_fn_inputs={
                    "target_tokens": types.TensorData.from_numpy(
                        np.array(full_tokens[1:], dtype=np.int64)
                    ),
                    "logprobs": types.TensorData.from_numpy(
                        np.array(logprobs[:-1], dtype=np.float32)
                    ),
                    "advantages": types.TensorData.from_numpy(
                        np.array(advantages[1:], dtype=np.float32)
                    ),
                    "mask": types.TensorData.from_numpy(
                        np.array(mask[1:], dtype=np.float32)
                    ),
                },
            )
            datums.append(datum)

        return datums

    def __len__(self) -> int:
        return (len(self.ds) + self.batch_size - 1) // self.batch_size


# --- Metrics ---


def compute_offpolicy_metrics(
    data_D: List[tinker.Datum],
    training_logprobs_D: List[torch.Tensor],
    kl_penalty_coef: float = 1.0,
) -> Dict[str, float]:
    """Compute metrics for off-policy training.

    Metrics:
    - optim/kl_sample_train_v1: KL between sampling and current training (importance weights)
    - optim/importance_weight_*: Stats on importance weights
    - optim/ppo_clip_fraction: Fraction of tokens clipped by PPO
    - teacher_kl: KL between current student and teacher (current - teacher)
    - optim/student_drift: Drift from original sampling (current - sampling)
    """
    metrics = {}

    # Use the existing compute_kl_sample_train for importance weight metrics
    iw_metrics = compute_kl_sample_train(data_D, training_logprobs_D)
    metrics.update(iw_metrics)

    # Compute teacher_kl (current_student - teacher) and drift (current - sampling)
    # We derive teacher from stored values:
    #   advantages = -kl_coef * (sampling - teacher)
    #   teacher = sampling + advantages / kl_coef
    total_teacher_kl = 0.0
    total_drift = 0.0
    total_tokens = 0

    for datum, train_logprobs in zip(data_D, training_logprobs_D):
        mask = datum.loss_fn_inputs["mask"].to_torch()
        response_length = int(mask.sum().item())

        if response_length == 0:
            continue

        current_response_logprobs = train_logprobs[-response_length:]
        sampling_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()[-response_length:]
        advantages = datum.loss_fn_inputs["advantages"].to_torch()[-response_length:]

        # Derive teacher logprobs from advantages:
        # advantages = -kl_coef * (sampling - teacher)
        # teacher = sampling + advantages / kl_coef
        if kl_penalty_coef > 0:
            teacher_logprobs = sampling_logprobs + advantages / kl_penalty_coef
            # teacher_kl: current_student - teacher (matches on-policy's reverse KL)
            teacher_kl = current_response_logprobs - teacher_logprobs
            total_teacher_kl += teacher_kl.sum().item()

        # drift: current - sampling (how much has policy changed)
        drift = current_response_logprobs - sampling_logprobs
        total_drift += drift.sum().item()

        total_tokens += response_length

    if total_tokens > 0:
        if kl_penalty_coef > 0:
            metrics["teacher_kl"] = total_teacher_kl / total_tokens
        metrics["optim/student_drift"] = total_drift / total_tokens

    return metrics


# --- Training ---


async def do_offpolicy_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    batches_per_epoch: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    dataset: OffPolicyDistillationDataset,
    ml_logger: ml_log.Logger,
):
    """Off-policy training: fixed trajectories, pre-computed teacher logprobs."""

    for i_batch in range(start_batch, end_batch):
        current_epoch = i_batch // batches_per_epoch
        batch_in_epoch = i_batch % batches_per_epoch

        # Shuffle at the start of each epoch
        if batch_in_epoch == 0 and i_batch > start_batch:
            dataset.shuffle_indices(seed=current_epoch)

        metrics: Dict[str, Any] = {
            "progress/batch": i_batch,
            "progress/epoch": current_epoch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Load pre-built datums (NO SAMPLING!)
        with timed("load_batch", metrics):
            data_D = dataset.get_batch(batch_in_epoch)

        metrics["batch/size"] = len(data_D)

        # Forward-backward-optim with importance_sampling loss
        with timed("train", metrics):
            training_logprobs_D = await train_step(
                data_D,
                training_client,
                cfg.learning_rate,
                cfg.num_substeps,
                cfg.loss_fn,
            )

        # Compute metrics
        with timed("compute_metrics", metrics):
            train_metrics = compute_offpolicy_metrics(
                data_D, training_logprobs_D, kl_penalty_coef=cfg.kl_penalty_coef
            )
            metrics.update(train_metrics)

        metrics["time/total"] = time.time() - t_start

        # Log metrics
        ml_logger.log_metrics(metrics, step=i_batch)

        # Log progress
        if i_batch % 10 == 0 or i_batch == end_batch - 1:
            teacher_kl = metrics.get("teacher_kl", 0.0)
            iw_max = metrics.get("optim/importance_weight_max", 0.0)
            clip_frac = metrics.get("optim/ppo_clip_fraction", 0.0)
            logger.info(
                f"Batch {i_batch}/{num_batches} | Epoch {current_epoch} | "
                f"teacher_kl: {teacher_kl:.4f} | IW_max: {iw_max:.2f} | Clip: {clip_frac:.3f}"
            )

        # Save checkpoint
        if (i_batch + 1) % cfg.save_every == 0 or i_batch == end_batch - 1:
            logger.info(f"Saving checkpoint at batch {i_batch + 1}...")
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"step_{i_batch + 1}",
                log_path=cfg.log_path,
                kind="both",
                loop_state={"batch": i_batch + 1},
            )


async def main(cfg: Config):
    """Main training loop for off-policy distillation."""

    # Save command line
    save_command_line(cfg.log_path)

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # Check for resume from log_path checkpoints
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    # Create training client
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    # Load checkpoint if resuming
    load_state_path = resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    if load_state_path:
        logger.info(f"Loading checkpoint from: {load_state_path}")
        future = await training_client.load_state_async(load_state_path)
        _ = await future.result_async()
        logger.info(f"Successfully loaded state from {load_state_path}")

    tokenizer = training_client.get_tokenizer()

    # Determine renderer
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    logger.info(f"Using renderer: {renderer_name}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create dataset
    dataset = OffPolicyDistillationDataset(
        dataset_path=cfg.dataset_path,
        batch_size=cfg.batch_size,
        renderer=renderer,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        shuffle=cfg.shuffle,
    )

    # Shuffle for first epoch if not resuming from middle
    if start_batch == 0:
        dataset.shuffle_indices(seed=0)

    batches_per_epoch = len(dataset)
    num_batches = batches_per_epoch * cfg.num_epochs

    logger.info(f"Off-policy distillation training:")
    logger.info(f"  Dataset: {cfg.dataset_path} ({len(dataset.ds)} examples)")
    logger.info(f"  Batch size: {cfg.batch_size}")
    logger.info(f"  Batches per epoch: {batches_per_epoch}")
    logger.info(f"  Epochs: {cfg.num_epochs}")
    logger.info(f"  Total batches: {num_batches}")
    logger.info(f"  KL penalty coefficient: {cfg.kl_penalty_coef}")
    logger.info(f"  Loss function: {cfg.loss_fn}")
    logger.info(f"  Learning rate: {cfg.learning_rate}")

    if start_batch > 0:
        logger.info(f"Resuming from batch {start_batch}")

    # Training loop
    await do_offpolicy_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        batches_per_epoch=batches_per_epoch,
        cfg=cfg,
        training_client=training_client,
        dataset=dataset,
        ml_logger=ml_logger,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    asyncio.run(main(cfg))
