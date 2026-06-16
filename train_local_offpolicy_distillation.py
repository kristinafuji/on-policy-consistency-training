#!/usr/bin/env python3

"""
Local PyTorch off-policy distillation training.

This script implements the same algorithm as train_offpolicy_distillation.py
but runs locally on a single GPU using Unsloth + LoRA, without requiring
the Tinker API.

Key features:
- Uses pre-computed trajectories with teacher logprobs (from prepare_offpolicy_data.py)
- Simple data loading -> train loop (no sampling, no teacher logprob computation)
- Memory efficient via gradient accumulation
- Compatible parameters with train_offpolicy_distillation.py
"""

# Early progress output (before heavy imports)
import sys
print("[1/6] Starting script, importing libraries...", flush=True)

# IMPORTANT: Import unsloth BEFORE any other ML libraries (transformers, torch, etc.)
# to ensure all Unsloth optimizations are applied correctly.
print("[2/6] Importing Unsloth...", flush=True)
try:
    import unsloth  # noqa: F401
except ImportError:
    print("  (Unsloth not available, will use standard PEFT)", flush=True)

print("[3/6] Importing PyTorch and other libraries...", flush=True)
import json
import logging
import os
import time
from typing import Any, Dict, List, Literal

import chz
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
print("[4/6] PyTorch loaded, importing local modules...", flush=True)

# Add tinker-cookbook to path for reusing renderers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tinker-cookbook"))

from tinker_cookbook import renderers
from tinker_cookbook.utils import ml_log

from local_ppo.checkpointing import (
    get_last_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_command_line,
    save_config,
    validate_resume_config,
    load_config,
)
from local_ppo.data_processing import (
    LocalDatum,
    collate_local_datums,
)
from local_ppo.model_manager import LocalModelManager, create_optimizer
from local_ppo.ppo_loss import (
    compute_ppo_loss,
    compute_importance_sampling_loss,
    compute_kl_sample_train,
)
from utils.renderer_utils import get_renderer_name_with_thinking_mode
print("[5/6] All imports complete, parsing config...", flush=True)

logger = logging.getLogger(__name__)


# --- Configuration ---

@chz.chz
class Config:
    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Thinking mode control for Qwen3 hybrid models
    thinking_mode: Literal["enable", "disable"] | None = None

    # Dataset settings (pre-computed off-policy data)
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
    ppo_clip_epsilon: float = 0.2  # Only used when loss_fn="ppo"
    num_substeps: int = 1  # Number of gradient accumulation steps per batch

    # Logging and checkpointing
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 20

    # Hardware settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_seq_length: int = 2048

    # Shuffle data each epoch
    shuffle: bool = True


def validate_config(cfg: Config) -> None:
    """Validate configuration parameters with defensive assertions."""
    errors = []

    # --- Config Validation (from plan) ---
    if cfg.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {cfg.batch_size}")
    if cfg.num_epochs <= 0:
        errors.append(f"num_epochs must be positive, got {cfg.num_epochs}")
    if cfg.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {cfg.learning_rate}")
    if cfg.lora_rank <= 0:
        errors.append(f"lora_rank must be positive, got {cfg.lora_rank}")
    if cfg.kl_penalty_coef < 0:
        errors.append(f"kl_penalty_coef must be non-negative, got {cfg.kl_penalty_coef}")
    if cfg.kl_discount_factor < 0 or cfg.kl_discount_factor > 1:
        errors.append(f"kl_discount_factor must be in [0, 1], got {cfg.kl_discount_factor}")
    if cfg.num_substeps <= 0:
        errors.append(f"num_substeps must be positive, got {cfg.num_substeps}")
    if cfg.save_every <= 0:
        errors.append(f"save_every must be positive, got {cfg.save_every}")
    if cfg.ppo_clip_epsilon <= 0:
        errors.append(f"ppo_clip_epsilon must be positive, got {cfg.ppo_clip_epsilon}")

    # Validate dataset path exists
    if not os.path.exists(cfg.dataset_path):
        errors.append(f"Dataset path does not exist: {cfg.dataset_path}")

    # Validate loss_fn
    if cfg.loss_fn not in ["importance_sampling", "ppo"]:
        errors.append(f"loss_fn must be 'importance_sampling' or 'ppo', got {cfg.loss_fn}")

    if not cfg.log_path:
        errors.append("log_path must be specified")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# --- Dataset ---

class LocalOffPolicyDataset:
    """
    Loads pre-computed off-policy trajectories from a HuggingFace dataset.

    Expected dataset columns:
    - prompt: str - the user prompt
    - response_tokens: list[int] - tokenized response
    - sampling_logprobs: list[float] - logprobs from sampling policy
    - teacher_logprobs: list[float] - logprobs from teacher (base + augmented prompt)
    """

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

        # --- Dataset Validation (from plan) ---
        required_cols = ["prompt", "response_tokens", "sampling_logprobs", "teacher_logprobs"]
        for col in required_cols:
            assert col in self.ds.column_names, f"Missing required column: {col}"

        self.batch_size = batch_size
        self.renderer = renderer
        self.kl_penalty_coef = kl_penalty_coef
        self.kl_discount_factor = kl_discount_factor
        self.shuffle = shuffle

        # Create index mapping (shuffled each epoch)
        self.indices = list(range(len(self.ds)))

    def shuffle_indices(self, seed: int | None = None) -> None:
        """Shuffle the index mapping for a new epoch."""
        if self.shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
            logger.info(f"Shuffled dataset indices (seed={seed})")

    def _row_to_datum(self, row: dict) -> LocalDatum:
        """
        Convert a dataset row to a LocalDatum.

        This implements the EXACT same logic as train_offpolicy_distillation.py:
        - Build prompt tokens via renderer
        - Build full sequence: prompt + response
        - Compute advantages: -kl_coef * (sampling_lp - teacher_lp)
        - Build LocalDatum with proper shifting
        """
        # 1. Build prompt tokens via renderer
        student_messages = [{"role": "user", "content": row["prompt"]}]
        student_prompt_input = self.renderer.build_generation_prompt(student_messages)
        prompt_tokens = student_prompt_input.to_ints()

        response_tokens = list(row["response_tokens"])
        sampling_logprobs = list(row["sampling_logprobs"])
        teacher_logprobs = list(row["teacher_logprobs"])

        # --- Per-row validation (from plan) ---
        assert len(response_tokens) == len(sampling_logprobs), (
            f"response_tokens length ({len(response_tokens)}) != "
            f"sampling_logprobs length ({len(sampling_logprobs)})"
        )
        assert len(response_tokens) == len(teacher_logprobs), (
            f"response_tokens length ({len(response_tokens)}) != "
            f"teacher_logprobs length ({len(teacher_logprobs)})"
        )

        # 2. Build full sequence: prompt + response
        full_tokens = prompt_tokens + response_tokens
        seq_len = len(full_tokens)

        # 3. Create input/target (shifted by 1)
        # model_input is tokens[:-1], targets are tokens[1:]
        input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(full_tokens[1:], dtype=torch.long)

        # 4. Create mask: 0 for prompt, 1 for response
        # In the shifted sequence:
        # - Positions 0 to prompt_len-2 are predicting prompt tokens (mask=0)
        # - Positions prompt_len-1 to end are predicting response tokens (mask=1)
        mask = torch.zeros(seq_len - 1, dtype=torch.float32)
        response_start = len(prompt_tokens) - 1
        mask[response_start:] = 1.0

        # 5. Build sampling logprobs tensor (padded with zeros for prompt positions)
        sampling_lp = torch.zeros(seq_len - 1, dtype=torch.float32)
        sampling_lp[response_start:] = torch.tensor(sampling_logprobs, dtype=torch.float32)

        # 6. Compute advantages: -kl_penalty_coef * (sampling_logprobs - teacher_logprobs)
        # This EXACTLY matches train_offpolicy_distillation.py:177-180:
        #   response_advantages = [
        #       -self.kl_penalty_coef * (s_lp - t_lp)
        #       for s_lp, t_lp in zip(sampling_logprobs, teacher_logprobs)
        #   ]
        response_advantages = [
            -self.kl_penalty_coef * (s_lp - t_lp)
            for s_lp, t_lp in zip(sampling_logprobs, teacher_logprobs)
        ]

        # Apply discount factor if configured
        if self.kl_discount_factor > 0:
            response_advantages = self._discounted_future_sum(
                response_advantages, self.kl_discount_factor
            )

        # Build advantages tensor (padded with zeros for prompt positions)
        advantages = torch.zeros(seq_len - 1, dtype=torch.float32)
        advantages[response_start:] = torch.tensor(response_advantages, dtype=torch.float32)

        # 7. Build teacher_logprobs tensor (for metrics computation)
        teacher_lp = torch.tensor(teacher_logprobs, dtype=torch.float32)

        # --- Numerical validation (from plan) ---
        assert input_ids.shape == target_tokens.shape, (
            f"input_ids shape {input_ids.shape} != target_tokens shape {target_tokens.shape}"
        )
        assert not torch.isnan(advantages).any(), "NaN in advantages"
        assert not torch.isnan(sampling_lp).any(), "NaN in sampling_logprobs"

        return LocalDatum(
            input_ids=input_ids,
            target_tokens=target_tokens,
            sampling_logprobs=sampling_lp,
            advantages=advantages,
            mask=mask,
            teacher_logprobs=teacher_lp,
            metadata={
                "prompt": row["prompt"],
                "response_len": len(response_tokens),
                "prompt_len": len(prompt_tokens),
            },
        )

    def _discounted_future_sum(
        self, values: list[float], discount_factor: float
    ) -> list[float]:
        """
        Compute discounted future sum.

        discounted[t] = sum_{s=t}^T (discount^(s-t) * values[s])

        Matches tinker_cookbook/rl/metrics.py:discounted_future_sum_vectorized
        """
        n = len(values)
        discounted = [0.0] * n
        running_sum = 0.0

        for t in range(n - 1, -1, -1):
            running_sum = values[t] + discount_factor * running_sum
            discounted[t] = running_sum

        return discounted

    def get_batch(self, batch_idx: int) -> List[LocalDatum]:
        """Returns pre-built LocalDatum objects ready for training."""
        datums = []

        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.ds))

        for i in range(start_idx, end_idx):
            # Use shuffled index
            actual_idx = self.indices[i % len(self.indices)]
            row = self.ds[actual_idx]
            datum = self._row_to_datum(row)
            datums.append(datum)

        return datums

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (len(self.ds) + self.batch_size - 1) // self.batch_size

    def num_examples(self) -> int:
        """Total number of examples in dataset."""
        return len(self.ds)


# --- Training Functions ---

def forward_backward_local(
    model,
    batch: Dict[str, torch.Tensor],
    loss_fn: str,
    clip_epsilon: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Compute forward pass and loss (backward handled by caller).

    Replaces Tinker's forward_backward_async.

    Args:
        model: The model to train.
        batch: Collated batch from collate_local_datums.
        loss_fn: "ppo" or "importance_sampling".
        clip_epsilon: PPO clipping parameter.

    Returns:
        Tuple of (loss, current_logprobs, metrics).
    """
    # Forward pass
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = outputs.logits  # (batch, seq_len, vocab)

    # Compute log softmax
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather logprobs for target tokens
    current_logprobs = torch.gather(
        log_probs,
        dim=-1,
        index=batch["target_tokens"].unsqueeze(-1),
    ).squeeze(-1)  # (batch, seq_len)

    # --- Numerical validation (from plan) ---
    assert not torch.isnan(current_logprobs).any(), "NaN in current_logprobs"
    # Note: logprobs can be -inf for padded positions, so we only check valid tokens
    valid_mask = batch["mask"] > 0
    if valid_mask.any():
        valid_logprobs = current_logprobs[valid_mask]
        assert (valid_logprobs <= 0).all(), f"Logprobs must be <= 0, got max {valid_logprobs.max().item()}"

    # Compute loss using existing local_ppo functions
    if loss_fn == "ppo":
        loss, metrics = compute_ppo_loss(
            current_logprobs=current_logprobs,
            sampling_logprobs=batch["sampling_logprobs"],
            advantages=batch["advantages"],
            mask=batch["mask"],
            clip_epsilon=clip_epsilon,
        )
    else:
        loss, metrics = compute_importance_sampling_loss(
            current_logprobs=current_logprobs,
            sampling_logprobs=batch["sampling_logprobs"],
            advantages=batch["advantages"],
            mask=batch["mask"],
        )

    # --- Numerical validation (from plan) ---
    assert not torch.isnan(loss), "NaN loss"
    assert not torch.isinf(loss), "Inf loss"

    return loss, current_logprobs.detach(), metrics


def train_step_local(
    data: List[LocalDatum],
    model,
    optimizer,
    num_substeps: int,
    loss_fn: str,
    clip_epsilon: float,
    pad_token_id: int,
    device: str,
) -> tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Single training step with gradient accumulation.

    Replaces Tinker's train_step.

    Args:
        data: List of LocalDatum for this batch.
        model: Model to train.
        optimizer: Optimizer.
        num_substeps: Number of gradient accumulation steps.
        loss_fn: "ppo" or "importance_sampling".
        clip_epsilon: PPO clipping parameter.
        pad_token_id: Token ID for padding.
        device: Device for tensors.

    Returns:
        Tuple of (training_logprobs list, averaged metrics).
    """
    model.train()

    # Split into mini-batches for gradient accumulation
    chunk_size = max(1, len(data) // num_substeps)
    mini_batches = [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]

    # Adjust num_substeps if we have fewer mini-batches
    actual_substeps = len(mini_batches)

    optimizer.zero_grad()

    all_training_logprobs: List[torch.Tensor] = []
    all_metrics: Dict[str, float] = {}
    total_loss = 0.0

    for mini_batch_data in mini_batches:
        # Collate into batch
        batch = collate_local_datums(
            mini_batch_data,
            pad_token_id=pad_token_id,
            device=device,
        )

        # Forward and compute loss
        loss, current_logprobs, metrics = forward_backward_local(
            model=model,
            batch=batch,
            loss_fn=loss_fn,
            clip_epsilon=clip_epsilon,
        )

        # Scale loss for gradient accumulation
        scaled_loss = loss / actual_substeps
        scaled_loss.backward()

        total_loss += loss.item()

        # Accumulate metrics
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v

        # Store logprobs for metrics (one tensor per datum in mini-batch)
        for i, datum in enumerate(mini_batch_data):
            seq_len = datum.input_ids.shape[0]
            all_training_logprobs.append(current_logprobs[i, :seq_len].cpu())

    # --- Importance weight sanity check (from plan) ---
    with torch.no_grad():
        # Check a sample of importance weights
        if all_training_logprobs and all_metrics.get("optim/importance_weight_max", 0) > 100.0:
            logger.warning(
                f"Very high importance ratio: {all_metrics['optim/importance_weight_max']:.2f} - "
                "training may be unstable"
            )

    # Compute gradient norm before optimizer step
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()

    # Average metrics
    for k in all_metrics:
        all_metrics[k] /= actual_substeps

    all_metrics["loss"] = total_loss / actual_substeps
    all_metrics["optim/grad_norm"] = grad_norm

    return all_training_logprobs, all_metrics


# --- Metrics ---

def compute_offpolicy_metrics_local(
    data: List[LocalDatum],
    training_logprobs: List[torch.Tensor],
    kl_penalty_coef: float = 1.0,
) -> Dict[str, float]:
    """
    Compute metrics for off-policy training.

    Matches compute_offpolicy_metrics from train_offpolicy_distillation.py.

    Metrics:
    - optim/kl_sample_train_v1: KL between sampling and current training
    - optim/kl_sample_train_v2: Second-order KL approximation
    - optim/importance_weight_*: Stats on importance weights
    - optim/ppo_clip_fraction: Fraction of tokens clipped by PPO
    - teacher_kl: KL between current student and teacher (current - teacher)
    - optim/student_drift: Drift from original sampling (current - sampling)
    """
    if not data or not training_logprobs:
        return {}

    # Compute teacher_kl and drift
    # We derive teacher from stored values:
    #   advantages = -kl_coef * (sampling - teacher)
    #   teacher = sampling + advantages / kl_coef
    total_teacher_kl = 0.0
    total_drift = 0.0
    total_tokens = 0

    for datum, train_lp in zip(data, training_logprobs):
        mask = datum.mask
        response_length = int(mask.sum().item())

        if response_length == 0:
            continue

        # Find response start position
        response_start = int((mask > 0).nonzero(as_tuple=True)[0][0])

        # Extract response logprobs
        current_response_logprobs = train_lp[response_start:response_start + response_length]
        sampling_logprobs = datum.sampling_logprobs[response_start:response_start + response_length]
        teacher_logprobs = datum.teacher_logprobs[:response_length]

        # teacher_kl: current_student - teacher (matches on-policy's reverse KL)
        teacher_kl = current_response_logprobs - teacher_logprobs
        total_teacher_kl += teacher_kl.sum().item()

        # drift: current - sampling (how much has policy changed)
        drift = current_response_logprobs - sampling_logprobs
        total_drift += drift.sum().item()

        total_tokens += response_length

    metrics = {}
    if total_tokens > 0:
        metrics["teacher_kl"] = total_teacher_kl / total_tokens
        metrics["optim/student_drift"] = total_drift / total_tokens

    # Compute KL sample-train metrics using existing function
    # First, stack the tensors for batch computation
    all_sampling = torch.cat([
        d.sampling_logprobs[d.mask > 0] for d in data
    ])
    all_training = torch.cat([
        train_lp[d.mask > 0] for d, train_lp in zip(data, training_logprobs)
    ])
    all_masks = torch.ones_like(all_sampling)  # All valid since we filtered

    kl_metrics = compute_kl_sample_train(
        sampling_logprobs=all_sampling.unsqueeze(0),
        training_logprobs=all_training.unsqueeze(0),
        mask=all_masks.unsqueeze(0),
    )
    metrics.update(kl_metrics)

    return metrics


# --- Main Training Loop ---

def do_offpolicy_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    batches_per_epoch: int,
    cfg: Config,
    model,
    optimizer,
    dataset: LocalOffPolicyDataset,
    tokenizer,
    ml_logger: ml_log.Logger,
):
    """Off-policy training: fixed trajectories, pre-computed teacher logprobs."""

    pad_token_id = tokenizer.pad_token_id or 0

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
        load_start = time.time()
        data_D = dataset.get_batch(batch_in_epoch)
        metrics["time/load_batch"] = time.time() - load_start
        metrics["batch/size"] = len(data_D)

        # Train step
        train_start = time.time()
        training_logprobs_D, train_metrics = train_step_local(
            data=data_D,
            model=model,
            optimizer=optimizer,
            num_substeps=cfg.num_substeps,
            loss_fn=cfg.loss_fn,
            clip_epsilon=cfg.ppo_clip_epsilon,
            pad_token_id=pad_token_id,
            device=cfg.device,
        )
        metrics["time/train"] = time.time() - train_start
        metrics.update(train_metrics)

        # Compute metrics
        compute_start = time.time()
        offpolicy_metrics = compute_offpolicy_metrics_local(
            data_D, training_logprobs_D, kl_penalty_coef=cfg.kl_penalty_coef
        )
        metrics.update(offpolicy_metrics)
        metrics["time/compute_metrics"] = time.time() - compute_start

        metrics["time/total"] = time.time() - t_start

        # Log metrics
        ml_logger.log_metrics(metrics, step=i_batch)

        # Log progress
        if i_batch % 10 == 0 or i_batch == end_batch - 1:
            teacher_kl = metrics.get("teacher_kl", 0.0)
            iw_max = metrics.get("optim/importance_weight_max", 0.0)
            clip_frac = metrics.get("optim/ppo_clip_fraction", 0.0)
            loss = metrics.get("loss", 0.0)
            grad_norm = metrics.get("optim/grad_norm", 0.0)
            logger.info(
                f"Batch {i_batch}/{num_batches} | Epoch {current_epoch} | "
                f"Loss: {loss:.4f} | teacher_kl: {teacher_kl:.4f} | "
                f"IW_max: {iw_max:.2f} | Clip: {clip_frac:.3f} | "
                f"Grad: {grad_norm:.4f}"
            )

        # Save checkpoint
        if (i_batch + 1) % cfg.save_every == 0 or i_batch == end_batch - 1:
            checkpoint_name = f"step_{i_batch + 1:06d}"
            logger.info(f"Saving checkpoint at batch {i_batch + 1}...")
            save_checkpoint(
                log_path=cfg.log_path,
                name=checkpoint_name,
                model=model,
                optimizer=optimizer,
                loop_state={"batch": i_batch + 1, "epoch": current_epoch},
            )


def main(cfg: Config):
    """Main training function."""
    print("[6/6] Config parsed, starting main()...", flush=True)

    # Validate configuration
    validate_config(cfg)
    print("  Config validated", flush=True)

    # Setup logging
    print("  Setting up logging and wandb...", flush=True)
    os.makedirs(cfg.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg,
    )
    save_config(cfg.log_path, chz.asdict(cfg))
    save_command_line(cfg.log_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    logger.info("Starting local off-policy distillation training")
    logger.info(f"Log path: {cfg.log_path}")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Learning rate: {cfg.learning_rate}")
    logger.info(f"KL penalty coefficient: {cfg.kl_penalty_coef}")
    logger.info(f"Loss function: {cfg.loss_fn}")
    logger.info(f"Num substeps: {cfg.num_substeps}")

    # Check for resume from log_path checkpoints
    resume_info = get_last_checkpoint(cfg.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    if resume_info:
        logger.info(f"Found checkpoint to resume from: {resume_info['name']}")
        logger.info(f"Resuming from batch {start_batch}")

        # Validate config matches for resume
        saved_config = load_config(cfg.log_path)
        if saved_config:
            validate_resume_config(
                chz.asdict(cfg),
                saved_config,
                critical_keys=[
                    "model_name",
                    "dataset_path",
                    "batch_size",
                    "lora_rank",
                    "loss_fn",
                    "kl_penalty_coef",
                    "thinking_mode",
                    "shuffle",
                ],
            )

    # Load model
    logger.info("Loading model...")
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    model_manager = LocalModelManager(
        model_name=cfg.model_name,
        lora_rank=cfg.lora_rank,
        device=cfg.device,
        dtype=dtype,
        max_seq_length=cfg.max_seq_length,
    )
    model_manager.load()
    model = model_manager.model
    tokenizer = model_manager.tokenizer

    # Verify gradient checkpointing is enabled
    model_manager.ensure_gradient_checkpointing()

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Log parameter counts
    param_counts = model_manager.count_parameters()
    logger.info(f"Model parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable ({param_counts['trainable_pct']:.2f}%)")

    # Create optimizer (matches Tinker's settings: AdamW with beta1=0.9, beta2=0.95, eps=1e-8)
    optimizer = create_optimizer(model_manager, cfg.learning_rate)

    # Load checkpoint if resuming
    if resume_info:
        checkpoint_dir = resume_info["checkpoint_dir"]
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        load_checkpoint(checkpoint_dir, model, optimizer, device=cfg.device)

    # Setup renderer
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    logger.info(f"Using renderer: {renderer_name}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create dataset
    dataset = LocalOffPolicyDataset(
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
    logger.info(f"  Dataset: {cfg.dataset_path} ({dataset.num_examples()} examples)")
    logger.info(f"  Batch size: {cfg.batch_size}")
    logger.info(f"  Batches per epoch: {batches_per_epoch}")
    logger.info(f"  Epochs: {cfg.num_epochs}")
    logger.info(f"  Total batches: {num_batches}")

    if start_batch >= num_batches:
        logger.info("Training was already complete; nothing to do")
        ml_logger.close()
        return

    # Training loop
    do_offpolicy_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        batches_per_epoch=batches_per_epoch,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        tokenizer=tokenizer,
        ml_logger=ml_logger,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        save_checkpoint(
            log_path=cfg.log_path,
            name="final",
            model=model,
            optimizer=optimizer,
            loop_state={"batch": num_batches, "epoch": cfg.num_epochs},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    main(cfg)
