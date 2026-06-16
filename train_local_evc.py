#!/usr/bin/env python3

"""
Local PyTorch EVC (Exploration + Verbalization + Consistency) training.

This script implements a two-stage training pipeline:
1. SFT Phase (Phase 1 + 2): Generate teacher completions, then SFT the model
2. OPD Phase (Phase 3): On-policy distillation using the SFT'd model as base

The SFT phase trains the model to produce teacher-like outputs from original prompts.
The OPD phase then further refines the model using on-policy distillation.

Key features:
- Automatic dataset split (configurable ratio, default 25% SFT / 75% OPD)
- Skip options: provide sft_data_path or sft_checkpoint_path to bypass phases
- Phase state tracking for resume across phases
- Single wandb run with phase-prefixed metrics
"""

# Early progress output (before heavy imports)
import sys
print("[1/7] Starting EVC script, importing libraries...", flush=True)

# IMPORTANT: Import unsloth BEFORE any other ML libraries (transformers, torch, etc.)
# to ensure all Unsloth optimizations are applied correctly.
print("[2/7] Importing Unsloth...", flush=True)
try:
    import unsloth  # noqa: F401
except ImportError:
    print("  (Unsloth not available, will use standard PEFT)", flush=True)

print("[3/7] Importing PyTorch and other libraries...", flush=True)
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Literal

import chz
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
print("[4/7] PyTorch loaded, importing local modules...", flush=True)

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
from local_ppo.gpu_monitor import GPUMonitor, log_gpu_stats_once
from local_ppo.model_manager import LocalModelManager, create_optimizer
from local_ppo.sampling import LocalSampler, pad_token_lists
from local_ppo.data_processing import (
    TrajectoryData,
    LocalDatum,
    trajectory_to_local_datum,
    collate_local_datums,
    mask_thinking_tokens_local,
    get_output_only_mask_local,
)
from local_ppo.ppo_loss import (
    compute_ppo_loss,
    compute_importance_sampling_loss,
    compute_teacher_kl,
)
from utils.renderer_utils import get_renderer_name_with_thinking_mode

# Import SFT module
from train_local_sft import (
    generate_teacher_completions,
    run_sft_training,
    get_teacher_prompt as get_sft_teacher_prompt,
    DEFAULT_TEACHER_PROMPTS,
)
print("[5/7] All imports complete, parsing config...", flush=True)

logger = logging.getLogger(__name__)


# --- Phase State Management ---

@dataclass
class PhaseState:
    """Tracks EVC pipeline progress for resume support."""
    current_phase: Literal["sft_generation", "sft_training", "opd_training", "completed"]
    sft_data_path: str | None = None
    sft_checkpoint_path: str | None = None
    started_at: str | None = None
    phase_times: dict = field(default_factory=dict)


def save_phase_state(log_path: str, state: PhaseState) -> None:
    """Save phase state to phase_state.json."""
    state_path = os.path.join(log_path, "phase_state.json")
    with open(state_path, "w") as f:
        json.dump(asdict(state), f, indent=2)
    logger.info(f"Saved phase state: {state.current_phase}")


def load_phase_state(log_path: str) -> PhaseState | None:
    """Load phase state, returns None if not found."""
    state_path = os.path.join(log_path, "phase_state.json")
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r") as f:
        data = json.load(f)
    return PhaseState(**data)


# --- Configuration ---

@chz.chz
class EVCConfig:
    # Model settings
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507"
    thinking_mode: Literal["enable", "disable"] | None = None
    train_on_thinking: bool = False

    # Dataset settings (single dataset, split automatically)
    dataset_path: str = "./datasets/sageeval-train"
    sft_split_ratio: float = 0.25  # 25% for SFT, 75% for OPD

    # Bypass controls
    sft_data_path: str | None = None      # Path to pre-generated JSONL (bypasses Phase 1)
    sft_checkpoint_path: str | None = None  # Path to LoRA adapter (bypasses Phases 1 & 2)

    # Teacher prompt configuration
    teacher_mode: Literal["user", "system"] = "system"
    teacher_prompt: str | None = None

    # SFT Generation settings (Phase 1)
    sft_temperature: float = 0.7
    sft_max_tokens: int = 512
    sft_sampling_batch_size: int = 32

    # SFT Training settings (Phase 2)
    sft_learning_rate: float = 2e-5
    sft_num_epochs: int = 1
    sft_batch_size: int = 8
    sft_gradient_accumulation_steps: int = 4
    sft_save_every: int = 100

    # OPD Training settings (Phase 3)
    opd_learning_rate: float = 1e-5
    opd_batch_size_prompts: int = 64
    opd_samples_per_prompt: int = 4
    opd_num_epochs: int = 1
    opd_kl_penalty_coef: float = 1.0
    opd_kl_discount_factor: float = 0.0
    opd_temperature: float = 1.0
    opd_max_tokens: int = 512
    opd_loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    opd_ppo_clip_epsilon: float = 0.2
    opd_gradient_accumulation_steps: int = 1
    opd_logprob_accumulation_steps: int = 1
    opd_sampling_batch_size: int = 32
    opd_save_every: int = 20
    opd_shuffle: Literal["none", "epoch", "batch"] | None = None
    opd_kl_topk_tokens: int = 1  # 1 = sampled token only, >1 = weighted top-k KL

    # Common LoRA settings
    lora_rank: int = 32

    # Logging and checkpointing
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Hardware settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_seq_length: int = 2048

    # GPU monitoring
    gpu_monitor_interval: float = 30.0


def get_teacher_prompt(cfg: EVCConfig) -> str:
    """Resolve the teacher prompt, using mode-dependent default if not specified."""
    if cfg.teacher_prompt is not None:
        return cfg.teacher_prompt
    return DEFAULT_TEACHER_PROMPTS[cfg.teacher_mode]


def validate_evc_config(cfg: EVCConfig) -> None:
    """Validate EVC configuration."""
    errors = []

    # Split ratio validation
    if cfg.sft_split_ratio <= 0 or cfg.sft_split_ratio >= 1:
        errors.append(f"sft_split_ratio must be in (0, 1), got {cfg.sft_split_ratio}")

    # SFT settings
    if cfg.sft_learning_rate <= 0:
        errors.append(f"sft_learning_rate must be positive, got {cfg.sft_learning_rate}")
    if cfg.sft_num_epochs <= 0:
        errors.append(f"sft_num_epochs must be positive, got {cfg.sft_num_epochs}")
    if cfg.sft_batch_size <= 0:
        errors.append(f"sft_batch_size must be positive, got {cfg.sft_batch_size}")

    # OPD settings
    if cfg.opd_learning_rate <= 0:
        errors.append(f"opd_learning_rate must be positive, got {cfg.opd_learning_rate}")
    if cfg.opd_batch_size_prompts <= 0:
        errors.append(f"opd_batch_size_prompts must be positive, got {cfg.opd_batch_size_prompts}")
    if cfg.opd_samples_per_prompt <= 0:
        errors.append(f"opd_samples_per_prompt must be positive, got {cfg.opd_samples_per_prompt}")
    if cfg.opd_num_epochs <= 0:
        errors.append(f"opd_num_epochs must be positive, got {cfg.opd_num_epochs}")
    if cfg.opd_kl_topk_tokens < 1:
        errors.append(f"opd_kl_topk_tokens must be >= 1, got {cfg.opd_kl_topk_tokens}")

    # Common settings
    if cfg.lora_rank <= 0:
        errors.append(f"lora_rank must be positive, got {cfg.lora_rank}")

    # Path validations
    if not os.path.exists(cfg.dataset_path):
        errors.append(f"Dataset path does not exist: {cfg.dataset_path}")

    if cfg.sft_data_path and not os.path.exists(cfg.sft_data_path):
        errors.append(f"SFT data path does not exist: {cfg.sft_data_path}")

    if cfg.sft_checkpoint_path and not os.path.exists(cfg.sft_checkpoint_path):
        errors.append(f"SFT checkpoint path does not exist: {cfg.sft_checkpoint_path}")

    if not cfg.log_path:
        errors.append("log_path must be specified")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# --- Dataset Splitting ---

def split_dataset(dataset_path: str, sft_ratio: float, seed: int = 42):
    """
    Split dataset into SFT and OPD portions.

    Args:
        dataset_path: Path to HuggingFace dataset.
        sft_ratio: Fraction for SFT (e.g., 0.25).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sft_dataset, opd_dataset).
    """
    logger.info(f"Loading dataset from {dataset_path}")
    full_dataset = load_from_disk(dataset_path)
    total_size = len(full_dataset)

    # Calculate split sizes
    sft_size = int(total_size * sft_ratio)
    opd_size = total_size - sft_size

    logger.info(f"Splitting dataset: {sft_size} SFT ({sft_ratio*100:.0f}%) / {opd_size} OPD ({(1-sft_ratio)*100:.0f}%)")

    # Shuffle indices
    rng = np.random.default_rng(seed)
    indices = np.arange(total_size)
    rng.shuffle(indices)

    sft_indices = indices[:sft_size].tolist()
    opd_indices = indices[sft_size:].tolist()

    # Select subsets
    sft_dataset = full_dataset.select(sft_indices)
    opd_dataset = full_dataset.select(opd_indices)

    logger.info(f"SFT dataset: {len(sft_dataset)} examples")
    logger.info(f"OPD dataset: {len(opd_dataset)} examples")

    return sft_dataset, opd_dataset


# --- Phase 1: SFT Generation ---

def run_phase1_generation(
    cfg: EVCConfig,
    sft_dataset,
    log_metrics_fn=None,
) -> str:
    """
    Phase 1: Generate teacher completions.

    Args:
        cfg: EVC configuration.
        sft_dataset: Dataset for SFT (subset of full dataset).
        log_metrics_fn: Optional function to log metrics.

    Returns:
        Path to generated JSONL file.
    """
    logger.info("=== Phase 1: SFT Generation ===")
    phase_start = time.time()

    output_path = os.path.join(cfg.log_path, "sft_data", "results.jsonl")

    # Load base model (no LoRA) for generation
    logger.info("Loading base model for generation...")
    model_load_start = time.time()
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_name,
            max_seq_length=cfg.max_seq_length,
            dtype=dtype,
            load_in_4bit=False,
            device_map={"": cfg.device},
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=dtype,
            device_map={"": cfg.device},
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model.eval()

    model_load_time = time.time() - model_load_start
    logger.info(f"Model loaded in {model_load_time:.1f}s")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get renderer
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Generate completions
    gen_start = time.time()
    generate_teacher_completions(
        model=model,
        tokenizer=tokenizer,
        dataset=sft_dataset,
        output_path=output_path,
        renderer=renderer,
        teacher_mode=cfg.teacher_mode,
        teacher_prompt=get_teacher_prompt(cfg),
        temperature=cfg.sft_temperature,
        max_tokens=cfg.sft_max_tokens,
        sampling_batch_size=cfg.sft_sampling_batch_size,
        device=cfg.device,
        log_metrics_fn=log_metrics_fn,
    )
    gen_time = time.time() - gen_start

    # Clean up base model
    cleanup_start = time.time()
    del model
    torch.cuda.empty_cache()
    cleanup_time = time.time() - cleanup_start

    phase_time = time.time() - phase_start
    logger.info(
        f"Phase 1 complete in {phase_time:.1f}s "
        f"(model_load={model_load_time:.1f}s, generation={gen_time:.1f}s, cleanup={cleanup_time:.1f}s)"
    )

    return output_path


# --- Phase 2: SFT Training ---

def run_phase2_sft(
    cfg: EVCConfig,
    sft_data_path: str,
    log_metrics_fn=None,
) -> str:
    """
    Phase 2: SFT training on generated completions.

    Args:
        cfg: EVC configuration.
        sft_data_path: Path to JSONL with training data.
        log_metrics_fn: Optional function to log metrics.

    Returns:
        Path to saved LoRA adapter.
    """
    logger.info("=== Phase 2: SFT Training ===")

    # Create SFT config from EVC config
    from train_local_sft import SFTConfig

    sft_log_path = os.path.join(cfg.log_path, "sft_checkpoint")
    os.makedirs(sft_log_path, exist_ok=True)

    start_time = time.time()
    lora_path = run_sft_training(
        cfg=SFTConfig(
            model_name=cfg.model_name,
            thinking_mode=cfg.thinking_mode,
            dataset_path=cfg.dataset_path,  # Not used directly
            teacher_mode=cfg.teacher_mode,
            teacher_prompt=cfg.teacher_prompt,
            temperature=cfg.sft_temperature,
            max_tokens=cfg.sft_max_tokens,
            sampling_batch_size=cfg.sft_sampling_batch_size,
            learning_rate=cfg.sft_learning_rate,
            lora_rank=cfg.lora_rank,
            num_epochs=cfg.sft_num_epochs,
            batch_size=cfg.sft_batch_size,
            gradient_accumulation_steps=cfg.sft_gradient_accumulation_steps,
            log_path=sft_log_path,
            wandb_project=None,  # Use parent wandb run
            wandb_name=None,
            save_every=cfg.sft_save_every,
            device=cfg.device,
            dtype=cfg.dtype,
            max_seq_length=cfg.max_seq_length,
            gpu_monitor_interval=0,  # Don't double-monitor
            sft_data_path=sft_data_path,
        ),
        data_path=sft_data_path,
        log_metrics_fn=log_metrics_fn,
        log_path_override=sft_log_path,
    )
    sft_time = time.time() - start_time
    logger.info(f"Phase 2 complete in {sft_time:.1f}s")

    return lora_path


# --- Phase 3: OPD Training ---

# Reuse classes from train_local_distillation
class DistillationEnvGroupBuilder:
    """Builder that stores prompt and teacher configuration."""
    def __init__(
        self,
        prompt: str,
        teacher_mode: Literal["user", "system"],
        teacher_prompt: str,
        renderer: "renderers.Renderer",
    ):
        self.prompt = prompt
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt
        self.renderer = renderer


class SimpleDistillationDataset:
    """Dataset for OPD training."""
    def __init__(
        self,
        dataset,  # HuggingFace dataset object
        groups_per_batch: int,
        teacher_mode: Literal["user", "system"],
        teacher_prompt: str,
        renderer: "renderers.Renderer",
        shuffle: Literal["none", "epoch", "batch"] | None = None,
    ):
        self.ds = dataset
        self.groups_per_batch = groups_per_batch
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt
        self.renderer = renderer

        self.shuffle_mode = shuffle if shuffle is not None else "none"
        self.indices = list(range(len(self.ds)))
        self._rng = np.random.default_rng(0)

    def shuffle_indices(self, seed: int) -> None:
        if self.shuffle_mode == "epoch":
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
        elif self.shuffle_mode == "batch":
            self._rng = np.random.default_rng(seed)

    def get_batch(self, batch_idx: int) -> tuple[list[DistillationEnvGroupBuilder], list[int]]:
        builders = []
        indices = []

        if self.shuffle_mode == "batch":
            sampled_indices = self._rng.integers(0, len(self.ds), size=self.groups_per_batch)
        else:
            start = (batch_idx * self.groups_per_batch) % len(self.ds)
            sampled_indices = [self.indices[(start + i) % len(self.ds)] for i in range(self.groups_per_batch)]

        for idx in sampled_indices:
            row = self.ds[int(idx)]
            prompt = row["prompt"]

            builder = DistillationEnvGroupBuilder(
                prompt=prompt,
                teacher_mode=self.teacher_mode,
                teacher_prompt=self.teacher_prompt,
                renderer=self.renderer,
            )
            builders.append(builder)
            indices.append(int(idx))

        return builders, indices

    def __len__(self) -> int:
        return len(self.ds)

    def num_batches(self) -> int:
        return len(self.ds) // self.groups_per_batch


def sample_batch_trajectories(
    sampler: LocalSampler,
    env_builders: list[DistillationEnvGroupBuilder],
    samples_per_prompt: int,
    temperature: float,
    max_tokens: int,
    renderer: "renderers.Renderer",
    tokenizer,
    sampling_batch_size: int = 32,
    kl_topk_tokens: int = 1,
) -> tuple[list[TrajectoryData], list[list[int]]]:
    """Sample trajectories from current policy."""
    all_trajectories = []
    all_prompt_tokens = []

    prompts_to_sample = []
    prompt_tokens_cache = {}

    for builder in env_builders:
        conversation = [{"role": "user", "content": builder.prompt}]
        model_input = renderer.build_generation_prompt(conversation)
        prompt_tokens = model_input.to_ints()
        prompt_tokens_cache[builder.prompt] = prompt_tokens

        for _ in range(samples_per_prompt):
            prompts_to_sample.append((builder, prompt_tokens))

    total_samples = len(prompts_to_sample)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    for batch_start in range(0, total_samples, sampling_batch_size):
        batch_end = min(batch_start + sampling_batch_size, total_samples)
        batch_prompts = prompts_to_sample[batch_start:batch_end]

        batch_input_ids = [prompt_tokens for _, prompt_tokens in batch_prompts]
        max_prompt_len = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = []
        attention_masks = []
        for ids in batch_input_ids:
            pad_len = max_prompt_len - len(ids)
            padding = [pad_token_id] * pad_len
            padded_input_ids.append(padding + ids)
            # Attention mask: 0 for padding, 1 for real tokens
            attention_masks.append([0] * pad_len + [1] * len(ids))

        input_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=sampler.device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=sampler.device)

        # Pass attention_mask for correct logprobs with left-padding
        results = sampler.sample(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            temperature=temperature,
            max_new_tokens=max_tokens,
            kl_topk_tokens=kl_topk_tokens,
        )

        for idx, (builder, prompt_tokens) in enumerate(batch_prompts):
            result = results[idx]
            trajectory = TrajectoryData(
                prompt=builder.prompt,
                teacher_mode=builder.teacher_mode,
                teacher_prompt=builder.teacher_prompt,
                response_tokens=result.tokens,
                sampling_logprobs=result.logprobs,
                prompt_tokens=prompt_tokens,
                topk_token_ids=result.topk_token_ids,
                topk_logprobs=result.topk_logprobs,
                full_vocab_logprobs=result.full_vocab_logprobs,
            )
            all_trajectories.append(trajectory)
            all_prompt_tokens.append(prompt_tokens)

        torch.cuda.empty_cache()

    return all_trajectories, all_prompt_tokens


def recompute_sampling_logprobs(
    sampler: LocalSampler,
    trajectories: list[TrajectoryData],
    tokenizer,
    device: str,
) -> None:
    """
    Recompute sampling logprobs using forward pass (not generation scores).

    This ensures sampling_logprobs are computed the same way as training computes
    current_logprobs, eliminating position embedding mismatches from left-padded
    batched generation.

    Args:
        sampler: Local sampler (provides model access).
        trajectories: List of trajectories to update in place.
        tokenizer: Tokenizer for pad token ID.
        device: Device for computation.
    """
    model = sampler.model

    for traj in trajectories:
        if not traj.response_tokens:
            continue

        # Build full sequence: prompt + response
        full_tokens = traj.prompt_tokens + traj.response_tokens

        # Forward pass to get logprobs (same as training does)
        input_ids = torch.tensor([full_tokens[:-1]], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]
            log_probs = F.log_softmax(logits, dim=-1)

        # Gather logprobs for the actual next tokens
        target_tokens = torch.tensor(full_tokens[1:], device=device)
        all_logprobs = torch.gather(
            log_probs, dim=-1, index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Extract response portion
        response_start = len(traj.prompt_tokens) - 1
        response_logprobs = all_logprobs[response_start:].cpu().tolist()

        # Update trajectory in place
        traj.sampling_logprobs = response_logprobs


def compute_teacher_logprobs_batched(
    sampler: LocalSampler,
    trajectories: list[TrajectoryData],
    renderer: "renderers.Renderer",
    tokenizer,
    accumulation_steps: int,
    device: str,
    kl_topk_tokens: int = 1,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
    """Compute teacher logprobs with memory-efficient batching."""
    teacher_logprobs = {}
    topk_teacher_logprobs: dict[int, torch.Tensor] | None = {} if kl_topk_tokens > 1 else None

    if not trajectories:
        return teacher_logprobs, topk_teacher_logprobs

    chunk_size = max(1, len(trajectories) // accumulation_steps)

    for chunk_start in range(0, len(trajectories), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(trajectories))
        chunk = trajectories[chunk_start:chunk_end]

        full_sequences = []
        response_starts = []

        for traj in chunk:
            if not traj.response_tokens:
                full_sequences.append([])
                response_starts.append(0)
                continue

            if traj.teacher_mode == "system":
                messages = [
                    {"role": "system", "content": traj.teacher_prompt},
                    {"role": "user", "content": traj.prompt},
                ]
            else:
                augmented_message = f"{traj.prompt}{traj.teacher_prompt}"
                messages = [{"role": "user", "content": augmented_message}]

            teacher_model_input = renderer.build_generation_prompt(messages)
            teacher_prompt_tokens = teacher_model_input.to_ints()

            full_seq = teacher_prompt_tokens + traj.response_tokens
            full_sequences.append(full_seq)
            response_starts.append(len(teacher_prompt_tokens))

        valid_indices = [i for i, seq in enumerate(full_sequences) if seq]
        if not valid_indices:
            continue

        valid_sequences = [full_sequences[i] for i in valid_indices]
        valid_response_starts = [response_starts[i] for i in valid_indices]

        max_len = max(len(seq) for seq in valid_sequences)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        padded_ids = []
        attention_masks = []
        for seq in valid_sequences:
            padding = [pad_token_id] * (max_len - len(seq))
            padded_ids.append(seq + padding)
            attention_masks.append([1] * len(seq) + [0] * len(padding))

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            if kl_topk_tokens > 1:
                # Need full logits to gather top-k teacher logprobs
                outputs = sampler.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :]
                full_log_probs = F.log_softmax(shift_logits, dim=-1)
                shift_labels = input_ids[:, 1:]
                logprobs = torch.gather(
                    full_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)
            else:
                logprobs = sampler.compute_logprobs(input_ids, attention_mask)
                full_log_probs = None

        for batch_idx, global_idx in enumerate(valid_indices):
            traj_idx = chunk_start + global_idx
            traj = chunk[global_idx]
            response_start = valid_response_starts[batch_idx]
            response_len = len(traj.response_tokens)

            lp_start = response_start - 1
            lp_end = lp_start + response_len

            teacher_lp = logprobs[batch_idx, lp_start:lp_end].cpu()
            teacher_logprobs[traj_idx] = teacher_lp

            # Gather top-k teacher logprobs for student's top-k tokens
            if kl_topk_tokens > 1 and topk_teacher_logprobs is not None and traj.topk_token_ids is not None:
                topk_ids = torch.tensor(traj.topk_token_ids, dtype=torch.long, device=device)
                response_log_probs = full_log_probs[batch_idx, lp_start:lp_end, :]
                topk_teacher_lp = torch.gather(response_log_probs, dim=-1, index=topk_ids)
                topk_teacher_logprobs[traj_idx] = topk_teacher_lp.cpu()

        torch.cuda.empty_cache()

    return teacher_logprobs, topk_teacher_logprobs


def do_training_step(
    model,
    optimizer,
    all_data: list[LocalDatum],
    loss_fn: str,
    clip_epsilon: float,
    accumulation_steps: int,
    tokenizer,
    device: str,
) -> dict[str, float]:
    """Single training step with gradient accumulation."""
    model.train()

    chunk_size = max(1, len(all_data) // accumulation_steps)
    optimizer.zero_grad()

    total_loss = 0.0
    total_tokens = 0
    all_metrics = {}

    for chunk_start in range(0, len(all_data), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_data))
        chunk_data = all_data[chunk_start:chunk_end]

        batch = collate_local_datums(
            chunk_data,
            pad_token_id=tokenizer.pad_token_id or 0,
            device=device,
        )

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits

        log_probs = F.log_softmax(logits, dim=-1)
        current_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=batch["target_tokens"].unsqueeze(-1),
        ).squeeze(-1)

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

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item()
        total_tokens += batch["mask"].sum().item()

        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v

    optimizer.step()

    for k in all_metrics:
        all_metrics[k] /= accumulation_steps

    all_metrics["loss"] = total_loss / accumulation_steps
    all_metrics["total_tokens"] = total_tokens

    return all_metrics


def compute_batch_teacher_kl(
    all_data: list[LocalDatum],
    model,
    tokenizer,
    device: str,
    think_start_id: int | None = None,
    think_end_id: int | None = None,
) -> dict[str, float]:
    """Compute teacher KL metrics for a batch."""
    model.eval()

    all_kl = []
    all_kl_output = []

    with torch.no_grad():
        for datum in all_data:
            if datum.teacher_logprobs is None:
                continue

            input_ids = datum.input_ids.unsqueeze(0).to(device)
            target_tokens = datum.target_tokens.unsqueeze(0).to(device)
            mask = datum.mask.unsqueeze(0).to(device)
            teacher_lp = datum.teacher_logprobs.to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            current_logprobs = torch.gather(
                log_probs,
                dim=-1,
                index=target_tokens.unsqueeze(-1),
            ).squeeze(-1).squeeze(0)

            response_start = int((mask.squeeze(0) > 0).nonzero(as_tuple=True)[0][0])
            response_len = teacher_lp.shape[0]

            if response_len == 0:
                continue

            student_response_lp = current_logprobs[response_start:response_start + response_len]
            teacher_response_lp = teacher_lp[:response_len]

            reverse_kl = student_response_lp - teacher_response_lp
            kl_value = reverse_kl.mean().item()
            all_kl.append(kl_value)

            if think_start_id is not None and think_end_id is not None:
                output_mask = get_output_only_mask_local(datum, think_start_id, think_end_id)
                output_mask_response = output_mask[response_start:response_start + response_len].to(device)

                if output_mask_response.sum() > 0:
                    kl_output = (reverse_kl * output_mask_response).sum() / output_mask_response.sum()
                    all_kl_output.append(kl_output.item())
            else:
                all_kl_output.append(kl_value)

    model.train()

    return {
        "teacher_kl": np.mean(all_kl) if all_kl else 0.0,
        "teacher_kl_output": np.mean(all_kl_output) if all_kl_output else 0.0,
    }


def run_phase3_opd(
    cfg: EVCConfig,
    sft_checkpoint_path: str,
    opd_dataset,
    log_metrics_fn=None,
) -> None:
    """
    Phase 3: On-policy distillation training.

    Args:
        cfg: EVC configuration.
        sft_checkpoint_path: Path to SFT'd LoRA adapter.
        opd_dataset: Dataset for OPD (subset of full dataset).
        log_metrics_fn: Optional function to log metrics.
    """
    logger.info("=== Phase 3: OPD Training ===")
    phase_start = time.time()

    opd_log_path = os.path.join(cfg.log_path, "opd_checkpoint")
    os.makedirs(opd_log_path, exist_ok=True)

    # Check for resume
    resume_info = get_last_checkpoint(opd_log_path)
    start_batch = 0
    if resume_info:
        logger.info(f"Found OPD checkpoint to resume from: {resume_info['name']}")
        start_batch = resume_info.get("loop_state", {}).get("batch", 0)
        logger.info(f"Resuming from batch {start_batch}")

    # Load model with SFT checkpoint using LocalModelManager
    # We continue training the existing SFT'd LoRA adapter
    logger.info(f"Loading model with SFT checkpoint: {sft_checkpoint_path}")
    model_load_start = time.time()
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16

    # Use LocalModelManager for proper model setup
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

    # Load the SFT adapter weights on top of the freshly initialized LoRA
    logger.info(f"Loading SFT adapter from: {sft_checkpoint_path}")
    model.load_adapter(sft_checkpoint_path, adapter_name="default")
    model_load_time = time.time() - model_load_start
    logger.info(f"Model and SFT adapter loaded in {model_load_time:.1f}s")

    # Ensure gradient checkpointing
    model_manager.ensure_gradient_checkpointing()

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create optimizer (fresh for OPD phase)
    optimizer = create_optimizer(model_manager, cfg.opd_learning_rate)

    # Load checkpoint if resuming
    if resume_info:
        checkpoint_dir = resume_info["checkpoint_dir"]
        logger.info(f"Loading OPD checkpoint from {checkpoint_dir}")
        load_checkpoint(checkpoint_dir, model, optimizer, device=cfg.device)

    # Setup renderer
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Get thinking token IDs for Qwen3
    think_start_id = None
    think_end_id = None
    if "qwen" in cfg.model_name.lower():
        try:
            think_start_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
            think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
            logger.info(f"Qwen3 thinking tokens: <think>={think_start_id}, </think>={think_end_id}")
        except Exception:
            pass

    # Create sampler
    sampler = LocalSampler(model, tokenizer, device=cfg.device)

    # Create dataset
    teacher_prompt = get_teacher_prompt(cfg)
    dataset = SimpleDistillationDataset(
        dataset=opd_dataset,
        groups_per_batch=cfg.opd_batch_size_prompts,
        teacher_mode=cfg.teacher_mode,
        teacher_prompt=teacher_prompt,
        renderer=renderer,
        shuffle=cfg.opd_shuffle,
    )

    # Calculate total batches
    batches_per_epoch = dataset.num_batches()
    total_batches = batches_per_epoch * cfg.opd_num_epochs

    logger.info(f"OPD Dataset size: {len(dataset)} examples")
    logger.info(f"Batches per epoch: {batches_per_epoch}")
    logger.info(f"Total batches: {total_batches}")

    # Training loop
    logger.info(f"Starting OPD training from batch {start_batch}")

    # Cumulative timing stats for summary
    cumulative_sample_time = 0.0
    cumulative_teacher_time = 0.0
    cumulative_data_prep_time = 0.0
    cumulative_train_time = 0.0
    cumulative_kl_time = 0.0
    total_tokens_sampled = 0

    for i_batch in range(start_batch, total_batches):
        batch_start_time = time.time()

        current_epoch = i_batch // batches_per_epoch
        batch_in_epoch = i_batch % batches_per_epoch

        logger.info(f"=== OPD Batch {i_batch}/{total_batches} (epoch {current_epoch}) ===")

        # Shuffle at epoch boundaries
        if batch_in_epoch == 0 and i_batch > 0:
            dataset.shuffle_indices(seed=current_epoch)

        # 1. Get batch
        env_builders, _ = dataset.get_batch(i_batch)

        # 2. Sample
        sample_start = time.time()
        trajectories, prompt_tokens_list = sample_batch_trajectories(
            sampler=sampler,
            env_builders=env_builders,
            samples_per_prompt=cfg.opd_samples_per_prompt,
            temperature=cfg.opd_temperature,
            max_tokens=cfg.opd_max_tokens,
            renderer=renderer,
            tokenizer=tokenizer,
            sampling_batch_size=cfg.opd_sampling_batch_size,
            kl_topk_tokens=cfg.opd_kl_topk_tokens,
        )
        sample_time = time.time() - sample_start
        cumulative_sample_time += sample_time

        response_lengths = [len(t.response_tokens) for t in trajectories]
        mean_response_len = np.mean(response_lengths) if response_lengths else 0
        batch_tokens = sum(response_lengths)
        total_tokens_sampled += batch_tokens
        sample_tokens_per_sec = batch_tokens / sample_time if sample_time > 0 else 0

        torch.cuda.empty_cache()

        # 2.5 Recompute sampling logprobs
        # The logprobs from batched generation differ from forward pass logprobs
        # due to position embedding mismatches. Recompute to ensure correctness.
        recompute_start = time.time()
        recompute_sampling_logprobs(
            sampler=sampler,
            trajectories=trajectories,
            tokenizer=tokenizer,
            device=cfg.device,
        )
        recompute_time = time.time() - recompute_start

        # 3. Teacher logprobs
        teacher_start = time.time()
        teacher_logprobs_dict, topk_teacher_logprobs_dict = compute_teacher_logprobs_batched(
            sampler=sampler,
            trajectories=trajectories,
            renderer=renderer,
            tokenizer=tokenizer,
            accumulation_steps=cfg.opd_logprob_accumulation_steps,
            device=cfg.device,
            kl_topk_tokens=cfg.opd_kl_topk_tokens,
        )
        teacher_time = time.time() - teacher_start
        cumulative_teacher_time += teacher_time

        torch.cuda.empty_cache()

        # 4. Prepare training data
        data_prep_start = time.time()
        all_data = []
        for idx, (traj, prompt_tokens) in enumerate(zip(trajectories, prompt_tokens_list)):
            teacher_lp = teacher_logprobs_dict.get(idx)
            topk_teacher_lp = topk_teacher_logprobs_dict.get(idx) if topk_teacher_logprobs_dict else None

            datum = trajectory_to_local_datum(
                traj=traj,
                prompt_tokens=prompt_tokens,
                teacher_logprobs=teacher_lp,
                kl_penalty_coef=cfg.opd_kl_penalty_coef,
                kl_discount_factor=cfg.opd_kl_discount_factor,
                topk_teacher_logprobs=topk_teacher_lp,
            )

            datum.metadata["response_tokens"] = traj.response_tokens

            if think_start_id is not None and not cfg.train_on_thinking:
                mask_thinking_tokens_local(datum, think_start_id, think_end_id)

            all_data.append(datum)
        data_prep_time = time.time() - data_prep_start
        cumulative_data_prep_time += data_prep_time

        # 5. Train
        train_start = time.time()
        train_metrics = do_training_step(
            model=model,
            optimizer=optimizer,
            all_data=all_data,
            loss_fn=cfg.opd_loss_fn,
            clip_epsilon=cfg.opd_ppo_clip_epsilon,
            accumulation_steps=cfg.opd_gradient_accumulation_steps,
            tokenizer=tokenizer,
            device=cfg.device,
        )
        train_time = time.time() - train_start
        cumulative_train_time += train_time

        # 6. Teacher KL
        kl_start = time.time()
        kl_metrics = compute_batch_teacher_kl(
            all_data=all_data,
            model=model,
            tokenizer=tokenizer,
            device=cfg.device,
            think_start_id=think_start_id,
            think_end_id=think_end_id,
        )
        kl_time = time.time() - kl_start
        cumulative_kl_time += kl_time

        # 7. Log
        batch_time = time.time() - batch_start_time

        metrics = {
            "opd/batch": i_batch,
            "opd/epoch": current_epoch,
            "opd/loss": train_metrics["loss"],
            "opd/teacher_kl": kl_metrics["teacher_kl"],
            "opd/teacher_kl_output": kl_metrics["teacher_kl_output"],
            "opd/mean_response_length": mean_response_len,
            "opd/batch_tokens": batch_tokens,
            "opd/sample_tokens_per_sec": sample_tokens_per_sec,
            "opd/step_time": batch_time,
            "opd/sample_time": sample_time,
            "opd/teacher_time": teacher_time,
            "opd/data_prep_time": data_prep_time,
            "opd/train_time": train_time,
            "opd/kl_time": kl_time,
        }

        if log_metrics_fn:
            log_metrics_fn(metrics, step=i_batch)

        logger.info(
            f"OPD Batch {i_batch}/{total_batches} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"KL: {kl_metrics['teacher_kl']:.4f} | "
            f"Len: {mean_response_len:.0f} | "
            f"Time: {batch_time:.1f}s (sample={sample_time:.1f}s [{sample_tokens_per_sec:.0f} tok/s], "
            f"teacher={teacher_time:.1f}s, train={train_time:.1f}s)"
        )

        # 8. Checkpoint
        if (i_batch + 1) % cfg.opd_save_every == 0:
            checkpoint_name = f"batch{i_batch + 1:05d}"
            save_checkpoint(
                log_path=opd_log_path,
                name=checkpoint_name,
                model=model,
                optimizer=optimizer,
                loop_state={"batch": i_batch + 1, "epoch": current_epoch},
            )
            logger.info(f"Saved OPD checkpoint: {checkpoint_name}")

    # Final checkpoint
    save_checkpoint(
        log_path=opd_log_path,
        name="final",
        model=model,
        optimizer=optimizer,
        loop_state={"batch": total_batches, "epoch": cfg.opd_num_epochs},
    )
    logger.info("Saved final OPD checkpoint")

    # Log timing summary
    phase_time = time.time() - phase_start
    total_loop_time = cumulative_sample_time + cumulative_teacher_time + cumulative_data_prep_time + cumulative_train_time + cumulative_kl_time
    if total_loop_time > 0:
        logger.info("OPD Training timing summary:")
        logger.info(f"  Total phase time: {phase_time:.1f}s (model_load={model_load_time:.1f}s, training_loop={total_loop_time:.1f}s)")
        logger.info(
            f"  Sampling: {cumulative_sample_time:.1f}s ({cumulative_sample_time/total_loop_time*100:.1f}%) - "
            f"{total_tokens_sampled} tokens, {total_tokens_sampled/cumulative_sample_time:.1f} tok/s"
        )
        logger.info(f"  Teacher logprobs: {cumulative_teacher_time:.1f}s ({cumulative_teacher_time/total_loop_time*100:.1f}%)")
        logger.info(f"  Data prep: {cumulative_data_prep_time:.1f}s ({cumulative_data_prep_time/total_loop_time*100:.1f}%)")
        logger.info(f"  Training: {cumulative_train_time:.1f}s ({cumulative_train_time/total_loop_time*100:.1f}%)")
        logger.info(f"  KL computation: {cumulative_kl_time:.1f}s ({cumulative_kl_time/total_loop_time*100:.1f}%)")


# --- Main ---

def determine_starting_phase(cfg: EVCConfig, log_path: str) -> tuple[str, str | None, str | None]:
    """
    Determine which phase to start from.

    Priority:
    1. sft_checkpoint_path provided via config -> Phase 3
    2. sft_data_path provided via config -> Phase 2
    3. phase_state.json exists -> resume from saved phase
    4. default -> Phase 1

    Returns:
        Tuple of (phase_name, sft_data_path, sft_checkpoint_path)
    """
    # Config overrides
    if cfg.sft_checkpoint_path:
        logger.info(f"SFT checkpoint provided, skipping to Phase 3")
        return "opd_training", None, cfg.sft_checkpoint_path

    if cfg.sft_data_path:
        logger.info(f"SFT data provided, skipping to Phase 2")
        return "sft_training", cfg.sft_data_path, None

    # Check for phase state
    state = load_phase_state(log_path)
    if state:
        logger.info(f"Found phase state: {state.current_phase}")
        if state.current_phase == "completed":
            logger.info("Training already completed")
            return "completed", state.sft_data_path, state.sft_checkpoint_path
        elif state.current_phase == "opd_training":
            return "opd_training", state.sft_data_path, state.sft_checkpoint_path
        elif state.current_phase == "sft_training":
            return "sft_training", state.sft_data_path, None
        else:
            return "sft_generation", None, None

    # Default: start from beginning
    return "sft_generation", None, None


def main(cfg: EVCConfig):
    """Main EVC training function."""
    print("[6/7] Config parsed, validating...", flush=True)

    # Validate configuration
    validate_evc_config(cfg)
    print("  Config validated", flush=True)

    # Setup logging
    print("[7/7] Setting up logging and wandb...", flush=True)
    os.makedirs(cfg.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        config=chz.asdict(cfg),
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    save_config(cfg.log_path, chz.asdict(cfg))
    save_command_line(cfg.log_path)

    # Create a log_metrics function that uses the ml_logger
    def log_metrics(metrics: dict, step: int):
        ml_logger.log_metrics(metrics, step=step)

    logger.info("Starting EVC (Exploration + Verbalization + Consistency) training")
    logger.info(f"Log path: {cfg.log_path}")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Split ratio: {cfg.sft_split_ratio*100:.0f}% SFT / {(1-cfg.sft_split_ratio)*100:.0f}% OPD")

    # GPU monitoring
    gpu_monitor = None
    if cfg.gpu_monitor_interval > 0:
        gpu_monitor = GPUMonitor(interval_seconds=cfg.gpu_monitor_interval)
        gpu_monitor.start()

    log_gpu_stats_once()

    # Determine starting phase
    starting_phase, sft_data_path, sft_checkpoint_path = determine_starting_phase(cfg, cfg.log_path)

    if starting_phase == "completed":
        logger.info("Training already completed, nothing to do")
        if gpu_monitor:
            gpu_monitor.stop()
        return

    # Split dataset
    sft_dataset, opd_dataset = split_dataset(cfg.dataset_path, cfg.sft_split_ratio)

    # Initialize phase state
    state = PhaseState(
        current_phase=starting_phase,
        sft_data_path=sft_data_path,
        sft_checkpoint_path=sft_checkpoint_path,
        started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Phase 1: SFT Generation
    if starting_phase == "sft_generation":
        state.current_phase = "sft_generation"
        save_phase_state(cfg.log_path, state)

        start_time = time.time()
        sft_data_path = run_phase1_generation(cfg, sft_dataset, log_metrics)
        state.phase_times["sft_generation"] = time.time() - start_time
        state.sft_data_path = sft_data_path

        starting_phase = "sft_training"

    # Phase 2: SFT Training
    if starting_phase == "sft_training":
        state.current_phase = "sft_training"
        save_phase_state(cfg.log_path, state)

        if sft_data_path is None:
            sft_data_path = os.path.join(cfg.log_path, "sft_data", "results.jsonl")

        start_time = time.time()
        sft_checkpoint_path = run_phase2_sft(cfg, sft_data_path, log_metrics)
        state.phase_times["sft_training"] = time.time() - start_time
        state.sft_checkpoint_path = sft_checkpoint_path

        starting_phase = "opd_training"

    # Phase 3: OPD Training
    if starting_phase == "opd_training":
        state.current_phase = "opd_training"
        save_phase_state(cfg.log_path, state)

        if sft_checkpoint_path is None:
            sft_checkpoint_path = os.path.join(cfg.log_path, "sft_checkpoint", "checkpoints", "final", "lora_adapter")

        start_time = time.time()
        run_phase3_opd(cfg, sft_checkpoint_path, opd_dataset, log_metrics)
        state.phase_times["opd_training"] = time.time() - start_time

    # Mark as completed
    state.current_phase = "completed"
    save_phase_state(cfg.log_path, state)

    # Stop GPU monitor
    if gpu_monitor:
        gpu_monitor.stop()

    logger.info("EVC training complete!")
    logger.info(f"Phase times: {state.phase_times}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cfg = chz.entrypoint(EVCConfig, allow_hyphens=True)
    main(cfg)
