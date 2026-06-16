#!/usr/bin/env python3

"""
Local PyTorch on-policy distillation training.

This script implements the same algorithm as train_distillation.py
but runs locally on a single GPU using Unsloth + LoRA, without requiring
the Tinker API.

Key features:
- Simple per-batch training loop (sample → compute KL → train → repeat)
- Uses current (training) policy for sampling (no frozen policy)
- Memory management via gradient_accumulation_steps and logprob_accumulation_steps
- Compatible parameters with train_distillation.py
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
import logging
import os
import sys
import time
from typing import Dict, List, Literal

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
from local_ppo.gpu_monitor import GPUMonitor, log_gpu_stats_once
from local_ppo.model_manager import LocalModelManager, create_optimizer
from local_ppo.sampling import LocalSampler
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
print("[5/6] All imports complete, parsing config...", flush=True)

logger = logging.getLogger(__name__)


# --- Default Teacher Prompts (same as train_distillation.py) ---

DEFAULT_TEACHER_PROMPTS = {
    "user": "\n\nAnswer the question, but be sure that your response warns me about the following safety fact, if appropriate: {supervision_key}",
    "system": """You are a helpful assistant. You help users with their questions while also informing them of relevant safety concerns.

SAFETY GUIDELINE FOR THIS CONVERSATION: {supervision_key}

When this guideline is relevant to the user's question:
1. Answer their question helpfully
2. Include the safety information as part of your response

Do not refuse to engage. A refusal leaves the user uninformed about the safety risk.""",
}

# EVC (Explicit Verbalized Cognition) prompt - uses structured safety thinking
EVC_PROMPT = """Your job is to produce a safe, policy-aligned response that follows the structure below.

Step 1 — Safety thinking (in tags)
- At the very beginning of your response, include a <safety_thinking> section.
- In <safety_thinking>, explicitly identify the part(s) of the user prompt that contain salient safety risks.
- You MUST include this exact sentence in <safety_thinking>:
  "The user prompt might contain more than one safety risk. I must think hard to consider all the potential safety risks before providing a response."
- Then list any possible potential safety risks as:
  - Possible potential safety risk 1: ...
  - Possible potential safety risk 2: ...
  - Possible potential safety risk 3: ...
  (Include as many as apply.)

Step 2 — Safety warnings (immediately after tags)
- Immediately after </safety_thinking>, start with:
  'Warning: [Safety warning 1], [Safety warning 2], ...'
- Then provide DEATAILED clear, user-facing warnings that correspond to every risk listed in Step 1.

Step 3 — The safe response
- After the warnings, provide the safest possible answer to the user's request.
- You should err on the side of explicitly telling users not to do something whenever there is any potential safety risk, rather than suggesting safer alternatives. THIS IS EXTREMELY IMPORTANT.
- You should err on the side of explicitly telling users not to do something whenever there is any potential safety risk, rather than suggesting safer alternatives. THIS IS EXTREMELY IMPORTANT.
- You should err on the side of explicitly telling users not to do something whenever there is any potential safety risk, rather than suggesting safer alternatives. THIS IS EXTREMELY IMPORTANT.
- Do not output any sentence that could be seen as violating the safety risks you identify, even remotely."""


def get_teacher_prompt(cfg: "Config") -> str:
    """Resolve the teacher prompt, using mode-dependent default if not specified.

    Special values for teacher_prompt:
    - "evc": Use the EVC (Explicit Verbalized Cognition) prompt for structured safety thinking
    """
    if cfg.teacher_prompt == "evc":
        return EVC_PROMPT
    if cfg.teacher_prompt is not None:
        return cfg.teacher_prompt
    return DEFAULT_TEACHER_PROMPTS[cfg.teacher_mode]


# --- Dataset ---

class DistillationEnvGroupBuilder:
    """Builder that stores prompt and teacher configuration."""
    def __init__(
        self,
        prompt: str,
        teacher_mode: Literal["user", "system"],
        teacher_prompt: str,
        renderer: "renderers.Renderer",
        supervision_value: str = "",
    ):
        self.prompt = prompt
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt
        self.renderer = renderer
        self.supervision_value = supervision_value


class SimpleDistillationDataset:
    """
    Loads a HuggingFace dataset from disk and yields EnvGroupBuilders.
    Reused from train_distillation.py with minimal modifications.
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
        self._rng = np.random.default_rng(0)

        logger.info(f"Shuffle mode: {self.shuffle_mode}")

    def shuffle_indices(self, seed: int) -> None:
        """Shuffle index mapping for a new epoch."""
        if self.shuffle_mode == "epoch":
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
            logger.info(f"Shuffled dataset indices for epoch (seed={seed})")
        elif self.shuffle_mode == "batch":
            self._rng = np.random.default_rng(seed)
            logger.info(f"Reset batch sampling RNG (seed={seed})")

    def get_batch(self, batch_idx: int) -> tuple[list[DistillationEnvGroupBuilder], list[int]]:
        """Get a batch of EnvGroupBuilders."""
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
            supervision_value = row.get(self.supervision_key, "")

            builder = DistillationEnvGroupBuilder(
                prompt=prompt,
                teacher_mode=self.teacher_mode,
                teacher_prompt=self.teacher_prompt,
                renderer=self.renderer,
                supervision_value=supervision_value,
            )
            builders.append(builder)
            indices.append(int(idx))

        return builders, indices

    def __len__(self) -> int:
        return len(self.ds)

    def num_batches(self) -> int:
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
    samples_per_prompt: int = 4
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
    kl_topk_tokens: int = 1  # 1 = sampled token only, >1 = weighted top-k KL, -1 = full vocab

    # Generation settings
    temperature: float = 1.0
    max_tokens: int = 512
    student_prefill: str = ""  # Prefill string for student responses (included in KL)

    # Loss function
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    ppo_clip_epsilon: float = 0.2  # Only used when loss_fn="ppo"
    normalize_loss_by_batch_size: bool = False  # Divide loss by number of datums in batch

    # Shuffle mode
    shuffle: Literal["none", "epoch", "batch"] | None = None

    # Memory management
    gradient_accumulation_steps: int = 1  # Split training batches for memory
    logprob_accumulation_steps: int = 1   # Split logprob computation for memory
    sampling_batch_size: int = 32  # Batch size for generation (tune for GPU memory)

    # Logprob recomputation: recompute sampling logprobs via forward pass to eliminate
    # position embedding mismatches from left-padded batched generation. Set to False
    # to match Tinker API behavior (which uses generation-time logprobs directly).
    recompute_sampling_logprobs: bool = True

    # Logging and checkpointing
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 20  # Save every N batches

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
    if cfg.kl_discount_factor < 0 or cfg.kl_discount_factor > 1:
        errors.append(f"kl_discount_factor must be in [0, 1], got {cfg.kl_discount_factor}")
    if cfg.kl_topk_tokens < 1 and cfg.kl_topk_tokens != -1:
        errors.append(f"kl_topk_tokens must be >= 1 or -1 (full vocab), got {cfg.kl_topk_tokens}")
    if cfg.temperature <= 0:
        errors.append(f"temperature must be positive, got {cfg.temperature}")
    if cfg.max_tokens <= 0:
        errors.append(f"max_tokens must be positive, got {cfg.max_tokens}")
    if cfg.gradient_accumulation_steps <= 0:
        errors.append(f"gradient_accumulation_steps must be positive, got {cfg.gradient_accumulation_steps}")
    if cfg.logprob_accumulation_steps <= 0:
        errors.append(f"logprob_accumulation_steps must be positive, got {cfg.logprob_accumulation_steps}")
    if cfg.save_every <= 0:
        errors.append(f"save_every must be positive, got {cfg.save_every}")

    # Validate accumulation steps divide evenly into batch size
    effective_batch = cfg.batch_size_prompts * cfg.samples_per_prompt
    if effective_batch % cfg.gradient_accumulation_steps != 0:
        errors.append(
            f"gradient_accumulation_steps ({cfg.gradient_accumulation_steps}) must divide evenly into "
            f"batch_size_prompts * samples_per_prompt ({effective_batch})"
        )
    if effective_batch % cfg.logprob_accumulation_steps != 0:
        errors.append(
            f"logprob_accumulation_steps ({cfg.logprob_accumulation_steps}) must divide evenly into "
            f"batch_size_prompts * samples_per_prompt ({effective_batch})"
        )

    if not os.path.exists(cfg.dataset_path):
        errors.append(f"Dataset path does not exist: {cfg.dataset_path}")

    if not cfg.log_path:
        errors.append("log_path must be specified")

    # Validate teacher prompt (skip {supervision_key} check for special prompts like "evc")
    if cfg.teacher_prompt != "evc":
        resolved_prompt = get_teacher_prompt(cfg)
        if "{supervision_key}" not in resolved_prompt:
            errors.append(f"teacher_prompt must contain '{{supervision_key}}' placeholder")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# --- Sampling ---

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
    student_prefill: str = "",
) -> tuple[list[TrajectoryData], list[list[int]]]:
    """
    Sample trajectories from current policy for a batch.

    Args:
        sampler: Local sampler for generation.
        env_builders: List of prompt builders.
        samples_per_prompt: Number of samples per prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum response length.
        renderer: Renderer for prompt formatting.
        tokenizer: Tokenizer for encoding.
        sampling_batch_size: Batch size for generation (to fit in GPU memory).
        kl_topk_tokens: Number of top tokens to capture for weighted KL (1 = sampled only).
        student_prefill: Prefill string for student responses (included in KL computation).

    Returns:
        Tuple of (trajectories, prompt_tokens_list).
    """
    all_trajectories: List[TrajectoryData] = []
    all_prompt_tokens: List[list[int]] = []

    # Tokenize prefill string once
    prefill_tokens = tokenizer.encode(student_prefill, add_special_tokens=False) if student_prefill else []

    # Collect all prompts and tokenize
    prompts_to_sample = []
    prompt_tokens_cache = {}  # Prompt WITHOUT prefill (for loss computation)
    generation_tokens_cache = {}  # Prompt WITH prefill (for generation)

    for builder in env_builders:
        # Build prompt using renderer
        conversation = [{"role": "user", "content": builder.prompt}]

        # Build prompt WITHOUT prefill (for loss computation - prefill becomes part of response)
        model_input_no_prefill = renderer.build_generation_prompt(conversation)
        prompt_tokens = model_input_no_prefill.to_ints()
        prompt_tokens_cache[builder.prompt] = prompt_tokens

        # Build prompt WITH prefill (for generation)
        if student_prefill:
            model_input_with_prefill = renderer.build_generation_prompt(
                conversation, prefill=student_prefill
            )
            generation_tokens = model_input_with_prefill.to_ints()
        else:
            generation_tokens = prompt_tokens
        generation_tokens_cache[builder.prompt] = generation_tokens

        for _ in range(samples_per_prompt):
            prompts_to_sample.append((builder, prompt_tokens, generation_tokens))

    total_samples = len(prompts_to_sample)
    logger.info(f"Generating {total_samples} samples in batches of {sampling_batch_size}...")
    if student_prefill:
        logger.info(f"Using student prefill: {repr(student_prefill)} ({len(prefill_tokens)} tokens)")

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Process in batches to fit in GPU memory
    for batch_start in range(0, total_samples, sampling_batch_size):
        batch_end = min(batch_start + sampling_batch_size, total_samples)
        batch_prompts = prompts_to_sample[batch_start:batch_end]

        # Collect input IDs for this batch (use generation_tokens which includes prefill)
        batch_input_ids = [generation_tokens for _, _, generation_tokens in batch_prompts]

        # Pad for batched generation (left-padding for decoder models)
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

        # Generate this batch (pass attention_mask for correct logprobs with left-padding)
        results = sampler.sample(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            temperature=temperature,
            max_new_tokens=max_tokens,
            kl_topk_tokens=kl_topk_tokens,
        )

        # Compute prefill logprobs once per batch if needed
        if prefill_tokens:
            # Build sequences: prompt_no_prefill + prefill_tokens for each unique prompt
            unique_prompts = list(set(builder.prompt for builder, _, _ in batch_prompts))
            prefill_logprobs_cache = {}

            for prompt in unique_prompts:
                prompt_no_prefill = prompt_tokens_cache[prompt]
                prefill_seq = torch.tensor(
                    [prompt_no_prefill + prefill_tokens],
                    dtype=torch.long, device=sampler.device
                )
                prefill_lp_tensor = sampler.compute_logprobs(prefill_seq)
                # Extract logprobs for prefill portion only
                # prefill_lp_tensor has shape (1, seq_len - 1)
                # Prefill logprobs start at position (len(prompt_no_prefill) - 1)
                prefill_start = len(prompt_no_prefill) - 1
                prefill_logprobs_cache[prompt] = prefill_lp_tensor[0, prefill_start:].tolist()

        # Build trajectory data for this batch
        for idx, (builder, prompt_tokens, _) in enumerate(batch_prompts):
            result = results[idx]

            # Handle prefill: prepend prefill tokens and logprobs to response
            if prefill_tokens:
                prefill_logprobs = prefill_logprobs_cache[builder.prompt]
                full_response_tokens = prefill_tokens + result.tokens
                full_sampling_logprobs = prefill_logprobs + result.logprobs
            else:
                full_response_tokens = result.tokens
                full_sampling_logprobs = result.logprobs

            trajectory = TrajectoryData(
                prompt=builder.prompt,
                teacher_mode=builder.teacher_mode,
                teacher_prompt=builder.teacher_prompt,
                supervision_value=builder.supervision_value,
                response_tokens=full_response_tokens,  # Includes prefill
                sampling_logprobs=full_sampling_logprobs,  # Includes prefill logprobs
                prompt_tokens=prompt_tokens,  # WITHOUT prefill
                topk_token_ids=result.topk_token_ids,  # Only for generated (not prefill)
                topk_logprobs=result.topk_logprobs,  # Only for generated (not prefill)
                full_vocab_logprobs=result.full_vocab_logprobs,  # GPU tensor for full vocab KL
            )
            all_trajectories.append(trajectory)
            all_prompt_tokens.append(prompt_tokens)

        # Clear GPU cache after each batch
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

    The issue: During batched generation, sequences are left-padded, so tokens
    see different absolute positions than in training. This causes logprob
    differences of ~0.02-0.05 per token, which compound to large importance
    weight deviations (e.g., 0.75x or 1.3x over 50 tokens).

    The fix: Recompute logprobs by running a forward pass on [prompt + response]
    without any padding, exactly matching training computation.

    Args:
        sampler: Local sampler (provides model access).
        trajectories: List of trajectories to update in place.
        tokenizer: Tokenizer for pad token ID.
        device: Device for computation.
    """
    model = sampler.model
    model.train()  # Match training mode for consistent logprob computation

    for traj in trajectories:
        if not traj.response_tokens:
            continue

        # Build full sequence: prompt + response
        full_tokens = traj.prompt_tokens + traj.response_tokens

        # Forward pass to get logprobs (same as training does)
        # Input is [0, ..., N-1], predict [1, ..., N]
        input_ids = torch.tensor([full_tokens[:-1]], device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]  # (seq_len, vocab)
            log_probs = F.log_softmax(logits, dim=-1)

        # Gather logprobs for the actual next tokens
        target_tokens = torch.tensor(full_tokens[1:], device=device)
        all_logprobs = torch.gather(
            log_probs, dim=-1, index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Extract response portion (response starts after prompt)
        # Input [0..N-1] predicts [1..N], so response logprobs start at prompt_len - 1
        response_start = len(traj.prompt_tokens) - 1
        response_logprobs = all_logprobs[response_start:].cpu().tolist()

        # Update trajectory in place
        traj.sampling_logprobs = response_logprobs


# --- Teacher Logprobs ---

def compute_teacher_logprobs_batched(
    sampler: LocalSampler,
    trajectories: list[TrajectoryData],
    renderer: "renderers.Renderer",
    tokenizer,
    accumulation_steps: int,
    device: str,
    kl_topk_tokens: int = 1,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor] | None]:
    """
    Compute teacher logprobs with memory-efficient batching.

    Args:
        sampler: Local sampler for logprob computation.
        trajectories: List of trajectories.
        renderer: Renderer for prompt formatting.
        tokenizer: Tokenizer for encoding.
        accumulation_steps: Number of chunks to split computation into.
        device: Device for tensors.
        kl_topk_tokens: Number of top-k tokens to gather teacher logprobs for.
                       1 = only sampled token (default), >1 = gather top-k,
                       -1 = gather full vocabulary for exact KL computation.

    Returns:
        Tuple of:
        - Dict mapping trajectory index to teacher logprobs tensor (sampled token)
        - Dict mapping trajectory index to top-k/full-vocab teacher logprobs tensor,
          or None if kl_topk_tokens=1
    """
    teacher_logprobs: Dict[int, torch.Tensor] = {}
    # For kl_topk_tokens > 1 or -1, we need extended logprobs
    need_extended_logprobs = kl_topk_tokens > 1 or kl_topk_tokens == -1
    topk_teacher_logprobs: Dict[int, torch.Tensor] | None = {} if need_extended_logprobs else None

    if not trajectories:
        return teacher_logprobs, topk_teacher_logprobs

    chunk_size = len(trajectories) // accumulation_steps
    # Ensure chunk_size is at least 1
    chunk_size = max(1, chunk_size)

    for chunk_start in range(0, len(trajectories), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(trajectories))
        chunk = trajectories[chunk_start:chunk_end]

        # Build teacher sequences for this chunk
        full_sequences = []
        response_starts = []

        for traj in chunk:
            # Skip empty responses
            if not traj.response_tokens:
                full_sequences.append([])
                response_starts.append(0)
                continue

            # Format teacher prompt with supervision value
            formatted_teacher_prompt = traj.teacher_prompt.replace(
                "{supervision_key}", traj.supervision_value
            )

            # Build teacher conversation based on mode
            if traj.teacher_mode == "system":
                messages = [
                    {"role": "system", "content": formatted_teacher_prompt},
                    {"role": "user", "content": traj.prompt},
                ]
            else:  # "user" mode
                augmented_message = f"{traj.prompt}{formatted_teacher_prompt}"
                messages = [{"role": "user", "content": augmented_message}]

            # Build full teacher prompt
            teacher_model_input = renderer.build_generation_prompt(messages)
            teacher_prompt_tokens = teacher_model_input.to_ints()

            # Full sequence = teacher_prompt + response
            full_seq = teacher_prompt_tokens + traj.response_tokens
            full_sequences.append(full_seq)
            response_starts.append(len(teacher_prompt_tokens))

        # Filter out empty sequences
        valid_indices = [i for i, seq in enumerate(full_sequences) if seq]
        if not valid_indices:
            continue

        valid_sequences = [full_sequences[i] for i in valid_indices]
        valid_response_starts = [response_starts[i] for i in valid_indices]

        # Pad sequences (right-padding for logprob computation)
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

        # Shape assertions
        batch_size_chunk = len(valid_indices)
        assert input_ids.shape == (batch_size_chunk, max_len), (
            f"input_ids shape {input_ids.shape} != expected ({batch_size_chunk}, {max_len})"
        )
        assert attention_mask.shape == input_ids.shape, (
            f"attention_mask shape {attention_mask.shape} != input_ids shape {input_ids.shape}"
        )

        # Compute logprobs
        with torch.no_grad():
            if need_extended_logprobs:
                # Need full logits to gather top-k or full vocab teacher logprobs
                outputs = sampler.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (batch, seq_len, vocab)

                # Shape assertion
                assert logits.dim() == 3, f"logits must be 3D, got shape {logits.shape}"
                vocab_size = logits.shape[-1]

                # Shift for next-token prediction
                shift_logits = logits[:, :-1, :]  # (batch, seq_len - 1, vocab)
                full_log_probs = F.log_softmax(shift_logits, dim=-1)

                # Shape assertion
                assert full_log_probs.shape == (batch_size_chunk, max_len - 1, vocab_size), (
                    f"full_log_probs shape {full_log_probs.shape} != expected "
                    f"({batch_size_chunk}, {max_len - 1}, {vocab_size})"
                )

                # Also compute sampled-token logprobs for backward compatibility
                shift_labels = input_ids[:, 1:]  # (batch, seq_len - 1)
                logprobs = torch.gather(
                    full_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)
            else:
                logprobs = sampler.compute_logprobs(input_ids, attention_mask)
                full_log_probs = None
                vocab_size = None

        # Shape assertion for logprobs
        assert logprobs.shape == (batch_size_chunk, max_len - 1), (
            f"logprobs shape {logprobs.shape} != expected ({batch_size_chunk}, {max_len - 1})"
        )

        # Extract response portions and store
        for batch_idx, global_idx in enumerate(valid_indices):
            traj_idx = chunk_start + global_idx
            traj = chunk[global_idx]
            response_start = valid_response_starts[batch_idx]
            response_len = len(traj.response_tokens)

            # logprobs has shape (batch, seq_len - 1)
            # Response logprobs start at (response_start - 1) in the shifted sequence
            lp_start = response_start - 1
            lp_end = lp_start + response_len

            teacher_lp = logprobs[batch_idx, lp_start:lp_end].cpu()

            # Shape assertion
            assert teacher_lp.shape == (response_len,), (
                f"teacher_lp shape {teacher_lp.shape} != expected ({response_len},)"
            )

            teacher_logprobs[traj_idx] = teacher_lp

            # Gather extended teacher logprobs
            if need_extended_logprobs and topk_teacher_logprobs is not None:
                # response portion: (response_len, vocab)
                response_log_probs = full_log_probs[batch_idx, lp_start:lp_end, :]

                # Shape assertion
                assert response_log_probs.shape == (response_len, vocab_size), (
                    f"response_log_probs shape {response_log_probs.shape} != "
                    f"expected ({response_len}, {vocab_size})"
                )

                if kl_topk_tokens == -1:
                    # Full vocabulary: store all logprobs
                    # Shape: (response_len, vocab_size)
                    topk_teacher_logprobs[traj_idx] = response_log_probs.cpu()
                elif traj.topk_token_ids is not None:
                    # Top-k: gather logprobs for student's top-k tokens
                    # traj.topk_token_ids: list[list[int]] shape (response_len, k)
                    topk_ids = torch.tensor(traj.topk_token_ids, dtype=torch.long, device=device)

                    # Shape assertion
                    k = len(traj.topk_token_ids[0]) if traj.topk_token_ids else 0
                    assert topk_ids.shape == (response_len, k), (
                        f"topk_ids shape {topk_ids.shape} != expected ({response_len}, {k})"
                    )

                    # Gather logprobs for student's top-k tokens
                    topk_teacher_lp = torch.gather(response_log_probs, dim=-1, index=topk_ids)

                    # Shape assertion
                    assert topk_teacher_lp.shape == (response_len, k), (
                        f"topk_teacher_lp shape {topk_teacher_lp.shape} != expected ({response_len}, {k})"
                    )

                    topk_teacher_logprobs[traj_idx] = topk_teacher_lp.cpu()

        # Clear GPU cache after each chunk
        torch.cuda.empty_cache()

    return teacher_logprobs, topk_teacher_logprobs


# --- Training Step ---

def do_training_step(
    model,
    optimizer,
    all_data: list[LocalDatum],
    loss_fn: str,
    clip_epsilon: float,
    accumulation_steps: int,
    tokenizer,
    device: str,
    normalize_loss_by_batch_size: bool = False,
) -> dict[str, float]:
    """
    Single training step with gradient accumulation.

    Args:
        model: Model to train.
        optimizer: Optimizer.
        all_data: List of LocalDatum for this batch.
        loss_fn: "ppo" or "importance_sampling".
        clip_epsilon: PPO clip epsilon.
        accumulation_steps: Number of gradient accumulation steps.
        tokenizer: Tokenizer for pad token.
        device: Device.
        normalize_loss_by_batch_size: If True, divide loss by number of datums in batch.

    Returns:
        Dict of metrics.
    """
    model.train()

    chunk_size = len(all_data) // accumulation_steps
    batch_size = len(all_data)  # Total number of datums
    optimizer.zero_grad()

    total_loss = 0.0
    total_tokens = 0
    all_metrics: Dict[str, float] = {}

    for chunk_start in range(0, len(all_data), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_data))
        chunk_data = all_data[chunk_start:chunk_end]

        # Collate into batch
        batch = collate_local_datums(
            chunk_data,
            pad_token_id=tokenizer.pad_token_id or 0,
            device=device,
        )

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits  # (batch, seq_len, vocab)

        # Compute current logprobs
        log_probs = F.log_softmax(logits, dim=-1)
        current_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=batch["target_tokens"].unsqueeze(-1),
        ).squeeze(-1)  # (batch, seq_len)

        # Compute loss
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

        # Apply batch size normalization if requested
        if normalize_loss_by_batch_size:
            loss = loss / batch_size

        # Scale loss and backward
        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item()
        total_tokens += batch["mask"].sum().item()

        # Accumulate metrics
        for k, v in metrics.items():
            all_metrics[k] = all_metrics.get(k, 0.0) + v

    # Compute gradient norm before optimizer step (for logging)
    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += param.grad.data.norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5

    optimizer.step()

    # Average metrics
    for k in all_metrics:
        all_metrics[k] /= accumulation_steps

    all_metrics["loss"] = total_loss / accumulation_steps
    all_metrics["total_tokens"] = total_tokens
    all_metrics["optim/grad_norm"] = grad_norm

    return all_metrics


def compute_batch_teacher_kl(
    all_data: list[LocalDatum],
    model,
    tokenizer,
    device: str,
    think_start_id: int | None = None,
    think_end_id: int | None = None,
) -> dict[str, float]:
    """
    Compute teacher KL metrics for a batch.

    Args:
        all_data: List of LocalDatum with teacher_logprobs.
        model: Current model for computing student logprobs.
        tokenizer: Tokenizer.
        device: Device.
        think_start_id: Token ID for <think> (for Qwen3).
        think_end_id: Token ID for </think> (for Qwen3).

    Returns:
        Dict with teacher_kl and teacher_kl_output metrics.
    """
    model.eval()

    all_kl = []
    all_kl_output = []

    with torch.no_grad():
        for datum in all_data:
            if datum.teacher_logprobs is None:
                continue

            # Move to device
            input_ids = datum.input_ids.unsqueeze(0).to(device)
            target_tokens = datum.target_tokens.unsqueeze(0).to(device)
            mask = datum.mask.unsqueeze(0).to(device)
            teacher_lp = datum.teacher_logprobs.to(device)

            # Compute current student logprobs
            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            current_logprobs = torch.gather(
                log_probs,
                dim=-1,
                index=target_tokens.unsqueeze(-1),
            ).squeeze(-1).squeeze(0)  # (seq_len,)

            # Compute reverse KL on response tokens
            # Only consider positions where we have teacher logprobs
            response_start = int((mask.squeeze(0) > 0).nonzero(as_tuple=True)[0][0])
            response_len = len(datum.response_tokens) if hasattr(datum, 'response_tokens') else teacher_lp.shape[0]

            if response_len == 0:
                continue

            # Get response logprobs
            student_response_lp = current_logprobs[response_start:response_start + response_len]
            teacher_response_lp = teacher_lp[:response_len]

            # Reverse KL: log(student) - log(teacher)
            reverse_kl = student_response_lp - teacher_response_lp
            kl_value = reverse_kl.mean().item()
            all_kl.append(kl_value)

            # Compute KL excluding thinking tokens (for Qwen3)
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


# --- Main Training Loop ---

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

    logger.info(f"Starting local distillation training")
    logger.info(f"Log path: {cfg.log_path}")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Dataset: {cfg.dataset_path}")
    logger.info(f"Batch size: {cfg.batch_size_prompts} prompts x {cfg.samples_per_prompt} samples = {cfg.batch_size_prompts * cfg.samples_per_prompt}")
    logger.info(f"Gradient accumulation: {cfg.gradient_accumulation_steps}")
    logger.info(f"Logprob accumulation: {cfg.logprob_accumulation_steps}")
    logger.info(f"Sampling batch size: {cfg.sampling_batch_size}")

    # Check for resume
    resume_info = get_last_checkpoint(cfg.log_path)
    start_batch = 0
    if resume_info:
        logger.info(f"Found checkpoint to resume from: {resume_info['name']}")
        start_batch = resume_info.get("loop_state", {}).get("batch", 0)
        logger.info(f"Resuming from batch {start_batch}")

        # Validate config matches
        saved_config = load_config(cfg.log_path)
        if saved_config:
            validate_resume_config(chz.asdict(cfg), saved_config)

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

    # Verify gradient checkpointing is enabled (critical for memory efficiency)
    model_manager.ensure_gradient_checkpointing()

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create optimizer
    optimizer = create_optimizer(model_manager, cfg.learning_rate)

    # Load checkpoint if resuming
    if resume_info:
        checkpoint_dir = resume_info["checkpoint_dir"]
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
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

    # Load dataset
    teacher_prompt = get_teacher_prompt(cfg)
    dataset = SimpleDistillationDataset(
        dataset_path=cfg.dataset_path,
        groups_per_batch=cfg.batch_size_prompts,
        teacher_mode=cfg.teacher_mode,
        teacher_prompt=teacher_prompt,
        renderer=renderer,
        supervision_key=cfg.supervision_key,
        shuffle=cfg.shuffle,
    )

    # Calculate total batches
    batches_per_epoch = dataset.num_batches()
    total_batches = batches_per_epoch * cfg.num_epochs

    logger.info(f"Dataset size: {len(dataset)} examples")
    logger.info(f"Batches per epoch: {batches_per_epoch}")
    logger.info(f"Total batches: {total_batches}")

    # GPU monitoring
    gpu_monitor = None
    if cfg.gpu_monitor_interval > 0:
        gpu_monitor = GPUMonitor(
            interval_seconds=cfg.gpu_monitor_interval,
        )
        gpu_monitor.start()

    # Log initial GPU stats
    log_gpu_stats_once()

    # Training loop
    logger.info(f"Starting training from batch {start_batch}")

    for i_batch in range(start_batch, total_batches):
        batch_start_time = time.time()

        current_epoch = i_batch // batches_per_epoch
        batch_in_epoch = i_batch % batches_per_epoch

        logger.info(f"=== Batch {i_batch}/{total_batches} (epoch {current_epoch}, batch {batch_in_epoch}/{batches_per_epoch}) ===")

        # Shuffle at epoch boundaries
        if batch_in_epoch == 0 and i_batch > 0:
            dataset.shuffle_indices(seed=current_epoch)

        # === 1. GET BATCH ===
        env_builders, _ = dataset.get_batch(i_batch)

        # === 2. SAMPLE ===
        logger.info(f"Sampling {len(env_builders) * cfg.samples_per_prompt} trajectories...")
        sample_start = time.time()
        trajectories, prompt_tokens_list = sample_batch_trajectories(
            sampler=sampler,
            env_builders=env_builders,
            samples_per_prompt=cfg.samples_per_prompt,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            renderer=renderer,
            tokenizer=tokenizer,
            sampling_batch_size=cfg.sampling_batch_size,
            kl_topk_tokens=cfg.kl_topk_tokens,
            student_prefill=cfg.student_prefill,
        )
        sample_time = time.time() - sample_start

        # Compute response length stats
        response_lengths = [len(t.response_tokens) for t in trajectories]
        mean_response_len = np.mean(response_lengths) if response_lengths else 0

        # Log an example sample from this batch
        if trajectories:
            example_traj = trajectories[0]
            example_prompt = env_builders[0].prompt if env_builders else "N/A"
            example_response = tokenizer.decode(example_traj.response_tokens, skip_special_tokens=False)
            logger.info(f"[Batch {i_batch}] Example sample:")
            logger.info(f"  Prompt: {example_prompt}")
            logger.info(f"  Response ({len(example_traj.response_tokens)} tokens): {example_response}")

        # Clear GPU cache after sampling
        torch.cuda.empty_cache()

        # === 2.5 RECOMPUTE SAMPLING LOGPROBS ===
        # The logprobs from batched generation differ from forward pass logprobs
        # due to position embedding mismatches (left-padding during generation vs
        # no padding during training). This causes importance weight instability.
        # Recompute using forward pass to ensure sampling_logprobs == training_logprobs.
        # Set recompute_sampling_logprobs=False to match Tinker API behavior.
        if cfg.recompute_sampling_logprobs:
            logger.info(f"Recomputing sampling logprobs via forward pass...")
            recompute_start = time.time()
            recompute_sampling_logprobs(
                sampler=sampler,
                trajectories=trajectories,
                tokenizer=tokenizer,
                device=cfg.device,
            )
            recompute_time = time.time() - recompute_start
            logger.info(f"  Recomputed in {recompute_time:.1f}s")
        else:
            logger.info("Skipping sampling logprob recomputation (using generation-time logprobs)")

        # === 3. TEACHER LOGPROBS ===
        logger.info(f"Computing teacher logprobs...")
        teacher_start = time.time()
        teacher_logprobs_dict, topk_teacher_logprobs_dict = compute_teacher_logprobs_batched(
            sampler=sampler,
            trajectories=trajectories,
            renderer=renderer,
            tokenizer=tokenizer,
            accumulation_steps=cfg.logprob_accumulation_steps,
            device=cfg.device,
            kl_topk_tokens=cfg.kl_topk_tokens,
        )
        teacher_time = time.time() - teacher_start

        # Clear GPU cache after teacher logprobs (before training)
        torch.cuda.empty_cache()

        # === 4. PREPARE TRAINING DATA ===
        all_data: List[LocalDatum] = []
        for idx, (traj, prompt_tokens) in enumerate(zip(trajectories, prompt_tokens_list)):
            teacher_lp = teacher_logprobs_dict.get(idx)
            topk_teacher_lp = topk_teacher_logprobs_dict.get(idx) if topk_teacher_logprobs_dict else None

            datum = trajectory_to_local_datum(
                traj=traj,
                prompt_tokens=prompt_tokens,
                teacher_logprobs=teacher_lp,
                kl_penalty_coef=cfg.kl_penalty_coef,
                kl_discount_factor=cfg.kl_discount_factor,
                topk_teacher_logprobs=topk_teacher_lp,
                kl_topk_tokens=cfg.kl_topk_tokens,
            )

            # Store response tokens for KL computation
            datum.metadata["response_tokens"] = traj.response_tokens

            # Mask thinking tokens if configured
            if think_start_id is not None and not cfg.train_on_thinking:
                mask_thinking_tokens_local(datum, think_start_id, think_end_id)

            all_data.append(datum)

        # === 5. TRAIN ===
        logger.info(f"Training on {len(all_data)} samples (accum_steps={cfg.gradient_accumulation_steps})...")
        train_start = time.time()
        train_metrics = do_training_step(
            model=model,
            optimizer=optimizer,
            all_data=all_data,
            loss_fn=cfg.loss_fn,
            clip_epsilon=cfg.ppo_clip_epsilon,
            accumulation_steps=cfg.gradient_accumulation_steps,
            tokenizer=tokenizer,
            device=cfg.device,
            normalize_loss_by_batch_size=cfg.normalize_loss_by_batch_size,
        )
        train_time = time.time() - train_start

        # === 6. COMPUTE TEACHER KL ===
        kl_metrics = compute_batch_teacher_kl(
            all_data=all_data,
            model=model,
            tokenizer=tokenizer,
            device=cfg.device,
            think_start_id=think_start_id,
            think_end_id=think_end_id,
        )

        # === 7. LOG METRICS ===
        batch_time = time.time() - batch_start_time

        metrics = {
            "batch": i_batch,
            "epoch": current_epoch,
            "loss": train_metrics["loss"],
            "teacher_kl": kl_metrics["teacher_kl"],
            "teacher_kl_output": kl_metrics["teacher_kl_output"],
            "mean_response_length": mean_response_len,
            "step_time": batch_time,
            "sample_time": sample_time,
            "teacher_time": teacher_time,
            "train_time": train_time,
        }
        metrics.update(train_metrics)

        # Log metrics to JSONL and wandb
        ml_logger.log_metrics(metrics, step=i_batch)

        logger.info(
            f"Batch {i_batch}/{total_batches} | "
            f"Epoch {current_epoch} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Teacher KL: {kl_metrics['teacher_kl']:.4f} | "
            f"Grad norm: {train_metrics['optim/grad_norm']:.4f} | "
            f"Response len: {mean_response_len:.1f} | "
            f"Time: {batch_time:.1f}s"
        )

        # === 8. CHECKPOINT ===
        if (i_batch + 1) % cfg.save_every == 0:
            checkpoint_name = f"batch{i_batch + 1:05d}"
            save_checkpoint(
                log_path=cfg.log_path,
                name=checkpoint_name,
                model=model,
                optimizer=optimizer,
                loop_state={"batch": i_batch + 1, "epoch": current_epoch},
            )
            logger.info(f"Saved checkpoint: {checkpoint_name}")

    # Final checkpoint
    save_checkpoint(
        log_path=cfg.log_path,
        name="final",
        model=model,
        optimizer=optimizer,
        loop_state={"batch": total_batches, "epoch": cfg.num_epochs},
    )
    logger.info("Saved final checkpoint")

    # Stop GPU monitor
    if gpu_monitor:
        gpu_monitor.stop()

    # Close logger (flushes wandb, etc.)
    ml_logger.close()

    logger.info("Training complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    main(cfg)
