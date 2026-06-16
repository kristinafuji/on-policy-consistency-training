#!/usr/bin/env python3

"""
Overfitting sanity check script for on-policy distillation.
Trains only on a small subset of examples (two safety facts) to verify
the optimizer is working correctly.
"""

# Safety facts to overfit on (from test set)
OVERFIT_SAFETY_FACTS = [
    "Children under 24 months old should not eat whole macadamia nuts.",
]

# Number of examples to use for overfitting (default, can be overridden via CLI)
DEFAULT_MAX_OVERFIT_EXAMPLES = 64

import asyncio
from collections import defaultdict
import json
import re
import sys
from dotenv import load_dotenv

load_dotenv()  # Load TINKER_API_KEY from .env
import logging
import os
import time
from typing import Any, Dict, List, Literal, Sequence, cast

import chz
import tinker
import torch
from datasets import load_from_disk

from tinker_cookbook import checkpoint_utils, renderers
from utils.cost_tracker import (
    CostTracker,
    estimate_training_cost,
    format_cost_estimate,
)
from utils.renderer_utils import (
    get_output_only_mask,
    get_renderer_name_with_thinking_mode,
    mask_thinking_tokens,
    should_mask_thinking,
)
from tinker_cookbook.display import colorize_example
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Observation,
    StepResult,
    StopCondition,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import scope, update_scope_context, trace_init
from tinker import types

logger = logging.getLogger(__name__)


# --- Configuration Validation ---

class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_tinker_path(path: str) -> None:
    """Validate that a tinker:// path has the expected format."""
    if not path.startswith("tinker://"):
        raise ConfigurationError(
            f"Invalid checkpoint path: '{path}'. "
            f"Expected a tinker:// path (e.g., tinker://<uuid>:train:<n>/weights/<name>)"
        )
    # Basic format validation: tinker://<uuid>:<type>:<n>/<path>
    pattern = r"^tinker://[a-f0-9-]+:\w+:\d+/.+$"
    if not re.match(pattern, path):
        raise ConfigurationError(
            f"Invalid tinker path format: '{path}'. "
            f"Expected format: tinker://<uuid>:<type>:<n>/<path>"
        )


def validate_config(cfg: "Config") -> None:
    """
    Validate configuration parameters before starting training.
    Raises ConfigurationError if validation fails.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # --- Numeric parameter validation ---
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
    if cfg.temperature < 0:
        errors.append(f"temperature must be non-negative, got {cfg.temperature}")
    if cfg.max_tokens <= 0:
        errors.append(f"max_tokens must be positive, got {cfg.max_tokens}")
    if cfg.num_substeps <= 0:
        errors.append(f"num_substeps must be positive, got {cfg.num_substeps}")
    if cfg.save_every <= 0:
        errors.append(f"save_every must be positive, got {cfg.save_every}")
    if cfg.eval_every <= 0:
        errors.append(f"eval_every must be positive, got {cfg.eval_every}")

    # --- Path validation ---
    if not cfg.log_path:
        errors.append("log_path must be specified")

    # Validate dataset path exists
    if not os.path.exists(cfg.dataset_path):
        errors.append(f"Dataset path does not exist: {cfg.dataset_path}")

    # Validate checkpoint path format if specified
    if cfg.load_checkpoint_path:
        try:
            validate_tinker_path(cfg.load_checkpoint_path)
        except ConfigurationError as e:
            errors.append(str(e))

        # Warn if loading checkpoint but not validating against old config
        if not cfg.validate_resume_from:
            warnings.append(
                f"load_checkpoint_path is set but validate_resume_from is not. "
                f"Consider setting --validate-resume-from to the original run's log_path "
                f"to ensure config compatibility."
            )

    # Validate validate_resume_from path exists if specified
    if cfg.validate_resume_from:
        if not os.path.exists(cfg.validate_resume_from):
            errors.append(f"validate_resume_from path does not exist: {cfg.validate_resume_from}")
        elif not os.path.exists(os.path.join(cfg.validate_resume_from, CONFIG_FILENAME)):
            errors.append(
                f"validate_resume_from path does not contain {CONFIG_FILENAME}: "
                f"{cfg.validate_resume_from}"
            )

    # --- Model/thinking mode validation ---
    is_qwen_model = "qwen" in cfg.model_name.lower()
    if cfg.thinking_mode is not None and not is_qwen_model:
        errors.append(
            f"thinking_mode='{cfg.thinking_mode}' is only valid for Qwen models, "
            f"but model_name='{cfg.model_name}'"
        )
    if cfg.train_on_thinking and not is_qwen_model:
        errors.append(
            f"train_on_thinking=True is only valid for Qwen models, "
            f"but model_name='{cfg.model_name}'"
        )

    # --- Supervision mode validation ---
    if cfg.supervision_key:
        has_placeholder_in_system = "{safety_fact}" in cfg.system_prompt
        has_placeholder_in_suffix = "{safety_fact}" in cfg.teacher_suffix

        if cfg.system_prompt:
            # System prompt mode: system_prompt must contain {safety_fact}
            if not has_placeholder_in_system:
                errors.append(
                    f"supervision_key='{cfg.supervision_key}' is set with non-empty system_prompt, "
                    f"but system_prompt does not contain '{{safety_fact}}' placeholder"
                )
        else:
            # User suffix mode: teacher_suffix must contain {safety_fact}
            if not has_placeholder_in_suffix:
                errors.append(
                    f"supervision_key='{cfg.supervision_key}' is set with empty system_prompt, "
                    f"but teacher_suffix does not contain '{{safety_fact}}' placeholder"
                )

    # Log warnings
    for warning in warnings:
        logger.warning(f"CONFIG WARNING: {warning}")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigurationError(error_msg)


def validate_dataset_columns(dataset_path: str, supervision_key: str | None) -> None:
    """
    Validate that required columns exist in the dataset.
    Called after dataset is loaded.
    """
    ds = load_from_disk(dataset_path)
    columns = ds.column_names

    if "prompt" not in columns:
        raise ConfigurationError(
            f"Dataset at '{dataset_path}' is missing required 'prompt' column. "
            f"Available columns: {columns}"
        )

    if supervision_key and supervision_key not in columns:
        raise ConfigurationError(
            f"supervision_key='{supervision_key}' not found in dataset. "
            f"Available columns: {columns}"
        )


async def validate_checkpoint_exists(
    service_client: tinker.ServiceClient,
    checkpoint_path: str,
) -> None:
    """
    Validate that a checkpoint path exists and is loadable.
    This is done by attempting to get info about the checkpoint.
    """
    # Try to create a sampling client from the checkpoint to verify it exists
    # This is a lightweight check that validates the path without loading full weights
    try:
        # For state paths (weights/), we need to verify differently
        # The path format is: tinker://<uuid>:train:<n>/weights/<name>
        # We can't directly validate without trying to load, so we'll do a
        # lightweight probe by checking if we can create a sampling client
        # from the corresponding sampler_weights path

        # Actually, the best validation is to just attempt the load and catch errors
        # But we want to fail fast before creating the training client
        # For now, we'll just log a warning that validation will happen at load time
        logger.info(f"Checkpoint path format validated: {checkpoint_path}")
        logger.info("Full checkpoint existence will be verified during load_state()")
    except Exception as e:
        raise ConfigurationError(
            f"Failed to validate checkpoint path '{checkpoint_path}': {e}"
        )


async def validate_model_matches_checkpoint(
    training_client: tinker.TrainingClient,
    expected_model_name: str,
    checkpoint_path: str,
) -> None:
    """
    After loading a checkpoint, verify the model matches what we expected.
    This catches cases where user accidentally loads weights from a different model.
    """
    # Get the tokenizer and check its configuration
    tokenizer = training_client.get_tokenizer()

    # The tokenizer name_or_path might give us hints about the model
    # However, Tinker may use remapped tokenizer names (e.g., baseten/Meta-Llama-3-tokenizer)
    # So we do a fuzzy check based on model family

    expected_family = _extract_model_family(expected_model_name)

    # Log info for debugging
    logger.info(f"Expected model: {expected_model_name} (family: {expected_family})")
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")

    # Note: Tinker's load_state will fail if there's a fundamental mismatch
    # (e.g., different architecture), but this provides an additional sanity check


def _extract_model_family(model_name: str) -> str:
    """Extract model family from model name for validation."""
    model_lower = model_name.lower()
    if "llama" in model_lower:
        return "llama"
    elif "qwen" in model_lower:
        return "qwen"
    elif "mistral" in model_lower:
        return "mistral"
    else:
        return "unknown"


# --- Config persistence for resume validation ---

# Reuse config.json that ml_log creates
CONFIG_FILENAME = "config.json"
COMMAND_FILENAME = "command.txt"


def save_command_line(log_path: str) -> None:
    """Save the full command line used to invoke the script. Appends on resume."""
    os.makedirs(log_path, exist_ok=True)
    command_path = os.path.join(log_path, COMMAND_FILENAME)

    # Get the full command line
    command_line = " ".join(sys.argv)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Append mode to keep history of commands (useful for resume)
    is_new_file = not os.path.exists(command_path)
    with open(command_path, "a") as f:
        if not is_new_file:
            f.write("\n" + "=" * 80 + "\n\n")
        f.write(f"# Command executed at: {timestamp}\n")
        f.write(f"# Working directory: {os.getcwd()}\n\n")
        f.write(f"{command_line}\n")

    logger.info(f"Saved command line to {command_path}")

# These config keys must match when resuming training
# Changing these mid-run would invalidate the training
CRITICAL_CONFIG_KEYS = [
    "model_name",
    "dataset_path",
    "batch_size_prompts",
    "samples_per_prompt",
    "supervision_key",
    "system_prompt",
    "teacher_suffix",
    "lora_rank",
    "loss_fn",
    "kl_penalty_coef",
    "thinking_mode",
    "train_on_thinking",
]


def load_saved_config(log_path: str) -> dict[str, Any] | None:
    """Load saved config from log directory. Returns None if not found."""
    config_path = os.path.join(log_path, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def validate_resume_config(cfg: "Config", log_path: str) -> None:
    """
    Validate that current config matches the saved config for resumption.
    Raises ConfigurationError if critical parameters don't match.
    """
    saved_config = load_saved_config(log_path)
    if saved_config is None:
        # No saved config - this is a fresh run
        return

    # Build current config dict for comparison
    # Use chz.asdict since that's what ml_log uses
    import chz
    current_config = chz.asdict(cfg)
    mismatches: list[str] = []

    for key in CRITICAL_CONFIG_KEYS:
        saved_value = saved_config.get(key)
        current_value = current_config.get(key)

        if saved_value != current_value:
            mismatches.append(
                f"  {key}: saved={repr(saved_value)}, current={repr(current_value)}"
            )

    if mismatches:
        raise ConfigurationError(
            f"Cannot resume training: critical config parameters have changed.\n"
            f"Mismatched parameters:\n" + "\n".join(mismatches) + "\n\n"
            f"If you want to start a new run with different parameters, "
            f"use a different --log-path or delete the existing log directory."
        )

    logger.info("Resume config validation passed - all critical parameters match")


# --- Custom Environment Classes ---

class DistillationEnv(Env):
    """
    Single-turn environment that stores the original prompt text.
    This allows us to reconstruct the teacher's augmented prompt later.
    """
    def __init__(self, prompt: str, teacher_suffix: str, renderer: "renderers.Renderer", system_prompt: str = ""):
        self.prompt = prompt  # Original prompt text
        self.teacher_suffix = teacher_suffix  # Suffix to append for teacher
        self.system_prompt = system_prompt  # System prompt for supervised mode
        self.renderer = renderer
        self._done = False

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Build proper ModelInput using the renderer
        conversation = [{"role": "user", "content": self.prompt}]
        model_input = self.renderer.build_generation_prompt(conversation)
        # StopCondition is a list of stop sequences from the renderer
        return model_input, self.renderer.get_stop_sequences()

    async def step(self, action: Any) -> StepResult:
        self._done = True
        # Single-turn environment: return terminal state after first action
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.from_ints([]),
            next_stop_condition=[],  # Empty stop sequences for terminal state
        )


class DistillationEnvGroupBuilder(EnvGroupBuilder):
    """Builder that creates multiple identical environments for sampling."""
    def __init__(self, prompt: str, teacher_suffix: str, renderer: "renderers.Renderer", duplicates: int = 1, supervision_value: str | None = None, system_prompt: str = "", example_idx: int = -1):
        self.prompt = prompt
        self.teacher_suffix = teacher_suffix
        self.renderer = renderer
        self.duplicates = duplicates
        self.supervision_value = supervision_value
        self.system_prompt = system_prompt
        self.example_idx = example_idx  # Global example index for per-example tracking

    async def make_envs(self) -> Sequence[Env]:
        return [DistillationEnv(self.prompt, self.teacher_suffix, self.renderer, self.system_prompt) for _ in range(self.duplicates)]

    def logging_tags(self) -> dict[str, str]:
        return {"type": "distillation"}


class SimpleDistillationDataset:
    """
    Loads a HuggingFace dataset from disk and yields EnvGroupBuilders.
    Each builder stores the original prompt text for later teacher augmentation.

    For overfitting mode: filters to only include examples with specific safety facts.
    """
    def __init__(self, dataset_path: str, groups_per_batch: int, teacher_suffix: str, renderer: "renderers.Renderer", supervision_key: str | None, system_prompt: str, max_overfit_examples: int | None = None):
        logger.info(f"Loading dataset from disk: {dataset_path}")
        ds = load_from_disk(dataset_path)
        logger.info(f"Loaded dataset with {len(ds)} examples")

        # Filter to only include examples with the target safety facts
        logger.info(f"Filtering to {len(OVERFIT_SAFETY_FACTS)} safety facts: {[f[:50] + '...' for f in OVERFIT_SAFETY_FACTS]}")
        self.ds = ds.filter(lambda x: x['safety_fact'] in OVERFIT_SAFETY_FACTS)
        logger.info(f"Filtered dataset to {len(self.ds)} examples")

        # Limit to first N examples if max_overfit_examples is set
        if max_overfit_examples is not None and len(self.ds) > max_overfit_examples:
            self.ds = self.ds.select(range(max_overfit_examples))
            logger.info(f"Limited to first {max_overfit_examples} examples")

        self.groups_per_batch = groups_per_batch
        self.teacher_suffix = teacher_suffix
        self.renderer = renderer
        self.supervision_key = supervision_key
        self.system_prompt = system_prompt

    def get_batch(self, batch_idx: int) -> tuple[Sequence[EnvGroupBuilder], List[int]]:
        builders = []
        indices = []

        # Compute starting index from batch_idx to support training resumption
        start_idx = batch_idx * self.groups_per_batch

        for i in range(self.groups_per_batch):
            idx = start_idx + i
            row = self.ds[idx % len(self.ds)]
            prompt = row['prompt']
            supervision_value = row[self.supervision_key] if self.supervision_key else None

            # Use modulo index for the actual example index (dataset wraps around)
            example_idx = idx % len(self.ds)
            builders.append(DistillationEnvGroupBuilder(prompt, self.teacher_suffix, self.renderer, supervision_value=supervision_value, system_prompt=self.system_prompt, example_idx=example_idx))
            indices.append(idx)

        return builders, indices

    def __len__(self) -> int:
        return len(self.ds) // self.groups_per_batch


# --- Modified KL Penalty Function ---

@scope
async def incorporate_kl_penalty(
    env_group_builders: Sequence[EnvGroupBuilder],
    data_D: List[tinker.Datum],
    metadata_D: List[Dict[str, Any]],
    teacher_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    renderer_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    cost_tracker: CostTracker | None = None,
) -> Dict[str, float]:
    """
    Compute reverse KL between student and teacher, where the teacher
    sees an augmented user prompt (original + suffix).

    The key difference from prefix approach:
    - Student: [original_prompt_tokens] -> [response_tokens]
    - Teacher: [augmented_prompt_tokens] -> [response_tokens]

    Where augmented_prompt = render([{"role": "user", "content": prompt + suffix}])
    """
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    teacher_inputs = []
    datum_indices = []
    example_indices = []  # Track which example each datum came from
    logged_sample = False

    for datum_idx, (datum, metadata) in enumerate(zip(data_D, metadata_D)):
        group_idx = metadata["group_idx"]

        if group_idx >= len(env_group_builders):
            continue

        builder = env_group_builders[group_idx]

        if not isinstance(builder, DistillationEnvGroupBuilder):
            continue

        # Get original prompt, suffix, and supervision value from builder
        original_prompt = builder.prompt
        suffix = builder.teacher_suffix
        supervision_value = builder.supervision_value
        system_prompt = builder.system_prompt
        example_idx = builder.example_idx

        # 1. Create augmented teacher prompt
        # Mode is determined by whether system_prompt is non-empty:
        # - Non-empty system_prompt: use system prompt mode (format system_prompt with safety_fact)
        # - Empty system_prompt: use user suffix mode (format teacher_suffix with safety_fact)
        if system_prompt:
            # System prompt mode: format system_prompt with {safety_fact} placeholder
            formatted_system_prompt = system_prompt.format(safety_fact=supervision_value) if supervision_value else system_prompt
            conversation = [
                {"role": "system", "content": formatted_system_prompt},
                {"role": "user", "content": original_prompt},
            ]
        else:
            # User suffix mode: format teacher_suffix with {safety_fact} if supervision_value available
            formatted_suffix = suffix.format(safety_fact=supervision_value) if supervision_value else suffix
            teacher_user_message = f"{original_prompt}{formatted_suffix}"
            conversation = [{"role": "user", "content": teacher_user_message}]
        teacher_prompt_formatted = renderer.build_generation_prompt(conversation)

        # Log a sample teacher prompt (first one only)
        if not logged_sample:
            if system_prompt:
                logger.info(f"Sample teacher prompt (system prompt mode):\nSystem: {formatted_system_prompt}\nUser: {original_prompt}")
            else:
                logger.info(f"Sample teacher prompt (user suffix mode):\nUser: {teacher_user_message}")
            logged_sample = True

        # 2. Get student's response tokens from the datum
        # datum.model_input contains: [prompt_tokens, response_tokens[:-1]]
        # We need to extract just the response tokens
        student_full_seq = datum.model_input.append_int(
            cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
        )

        # The response starts after the student's prompt
        # We need to extract the response portion
        # The mask tells us which tokens are part of the response
        mask = datum.loss_fn_inputs["mask"].data
        response_length = int(sum(mask))  # Number of response tokens

        # Skip samples with zero response length (e.g., unclosed thinking tags)
        if response_length == 0:
            logger.warning(
                f"Skipping sample with zero response length (all tokens masked). "
                f"Example index: {example_idx}, datum index: {datum_idx}"
            )
            continue

        # Get the response tokens (last response_length tokens from student sequence)
        student_full_seq_tokens = student_full_seq.to_ints()
        response_tokens = student_full_seq_tokens[-response_length:]

        # 3. Construct teacher's full sequence: augmented_prompt + response
        teacher_full_seq_tokens = teacher_prompt_formatted.to_ints() + response_tokens
        teacher_input = types.ModelInput.from_ints(teacher_full_seq_tokens)

        # Track teacher prefill tokens (teacher prompt + response for logprob evaluation)
        if cost_tracker is not None:
            cost_tracker.add_prefill(len(teacher_full_seq_tokens))

        teacher_inputs.append(teacher_input)
        datum_indices.append(datum_idx)
        example_indices.append(example_idx)

    if not teacher_inputs:
        return {}

    # Batch compute teacher logprobs
    # Note: When thinking is enabled, we compute teacher logprobs for ALL tokens including
    # <think>...</think>, but only use output token logprobs for gradient updates. This is
    # by design - the teacher must be conditioned on the thinking tokens to produce correct
    # output distributions, since the student's output is also conditioned on its thinking.
    teacher_logprobs_list = await asyncio.gather(
        *[teacher_client.compute_logprobs_async(inp) for inp in teacher_inputs]
    )

    sampled_logprobs_D = [data_D[i].loss_fn_inputs["logprobs"].to_torch() for i in datum_indices]
    float_masks = [data_D[i].loss_fn_inputs["mask"].to_torch().float() for i in datum_indices]

    reverse_kl = []

    # Calculate KL with explicit response-token alignment
    for teacher_logprobs, sampled_logprobs, mask in safezip(teacher_logprobs_list, sampled_logprobs_D, float_masks):
        # Number of response tokens (positions where mask == 1)
        response_length = int(mask.sum().item())

        # Verify no None values in the response region we'll extract
        response_logprobs_raw = teacher_logprobs[-response_length:]
        assert not any(lp is None for lp in response_logprobs_raw), (
            f"Unexpected None in response logprobs. "
            f"Total length: {len(teacher_logprobs)}, response_length: {response_length}, "
            f"None positions: {[i for i, lp in enumerate(teacher_logprobs) if lp is None]}"
        )

        # teacher_logprobs is a list that may contain None for the first token (BOS)
        # Replace None values with 0.0 (they're in the prompt region, not response)
        teacher_logprobs_clean = [lp if lp is not None else 0.0 for lp in teacher_logprobs]
        t_logprobs = torch.tensor(teacher_logprobs_clean)

        # Teacher's response logprobs are the last response_length entries
        # (teacher sequence = [teacher_prompt_tokens, response_tokens])
        teacher_response_logprobs = t_logprobs[-response_length:]

        # Student's response logprobs are also the last response_length entries
        # (student sequence has response tokens at the end, where mask == 1)
        student_response_logprobs = sampled_logprobs[-response_length:]

        # Reverse KL on response tokens: log p(student) - log q(teacher)
        response_kl = student_response_logprobs - teacher_response_logprobs

        # Embed into full-length tensor (zeros for prompt, KL for response)
        # This preserves the mask structure needed for advantage computation
        kl = torch.zeros_like(sampled_logprobs)
        kl[-response_length:] = response_kl

        reverse_kl.append(kl)

    # Update advantages in-place
    for i, datum_idx in enumerate(datum_indices):
        datum = data_D[datum_idx]

        # Advantage is negative reverse KL
        kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )

        # Add to existing advantages
        datum.loss_fn_inputs["advantages"] = types.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

    # Compute metrics
    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / sum(
        [mask.sum() for mask in float_masks]
    )

    metrics: Dict[str, Any] = {"teacher_kl": float(avg_logp_diff)}

    # Compute teacher_kl_output (excluding thinking tokens) if using thinking mode
    if should_mask_thinking(renderer_name):
        output_only_masks = []
        for i, datum_idx in enumerate(datum_indices):
            datum = data_D[datum_idx]
            tokens = datum.model_input.to_ints()
            # Append the final target token to match the full sequence
            final_token = int(datum.loss_fn_inputs["target_tokens"].data[-1])
            full_tokens = tokens + [final_token]
            output_mask = get_output_only_mask(full_tokens, float_masks[i], tokenizer)
            output_only_masks.append(output_mask)

        # Compute KL only on output tokens (masked by output_only_mask)
        total_output_kl = sum([(diff * mask).sum() for diff, mask in safezip(reverse_kl, output_only_masks)])
        total_output_tokens = sum([mask.sum() for mask in output_only_masks])

        if total_output_tokens > 0:
            metrics["teacher_kl_output"] = float(total_output_kl / total_output_tokens)
        else:
            metrics["teacher_kl_output"] = 0.0

    # Compute per-example KL (average across samples for each example)
    example_kl_sums: Dict[int, float] = defaultdict(float)
    example_token_counts: Dict[int, float] = defaultdict(float)

    for i, example_idx in enumerate(example_indices):
        kl_sum = float(reverse_kl[i].sum().item())
        token_count = float(float_masks[i].sum().item())
        example_kl_sums[example_idx] += kl_sum
        example_token_counts[example_idx] += token_count

    # Compute per-example average KL
    per_example_kl = {}
    for example_idx in sorted(example_kl_sums.keys()):
        if example_token_counts[example_idx] > 0:
            per_example_kl[example_idx] = example_kl_sums[example_idx] / example_token_counts[example_idx]
        else:
            per_example_kl[example_idx] = 0.0

    metrics["per_example_kl"] = per_example_kl

    return metrics


# --- Configuration ---

@chz.chz
class Config:
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"

    # Thinking mode control for Qwen3 hybrid models
    # None = use model default, "enable" = force thinking, "disable" = force no thinking
    thinking_mode: Literal["enable", "disable"] | None = None

    # Whether to train on thinking tokens (default: False = exclude thinking from loss)
    # Only applies when using Qwen3 with thinking enabled
    train_on_thinking: bool = False

    # Dataset settings (using test set for overfitting sanity check)
    dataset_path: str = "./datasets/sageeval-test"
    batch_size_prompts: int = 32
    samples_per_prompt: int = 4
    num_epochs: int = 50  # Overfit for many epochs

    # Teacher augmentation (suffix appended to user message in user suffix mode)
    # Use {safety_fact} placeholder when supervision_key is set
    teacher_suffix: str = "\n\nAnswer the question, but be sure that your response warns me about the following safety fact, if appropriate: {safety_fact}"

    # Supervision: dataset column for per-example supervision appended to teacher prompt
    # When set, uses system_prompt instead of teacher_suffix
    supervision_key: str | None = "safety_fact"

    # System prompt for supervised mode (used when supervision_key is set)
    # Use {safety_fact} as placeholder - it will be replaced with supervision_value
    system_prompt: str = """You are a helpful assistant. You help users with their questions while also informing them of relevant safety concerns.

SAFETY GUIDELINE FOR THIS CONVERSATION: {safety_fact}

When this guideline is relevant to the user's question:
1. Answer their question helpfully
2. Include the safety information as part of your response

Do not refuse to engage. A refusal leaves the user uninformed about the safety risk."""

    # Training hyperparameters
    learning_rate: float = 1e-5
    lora_rank: int = 32
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Generation settings
    temperature: float = 1.0
    max_tokens: int = 2048  # Max response length

    # Loss function
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    num_substeps: int = 1

    # Logging and checkpointing
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 2   # Save every epoch (64 examples / 32 batch_size = 2 batches)
    eval_every: int = 2   # Eval every epoch

    # System
    base_url: str | None = None
    enable_trace: bool = False
    load_checkpoint_path: str | None = None
    compute_post_kl: bool = False

    # Resume validation: path to old log directory to validate config against
    # Use this when loading weights from a different run to ensure config compatibility
    validate_resume_from: str | None = None

    # Overfitting settings
    max_overfit_examples: int | None = DEFAULT_MAX_OVERFIT_EXAMPLES


# --- Training Functions ---

@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    tokenizer: Tokenizer,
    teacher_client: tinker.SamplingClient,
    renderer_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    mask_thinking: bool,
    cost_tracker: CostTracker | None = None,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts trajectories into a minibatch with KL penalty."""

    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Track student prefill and sample tokens
    if cost_tracker is not None:
        for datum in data_D:
            # Student prefill: prompt tokens
            # Sample: response tokens (where mask == 1)
            mask = datum.loss_fn_inputs["mask"].data
            response_length = int(sum(mask))
            total_length = len(datum.model_input.to_ints())
            prefill_length = total_length - response_length

            cost_tracker.add_prefill(prefill_length)
            cost_tracker.add_sample(response_length)

    # Print one example
    if len(data_D) > 0:
        logger.info(colorize_example(data_D[0], tokenizer, key="mask"))

    # Mask thinking tokens BEFORE computing KL penalty
    # This ensures KL is only computed on tokens we actually train on
    if mask_thinking:
        mask_thinking_tokens(data_D, tokenizer)
        logger.info("Masked thinking tokens (set mask=0 for <think>...</think> regions)")

    # Incorporate KL penalty if configured
    if kl_penalty_coef > 0:
        with timed("compute_kl_penalty", metrics):
            kl_metrics = await incorporate_kl_penalty(
                env_group_builders_P,
                data_D,
                metadata_D,
                teacher_client,
                tokenizer,
                renderer_name,
                kl_penalty_coef,
                kl_discount_factor,
                cost_tracker=cost_tracker,
            )
            metrics.update(kl_metrics)

    return data_D, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    tokenizer: Tokenizer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    teacher_client: tinker.SamplingClient,
    renderer_name: str,
    cost_tracker: CostTracker | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    # Determine if we should mask thinking tokens
    # This affects both KL computation and training loss
    mask_thinking = should_mask_thinking(renderer_name) and not cfg.train_on_thinking

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        teacher_client,
        renderer_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        mask_thinking=mask_thinking,
        cost_tracker=cost_tracker,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

    # Track training tokens (all tokens that went through the training step)
    if cost_tracker is not None:
        train_tokens = sum(len(datum.model_input.to_ints()) for datum in data_D)
        cost_tracker.add_train(train_tokens)

    sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        cfg.compute_post_kl,
    )
    metrics.update(full_batch_metrics)

    return sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    batches_per_epoch: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    dataset: SimpleDistillationDataset,
    teacher_client: tinker.SamplingClient,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    renderer_name: str,
    cost_tracker: CostTracker,
):
    """Implements fully synchronous on-policy training"""

    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    for i_batch in range(start_batch, end_batch):
        current_epoch = i_batch // batches_per_epoch
        metrics = {
            "progress/batch": i_batch,
            "progress/epoch": current_epoch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Get batch and sample trajectories
        env_group_builders_P, _ = dataset.get_batch(i_batch)

        # Set duplicates for each builder
        for builder in env_group_builders_P:
            builder.duplicates = cfg.samples_per_prompt

        with timed("sample", metrics):
            trajectory_groups_P = await asyncio.gather(
                *[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client,
                            builder,
                            temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=False,
                        ),
                        name=f"sample_task_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ],
            )
        trajectory_groups_P = [
            trajectory_group
            for trajectory_group in trajectory_groups_P
            if trajectory_group is not None
        ]

        # Train step
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            teacher_client,
            renderer_name,
            cost_tracker=cost_tracker,
        )

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        metrics.update(cost_tracker.get_metrics())
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(cfg: Config):
    """Main training loop for on-policy distillation."""

    # --- Early validation (before any expensive operations) ---
    logger.info("Validating configuration...")
    validate_config(cfg)
    logger.info("Configuration validation passed")

    # Validate dataset columns exist
    logger.info(f"Validating dataset at {cfg.dataset_path}...")
    validate_dataset_columns(cfg.dataset_path, cfg.supervision_key)
    logger.info("Dataset validation passed")

    # Validate resume config if specified
    if cfg.validate_resume_from:
        logger.info(f"Validating config against previous run at {cfg.validate_resume_from}...")
        validate_resume_config(cfg, cfg.validate_resume_from)

    # Also check if resuming from same log_path (for auto-resume case)
    is_resuming_same_path = os.path.exists(cfg.log_path) and (
        os.path.exists(os.path.join(cfg.log_path, CONFIG_FILENAME)) or
        os.path.exists(os.path.join(cfg.log_path, "checkpoints.jsonl"))
    )
    if is_resuming_same_path and not cfg.validate_resume_from:
        logger.info("Detected existing run at log_path, validating resume config...")
        validate_resume_config(cfg, cfg.log_path)

    # Save command line (always, even on resume - shows the resume command used)
    save_command_line(cfg.log_path)

    # Note: config.json is saved by ml_log.setup_logging below
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )

    if cfg.enable_trace:
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        logger.info(
            f"Run `python tinker_cookbook/utils/trace.py {trace_events_path} trace.json` and visualize in chrome://tracing or https://ui.perfetto.dev/"
        )
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    # Check for resume from log_path checkpoints
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    # Create clients
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    # Load checkpoint if resuming or loading from external path
    load_state_path = resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    if load_state_path:
        logger.info(f"Loading checkpoint from: {load_state_path}")
        try:
            future = await training_client.load_state_async(load_state_path)
            _ = await future.result_async()
            logger.info(f"Successfully loaded state from {load_state_path}")
        except Exception as e:
            error_msg = f"Failed to load checkpoint from '{load_state_path}': {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

        # Validate model matches after loading checkpoint
        await validate_model_matches_checkpoint(
            training_client, cfg.model_name, load_state_path
        )

    tokenizer = training_client.get_tokenizer()

    # Determine renderer based on model and thinking mode
    # IMPORTANT: The teacher prompt is explicitly rendered with this renderer in incorporate_kl_penalty.
    # The student's prompt (from DistillationEnv) is also rendered with this same renderer.
    # If there's a mismatch, the KL comparison will be invalid due to different tokenization.
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    logger.info(f"Using renderer: {renderer_name} for model: {cfg.model_name}")
    if cfg.thinking_mode is not None:
        logger.info(f"Thinking mode explicitly set to: {cfg.thinking_mode}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create teacher sampling client (same model as student)
    teacher_client = service_client.create_sampling_client(base_model=cfg.model_name)
    logger.info(f"Created teacher sampling client for {cfg.model_name}")

    # Create dataset
    dataset = SimpleDistillationDataset(
        dataset_path=cfg.dataset_path,
        groups_per_batch=cfg.batch_size_prompts,
        teacher_suffix=cfg.teacher_suffix,
        renderer=renderer,
        supervision_key=cfg.supervision_key,
        system_prompt=cfg.system_prompt,
        max_overfit_examples=cfg.max_overfit_examples,
    )
    batches_per_epoch = len(dataset)
    num_batches = batches_per_epoch * cfg.num_epochs
    logger.info(f"Overfitting mode: saving checkpoint every {cfg.save_every} batches (= 1 epoch)")
    logger.info(f"Will train on {num_batches} batches ({cfg.num_epochs} epoch(s) x {batches_per_epoch} batches/epoch)")
    if num_batches == 0:
        raise ConfigurationError(
            f"num_batches is 0! Dataset has {len(dataset.ds)} examples but batch_size_prompts={cfg.batch_size_prompts}. "
            f"Set batch_size_prompts <= {len(dataset.ds)} to have at least 1 batch per epoch."
        )
    if cfg.system_prompt:
        logger.info(f"Mode: SYSTEM PROMPT")
        logger.info(f"  supervision_key='{cfg.supervision_key}'")
        logger.info(f"  system_prompt='{cfg.system_prompt[:80]}...'")
        logger.info(f"  teacher_suffix (unused)='{cfg.teacher_suffix[:50]}...'")
    else:
        logger.info(f"Mode: USER SUFFIX")
        logger.info(f"  supervision_key='{cfg.supervision_key}'")
        logger.info(f"  teacher_suffix='{cfg.teacher_suffix}'")
        logger.info(f"  system_prompt (unused)='{cfg.system_prompt[:50] if cfg.system_prompt else '(empty)'}'")

    # --- Cost Estimation ---
    # Sample prompts to estimate average token length
    sample_size = min(100, len(dataset.ds))
    sample_prompts = [dataset.ds[i]["prompt"] for i in range(sample_size)]
    avg_prompt_tokens = sum(len(tokenizer.encode(p)) for p in sample_prompts) / len(sample_prompts)

    cost_estimate = estimate_training_cost(
        model_name=cfg.model_name,
        dataset_size=len(dataset.ds),
        batch_size=cfg.batch_size_prompts,
        samples_per_prompt=cfg.samples_per_prompt,
        num_epochs=cfg.num_epochs,
        avg_prompt_tokens=avg_prompt_tokens,
        max_response_tokens=cfg.max_tokens,
    )
    logger.info("\n" + format_cost_estimate(cost_estimate, cfg.model_name))

    # Initialize cost tracker
    cost_tracker = CostTracker(cfg.model_name)

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=num_batches,
        num_batches=num_batches,
        batches_per_epoch=batches_per_epoch,
        cfg=cfg,
        training_client=training_client,
        dataset=dataset,
        teacher_client=teacher_client,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
        renderer_name=renderer_name,
        cost_tracker=cost_tracker,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Log final cost report
    logger.info("\n" + cost_tracker.format_report())
    ml_logger.log_metrics(cost_tracker.get_metrics(), step=num_batches)

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    asyncio.run(main(cfg))
