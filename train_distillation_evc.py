#!/usr/bin/env python3

"""
EVC (Explicit Verbal Cues) distillation training with student prefill support.

This script extends the base distillation training to support:
1. Student prefill: The student's response starts with a configurable prefix (e.g., "<safety_thinking>")
2. EVC teacher prompt: The teacher uses a structured prompt that expects <safety_thinking> tags

Key differences from train_distillation.py:
- student_prefill parameter: Prefill string for student responses (default: "<safety_thinking>")
- EVC_TEACHER_PROMPT: Full EVC prompt that doesn't use {supervision_key} placeholder
- Prefill logprobs computed via separate forward pass (Tinker sampling only returns generated token logprobs)
"""

import asyncio
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
import numpy as np
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
    if cfg.kl_topk_tokens < 1:
        errors.append(f"kl_topk_tokens must be >= 1, got {cfg.kl_topk_tokens}")
    if cfg.kl_topk_tokens > 20:
        errors.append(
            f"kl_topk_tokens={cfg.kl_topk_tokens} exceeds Tinker API limit of 20. "
            f"The Tinker sampling API only supports topk_prompt_logprobs up to 20. "
            f"Use kl_topk_tokens=20 for best approximation, or use train_local_distillation.py for full vocab KL."
        )
    if cfg.temperature <= 0:
        errors.append(f"temperature must be positive, got {cfg.temperature}")
    if cfg.max_tokens <= 0:
        errors.append(f"max_tokens must be positive, got {cfg.max_tokens}")
    if cfg.num_substeps <= 0:
        errors.append(f"num_substeps must be positive, got {cfg.num_substeps}")
    if cfg.save_every <= 0:
        errors.append(f"save_every must be positive, got {cfg.save_every}")
    if cfg.eval_every <= 0:
        errors.append(f"eval_every must be positive, got {cfg.eval_every}")
    if cfg.max_steps is not None and cfg.max_steps <= 0:
        errors.append(f"max_steps must be positive, got {cfg.max_steps}")
    if cfg.policy_update_interval <= 0:
        errors.append(f"policy_update_interval must be positive, got {cfg.policy_update_interval}")

    # --- Shuffle mode validation ---
    valid_shuffle_modes = [None, "none", "epoch", "batch"]
    if cfg.shuffle not in valid_shuffle_modes:
        errors.append(f"shuffle must be one of {valid_shuffle_modes}, got {cfg.shuffle}")

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
    is_openai_model = "openai" in cfg.model_name.lower() or "gpt-oss" in cfg.model_name.lower()
    supports_thinking_mode = is_qwen_model or is_openai_model

    if cfg.thinking_mode is not None and not supports_thinking_mode:
        errors.append(
            f"thinking_mode='{cfg.thinking_mode}' is only valid for Qwen or OpenAI models, "
            f"but model_name='{cfg.model_name}'"
        )
    # Validate reasoning_effort only applies to OpenAI with thinking enabled
    if cfg.reasoning_effort is not None:
        if not is_openai_model:
            errors.append(
                f"reasoning_effort='{cfg.reasoning_effort}' is only valid for OpenAI models, "
                f"but model_name='{cfg.model_name}'"
            )
        elif cfg.thinking_mode != "enable":
            errors.append(
                f"reasoning_effort='{cfg.reasoning_effort}' requires thinking_mode='enable', "
                f"but thinking_mode='{cfg.thinking_mode}'"
            )

    # --- EVC-specific: No {supervision_key} validation needed ---
    # EVC teacher prompt doesn't use {supervision_key} placeholder

    # Log warnings
    for warning in warnings:
        logger.warning(f"CONFIG WARNING: {warning}")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigurationError(error_msg)


def validate_dataset_columns(dataset_path: str, supervision_key: str) -> None:
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

    if supervision_key not in columns:
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
    try:
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
    tokenizer = training_client.get_tokenizer()
    expected_family = _extract_model_family(expected_model_name)
    logger.info(f"Expected model: {expected_model_name} (family: {expected_family})")
    logger.info(f"Loaded checkpoint from: {checkpoint_path}")


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

CONFIG_FILENAME = "config.json"
COMMAND_FILENAME = "command.txt"


def save_command_line(log_path: str) -> None:
    """Save the full command line used to invoke the script. Appends on resume."""
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


# These config keys must match when resuming training
CRITICAL_CONFIG_KEYS = [
    "model_name",
    "dataset_path",
    "batch_size_prompts",
    "samples_per_prompt",
    "teacher_mode",
    "teacher_prompt",
    "supervision_key",
    "lora_rank",
    "loss_fn",
    "kl_penalty_coef",
    "thinking_mode",
    "train_on_thinking",
    "reasoning_effort",  # For OpenAI models
    "shuffle",
    "student_prefill",  # EVC-specific: prefill must match
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
        return

    import chz
    current_config = chz.asdict(cfg)
    mismatches: list[str] = []

    for key in CRITICAL_CONFIG_KEYS:
        saved_value = saved_config.get(key)
        if saved_value is None:
            continue  # Key not in saved config (e.g. cross-pipeline resume) — skip
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
    Supports student prefill for EVC-style training.
    """
    def __init__(
        self,
        prompt: str,
        teacher_mode: Literal["user", "system"],
        teacher_prompt: str,
        renderer: "renderers.Renderer",
        prefill: str = "",
    ):
        self.prompt = prompt
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt
        self.renderer = renderer
        self.prefill = prefill
        self._done = False

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        conversation = [{"role": "user", "content": self.prompt}]
        # Build prompt with prefill if specified
        model_input = self.renderer.build_generation_prompt(
            conversation,
            prefill=self.prefill if self.prefill else None
        )
        return model_input, self.renderer.get_stop_sequences()

    async def step(self, action: Any) -> StepResult:
        self._done = True
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.from_ints([]),
            next_stop_condition=[],
        )


class DistillationEnvGroupBuilder(EnvGroupBuilder):
    """Builder that creates multiple identical environments for sampling."""
    def __init__(
        self,
        prompt: str,
        teacher_mode: Literal["user", "system"],
        teacher_prompt: str,
        renderer: "renderers.Renderer",
        duplicates: int = 1,
        supervision_value: str = "",
        prefill: str = "",
    ):
        self.prompt = prompt
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt
        self.renderer = renderer
        self.duplicates = duplicates
        self.supervision_value = supervision_value
        self.prefill = prefill

    async def make_envs(self) -> Sequence[Env]:
        return [
            DistillationEnv(
                self.prompt,
                self.teacher_mode,
                self.teacher_prompt,
                self.renderer,
                self.prefill,
            )
            for _ in range(self.duplicates)
        ]

    def logging_tags(self) -> dict[str, str]:
        return {"type": "distillation"}


class SimpleDistillationDataset:
    """
    Loads a HuggingFace dataset from disk and yields EnvGroupBuilders.
    Each builder stores the original prompt text for later teacher augmentation.
    Supports student prefill for EVC-style training.
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
        prefill: str = "",
    ):
        logger.info(f"Loading dataset from disk: {dataset_path}")
        self.ds = load_from_disk(dataset_path)
        logger.info(f"Loaded dataset with {len(self.ds)} examples")

        self.groups_per_batch = groups_per_batch
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt
        self.renderer = renderer
        self.supervision_key = supervision_key
        self.prefill = prefill

        self.shuffle_mode = shuffle if shuffle is not None else "none"
        self.indices = list(range(len(self.ds)))
        self._rng = np.random.default_rng(0)

        logger.info(f"Shuffle mode: {self.shuffle_mode}")
        if prefill:
            logger.info(f"Student prefill: {repr(prefill)}")

    def shuffle_indices(self, seed: int) -> None:
        """Shuffle index mapping for a new epoch."""
        if self.shuffle_mode == "epoch":
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
            logger.info(f"Shuffled dataset indices for epoch (seed={seed})")
        elif self.shuffle_mode == "batch":
            self._rng = np.random.default_rng(seed)
            logger.info(f"Reset batch sampling RNG (seed={seed})")

    def get_batch(self, batch_idx: int) -> tuple[Sequence[EnvGroupBuilder], List[int]]:
        builders = []
        indices = []

        if self.shuffle_mode == "batch":
            sampled_indices = self._rng.integers(0, len(self.ds), size=self.groups_per_batch)

            for idx in sampled_indices:
                idx = int(idx)
                row = self.ds[idx]
                prompt = row['prompt']
                supervision_value = row[self.supervision_key]

                builders.append(DistillationEnvGroupBuilder(
                    prompt, self.teacher_mode, self.teacher_prompt, self.renderer,
                    supervision_value=supervision_value,
                    prefill=self.prefill,
                ))
                indices.append(idx)
        else:
            start_idx = batch_idx * self.groups_per_batch

            for i in range(self.groups_per_batch):
                idx = start_idx + i
                actual_idx = self.indices[idx % len(self.indices)]
                row = self.ds[actual_idx]
                prompt = row['prompt']
                supervision_value = row[self.supervision_key]

                builders.append(DistillationEnvGroupBuilder(
                    prompt, self.teacher_mode, self.teacher_prompt, self.renderer,
                    supervision_value=supervision_value,
                    prefill=self.prefill,
                ))
                indices.append(actual_idx)

        return builders, indices

    def __len__(self) -> int:
        return len(self.ds) // self.groups_per_batch


# --- Prefill Logprob Computation ---

async def compute_prefill_logprobs(
    student_client: tinker.SamplingClient,
    prompt_tokens: list[int],
    prefill_tokens: list[int],
) -> list[float]:
    """
    Compute logprobs for prefill tokens via forward pass.

    The Tinker sampling API returns logprobs only for GENERATED tokens, not prefill.
    To get prefill logprobs, we do a forward pass with the full sequence
    (prompt + prefill) and extract logprobs for the prefill portion.

    Args:
        student_client: Sampling client for the student model.
        prompt_tokens: Tokens for the prompt (without prefill).
        prefill_tokens: Tokens for the prefill string.

    Returns:
        List of logprobs for each prefill token.
    """
    if not prefill_tokens:
        return []

    full_seq = prompt_tokens + prefill_tokens
    inp = types.ModelInput.from_ints(full_seq)
    sampling_params = types.SamplingParams(max_tokens=1)

    resp = await student_client.sample_async(
        prompt=inp,
        num_samples=1,
        sampling_params=sampling_params,
        include_prompt_logprobs=True,
    )

    # prompt_logprobs[i] = logprob of token i given tokens [0:i]
    # Prefill logprobs are at indices [len(prompt_tokens) : len(prompt_tokens) + len(prefill_tokens)]
    prompt_len = len(prompt_tokens)
    prefill_logprobs = resp.prompt_logprobs[prompt_len:prompt_len + len(prefill_tokens)]
    return [lp if lp is not None else 0.0 for lp in prefill_logprobs]


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
    kl_topk_tokens: int = 1,
    student_client: tinker.SamplingClient | None = None,
    prefill: str = "",
) -> Dict[str, float]:
    """
    Compute reverse KL between student and teacher, where the teacher
    sees an augmented user prompt (original + suffix).

    For EVC training with prefill:
    - Student generates: [prompt] -> [prefill + response]
    - Sampling returns logprobs only for [response] (generated tokens)
    - We compute prefill logprobs separately via forward pass
    - Full student logprobs = [prefill_logprobs] + [response_logprobs]
    - Teacher evaluates: [augmented_prompt] + [prefill + response]

    Args:
        prefill: Prefill string for student responses (e.g., "<safety_thinking>").
                 If provided, prefill logprobs are computed separately.
    """
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    use_topk = kl_topk_tokens > 1

    if use_topk:
        assert student_client is not None, (
            f"student_client is required when kl_topk_tokens={kl_topk_tokens}. "
            f"Pass the current sampling_client to get student's top-k logprobs."
        )

    # Tokenize prefill once
    prefill_tokens = tokenizer.encode(prefill, add_special_tokens=False) if prefill else []
    if prefill_tokens:
        logger.info(f"Computing KL with prefill: {repr(prefill)} ({len(prefill_tokens)} tokens)")

    teacher_inputs = []
    student_inputs = []
    datum_indices = []
    response_lengths = []
    prefill_requests = []  # Track which data need prefill logprob computation
    logged_sample = False

    for datum_idx, (datum, metadata) in enumerate(zip(data_D, metadata_D)):
        group_idx = metadata["group_idx"]

        if group_idx >= len(env_group_builders):
            continue

        builder = env_group_builders[group_idx]

        if not isinstance(builder, DistillationEnvGroupBuilder):
            continue

        original_prompt = builder.prompt
        teacher_mode = builder.teacher_mode
        teacher_prompt_template = builder.teacher_prompt
        supervision_value = builder.supervision_value

        # Create augmented teacher prompt
        # For EVC, teacher_prompt doesn't use {supervision_key}, but we keep the format call
        # in case users want to customize with supervision
        if "{supervision_key}" in teacher_prompt_template:
            formatted_teacher_prompt = teacher_prompt_template.format(supervision_key=supervision_value)
        else:
            formatted_teacher_prompt = teacher_prompt_template

        if teacher_mode == "system":
            conversation = [
                {"role": "system", "content": formatted_teacher_prompt},
                {"role": "user", "content": original_prompt},
            ]
        else:
            teacher_user_message = f"{original_prompt}{formatted_teacher_prompt}"
            conversation = [{"role": "user", "content": teacher_user_message}]
        teacher_prompt_formatted = renderer.build_generation_prompt(conversation)

        if not logged_sample:
            if teacher_mode == "system":
                logger.info(f"Sample teacher prompt (system mode):\nSystem: {formatted_teacher_prompt[:200]}...")
            else:
                logger.info(f"Sample teacher prompt (user mode):\nUser: {teacher_user_message[:200]}...")
            logged_sample = True

        # Get student's response tokens from the datum
        student_full_seq = datum.model_input.append_int(
            cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
        )

        mask = datum.loss_fn_inputs["mask"].data
        response_length = int(sum(mask))

        if response_length == 0:
            logger.warning(
                f"Skipping sample with zero response length (all tokens masked). "
                f"Datum index: {datum_idx}"
            )
            continue

        student_full_seq_tokens = student_full_seq.to_ints()
        response_tokens = student_full_seq_tokens[-response_length:]

        # For EVC with prefill:
        # - response_tokens includes prefill (it's part of the response the student "generated")
        # - The sampling logprobs in datum only cover the generated portion (after prefill)
        # - We need to prepend prefill logprobs to get the full response logprobs

        # Teacher's full sequence: augmented_prompt + full_response (including prefill)
        teacher_full_seq_tokens = teacher_prompt_formatted.to_ints() + response_tokens
        teacher_input = types.ModelInput.from_ints(teacher_full_seq_tokens)

        student_input = types.ModelInput.from_ints(student_full_seq_tokens)

        if cost_tracker is not None:
            cost_tracker.add_prefill(len(teacher_full_seq_tokens))
            if use_topk:
                cost_tracker.add_prefill(len(student_full_seq_tokens))

        teacher_inputs.append(teacher_input)
        student_inputs.append(student_input)
        datum_indices.append(datum_idx)
        response_lengths.append(response_length)

        # Track prefill request info
        if prefill_tokens:
            # Build prompt tokens WITHOUT prefill for prefill logprob computation
            student_conversation = [{"role": "user", "content": original_prompt}]
            prompt_no_prefill = renderer.build_generation_prompt(student_conversation)
            prefill_requests.append({
                "datum_idx": datum_idx,
                "prompt_tokens": prompt_no_prefill.to_ints(),
            })

    if not teacher_inputs:
        return {}

    effective_topk = kl_topk_tokens

    # Compute prefill logprobs for all samples in parallel
    prefill_logprobs_map = {}
    if prefill_tokens and student_client is not None:
        # Deduplicate by prompt tokens (same prompt = same prefill logprobs)
        unique_prompts = {}
        for req in prefill_requests:
            prompt_key = tuple(req["prompt_tokens"])
            if prompt_key not in unique_prompts:
                unique_prompts[prompt_key] = req["prompt_tokens"]

        # Compute prefill logprobs for unique prompts
        prefill_tasks = [
            compute_prefill_logprobs(student_client, prompt_tokens, prefill_tokens)
            for prompt_tokens in unique_prompts.values()
        ]
        prefill_results = await asyncio.gather(*prefill_tasks)

        # Map results back
        for prompt_key, logprobs in zip(unique_prompts.keys(), prefill_results):
            prefill_logprobs_map[prompt_key] = logprobs

        if cost_tracker is not None:
            # Track cost for prefill logprob computation
            for prompt_tokens in unique_prompts.values():
                cost_tracker.add_prefill(len(prompt_tokens) + len(prefill_tokens))

    # Batch compute logprobs
    if use_topk:
        sampling_params = types.SamplingParams(max_tokens=1)

        all_responses = await asyncio.gather(
            *[
                student_client.sample_async(
                    prompt=inp,
                    num_samples=1,
                    sampling_params=sampling_params,
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=effective_topk,
                )
                for inp in student_inputs
            ],
            *[
                teacher_client.sample_async(
                    prompt=inp,
                    num_samples=1,
                    sampling_params=sampling_params,
                    include_prompt_logprobs=True,
                    topk_prompt_logprobs=effective_topk,
                )
                for inp in teacher_inputs
            ],
        )

        n = len(student_inputs)
        student_responses = all_responses[:n]
        teacher_responses = all_responses[n:]

        teacher_logprobs_list = []
        teacher_topk_data_list = []
        student_topk_data_list = []

        for student_resp, teacher_resp in zip(student_responses, teacher_responses):
            if teacher_resp.prompt_logprobs is not None:
                teacher_logprobs_list.append(teacher_resp.prompt_logprobs)
            else:
                teacher_logprobs_list.append([])

            student_topk = getattr(student_resp, 'topk_prompt_logprobs', None)
            teacher_topk = getattr(teacher_resp, 'topk_prompt_logprobs', None)

            assert student_topk is not None
            assert teacher_topk is not None

            student_topk_data_list.append(student_topk)
            teacher_topk_data_list.append(teacher_topk)
    else:
        teacher_logprobs_list = await asyncio.gather(
            *[teacher_client.compute_logprobs_async(inp) for inp in teacher_inputs]
        )
        teacher_topk_data_list = [None] * len(teacher_inputs)
        student_topk_data_list = [None] * len(teacher_inputs)

    # Get sampling logprobs from datums (these are for generated tokens only, not prefill)
    sampled_logprobs_D = [data_D[i].loss_fn_inputs["logprobs"].to_torch() for i in datum_indices]
    float_masks = [data_D[i].loss_fn_inputs["mask"].to_torch().float() for i in datum_indices]

    reverse_kl = []

    for idx, (teacher_logprobs, teacher_topk_data, student_topk_data, sampled_logprobs, mask, response_length) in enumerate(
        safezip(teacher_logprobs_list, teacher_topk_data_list, student_topk_data_list, sampled_logprobs_D, float_masks, response_lengths)
    ):
        assert sampled_logprobs.dim() == 1
        assert mask.dim() == 1
        assert sampled_logprobs.shape == mask.shape

        # For prefill: prepend prefill logprobs to sampled_logprobs
        if prefill_tokens and prefill_requests:
            # Find the prefill request for this datum
            datum_idx = datum_indices[idx]
            req = next((r for r in prefill_requests if r["datum_idx"] == datum_idx), None)
            if req:
                prompt_key = tuple(req["prompt_tokens"])
                prefill_lps = prefill_logprobs_map.get(prompt_key, [])
                if prefill_lps:
                    # The sampled_logprobs from datum covers the generated portion
                    # We need to account for prefill tokens in the response
                    # response_length includes prefill tokens
                    # But sampled_logprobs only has (response_length - len(prefill_tokens)) entries
                    #
                    # Actually, looking at the mask and trajectory assembly:
                    # - The datum's response includes prefill (since prefill was in initial_observation)
                    # - But the sampling logprobs from Tinker only cover tokens AFTER the prefill
                    # - We need to insert prefill logprobs at the right position

                    # The mask has 1s for all response tokens (including prefill)
                    # The sampled_logprobs tensor should match mask length
                    # But the ACTUAL logprobs from sampling only cover generated tokens

                    # Let's verify: response_length = sum(mask) = total response tokens
                    # sampled_logprobs should be same length as model_input sequence
                    pass  # We handle this below

        if use_topk:
            assert student_topk_data is not None
            assert teacher_topk_data is not None

            assert len(student_topk_data) >= response_length
            assert len(teacher_topk_data) >= response_length
            student_response_topk = student_topk_data[-response_length:]
            teacher_response_topk = teacher_topk_data[-response_length:]

            weighted_kl_per_token = []
            for pos_idx, (student_pos_topk, teacher_pos_topk) in enumerate(
                zip(student_response_topk, teacher_response_topk)
            ):
                assert student_pos_topk is not None
                assert teacher_pos_topk is not None

                def to_dict(pos_topk):
                    if isinstance(pos_topk, dict):
                        return pos_topk
                    elif isinstance(pos_topk, list):
                        return {token_id: lp for token_id, lp in pos_topk}
                    else:
                        raise ValueError(f"Unexpected format: {type(pos_topk)}")

                student_topk_dict = to_dict(student_pos_topk)
                teacher_topk_dict = to_dict(teacher_pos_topk)

                weighted_kl = 0.0
                for token_id, student_lp in student_topk_dict.items():
                    if student_lp is None:
                        continue

                    student_prob = np.exp(student_lp)
                    teacher_lp = teacher_topk_dict.get(token_id)
                    if teacher_lp is None:
                        teacher_lp = -100.0

                    weighted_kl += student_prob * (student_lp - teacher_lp)

                weighted_kl_per_token.append(weighted_kl)

            response_kl = torch.tensor(weighted_kl_per_token, dtype=torch.float32)
        else:
            response_logprobs_raw = teacher_logprobs[-response_length:] if teacher_logprobs else []
            if response_logprobs_raw:
                assert not any(lp is None for lp in response_logprobs_raw)

            teacher_logprobs_clean = [lp if lp is not None else 0.0 for lp in teacher_logprobs]
            t_logprobs = torch.tensor(teacher_logprobs_clean)

            assert t_logprobs.dim() == 1
            assert len(t_logprobs) >= response_length

            teacher_response_logprobs = t_logprobs[-response_length:]
            student_response_logprobs = sampled_logprobs[-response_length:]

            assert teacher_response_logprobs.shape == student_response_logprobs.shape

            response_kl = student_response_logprobs - teacher_response_logprobs

        assert response_kl.shape == (response_length,)

        kl = torch.zeros_like(sampled_logprobs)
        kl[-response_length:] = response_kl

        assert kl.shape == sampled_logprobs.shape

        reverse_kl.append(kl)

    # Update advantages in-place
    for i, datum_idx in enumerate(datum_indices):
        datum = data_D[datum_idx]

        existing_advantages = datum.loss_fn_inputs["advantages"].to_torch()
        assert float_masks[i].shape == reverse_kl[i].shape
        assert existing_advantages.shape == reverse_kl[i].shape

        kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]

        assert kl_advantages.shape == existing_advantages.shape

        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )
            assert kl_advantages.shape == existing_advantages.shape

        datum.loss_fn_inputs["advantages"] = types.TensorData.from_torch(
            existing_advantages + kl_advantages
        )

    # Compute metrics
    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / sum(
        [mask.sum() for mask in float_masks]
    )

    metrics = {"teacher_kl": float(avg_logp_diff)}

    if should_mask_thinking(renderer_name):
        output_only_masks = []
        for i, datum_idx in enumerate(datum_indices):
            datum = data_D[datum_idx]
            tokens = datum.model_input.to_ints()
            final_token = int(datum.loss_fn_inputs["target_tokens"].data[-1])
            full_tokens = tokens + [final_token]
            output_mask = get_output_only_mask(full_tokens, float_masks[i], tokenizer, renderer_name)
            output_only_masks.append(output_mask)

        total_output_kl = sum([(diff * mask).sum() for diff, mask in safezip(reverse_kl, output_only_masks)])
        total_output_tokens = sum([mask.sum() for mask in output_only_masks])

        if total_output_tokens > 0:
            metrics["teacher_kl_output"] = float(total_output_kl / total_output_tokens)
        else:
            metrics["teacher_kl_output"] = 0.0

    return metrics


# --- EVC Teacher Prompt ---

EVC_TEACHER_PROMPT = """Your job is to produce a safe, policy-aligned response that follows the structure below.

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
- You should err on the side of explicitly telling users not to do something whenever there is any potential safety risk, rather than suggesting safer alternatives. THIS IS EXTREMELY IMPORTANT."""

DEFAULT_TEACHER_PROMPTS = {
    "user": "\n\nAnswer the question safely and helpfully.",
    "system": EVC_TEACHER_PROMPT,
}


def get_teacher_prompt(cfg: "Config") -> str:
    """Resolve the teacher prompt, using mode-dependent default if not specified."""
    if cfg.teacher_prompt is not None:
        return cfg.teacher_prompt
    return DEFAULT_TEACHER_PROMPTS[cfg.teacher_mode]


# --- Configuration ---

@chz.chz
class Config:
    # Model settings
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Thinking mode control for Qwen3 hybrid models and OpenAI models
    thinking_mode: Literal["enable", "disable"] | None = None

    # Whether to train on thinking tokens (default: True = include thinking in loss)
    train_on_thinking: bool = True

    # Reasoning effort for OpenAI models (only valid when thinking_mode="enable")
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    # Dataset settings
    dataset_path: str = "./datasets/sageeval-train"
    batch_size_prompts: int = 8
    samples_per_prompt: int = 3
    num_epochs: int = 3

    # Teacher prompt configuration
    teacher_mode: Literal["user", "system"] = "system"
    supervision_key: str = "safety_fact"
    teacher_prompt: str | None = None

    # EVC-specific: Student prefill
    # The student's response will start with this string (e.g., "<safety_thinking>")
    # This is included in the KL computation
    student_prefill: str = "<safety_thinking>"

    # Training hyperparameters
    learning_rate: float = 5e-6
    lora_rank: int = 32
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0
    kl_topk_tokens: int = 1

    # Early stopping
    max_steps: int | None = None

    # Generation settings
    temperature: float = 1.0
    max_tokens: int = 1024

    # Loss function
    loss_fn: Literal["importance_sampling", "ppo"] = "importance_sampling"
    num_substeps: int = 1

    # Policy update interval
    policy_update_interval: int = 1

    # Data shuffling strategy
    shuffle: Literal["none", "epoch", "batch"] | None = None

    # Logging and checkpointing
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 20
    eval_every: int = 20

    # System
    base_url: str | None = None
    enable_trace: bool = False
    load_checkpoint_path: str | None = None
    compute_post_kl: bool = False
    validate_resume_from: str | None = None


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
    cost_tracker: CostTracker | None = None,
    kl_topk_tokens: int = 1,
    student_client: tinker.SamplingClient | None = None,
    prefill: str = "",
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts trajectories into a minibatch with KL penalty."""

    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    if cost_tracker is not None:
        for datum in data_D:
            mask = datum.loss_fn_inputs["mask"].data
            response_length = int(sum(mask))
            total_length = len(datum.model_input.to_ints())
            prefill_length = total_length - response_length

            cost_tracker.add_prefill(prefill_length)
            cost_tracker.add_sample(response_length)

    if len(data_D) > 0:
        logger.info(colorize_example(data_D[0], tokenizer, key="mask"))

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
                kl_topk_tokens=kl_topk_tokens,
                student_client=student_client,
                prefill=prefill,
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
    student_client: tinker.SamplingClient | None = None,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        tokenizer,
        teacher_client,
        renderer_name,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
        cost_tracker=cost_tracker,
        kl_topk_tokens=cfg.kl_topk_tokens,
        student_client=student_client,
        prefill=cfg.student_prefill,
    )
    metrics.update(prepare_minibatch_metrics)

    if should_mask_thinking(renderer_name) and not cfg.train_on_thinking:
        mask_thinking_tokens(data_D, tokenizer, renderer_name)
        logger.info(f"Masked thinking tokens for renderer '{renderer_name}'")

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

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
    """Implements fully synchronous on-policy training with policy_update_interval and shuffle support."""

    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    if start_batch > 0:
        resume_epoch = start_batch // batches_per_epoch
        dataset.shuffle_indices(seed=resume_epoch)

    for i_batch in range(start_batch, end_batch):
        current_epoch = i_batch // batches_per_epoch
        batch_in_epoch = i_batch % batches_per_epoch

        if batch_in_epoch == 0:
            dataset.shuffle_indices(seed=current_epoch)

        metrics = {
            "progress/batch": i_batch,
            "progress/epoch": current_epoch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
            "progress/policy_version": i_batch // cfg.policy_update_interval,
        }
        t_start = time.time()

        env_group_builders_P, _ = dataset.get_batch(i_batch)

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

        if trajectory_groups_P and trajectory_groups_P[0].trajectories_G:
            sample_trajectory = trajectory_groups_P[0].trajectories_G[0]
            sample_prompt = env_group_builders_P[0].prompt
            sample_tokens = []
            for transition in sample_trajectory.transitions:
                sample_tokens.extend(transition.ac.tokens)
            sample_response = tokenizer.decode(sample_tokens)
            logger.info(f"Sample from batch {i_batch}:\n--- PROMPT ---\n{sample_prompt}\n--- RESPONSE ---\n{sample_response}\n--- END ---")

        new_sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            tokenizer,
            env_group_builders_P,
            trajectory_groups_P,
            teacher_client,
            renderer_name,
            cost_tracker=cost_tracker,
            student_client=sampling_client,
        )

        should_update_policy = (
            (i_batch + 1) % cfg.policy_update_interval == 0
        ) or (i_batch == end_batch - 1)

        if should_update_policy:
            sampling_client = new_sampling_client
            metrics["progress/policy_updated"] = 1
        else:
            metrics["progress/policy_updated"] = 0

        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        metrics.update(cost_tracker.get_metrics())
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(cfg: Config):
    """Main training loop for EVC on-policy distillation."""

    logger.info("Validating configuration...")
    validate_config(cfg)
    logger.info("Configuration validation passed")

    logger.info(f"Validating dataset at {cfg.dataset_path}...")
    validate_dataset_columns(cfg.dataset_path, cfg.supervision_key)
    logger.info("Dataset validation passed")

    if cfg.validate_resume_from:
        logger.info(f"Validating config against previous run at {cfg.validate_resume_from}...")
        validate_resume_config(cfg, cfg.validate_resume_from)

    is_resuming_same_path = os.path.exists(cfg.log_path) and (
        os.path.exists(os.path.join(cfg.log_path, CONFIG_FILENAME)) or
        os.path.exists(os.path.join(cfg.log_path, "checkpoints.jsonl"))
    )
    if is_resuming_same_path and not cfg.validate_resume_from:
        logger.info("Detected existing run at log_path, validating resume config...")
        validate_resume_config(cfg, cfg.log_path)

    save_command_line(cfg.log_path)

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
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    start_batch = resume_info["batch"] if resume_info else 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

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

        await validate_model_matches_checkpoint(
            training_client, cfg.model_name, load_state_path
        )

    tokenizer = training_client.get_tokenizer()

    renderer_name = get_renderer_name_with_thinking_mode(
        cfg.model_name, cfg.thinking_mode, cfg.reasoning_effort
    )
    logger.info(f"Using renderer: {renderer_name} for model: {cfg.model_name}")
    if cfg.thinking_mode is not None:
        logger.info(f"Thinking mode explicitly set to: {cfg.thinking_mode}")
    if cfg.reasoning_effort is not None:
        logger.info(f"Reasoning effort: {cfg.reasoning_effort}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    teacher_client = service_client.create_sampling_client(base_model=cfg.model_name)
    logger.info(f"Created teacher sampling client for {cfg.model_name}")

    resolved_teacher_prompt = get_teacher_prompt(cfg)

    # Log EVC-specific settings
    logger.info(f"EVC Training Configuration:")
    logger.info(f"  student_prefill: {repr(cfg.student_prefill)}")
    logger.info(f"  Teacher prompt (first 200 chars): {resolved_teacher_prompt[:200]}...")

    dataset = SimpleDistillationDataset(
        dataset_path=cfg.dataset_path,
        groups_per_batch=cfg.batch_size_prompts,
        teacher_mode=cfg.teacher_mode,
        teacher_prompt=resolved_teacher_prompt,
        renderer=renderer,
        supervision_key=cfg.supervision_key,
        shuffle=cfg.shuffle,
        prefill=cfg.student_prefill,
    )
    batches_per_epoch = len(dataset)
    num_batches = batches_per_epoch * cfg.num_epochs

    if cfg.max_steps is not None:
        if cfg.max_steps < num_batches:
            logger.info(f"max_steps={cfg.max_steps} limits training (would be {num_batches} batches for full epochs)")
            num_batches = cfg.max_steps
        else:
            logger.info(f"max_steps={cfg.max_steps} has no effect (only {num_batches} batches needed)")

    logger.info(f"Will train on {num_batches} batches ({cfg.num_epochs} epoch(s) x {batches_per_epoch} batches/epoch)")
    logger.info(f"Teacher mode: {cfg.teacher_mode.upper()}")

    # Cost estimation
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

    cost_tracker = CostTracker(cfg.model_name)

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

    logger.info("\n" + cost_tracker.format_report())
    ml_logger.log_metrics(cost_tracker.get_metrics(), step=num_batches)

    ml_logger.close()
    logger.info("EVC distillation training completed successfully")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    asyncio.run(main(cfg))
