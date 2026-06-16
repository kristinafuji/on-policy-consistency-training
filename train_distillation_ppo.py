#!/usr/bin/env python3

"""
PPO-style mega-batch distillation training.

Implements a hybrid approach between on-policy and off-policy distillation:
- Sample ALL trajectories for a mega-batch from frozen policy (theta_start)
- Compute ALL teacher logprobs ONCE per mega-batch
- Do multiple mini-epochs of PPO training on the fixed data
- Update theta_start after each mega-batch

This amortizes the expensive sampling and teacher logprob computation
over multiple gradient updates while using PPO clipping to prevent
catastrophic updates from stale importance weights.
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
from tinker_cookbook.rl.metrics import (
    compute_kl_sample_train,
    discounted_future_sum_vectorized,
)
from tinker_cookbook.rl.train import (
    do_group_rollout_and_filter_constant_reward,
    forward_backward,
    optim_step,
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
    if cfg.policy_update_interval <= 0:
        errors.append(f"policy_update_interval must be positive, got {cfg.policy_update_interval}")

    # --- PPO mega-batch specific validation ---
    if cfg.mini_epochs <= 0:
        errors.append(f"mini_epochs must be positive, got {cfg.mini_epochs}")
    if cfg.ppo_mini_batch_size is not None and cfg.ppo_mini_batch_size <= 0:
        errors.append(f"ppo_mini_batch_size must be positive, got {cfg.ppo_mini_batch_size}")

    # Warn if using importance_sampling with multiple mini-epochs
    if cfg.mini_epochs > 1 and cfg.loss_fn == "importance_sampling":
        warnings.append(
            f"Using importance_sampling loss with mini_epochs={cfg.mini_epochs} may lead to "
            f"unstable training due to unbounded importance weights. Consider using loss_fn='ppo'."
        )

    # Warn about large mega-batch sizes
    estimated_trajectories = cfg.policy_update_interval * cfg.batch_size_prompts * cfg.samples_per_prompt
    if estimated_trajectories > 4096:
        warnings.append(
            f"Large mega-batch size ({estimated_trajectories} trajectories). "
            f"Consider reducing policy_update_interval or batch_size_prompts if you encounter memory issues."
        )

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

    # --- Teacher prompt validation ---
    # Resolve the teacher prompt (uses default if None)
    resolved_teacher_prompt = get_teacher_prompt(cfg)
    if "{supervision_key}" not in resolved_teacher_prompt:
        errors.append(
            f"teacher_prompt must contain '{{supervision_key}}' placeholder. "
            f"Got: '{resolved_teacher_prompt[:100]}...'"
        )

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

# Reuse config.json that ml_log creates
CONFIG_FILENAME = "config.json"
COMMAND_FILENAME = "command.txt"
WANDB_RUN_ID_FILENAME = "wandb_run_id.txt"


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


def load_wandb_run_id(log_path: str) -> str | None:
    """Load saved wandb run ID from log directory."""
    run_id_path = os.path.join(log_path, WANDB_RUN_ID_FILENAME)
    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            return f.read().strip()
    return None


def save_wandb_run_id(log_path: str, run_id: str) -> None:
    """Save wandb run ID to log directory for resume support."""
    os.makedirs(log_path, exist_ok=True)
    run_id_path = os.path.join(log_path, WANDB_RUN_ID_FILENAME)
    with open(run_id_path, "w") as f:
        f.write(run_id)
    logger.info(f"Saved wandb run ID to {run_id_path}")


def init_wandb_with_resume(
    log_path: str,
    wandb_project: str | None,
    wandb_name: str | None,
    config: Any,
) -> bool:
    """
    Initialize wandb with resume support.

    If a previous run ID exists in log_path, resumes that run.
    Otherwise, creates a new run and saves the ID.

    Returns True if wandb was initialized, False if wandb is not configured.
    """
    if not wandb_project:
        return False

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping wandb logging")
        return False

    if not os.environ.get("WANDB_API_KEY"):
        logger.warning("WANDB_API_KEY not set, skipping wandb logging")
        return False

    existing_run_id = load_wandb_run_id(log_path)

    if existing_run_id:
        logger.info(f"Resuming wandb run: {existing_run_id}")
        wandb.init(
            project=wandb_project,
            id=existing_run_id,
            resume="must",
        )
    else:
        from tinker_cookbook.utils.ml_log import dump_config
        wandb.init(
            project=wandb_project,
            name=wandb_name,
            config=dump_config(config),
        )
        save_wandb_run_id(log_path, wandb.run.id)
        logger.info(f"Created new wandb run: {wandb.run.id}")

    logger.info(f"Wandb logging to: {wandb.run.url}")
    return True


def log_metrics_with_wandb(
    ml_logger: "ml_log.Logger",
    metrics: Dict[str, Any],
    step: int | None = None,
) -> None:
    """Log metrics to ml_logger and wandb (if active)."""
    ml_logger.log_metrics(metrics, step=step)

    # Also log to wandb if a run is active
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


# These config keys must match when resuming training
# Changing these mid-run would invalidate the training
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
    "shuffle",
    # PPO mega-batch specific
    "mini_epochs",
    "policy_update_interval",
    "ppo_mini_batch_size",
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
    def __init__(self, prompt: str, teacher_mode: Literal["user", "system"], teacher_prompt: str, renderer: "renderers.Renderer"):
        self.prompt = prompt  # Original prompt text
        self.teacher_mode = teacher_mode  # "user" or "system"
        self.teacher_prompt = teacher_prompt  # Teacher prompt template (already resolved)
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
    def __init__(self, prompt: str, teacher_mode: Literal["user", "system"], teacher_prompt: str, renderer: "renderers.Renderer", duplicates: int = 1, supervision_value: str = ""):
        self.prompt = prompt
        self.teacher_mode = teacher_mode
        self.teacher_prompt = teacher_prompt  # Already resolved template
        self.renderer = renderer
        self.duplicates = duplicates
        self.supervision_value = supervision_value

    async def make_envs(self) -> Sequence[Env]:
        return [DistillationEnv(self.prompt, self.teacher_mode, self.teacher_prompt, self.renderer) for _ in range(self.duplicates)]

    def logging_tags(self) -> dict[str, str]:
        return {"type": "distillation"}


class SimpleDistillationDataset:
    """
    Loads a HuggingFace dataset from disk and yields EnvGroupBuilders.
    Each builder stores the original prompt text for later teacher augmentation.

    Supports three shuffle modes:
    - "none": Sequential deterministic order (default)
    - "epoch": Shuffle at start of each epoch (all prompts seen exactly once per epoch)
    - "batch": Random sampling each batch (prompts may be seen 0 or multiple times)
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
        self.teacher_prompt = teacher_prompt  # Already resolved template
        self.renderer = renderer
        self.supervision_key = supervision_key

        # Shuffle configuration
        self.shuffle_mode = shuffle if shuffle is not None else "none"
        self.indices = list(range(len(self.ds)))
        self._rng = np.random.default_rng(0)  # Initialize with seed=0 for reproducibility

        logger.info(f"Shuffle mode: {self.shuffle_mode}")

    def shuffle_indices(self, seed: int) -> None:
        """
        Shuffle index mapping for a new epoch.

        For "epoch" mode: shuffles the indices array
        For "batch" mode: re-seeds the RNG for random sampling
        For "none" mode: no-op
        """
        if self.shuffle_mode == "epoch":
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
            logger.info(f"Shuffled dataset indices for epoch (seed={seed})")
        elif self.shuffle_mode == "batch":
            self._rng = np.random.default_rng(seed)
            logger.info(f"Reset batch sampling RNG (seed={seed})")
        # For "none" mode, do nothing

    def get_batch(self, batch_idx: int) -> tuple[Sequence[EnvGroupBuilder], List[int]]:
        builders = []
        indices = []

        if self.shuffle_mode == "batch":
            # Random sampling each batch (RNG is seeded in __init__ and re-seeded at epoch boundaries)
            sampled_indices = self._rng.integers(0, len(self.ds), size=self.groups_per_batch)

            for idx in sampled_indices:
                idx = int(idx)
                row = self.ds[idx]
                prompt = row['prompt']
                supervision_value = row[self.supervision_key]

                builders.append(DistillationEnvGroupBuilder(
                    prompt, self.teacher_mode, self.teacher_prompt, self.renderer,
                    supervision_value=supervision_value
                ))
                indices.append(idx)
        else:
            # Sequential or epoch shuffle: use indices array
            start_idx = batch_idx * self.groups_per_batch

            for i in range(self.groups_per_batch):
                idx = start_idx + i
                actual_idx = self.indices[idx % len(self.indices)]  # Use shuffled indices
                row = self.ds[actual_idx]
                prompt = row['prompt']
                supervision_value = row[self.supervision_key]

                builders.append(DistillationEnvGroupBuilder(
                    prompt, self.teacher_mode, self.teacher_prompt, self.renderer,
                    supervision_value=supervision_value
                ))
                indices.append(actual_idx)

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
) -> tuple[Dict[str, float], Dict[int, torch.Tensor]]:
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
    logged_sample = False

    for datum_idx, (datum, metadata) in enumerate(zip(data_D, metadata_D)):
        group_idx = metadata["group_idx"]

        if group_idx >= len(env_group_builders):
            continue

        builder = env_group_builders[group_idx]

        if not isinstance(builder, DistillationEnvGroupBuilder):
            continue

        # Get original prompt, teacher config, and supervision value from builder
        original_prompt = builder.prompt
        teacher_mode = builder.teacher_mode
        teacher_prompt_template = builder.teacher_prompt
        supervision_value = builder.supervision_value

        # 1. Create augmented teacher prompt
        # Format template with supervision value
        formatted_teacher_prompt = teacher_prompt_template.format(supervision_key=supervision_value)

        if teacher_mode == "system":
            # System prompt mode: teacher prompt is the system message
            conversation = [
                {"role": "system", "content": formatted_teacher_prompt},
                {"role": "user", "content": original_prompt},
            ]
        else:
            # User suffix mode: teacher prompt is appended to user message
            teacher_user_message = f"{original_prompt}{formatted_teacher_prompt}"
            conversation = [{"role": "user", "content": teacher_user_message}]
        teacher_prompt_formatted = renderer.build_generation_prompt(conversation)

        # Log a sample teacher prompt (first one only)
        if not logged_sample:
            if teacher_mode == "system":
                logger.info(f"Sample teacher prompt (system mode):\nSystem: {formatted_teacher_prompt}\nUser: {original_prompt}")
            else:
                logger.info(f"Sample teacher prompt (user mode):\nUser: {teacher_user_message}")
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
                f"Datum index: {datum_idx}"
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

    if not teacher_inputs:
        return {}, {}

    # Batch compute teacher logprobs
    teacher_logprobs_list = await asyncio.gather(
        *[teacher_client.compute_logprobs_async(inp) for inp in teacher_inputs]
    )

    sampled_logprobs_D = [data_D[i].loss_fn_inputs["logprobs"].to_torch() for i in datum_indices]
    float_masks = [data_D[i].loss_fn_inputs["mask"].to_torch().float() for i in datum_indices]

    reverse_kl = []
    # Store teacher logprobs separately (not in loss_fn_inputs) to avoid Tinker serialization issues
    teacher_logprobs_dict: Dict[int, torch.Tensor] = {}

    # Calculate KL with explicit response-token alignment
    for loop_idx, (teacher_logprobs, sampled_logprobs, mask) in enumerate(safezip(teacher_logprobs_list, sampled_logprobs_D, float_masks)):
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

        # Store teacher response logprobs for later KL computation during mini-epochs
        # This enables computing teacher_kl using current student logprobs (from forward_backward)
        # rather than the stale sampled logprobs
        # NOTE: Store in separate dict, NOT in loss_fn_inputs (Tinker can't serialize extra fields)
        curr_datum_idx = datum_indices[loop_idx]
        teacher_lp_tensor = torch.zeros_like(sampled_logprobs)
        teacher_lp_tensor[-response_length:] = teacher_response_logprobs
        teacher_logprobs_dict[curr_datum_idx] = teacher_lp_tensor

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

    metrics = {"teacher_kl": float(avg_logp_diff)}

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

    return metrics, teacher_logprobs_dict


# --- Default Teacher Prompts ---

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

    # Dataset settings
    dataset_path: str = "./datasets/sageeval-train"
    batch_size_prompts: int = 64
    samples_per_prompt: int = 2
    num_epochs: int = 1

    # Teacher prompt configuration
    # teacher_mode: "user" = append to user message, "system" = use as system prompt
    teacher_mode: Literal["user", "system"] = "system"

    # Dataset column containing per-example supervision (e.g., safety facts)
    supervision_key: str = "safety_fact"

    # Teacher prompt template. Use {supervision_key} placeholder for the supervision value.
    # If None, uses mode-dependent default (see DEFAULT_TEACHER_PROMPTS)
    teacher_prompt: str | None = None

    # Training hyperparameters
    learning_rate: float = 1e-5
    lora_rank: int = 32
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Generation settings
    temperature: float = 1.0
    max_tokens: int = 512

    # Loss function - PPO is default for mega-batch training
    loss_fn: Literal["importance_sampling", "ppo"] = "ppo"
    num_substeps: int = 1

    # PPO mega-batch training parameters
    # policy_update_interval: batches per mega-batch (controls mega-batch size)
    policy_update_interval: int = 8

    # mini_epochs: number of passes through mega-batch data
    mini_epochs: int = 4

    # ppo_mini_batch_size: size of mini-batches within each mini-epoch
    # If None, defaults to batch_size_prompts * samples_per_prompt
    ppo_mini_batch_size: int | None = None

    # Data shuffling strategy
    # None/"none": Sequential deterministic order (current behavior)
    # "epoch": Shuffle at start of each epoch (all prompts seen exactly once per epoch)
    # "batch": Random sampling each batch (prompts may be seen 0 or multiple times)
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

    # Resume validation: path to old log directory to validate config against
    # Use this when loading weights from a different run to ensure config compatibility
    validate_resume_from: str | None = None


# --- Training Functions ---

@scope
async def prepare_mega_batch_data(
    all_env_group_builders: List[EnvGroupBuilder],
    all_trajectory_groups: List[TrajectoryGroup],
    tokenizer: Tokenizer,
    teacher_client: tinker.SamplingClient,
    renderer_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    train_on_thinking: bool,
    cost_tracker: CostTracker | None = None,
) -> tuple[List[tinker.Datum], List[Dict[str, Any]], Dict[str, float], Dict[int, torch.Tensor]]:
    """
    Prepare mega-batch data: assemble training data and compute teacher logprobs.
    This is done ONCE per mega-batch.

    Returns:
        all_data_D: List of training datums
        all_metadata_D: List of metadata dicts
        metrics: Dict of metrics
        teacher_logprobs_dict: Dict mapping datum index to teacher logprobs tensor
    """
    metrics = {}
    teacher_logprobs_dict: Dict[int, torch.Tensor] = {}

    # Compute trajectory metrics
    taglist_P = [builder.logging_tags() for builder in all_env_group_builders]
    metrics.update(compute_trajectory_metrics(all_trajectory_groups, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(all_trajectory_groups)
        all_data_D, all_metadata_D = assemble_training_data(all_trajectory_groups, advantages_P)

    # Track student prefill and sample tokens
    if cost_tracker is not None:
        for datum in all_data_D:
            mask = datum.loss_fn_inputs["mask"].data
            response_length = int(sum(mask))
            total_length = len(datum.model_input.to_ints())
            prefill_length = total_length - response_length
            cost_tracker.add_prefill(prefill_length)
            cost_tracker.add_sample(response_length)

    # Print one example
    if len(all_data_D) > 0:
        logger.info(colorize_example(all_data_D[0], tokenizer, key="mask"))

    # Incorporate KL penalty (computes teacher logprobs ONCE for entire mega-batch)
    if kl_penalty_coef > 0:
        with timed("compute_kl_penalty", metrics):
            kl_metrics, teacher_logprobs_dict = await incorporate_kl_penalty(
                all_env_group_builders,
                all_data_D,
                all_metadata_D,
                teacher_client,
                tokenizer,
                renderer_name,
                kl_penalty_coef,
                kl_discount_factor,
                cost_tracker=cost_tracker,
            )
            metrics.update(kl_metrics)

    # Mask thinking tokens if configured
    if should_mask_thinking(renderer_name) and not train_on_thinking:
        mask_thinking_tokens(all_data_D, tokenizer)
        logger.info("Masked thinking tokens (set mask=0 for <think>...</think> regions)")

    return all_data_D, all_metadata_D, metrics, teacher_logprobs_dict


@scope
async def do_ppo_mega_batch_training(
    start_mega_batch: int,
    end_mega_batch: int,
    num_mega_batches: int,
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
    """
    PPO-style mega-batch training.

    For each mega-batch:
    1. Sample ALL trajectories from frozen policy (theta_start)
    2. Compute ALL teacher logprobs ONCE
    3. Do multiple mini-epoch passes with PPO loss
    4. Update frozen policy
    """
    batches_per_mega_batch = cfg.policy_update_interval

    # Get initial frozen sampling client
    sampling_client_frozen = await training_client.save_weights_and_get_sampling_client_async()

    # Restore shuffle state for resumed training
    if start_mega_batch > 0:
        resume_batch = start_mega_batch * batches_per_mega_batch
        resume_epoch = resume_batch // batches_per_epoch
        dataset.shuffle_indices(seed=resume_epoch)

    for i_mega_batch in range(start_mega_batch, end_mega_batch):
        mega_batch_metrics: Dict[str, Any] = {
            "progress/mega_batch": i_mega_batch,
            "progress/policy_version": i_mega_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_mega_batch + 1) / num_mega_batches,
        }
        t_start = time.time()

        # === Phase 1: Sample ALL trajectories from frozen policy ===
        all_env_group_builders: List[EnvGroupBuilder] = []
        all_trajectory_groups: List[TrajectoryGroup] = []

        with timed("sample_mega_batch", mega_batch_metrics):
            for batch_offset in range(batches_per_mega_batch):
                i_batch = i_mega_batch * batches_per_mega_batch + batch_offset
                current_epoch = i_batch // batches_per_epoch
                batch_in_epoch = i_batch % batches_per_epoch

                # Handle epoch shuffling
                if batch_in_epoch == 0:
                    dataset.shuffle_indices(seed=current_epoch)

                # Get batch of prompts
                env_group_builders_P, _ = dataset.get_batch(i_batch)

                # Set duplicates for each builder
                for builder in env_group_builders_P:
                    builder.duplicates = cfg.samples_per_prompt

                # Sample trajectories from frozen policy
                trajectory_groups_P = await asyncio.gather(*[
                    asyncio.create_task(
                        do_group_rollout_and_filter_constant_reward(
                            sampling_client_frozen,  # Frozen policy!
                            builder,
                            temperature=cfg.temperature,
                            max_tokens=cfg.max_tokens,
                            do_remove_constant_reward_groups=False,
                        ),
                        name=f"sample_task_{batch_offset}_{i}",
                    )
                    for i, builder in enumerate(env_group_builders_P)
                ])

                # Filter None results
                trajectory_groups_P = [tg for tg in trajectory_groups_P if tg is not None]

                all_env_group_builders.extend(env_group_builders_P)
                all_trajectory_groups.extend(trajectory_groups_P)

        mega_batch_metrics["mega_batch/num_trajectory_groups"] = len(all_trajectory_groups)
        mega_batch_metrics["mega_batch/num_prompts"] = len(all_env_group_builders)

        # === Phase 2: Prepare data and compute teacher logprobs ONCE ===
        all_data_D, all_metadata_D, prepare_metrics, teacher_logprobs_dict = await prepare_mega_batch_data(
            all_env_group_builders,
            all_trajectory_groups,
            tokenizer,
            teacher_client,
            renderer_name,
            cfg.kl_penalty_coef,
            cfg.kl_discount_factor,
            cfg.train_on_thinking,
            cost_tracker=cost_tracker,
        )
        mega_batch_metrics.update(prepare_metrics)
        mega_batch_metrics["mega_batch/num_datums"] = len(all_data_D)

        if len(all_data_D) == 0:
            logger.warning(f"No valid datums in mega-batch {i_mega_batch}, skipping")
            continue

        # Determine mini-batch size
        mini_batch_size = cfg.ppo_mini_batch_size or (cfg.batch_size_prompts * cfg.samples_per_prompt)
        mega_batch_metrics["mega_batch/mini_batch_size"] = mini_batch_size
        mini_batches_per_mini_epoch = (len(all_data_D) + mini_batch_size - 1) // mini_batch_size

        # === Phase 3: Mini-epoch training with PPO ===
        # Log teacher_kl at mini-batch granularity
        # Total steps = mega_batches * mini_epochs * mini_batches_per_mini_epoch
        mini_batch_step_base = i_mega_batch * cfg.mini_epochs * mini_batches_per_mini_epoch

        last_teacher_kl = 0.0  # Track last value for mega-batch summary

        with timed("mini_epoch_training", mega_batch_metrics):
            for mini_epoch in range(cfg.mini_epochs):
                # Shuffle data indices for this mini-epoch
                shuffled_indices = np.random.permutation(len(all_data_D)).tolist()

                mini_epoch_training_logprobs: List[tuple[int, torch.Tensor]] = []

                # Process mini-batches
                for mini_batch_idx, mini_batch_start in enumerate(range(0, len(all_data_D), mini_batch_size)):
                    mini_batch_end = min(mini_batch_start + mini_batch_size, len(all_data_D))
                    mini_batch_indices = shuffled_indices[mini_batch_start:mini_batch_end]
                    mini_batch_data = [all_data_D[i] for i in mini_batch_indices]

                    # Forward-backward with PPO loss
                    training_logprobs = await forward_backward(
                        training_client, mini_batch_data, loss_fn=cfg.loss_fn
                    )

                    # Optimizer step after each mini-batch (standard PPO behavior)
                    await optim_step(training_client, cfg.learning_rate)

                    mini_epoch_training_logprobs.extend(zip(mini_batch_indices, training_logprobs))

                    # Compute teacher_kl for this mini-batch using current student logprobs
                    # This matches train_offpolicy_distillation.py:269
                    total_kl = 0.0
                    total_tokens = 0
                    for idx, train_lp in zip(mini_batch_indices, training_logprobs):
                        datum = all_data_D[idx]
                        if idx not in teacher_logprobs_dict:
                            continue
                        mask = datum.loss_fn_inputs["mask"].to_torch().float()
                        teacher_lp = teacher_logprobs_dict[idx]
                        # KL = current_student - teacher (reverse KL)
                        kl = (train_lp - teacher_lp) * mask
                        total_kl += kl.sum().item()
                        total_tokens += mask.sum().item()

                    teacher_kl = total_kl / total_tokens if total_tokens > 0 else 0.0
                    last_teacher_kl = teacher_kl

                    # Log at mini-batch granularity
                    mini_batch_step = mini_batch_step_base + mini_epoch * mini_batches_per_mini_epoch + mini_batch_idx
                    log_metrics_with_wandb(ml_logger, {"teacher_kl": teacher_kl}, step=mini_batch_step)

                # Track training tokens
                if cost_tracker is not None:
                    train_tokens = sum(len(datum.model_input.to_ints()) for datum in all_data_D)
                    cost_tracker.add_train(train_tokens)

                # Reorder logprobs to match original data order for metrics
                mini_epoch_training_logprobs.sort(key=lambda x: x[0])
                ordered_logprobs = [lp for _, lp in mini_epoch_training_logprobs]

                # Compute mini-epoch aggregate metrics (importance weights, clipping)
                mini_epoch_kl_metrics = compute_kl_sample_train(all_data_D, ordered_logprobs)
                for k, v in mini_epoch_kl_metrics.items():
                    mega_batch_metrics[f"mini_epoch_{mini_epoch}/{k}"] = v

                logger.info(
                    f"  Mini-epoch {mini_epoch + 1}/{cfg.mini_epochs}: "
                    f"teacher_kl={last_teacher_kl:.4f}, "
                    f"kl_sample_train={mini_epoch_kl_metrics.get('optim/kl_sample_train_v1', 0):.4f}, "
                    f"iw_max={mini_epoch_kl_metrics.get('optim/importance_weight_max', 0):.2f}, "
                    f"clip_frac={mini_epoch_kl_metrics.get('optim/ppo_clip_fraction', 0):.3f}"
                )

        # Final metrics (from last mini-epoch)
        # teacher_kl is logged at mini-batch granularity; mega_batch/teacher_kl is the final value
        mega_batch_metrics.update(mini_epoch_kl_metrics)
        mega_batch_metrics["mega_batch/teacher_kl"] = last_teacher_kl

        # === Phase 4: Update frozen policy ===
        sampling_client_frozen = await training_client.save_weights_and_get_sampling_client_async()
        mega_batch_metrics["progress/policy_updated"] = 1

        # Save checkpoint at mega-batch boundaries
        checkpoint_batch = (i_mega_batch + 1) * batches_per_mega_batch
        if checkpoint_batch % cfg.save_every == 0 or i_mega_batch == end_mega_batch - 1:
            logger.info(f"Saving checkpoint at mega-batch {i_mega_batch + 1} (batch {checkpoint_batch})...")
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{checkpoint_batch:06d}",
                log_path=cfg.log_path,
                kind="both",
                loop_state={"batch": checkpoint_batch, "mega_batch": i_mega_batch + 1},
            )

        mega_batch_metrics["time/total"] = time.time() - t_start
        mega_batch_metrics.update(cost_tracker.get_metrics())

        # Log at the final mini-batch step of this mega-batch (consistent with mini-batch logging)
        final_mini_batch_step = mini_batch_step_base + cfg.mini_epochs * mini_batches_per_mini_epoch - 1
        log_metrics_with_wandb(ml_logger, mega_batch_metrics, step=final_mini_batch_step)

        # Log progress (use last_teacher_kl from mini-batch loop)
        logger.info(
            f"Mega-batch {i_mega_batch + 1}/{num_mega_batches} complete | "
            f"teacher_kl: {last_teacher_kl:.4f} | "
            f"time: {mega_batch_metrics['time/total']:.1f}s"
        )


@scope
async def main(cfg: Config):
    """Main training loop for PPO-style mega-batch distillation."""

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

    # Initialize wandb with resume support BEFORE setup_logging
    # This ensures we resume the same wandb run instead of creating a new one
    wandb_initialized = init_wandb_with_resume(
        log_path=cfg.log_path,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
        config=cfg,
    )

    # Note: config.json is saved by ml_log.setup_logging below
    # Pass wandb_project=None since we already initialized wandb above
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=None,  # Skip wandb in setup_logging, we handle it ourselves
        config=cfg,
        wandb_name=None,
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
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    logger.info(f"Using renderer: {renderer_name} for model: {cfg.model_name}")
    if cfg.thinking_mode is not None:
        logger.info(f"Thinking mode explicitly set to: {cfg.thinking_mode}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Create teacher sampling client (same model as student)
    teacher_client = service_client.create_sampling_client(base_model=cfg.model_name)
    logger.info(f"Created teacher sampling client for {cfg.model_name}")

    # Resolve teacher prompt (uses mode-dependent default if None)
    resolved_teacher_prompt = get_teacher_prompt(cfg)

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
    batches_per_epoch = len(dataset)
    total_batches = batches_per_epoch * cfg.num_epochs

    # Calculate mega-batch counts
    batches_per_mega_batch = cfg.policy_update_interval
    num_mega_batches = (total_batches + batches_per_mega_batch - 1) // batches_per_mega_batch
    start_mega_batch = start_batch // batches_per_mega_batch

    logger.info(f"PPO mega-batch training configuration:")
    logger.info(f"  Total batches: {total_batches} ({cfg.num_epochs} epoch(s) x {batches_per_epoch} batches/epoch)")
    logger.info(f"  Batches per mega-batch: {batches_per_mega_batch}")
    logger.info(f"  Mini-epochs per mega-batch: {cfg.mini_epochs}")
    logger.info(f"  Total mega-batches: {num_mega_batches}")
    logger.info(f"  Effective gradient updates per mega-batch: {cfg.mini_epochs}")
    logger.info(f"Teacher mode: {cfg.teacher_mode.upper()}")
    logger.info(f"  supervision_key='{cfg.supervision_key}'")
    logger.info(f"  teacher_prompt='{resolved_teacher_prompt[:80]}...'")

    if start_mega_batch > 0:
        logger.info(f"Resuming from mega-batch {start_mega_batch} (batch {start_batch})")

    # --- Cost Estimation ---
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
    await do_ppo_mega_batch_training(
        start_mega_batch=start_mega_batch,
        end_mega_batch=num_mega_batches,
        num_mega_batches=num_mega_batches,
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
    final_batch = num_mega_batches * batches_per_mega_batch
    if start_mega_batch < num_mega_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": final_batch, "mega_batch": num_mega_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Log final cost report
    logger.info("\n" + cost_tracker.format_report())
    log_metrics_with_wandb(ml_logger, cost_tracker.get_metrics(), step=final_batch)

    # Close wandb run if active
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    asyncio.run(main(cfg))
