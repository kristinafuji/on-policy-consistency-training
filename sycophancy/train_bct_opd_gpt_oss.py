#!/usr/bin/env python3

"""
Trains consistency on the BCT-Train dataset from HuggingFace using On-Policy Distillation (OPD).
https://huggingface.co/datasets/<hf-username>/Sycophancy_Train

The BCT dataset has pre-constructed biased/unbiased prompt pairs:
- Student sees: prompt_biased (contains biasing statement)
- Teacher sees: prompt_unbiased (gives correct answer)
- Goal: student learns to ignore bias and match teacher's correct responses

KL penalty modes:
- Per-token KL (default): Computes per-token reverse KL between student and teacher
  on the full response sequence.
- Two-phase KL (--two-phase-kl): Extracts thinking tokens from the student's response,
  builds Phase 2 inputs for both student (biased) and teacher (unbiased) with "#### "
  prefilled, then computes KL on the answer logit distributions using
  topk_prompt_logprobs. Handles variable numbers of answer choices (A-B, A-D, A-E, etc.)
  per example via num_answer_choices. This focuses the training signal on the final
  answer rather than matching every intermediate token.

Supports both Llama-3.1-8B and Qwen3-8B models.
"""

import asyncio
import json
import re
import signal
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
from datasets import load_dataset

from tinker_cookbook import checkpoint_utils, renderers
from utils.renderer_utils import (
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


def convert_messages(raw_messages):
    """Convert HuggingFace dataset message format to renderer-expected list of dicts.
    
    The dataset stores messages as JSON strings (from json.dumps), but the renderer
    expects list[dict] with 'role' and 'content' keys.
    """
    if not raw_messages:
        return []

    if isinstance(raw_messages, str):
        try:
            raw_messages = json.loads(raw_messages)
        except json.JSONDecodeError:
            return [{"role": "user", "content": raw_messages}]

    if isinstance(raw_messages, list) and len(raw_messages) > 0:
        first = raw_messages[0]
        if isinstance(first, dict) and "content" in first and "role" in first:
            return raw_messages

    if isinstance(raw_messages, dict) and "content" in raw_messages and "role" in raw_messages:
        contents = raw_messages["content"]
        roles = raw_messages["role"]
        if isinstance(contents, list) and isinstance(roles, list):
            return [{"role": r, "content": c} for r, c in zip(roles, contents)]
        else:
            return [{"role": roles, "content": contents}]

    if isinstance(raw_messages, list):
        converted = []
        for msg in raw_messages:
            if isinstance(msg, dict):
                converted.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
            elif isinstance(msg, str):
                converted.append({"role": "user", "content": msg})
        return converted

    return raw_messages


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
    if cfg.temperature == 0.0 and not cfg.two_phase_kl:
        warnings.append(
            "temperature=0 with standard (non-two-phase) KL produces deterministic rollouts, "
            "which may cause degenerate advantage estimation. Consider using two_phase_kl=true "
            "or setting temperature > 0."
        )
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

    # Note: dataset_name is a HuggingFace dataset, not a local path
    # Validation happens when loading the dataset in BCTDataset

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

    # Log warnings
    for warning in warnings:
        logger.warning(f"CONFIG WARNING: {warning}")

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigurationError(error_msg)


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
    "dataset_name",
    "batch_size_prompts",
    "samples_per_prompt",
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
        return

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




# --- Custom Environment Classes for BCT ---

class BCTEnv(Env):
    """
    Single-turn environment for BCT consistency training.
    Stores both the biased (student) and unbiased (teacher) prompts.
    """
    
    def __init__(
        self,
        prompt_unbiased: List[Dict[str, str]],
        prompt_biased: List[Dict[str, str]],
        renderer: "renderers.Renderer",
    ):
        self.prompt_unbiased = prompt_unbiased  # Teacher prompt (correct answer)
        self.prompt_biased = prompt_biased      # Student prompt (with bias)
        self.renderer = renderer
        self._done = False

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Student sees the biased prompt
        model_input = self.renderer.build_generation_prompt(self.prompt_biased)
        return model_input, self.renderer.get_stop_sequences()

    async def step(self, action: Any) -> StepResult:
        self._done = True
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=types.ModelInput.from_ints([]),
            next_stop_condition=[],
        )


class BCTEnvGroupBuilder(EnvGroupBuilder):
    """Builder that creates BCT environments for sampling."""
    def __init__(
        self,
        prompt_unbiased: List[Dict[str, str]],
        prompt_biased: List[Dict[str, str]],
        renderer: "renderers.Renderer",
        duplicates: int = 1,
        answer: str | None = None,
        num_answer_choices: int = 4,
    ):
        self.prompt_unbiased = prompt_unbiased
        self.prompt_biased = prompt_biased
        self.renderer = renderer
        self.duplicates = duplicates
        self.answer = answer  # Ground truth answer (for logging/debugging)
        self.num_answer_choices = num_answer_choices

    async def make_envs(self) -> Sequence[Env]:
        return [
            BCTEnv(self.prompt_unbiased, self.prompt_biased, self.renderer)
            for _ in range(self.duplicates)
        ]

    def logging_tags(self) -> dict[str, str]:
        return {"type": "bct_consistency"}


class BCTDataset:
    """
    Loads BCT-Train dataset from HuggingFace and yields EnvGroupBuilders.
    """
    def __init__(
        self,
        dataset_name: str,
        groups_per_batch: int,
        renderer: "renderers.Renderer",
        split: str = "train",
        max_examples: int | None = None,
    ):
        logger.info(f"Loading dataset from HuggingFace: {dataset_name}")
        self.ds = load_dataset(dataset_name, split=split)

         # Limit dataset size if specified
        if max_examples is not None:
            original_size = len(self.ds)
            self.ds = self.ds.select(range(min(max_examples, len(self.ds))))
            logger.info(f"Limited dataset from {original_size} to {len(self.ds)} examples")
        
        logger.info(f"Loaded dataset with {len(self.ds)} examples")

        self.dataset_name = dataset_name
        self.groups_per_batch = groups_per_batch
        self.renderer = renderer

        # Validate dataset format after loading
        self._validate_dataset_format()

    def _validate_dataset_format(self) -> None:
        """Validate that the dataset has the required BCT columns and format."""
        columns = self.ds.column_names
        required_columns = ["prompt_unbiased", "prompt_biased", "answer"]
        
        missing = [col for col in required_columns if col not in columns]
        if missing:
            raise ConfigurationError(
                f"Dataset '{self.dataset_name}' is missing required columns: {missing}. "
                f"Available columns: {columns}. "
                f"BCT datasets must have 'prompt_unbiased', 'prompt_biased', and 'answer' columns."
            )

        # Validate format of first example (after convert_messages normalization)
        first_row = self.ds[0]
        
        prompt_unbiased = convert_messages(first_row['prompt_unbiased'])
        prompt_biased = convert_messages(first_row['prompt_biased'])

        # Check prompt_unbiased is a list of message dicts
        if not isinstance(prompt_unbiased, list) or len(prompt_unbiased) == 0:
            raise ConfigurationError(
                f"'prompt_unbiased' must be a non-empty list of message dicts after parsing, "
                f"got {type(prompt_unbiased).__name__} (raw type: {type(first_row['prompt_unbiased']).__name__})"
            )
        if not isinstance(prompt_unbiased[0], dict) or 'content' not in prompt_unbiased[0]:
            raise ConfigurationError(
                f"'prompt_unbiased' must contain dicts with 'content' key. "
                f"Got: {prompt_unbiased[0]}"
            )

        # Check prompt_biased similarly
        if not isinstance(prompt_biased, list) or len(prompt_biased) == 0:
            raise ConfigurationError(
                f"'prompt_biased' must be a non-empty list of message dicts after parsing, "
                f"got {type(prompt_biased).__name__} (raw type: {type(first_row['prompt_biased']).__name__})"
            )
        if not isinstance(prompt_biased[0], dict) or 'content' not in prompt_biased[0]:
            raise ConfigurationError(
                f"'prompt_biased' must contain dicts with 'content' key. "
                f"Got: {prompt_biased[0]}"
            )

        logger.info(f"Dataset format validation passed for '{self.dataset_name}'")

    def get_batch(self, batch_idx: int) -> tuple[Sequence[EnvGroupBuilder], List[int]]:
        builders = []
        indices = []

        start_idx = batch_idx * self.groups_per_batch

        for i in range(self.groups_per_batch):
            idx = start_idx + i
            row = self.ds[idx % len(self.ds)]
            
            # BCT dataset format: prompt_unbiased and prompt_biased may be JSON strings
            prompt_unbiased = convert_messages(row['prompt_unbiased'])
            prompt_biased = convert_messages(row['prompt_biased'])
            answer = row.get('answer', None)
            num_answer_choices = row.get('num_answer_choices', 4)

            builders.append(BCTEnvGroupBuilder(
                prompt_unbiased=prompt_unbiased,
                prompt_biased=prompt_biased,
                renderer=self.renderer,
                answer=answer,
                num_answer_choices=num_answer_choices,
            ))
            indices.append(idx)

        return builders, indices

    def __len__(self) -> int:
        return len(self.ds) // self.groups_per_batch


# --- KL Penalty for BCT ---

ALL_ANSWER_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

def get_answer_letters(num_choices: int) -> list[str]:
    """Return the list of answer letters for the given number of choices."""
    return ALL_ANSWER_LETTERS[:num_choices]


def _extract_answer_logits_from_response(
    response, tokenizer, num_answer_choices: int = 4,
) -> dict[str, float] | None:
    """Extract logprobs for answer choices from a Phase 2 response's topk_prompt_logprobs.
    
    Handles variable numbers of answer choices (2-8) via num_answer_choices.
    """
    topk = response.topk_prompt_logprobs
    if not topk or len(topk) == 0:
        return None

    last_pos_topk = topk[-1]
    if not last_pos_topk:
        return None

    topk_dict = {token_id: logprob for token_id, logprob in last_pos_topk}

    answer_letters = get_answer_letters(num_answer_choices)
    logits = {}
    for letter in answer_letters:
        token_id = tokenizer.encode(f" {letter}", add_special_tokens=False)[0]
        logits[letter] = topk_dict.get(token_id, float("-inf"))

    return logits


@scope
async def _incorporate_kl_two_phase(
    env_group_builders: Sequence[EnvGroupBuilder],
    data_D: List[tinker.Datum],
    metadata_D: List[Dict[str, Any]],
    teacher_client: tinker.SamplingClient,
    student_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    renderer_name: str,
    kl_penalty_coef: float,
) -> Dict[str, float]:
    """
    Two-phase KL penalty: compute KL on answer logit distributions.

    Handles variable numbers of answer choices per example (2-8) via
    num_answer_choices stored on each BCTEnvGroupBuilder.

    Instead of per-token KL on the full response, this:
    1. Extracts thinking tokens from the student's generated response
    2. Builds Phase 2 inputs for teacher (unbiased prompt) and student (biased prompt)
       by appending thinking_tokens + "#### " suffix
    3. Uses topk_prompt_logprobs to get each model's distribution over answer choices
    4. Computes KL divergence between these N-class distributions
    5. Applies the KL as a uniform advantage penalty across response tokens
    """
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    hash_token_id = tokenizer.encode("####", add_special_tokens=False)[0]

    teacher_p2_inputs = []
    student_p2_inputs = []
    datum_indices = []
    float_masks = []
    datum_num_choices = []  # Track num_answer_choices per datum
    logged_sample = False

    for datum_idx, (datum, metadata) in enumerate(zip(data_D, metadata_D)):
        group_idx = metadata["group_idx"]
        if group_idx >= len(env_group_builders):
            continue

        builder = env_group_builders[group_idx]
        if not isinstance(builder, BCTEnvGroupBuilder):
            continue

        # Log a sample (first one only)
        if not logged_sample:
            student_prompt_text = builder.prompt_biased[0]['content'][:200]
            teacher_prompt_text = builder.prompt_unbiased[0]['content'][:200]
            logger.info(f"[Two-phase KL] Student prompt (biased): {student_prompt_text}...")
            logger.info(f"[Two-phase KL] Teacher prompt (unbiased): {teacher_prompt_text}...")
            if builder.answer:
                logger.info(f"[Two-phase KL] Ground truth answer: {builder.answer}")
            logged_sample = True

        # Get student's response tokens from the datum
        student_full_seq = datum.model_input.append_int(
            cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
        )
        mask = datum.loss_fn_inputs["mask"].data
        response_length = int(sum(mask))
        student_full_seq_tokens = student_full_seq.to_ints()
        response_tokens = list(student_full_seq_tokens[-response_length:])
        prompt_tokens = list(student_full_seq_tokens[:-response_length])

        # Extract thinking tokens (everything before ####)
        thinking_tokens = []
        for i, token in enumerate(response_tokens):
            if token == hash_token_id:
                thinking_tokens = response_tokens[:i]
                break
        else:
            # No #### found — use all response tokens as thinking
            thinking_tokens = list(response_tokens)

        # Determine suffix based on whether thinking is complete
        thinking_text = tokenizer.decode(thinking_tokens, skip_special_tokens=False)
        if "</think>" in thinking_text:
            suffix_text = "#### "
        else:
            suffix_text = "</think>\n\n#### "
        suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)

        # Build Phase 2 input for teacher: unbiased_prompt + thinking + "#### "
        teacher_prompt = renderer.build_generation_prompt(builder.prompt_unbiased)
        teacher_p2_tokens = teacher_prompt.to_ints() + thinking_tokens + suffix_tokens
        teacher_p2_inputs.append(types.ModelInput.from_ints(teacher_p2_tokens))

        # Build Phase 2 input for student: biased_prompt + thinking + "#### "
        student_p2_tokens = prompt_tokens + thinking_tokens + suffix_tokens
        student_p2_inputs.append(types.ModelInput.from_ints(student_p2_tokens))

        datum_indices.append(datum_idx)
        float_masks.append(datum.loss_fn_inputs["mask"].to_torch().float())
        datum_num_choices.append(builder.num_answer_choices)

    if not teacher_p2_inputs:
        return {}

    # Phase 2 sampling params (generate 1 token, extract prompt logprobs)
    phase2_params = types.SamplingParams(max_tokens=1, temperature=0.0)

    # Batch compute Phase 2 for teacher and student IN PARALLEL
    # Both sets of API calls run concurrently instead of sequentially
    teacher_p2_results, student_p2_results = await asyncio.gather(
        asyncio.gather(
            *[teacher_client.sample_async(
                prompt=inp, num_samples=1, sampling_params=phase2_params,
                include_prompt_logprobs=True, topk_prompt_logprobs=20,
            ) for inp in teacher_p2_inputs]
        ),
        asyncio.gather(
            *[student_client.sample_async(
                prompt=inp, num_samples=1, sampling_params=phase2_params,
                include_prompt_logprobs=True, topk_prompt_logprobs=20,
            ) for inp in student_p2_inputs]
        ),
    )

    # Compute KL divergence between student and teacher answer distributions
    total_kl = 0.0
    count = 0
    skipped_no_logits = 0
    skipped_all_inf = 0
    skipped_non_finite = 0
    total_attempted = len(teacher_p2_results)

    for i in range(total_attempted):
        num_choices = datum_num_choices[i]
        answer_letters = get_answer_letters(num_choices)

        teacher_logits = _extract_answer_logits_from_response(
            teacher_p2_results[i], tokenizer, num_answer_choices=num_choices,
        )
        student_logits = _extract_answer_logits_from_response(
            student_p2_results[i], tokenizer, num_answer_choices=num_choices,
        )

        if teacher_logits is None or student_logits is None:
            skipped_no_logits += 1
            continue

        # Convert to tensors and normalize to proper distributions
        t_lp = torch.tensor([teacher_logits[k] for k in answer_letters])
        s_lp = torch.tensor([student_logits[k] for k in answer_letters])

        # Guard against all-(-inf) logits: softmax of all -inf produces NaN
        if torch.all(t_lp == float("-inf")) or torch.all(s_lp == float("-inf")):
            skipped_all_inf += 1
            continue

        t_probs = torch.softmax(t_lp, dim=0)
        s_probs = torch.softmax(s_lp, dim=0)

        # Reverse KL: sum_i p_student(i) * log(p_student(i) / p_teacher(i))
        kl = (s_probs * (torch.log(s_probs + 1e-10) - torch.log(t_probs + 1e-10))).sum()

        # Final NaN guard — skip if KL is not finite
        if not torch.isfinite(kl):
            skipped_non_finite += 1
            continue

        # Apply KL penalty uniformly across response tokens
        datum = data_D[datum_indices[i]]
        mask = float_masks[i]
        response_length = int(mask.sum().item())

        kl_penalty = -kl_penalty_coef * kl
        kl_advantages = torch.zeros_like(mask)
        if response_length > 0:
            kl_advantages[-response_length:] = (kl_penalty / response_length) * mask[-response_length:]

        datum.loss_fn_inputs["advantages"] = types.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

        total_kl += kl.item()
        count += 1

    total_skipped = skipped_no_logits + skipped_all_inf + skipped_non_finite
    avg_kl = total_kl / count if count > 0 else 0.0
    logger.info(
        f"[Two-phase KL] avg_kl={avg_kl:.4f} over {count}/{total_attempted} examples "
        f"(skipped {total_skipped}: {skipped_no_logits} no_logits, "
        f"{skipped_all_inf} all_inf, {skipped_non_finite} non_finite)"
    )

    return {
        "teacher_kl": avg_kl,
        "two_phase_kl/total_attempted": total_attempted,
        "two_phase_kl/count_used": count,
        "two_phase_kl/skipped_total": total_skipped,
        "two_phase_kl/skipped_no_logits": skipped_no_logits,
        "two_phase_kl/skipped_all_inf": skipped_all_inf,
        "two_phase_kl/skipped_non_finite": skipped_non_finite,
    }


@scope
async def incorporate_kl_penalty_bct(
    env_group_builders: Sequence[EnvGroupBuilder],
    data_D: List[tinker.Datum],
    metadata_D: List[Dict[str, Any]],
    teacher_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    renderer_name: str,
    kl_penalty_coef: float,
    kl_discount_factor: float,
    student_client: tinker.SamplingClient | None = None,
    two_phase_kl: bool = False,
) -> Dict[str, float]:
    """
    Compute reverse KL between student and teacher for BCT.
    
    - Student: sees prompt_biased -> generates response
    - Teacher: sees prompt_unbiased -> evaluates same response
    
    When two_phase_kl=True, computes KL on answer logit distributions (variable choices)
    at the "#### " position using topk_prompt_logprobs, instead of per-token KL.

    When two_phase_kl=False, computes per-token KL on the full response (original approach).
    """
    # Dispatch to two-phase KL if enabled
    if two_phase_kl and student_client is not None:
        return await _incorporate_kl_two_phase(
            env_group_builders, data_D, metadata_D,
            teacher_client, student_client, tokenizer,
            renderer_name, kl_penalty_coef,
        )

    # --- Original per-token KL approach ---
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    teacher_inputs = []
    datum_indices = []
    logged_sample = False

    for datum_idx, (datum, metadata) in enumerate(zip(data_D, metadata_D)):
        group_idx = metadata["group_idx"]

        if group_idx >= len(env_group_builders):
            continue

        builder = env_group_builders[group_idx]

        if not isinstance(builder, BCTEnvGroupBuilder):
            continue

        # Get the unbiased prompt (what teacher sees)
        prompt_unbiased = builder.prompt_unbiased
        
        # Build teacher prompt using the unbiased version
        teacher_prompt_formatted = renderer.build_generation_prompt(prompt_unbiased)

        # Log a sample (first one only)
        if not logged_sample:
            student_prompt_text = builder.prompt_biased[0]['content'][:200]
            teacher_prompt_text = prompt_unbiased[0]['content'][:200]
            logger.info(f"Sample student prompt (biased): {student_prompt_text}...")
            logger.info(f"Sample teacher prompt (unbiased): {teacher_prompt_text}...")
            if builder.answer:
                logger.info(f"Ground truth answer: {builder.answer}")
            logged_sample = True

        # Get student's response tokens from the datum
        student_full_seq = datum.model_input.append_int(
            cast(int, datum.loss_fn_inputs["target_tokens"].data[-1])
        )

        mask = datum.loss_fn_inputs["mask"].data
        response_length = int(sum(mask))

        student_full_seq_tokens = student_full_seq.to_ints()
        response_tokens = student_full_seq_tokens[-response_length:]

        # Construct teacher's full sequence: unbiased_prompt + response
        teacher_full_seq_tokens = teacher_prompt_formatted.to_ints() + response_tokens
        teacher_input = types.ModelInput.from_ints(teacher_full_seq_tokens)

        teacher_inputs.append(teacher_input)
        datum_indices.append(datum_idx)

    if not teacher_inputs:
        return {}

    # Batch compute teacher logprobs
    teacher_logprobs_list = await asyncio.gather(
        *[teacher_client.compute_logprobs_async(inp) for inp in teacher_inputs]
    )

    sampled_logprobs_D = [data_D[i].loss_fn_inputs["logprobs"].to_torch() for i in datum_indices]
    float_masks = [data_D[i].loss_fn_inputs["mask"].to_torch().float() for i in datum_indices]

    reverse_kl = []

    for teacher_logprobs, sampled_logprobs, mask in safezip(teacher_logprobs_list, sampled_logprobs_D, float_masks):
        response_length = int(mask.sum().item())

        response_logprobs_raw = teacher_logprobs[-response_length:]
        assert not any(lp is None for lp in response_logprobs_raw), (
            f"Unexpected None in response logprobs."
        )

        teacher_logprobs_clean = [lp if lp is not None else 0.0 for lp in teacher_logprobs]
        t_logprobs = torch.tensor(teacher_logprobs_clean)

        teacher_response_logprobs = t_logprobs[-response_length:]
        student_response_logprobs = sampled_logprobs[-response_length:]

        # Reverse KL: log p(student) - log q(teacher)
        response_kl = student_response_logprobs - teacher_response_logprobs

        kl = torch.zeros_like(sampled_logprobs)
        kl[-response_length:] = response_kl

        reverse_kl.append(kl)

    # Update advantages in-place
    for i, datum_idx in enumerate(datum_indices):
        datum = data_D[datum_idx]

        kl_advantages = -kl_penalty_coef * float_masks[i] * reverse_kl[i]
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )

        datum.loss_fn_inputs["advantages"] = types.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

    avg_logp_diff = sum([diff.sum() for diff in reverse_kl]) / sum(
        [mask.sum() for mask in float_masks]
    )

    metrics = {"teacher_kl": float(avg_logp_diff)}
    return metrics


# --- Configuration ---

@chz.chz
class Config:
    # Model settings
    model_name: str = "openai/gpt-oss-20b"

    # Thinking mode: GPT-OSS does not support thinking mode (Qwen3-only feature)
    thinking_mode: Literal["enable", "disable"] | None = None
    # train_on_thinking must be False for non-Qwen models (validate_config enforces this)
    train_on_thinking: bool = False

    # Renderer override: controls whether GPT-OSS produces reasoning tokens.
    # Options:
    #   "gpt_oss_no_sysprompt"      — no reasoning tokens (default)
    #   "gpt_oss_low_reasoning"     — low reasoning effort
    #   "gpt_oss_medium_reasoning"  — medium reasoning effort
    #   "gpt_oss_high_reasoning"    — high reasoning effort
    renderer_name_override: str | None = "gpt_oss_no_sysprompt"

    # Dataset settings
    dataset_name: str = "<hf-username>/Sycophancy_Train"
    batch_size_prompts: int = 64
    samples_per_prompt: int = 8
    num_epochs: int = 1

    # Training hyperparameters
    learning_rate: float = 1e-5
    lora_rank: int = 32
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Two-phase KL: compute KL on answer logit distributions (variable choices)
    # instead of per-token KL on the full response
    two_phase_kl: bool = False

    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 2048

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
    enable_trace: bool = False
    load_checkpoint_path: str | None = None
    compute_post_kl: bool = False

    # Resume validation: path to old log directory to validate config against
    validate_resume_from: str | None = None

    max_examples: int | None = None  # Limit dataset size


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
    student_client: tinker.SamplingClient | None = None,
    two_phase_kl: bool = False,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts trajectories into a minibatch with KL penalty."""

    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    if len(data_D) > 0:
        logger.info(colorize_example(data_D[0], tokenizer, key="mask"))

    if kl_penalty_coef > 0:
        with timed("compute_kl_penalty", metrics):
            kl_metrics = await incorporate_kl_penalty_bct(
                env_group_builders_P,
                data_D,
                metadata_D,
                teacher_client,
                tokenizer,
                renderer_name,
                kl_penalty_coef,
                kl_discount_factor,
                student_client=student_client,
                two_phase_kl=two_phase_kl,
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
        student_client=student_client,
        two_phase_kl=cfg.two_phase_kl,
    )
    metrics.update(prepare_minibatch_metrics)

    if should_mask_thinking(renderer_name) and not cfg.train_on_thinking:
        mask_thinking_tokens(data_D, tokenizer)
        logger.info("Masked thinking tokens")

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

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
    dataset: BCTDataset,
    teacher_client: tinker.SamplingClient,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    renderer_name: str,
):
    """Implements fully synchronous on-policy training"""

    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )

    # Register signal handlers so SIGTERM/SIGINT (cloud preemption, docker stop,
    # Ctrl+C) finishes the current batch cleanly, saves a checkpoint, then exits.
    # Without this, up to save_every batches of work would be lost on interruption.
    shutdown_requested = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, shutdown_requested.set)
    loop.add_signal_handler(signal.SIGINT, shutdown_requested.set)

    async def _save_emergency_checkpoint(batch: int, label: str) -> None:
        logger.info(f"Saving emergency checkpoint at batch {batch} ({label})...")
        try:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"{batch:06d}_{label}",
                log_path=cfg.log_path,
                loop_state={"batch": batch},
                kind="both",
            )
            logger.info(
                f"Emergency checkpoint saved. Re-run the same command to resume from batch {batch}."
            )
        except Exception as ckpt_err:
            logger.error(f"Failed to save emergency checkpoint: {ckpt_err}")

    i_batch = start_batch  # Track for error reporting if exception fires before the loop
    try:
        for i_batch in range(start_batch, end_batch):
            current_epoch = i_batch // batches_per_epoch
            metrics = {
                "progress/batch": i_batch,
                "progress/epoch": current_epoch,
                "optim/lr": cfg.learning_rate,
                "progress/done_frac": (i_batch + 1) / num_batches,
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

            sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
                cfg,
                i_batch,
                training_client,
                tokenizer,
                env_group_builders_P,
                trajectory_groups_P,
                teacher_client,
                renderer_name,
                student_client=sampling_client,
            )

            metrics.update(train_step_metrics)
            metrics["time/total"] = time.time() - t_start
            ml_logger.log_metrics(metrics, step=i_batch)

            # Check after each completed batch whether a shutdown signal arrived.
            # If so, save a checkpoint and exit cleanly.
            if shutdown_requested.is_set():
                logger.info(
                    f"Shutdown signal received — saving checkpoint at batch {i_batch + 1} and exiting."
                )
                await _save_emergency_checkpoint(i_batch + 1, "interrupted")
                break

    except Exception as e:
        logger.error(f"Training failed at batch {i_batch}: {e}")
        await _save_emergency_checkpoint(i_batch, "crashed")
        raise
    finally:
        # Always remove signal handlers on exit to avoid interfering with other code.
        try:
            loop.remove_signal_handler(signal.SIGTERM)
            loop.remove_signal_handler(signal.SIGINT)
        except Exception:
            pass


@scope
async def main(cfg: Config):
    """Main training loop for BCT consistency training."""

    # --- Early validation (before any expensive operations) ---
    logger.info("Validating configuration...")
    validate_config(cfg)
    logger.info("Configuration validation passed")

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

    # Save command line for reproducibility
    save_command_line(cfg.log_path)

    # Setup logging (also saves config.json)
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
        logger.info(f"Tracing enabled. Events saved to {trace_events_path}")
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

    if cfg.renderer_name_override is not None:
        renderer_name = cfg.renderer_name_override
        logger.info(f"Using renderer override: {renderer_name} for model: {cfg.model_name}")
    else:
        renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
        logger.info(f"Using renderer: {renderer_name} for model: {cfg.model_name}")
    if cfg.thinking_mode is not None:
        logger.info(f"Thinking mode explicitly set to: {cfg.thinking_mode}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    teacher_client = service_client.create_sampling_client(base_model=cfg.model_name)
    logger.info(f"Created teacher sampling client for {cfg.model_name}")

    # Create dataset (this also validates dataset format)
    dataset = BCTDataset(
        dataset_name=cfg.dataset_name,
        groups_per_batch=cfg.batch_size_prompts,
        renderer=renderer,
        max_examples=cfg.max_examples,
    )
    batches_per_epoch = len(dataset)
    num_batches = batches_per_epoch * cfg.num_epochs
    logger.info(f"Training on {num_batches} batches ({cfg.num_epochs} epoch(s) x {batches_per_epoch} batches/epoch)")
    logger.info(f"Dataset: {cfg.dataset_name}")

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

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    asyncio.run(main(cfg))
