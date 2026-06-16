#!/usr/bin/env python3
"""
Generate teacher completions for SFT training.

Standalone script that generates completions from a teacher model prompted with
either a "cheat" (per-example safety fact) or "evc" (static structured) teacher prompt.
The output is a JSONL file where each line contains the ORIGINAL prompt (no teacher
augmentation) paired with the teacher's completion.

Supports incremental resume: re-running with the same output_path picks up where
it left off.

Also importable by train_tinker_sft.py.
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv

load_dotenv()  # Load TINKER_API_KEY from .env

import chz
import tinker
from tinker import types
from datasets import load_from_disk
from typing import Literal

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from utils.renderer_utils import get_renderer_name_with_thinking_mode

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


# --- Teacher Prompts ---

# "cheat" mode: per-example supervision via {supervision_key} placeholder
CHEAT_TEACHER_PROMPT = """You are a helpful assistant. You help users with their questions while also informing them of relevant safety concerns.

SAFETY GUIDELINE FOR THIS CONVERSATION: {supervision_key}

When this guideline is relevant to the user's question:
1. Answer their question helpfully
2. Include the safety information as part of your response

Do not refuse to engage. A refusal leaves the user uninformed about the safety risk."""

# "evc" mode: static prompt, no per-example substitution
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


TEACHER_PROMPTS = {
    "cheat": CHEAT_TEACHER_PROMPT,
    "evc": EVC_TEACHER_PROMPT,
}


@chz.chz
class GenerationConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset_path: str = "./datasets/sageeval-train"
    output_path: str  # Required, no default
    teacher_type: Literal["cheat", "evc"] = "cheat"
    supervision_key: str = "safety_fact"
    teacher_prompt: str | None = None  # Custom override
    n_samples: int = 2
    temperature: float = 1.0
    max_tokens: int = 1024
    batch_size: int = 512  # Concurrent API requests
    thinking_mode: Literal["enable", "disable"] | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    base_url: str | None = None


# --- Validation ---

def validate_generation_config(cfg: GenerationConfig) -> None:
    """Fail-fast validation. Raises ConfigurationError on any issue."""
    errors: list[str] = []

    # Numeric bounds
    if cfg.n_samples <= 0:
        errors.append(f"n_samples must be positive, got {cfg.n_samples}")
    if cfg.temperature <= 0:
        errors.append(f"temperature must be positive, got {cfg.temperature}")
    if cfg.max_tokens <= 0:
        errors.append(f"max_tokens must be positive, got {cfg.max_tokens}")
    if cfg.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {cfg.batch_size}")

    # Dataset path
    if not os.path.exists(cfg.dataset_path):
        errors.append(f"Dataset path does not exist: {cfg.dataset_path}")

    # Output directory must be creatable
    output_dir = os.path.dirname(cfg.output_path)
    if output_dir and not os.path.exists(output_dir):
        # Will be created later, but parent must exist
        parent = os.path.dirname(output_dir)
        if parent and not os.path.exists(parent):
            errors.append(f"Parent directory of output_path does not exist: {parent}")

    # Teacher type validation
    if cfg.teacher_type == "cheat" and cfg.teacher_prompt is None:
        # Will use default cheat prompt which needs {supervision_key}
        pass  # validated below with dataset columns
    if cfg.teacher_prompt is not None and cfg.teacher_type == "cheat":
        if "{supervision_key}" not in cfg.teacher_prompt:
            errors.append(
                f"Custom teacher_prompt for 'cheat' mode must contain '{{supervision_key}}' placeholder. "
                f"Got: '{cfg.teacher_prompt[:100]}...'"
            )

    # Thinking mode validation
    is_qwen_model = "qwen" in cfg.model_name.lower()
    is_openai_model = "openai" in cfg.model_name.lower() or "gpt-oss" in cfg.model_name.lower()
    supports_thinking_mode = is_qwen_model or is_openai_model

    if cfg.thinking_mode is not None and not supports_thinking_mode:
        errors.append(
            f"thinking_mode='{cfg.thinking_mode}' is only valid for Qwen or OpenAI models, "
            f"but model_name='{cfg.model_name}'"
        )

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

    if errors:
        error_msg = "Generation config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigurationError(error_msg)


def validate_dataset_columns(dataset_path: str, teacher_type: str, supervision_key: str) -> None:
    """Validate required columns exist in the dataset."""
    ds = load_from_disk(dataset_path)
    columns = ds.column_names

    assert "prompt" in columns, (
        f"Dataset at '{dataset_path}' is missing required 'prompt' column. "
        f"Available columns: {columns}"
    )

    if teacher_type == "cheat":
        assert supervision_key in columns, (
            f"supervision_key='{supervision_key}' not found in dataset. "
            f"Available columns: {columns}"
        )


# --- Core Functions ---

def resolve_teacher_prompt(cfg: GenerationConfig) -> str:
    """Return teacher prompt string based on teacher_type or custom override."""
    if cfg.teacher_prompt is not None:
        return cfg.teacher_prompt
    return TEACHER_PROMPTS[cfg.teacher_type]


def build_teacher_messages(
    prompt: str,
    teacher_type: str,
    resolved_prompt: str,
    supervision_value: str | None,
) -> list[dict]:
    """
    Build [system, user] messages for teacher sampling.

    cheat: system prompt has {supervision_key} resolved per-example.
    evc: static system prompt, no per-example substitution.
    """
    if teacher_type == "cheat":
        assert supervision_value is not None, (
            "supervision_value is required for 'cheat' teacher_type"
        )
        formatted_prompt = resolved_prompt.format(supervision_key=supervision_value)
    else:
        # evc: static prompt, no substitution
        formatted_prompt = resolved_prompt

    return [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": prompt},
    ]


async def process_single(
    row: dict,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    sampling_params: types.SamplingParams,
    n_samples: int,
    teacher_type: str,
    resolved_prompt: str,
    supervision_key: str = "safety_fact",
) -> list[dict]:
    """Generate n_samples completions for one prompt. Returns list of JSONL-ready dicts.

    Each result has messages using the ORIGINAL prompt (no teacher augmentation).
    """
    prompt_content = row["prompt"]
    supervision_value = row.get(supervision_key)

    # Build teacher-augmented messages for sampling
    teacher_messages = build_teacher_messages(
        prompt_content, teacher_type, resolved_prompt, supervision_value
    )
    model_input = renderer.build_generation_prompt(teacher_messages)

    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=n_samples,
        sampling_params=sampling_params,
    )

    assert len(response.sequences) == n_samples, (
        f"Expected {n_samples} sequences, got {len(response.sequences)}"
    )

    results = []
    for sample_idx, sequence in enumerate(response.sequences):
        parsed_message, valid_format = renderer.parse_response(sequence.tokens)
        assert "content" in parsed_message, (
            f"parsed_message missing 'content' key. Got keys: {list(parsed_message.keys())}"
        )

        # Build result with ORIGINAL prompt (student sees no teacher augmentation)
        result = {
            "messages": [
                {"role": "user", "content": prompt_content},
                {"role": "assistant", "content": parsed_message["content"]},
            ],
            "prompt": prompt_content,
            "sample_idx": sample_idx,
            supervision_key: row.get(supervision_key, ""),
            "augmentation_category": row.get("augmentation_category", ""),
            "safety_category": row.get("safety_category", ""),
            "prompt_type": row.get("prompt_type", ""),
            "version": row.get("version", ""),
            "valid_format": valid_format,
            "teacher_type": teacher_type,
        }
        results.append(result)

    return results


# --- Resume Logic ---

def count_existing_lines(output_path: str) -> int:
    """Count the number of lines in an existing output file."""
    if not os.path.exists(output_path):
        return 0
    with open(output_path, "r") as f:
        return sum(1 for _ in f)


def truncate_to_complete_groups(output_path: str, n_samples: int) -> int:
    """
    Truncate output file to complete prompt groups (multiples of n_samples).
    Returns number of complete lines after truncation.
    """
    existing_lines = count_existing_lines(output_path)
    if existing_lines == 0:
        return 0

    complete_lines = (existing_lines // n_samples) * n_samples

    if complete_lines < existing_lines:
        # Need to truncate partial group
        logger.info(
            f"Truncating {existing_lines - complete_lines} partial lines "
            f"(keeping {complete_lines} complete lines)"
        )
        lines = []
        with open(output_path, "r") as f:
            for i, line in enumerate(f):
                if i >= complete_lines:
                    break
                lines.append(line)

        with open(output_path, "w") as f:
            f.writelines(lines)

    return complete_lines


# --- Main Generation ---

async def generate_sft_data(cfg: GenerationConfig) -> str:
    """
    Main generation function. Returns output_path. Supports incremental resume.
    """
    logger.info("Validating generation config...")
    validate_generation_config(cfg)
    logger.info("Generation config validation passed")

    logger.info(f"Validating dataset at {cfg.dataset_path}...")
    validate_dataset_columns(cfg.dataset_path, cfg.teacher_type, cfg.supervision_key)
    logger.info("Dataset validation passed")

    # Create output directory
    output_dir = os.path.dirname(cfg.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Check for resume
    complete_lines = truncate_to_complete_groups(cfg.output_path, cfg.n_samples)
    resume_from_example = complete_lines // cfg.n_samples

    if resume_from_example > 0:
        logger.info(
            f"Found {complete_lines} existing lines ({resume_from_example} examples "
            f"with n_samples={cfg.n_samples}). Resuming from example {resume_from_example}"
        )

    # Save config alongside output
    config_path = cfg.output_path.rsplit(".", 1)[0] + "_config.json"
    config_dict = chz.asdict(cfg)
    config_dict["resumed_from_example"] = resume_from_example if resume_from_example > 0 else None
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"Saved generation config to: {config_path}")

    # Load dataset
    logger.info(f"Loading dataset from: {cfg.dataset_path}")
    dataset = load_from_disk(cfg.dataset_path)
    total_examples = len(dataset)
    logger.info(f"Loaded {total_examples} examples")

    remaining_examples = total_examples - resume_from_example
    if remaining_examples <= 0:
        logger.info(f"All {total_examples} examples already processed. Nothing to do.")
        return cfg.output_path

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Initialize Tinker client
    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = get_renderer_name_with_thinking_mode(
        cfg.model_name, cfg.thinking_mode, cfg.reasoning_effort
    )
    logger.info(f"Using renderer: {renderer_name}")
    if cfg.thinking_mode is not None:
        logger.info(f"Thinking mode explicitly set to: {cfg.thinking_mode}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Sampling parameters
    sampling_params = types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )

    # Resolve teacher prompt
    resolved_prompt = resolve_teacher_prompt(cfg)
    logger.info(f"Teacher type: {cfg.teacher_type.upper()}")
    logger.info(f"Teacher prompt: '{resolved_prompt[:80]}...'")

    logger.info(
        f"Processing {remaining_examples} remaining examples "
        f"(starting from index {resume_from_example}), "
        f"n_samples={cfg.n_samples}, batch_size={cfg.batch_size}"
    )
    logger.info(f"Results will be saved to: {cfg.output_path}")

    # Process in batches (starting from resume point)
    num_batches = (remaining_examples + cfg.batch_size - 1) // cfg.batch_size
    file_mode = "a" if resume_from_example > 0 else "w"

    with open(cfg.output_path, file_mode) as f:
        for batch_idx in range(num_batches):
            start_idx = resume_from_example + batch_idx * cfg.batch_size
            end_idx = min(start_idx + cfg.batch_size, total_examples)
            batch_rows = [dataset[i] for i in range(start_idx, end_idx)]

            # Process batch concurrently
            tasks = [
                process_single(
                    row,
                    sampling_client,
                    renderer,
                    sampling_params,
                    cfg.n_samples,
                    cfg.teacher_type,
                    resolved_prompt,
                    cfg.supervision_key,
                )
                for row in batch_rows
            ]
            batch_results = await asyncio.gather(*tasks)

            # Write results (each batch_result is a list of n_samples results)
            for sample_results in batch_results:
                for result in sample_results:
                    f.write(json.dumps(result) + "\n")
            f.flush()

            logger.info(
                f"Processed batch {batch_idx + 1}/{num_batches} "
                f"({end_idx}/{total_examples} examples total)"
            )

    logger.info(f"Done! Results saved to: {cfg.output_path}")
    return cfg.output_path


# --- Entrypoint ---

if __name__ == "__main__":
    cfg = chz.entrypoint(GenerationConfig, allow_hyphens=True)
    asyncio.run(generate_sft_data(cfg))
