#!/usr/bin/env python3
"""
Prepare off-policy distillation data by pre-computing teacher logprobs.

Takes baseline student trajectories (from eval_sageeval.py) and computes:
1. response_tokens - tokenized response
2. sampling_logprobs - base model logprobs on response (student prompt)
3. teacher_logprobs - base model logprobs on response (teacher augmented prompt)

Output is a HuggingFace dataset ready for off-policy training.
"""

import asyncio
import json
import logging
import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

import chz
import tinker
from datasets import Dataset
from tqdm import tqdm

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from utils.renderer_utils import get_renderer_name_with_thinking_mode

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --- Default Teacher Prompts (must match train_distillation.py) ---

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


@chz.chz
class Config:
    # Model settings (must match the model used to generate trajectories)
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"

    # Thinking mode control for Qwen3 hybrid models
    thinking_mode: Literal["enable", "disable"] | None = None

    # Input/output paths
    input_path: str = "./eval_results_qwen3_4b_baseline_train/results.jsonl"
    output_path: str = "./datasets/offpolicy-qwen3-4b"

    # Teacher prompt configuration (must match train_distillation.py settings)
    teacher_mode: Literal["user", "system"] = "user"
    supervision_key: str = "safety_fact"
    teacher_prompt: str | None = None

    # Processing settings
    batch_size: int = 64  # Concurrent API calls


async def compute_logprobs_batch(
    client: tinker.SamplingClient,
    sequences: list[list[int]],
    batch_size: int = 64,
) -> list[list[float]]:
    """Compute logprobs for a batch of sequences."""
    results = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        model_inputs = [tinker.types.ModelInput.from_ints(seq) for seq in batch]

        batch_logprobs = await asyncio.gather(
            *[client.compute_logprobs_async(inp) for inp in model_inputs]
        )

        # Convert to list, replacing None with 0.0 (first token has no logprob)
        for logprobs in batch_logprobs:
            clean_logprobs = [lp if lp is not None else 0.0 for lp in logprobs]
            results.append(clean_logprobs)

    return results


async def main(cfg: Config):
    """Process baseline trajectories and compute teacher logprobs."""

    # Load input data
    logger.info(f"Loading trajectories from: {cfg.input_path}")
    with open(cfg.input_path, "r") as f:
        trajectories = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(trajectories)} trajectories")

    # Initialize Tinker client
    logger.info(f"Initializing model: {cfg.model_name}")
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)

    # Get tokenizer and renderer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    logger.info(f"Using renderer: {renderer_name}")
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    # Resolve teacher prompt
    resolved_teacher_prompt = get_teacher_prompt(cfg)
    logger.info(f"Teacher mode: {cfg.teacher_mode.upper()}")
    logger.info(f"Teacher prompt: '{resolved_teacher_prompt[:80]}...'")

    # Process trajectories
    processed_data = []

    # Build all sequences first
    logger.info("Building sequences...")
    student_sequences = []
    teacher_sequences = []

    for traj in tqdm(trajectories, desc="Tokenizing"):
        prompt = traj["prompt"]
        response = traj["response"]
        supervision_value = traj[cfg.supervision_key]

        # Tokenize response
        response_tokens = tokenizer.encode(response, add_special_tokens=False)

        # Build student prompt (plain prompt)
        student_messages = [{"role": "user", "content": prompt}]
        student_prompt_input = renderer.build_generation_prompt(student_messages)
        student_prompt_tokens = student_prompt_input.to_ints()

        # Build teacher prompt (augmented with supervision)
        formatted_teacher_prompt = resolved_teacher_prompt.format(
            supervision_key=supervision_value
        )

        if cfg.teacher_mode == "system":
            teacher_messages = [
                {"role": "system", "content": formatted_teacher_prompt},
                {"role": "user", "content": prompt},
            ]
        else:  # "user" mode
            teacher_messages = [
                {"role": "user", "content": f"{prompt}{formatted_teacher_prompt}"}
            ]

        teacher_prompt_input = renderer.build_generation_prompt(teacher_messages)
        teacher_prompt_tokens = teacher_prompt_input.to_ints()

        # Full sequences: prompt + response
        student_full = student_prompt_tokens + response_tokens
        teacher_full = teacher_prompt_tokens + response_tokens

        student_sequences.append(student_full)
        teacher_sequences.append(teacher_full)

        # Store metadata
        processed_data.append({
            "prompt": prompt,
            "response": response,
            "response_tokens": response_tokens,
            "student_prompt_length": len(student_prompt_tokens),
            "teacher_prompt_length": len(teacher_prompt_tokens),
            cfg.supervision_key: supervision_value,
            # Include other metadata from original trajectory
            "trial_idx": traj.get("trial_idx", 0),
            "valid_format": traj.get("valid_format", True),
            "augmentation_category": traj.get("augmentation_category", ""),
            "safety_category": traj.get("safety_category", ""),
            "prompt_type": traj.get("prompt_type", ""),
        })

    # Compute logprobs in batches
    logger.info(f"Computing student logprobs ({len(student_sequences)} sequences)...")
    student_logprobs_all = await compute_logprobs_batch(
        sampling_client, student_sequences, cfg.batch_size
    )

    logger.info(f"Computing teacher logprobs ({len(teacher_sequences)} sequences)...")
    teacher_logprobs_all = await compute_logprobs_batch(
        sampling_client, teacher_sequences, cfg.batch_size
    )

    # Extract response-region logprobs and add to processed data
    logger.info("Extracting response logprobs...")
    for i, data in enumerate(tqdm(processed_data, desc="Processing")):
        response_length = len(data["response_tokens"])
        student_prompt_length = data["student_prompt_length"]
        teacher_prompt_length = data["teacher_prompt_length"]

        # Student logprobs: last response_length values from student sequence
        # Note: logprobs[i] is the logprob of token[i] given tokens[:i]
        # So for response tokens, we want logprobs[student_prompt_length:]
        student_response_logprobs = student_logprobs_all[i][student_prompt_length:]

        # Teacher logprobs: last response_length values from teacher sequence
        teacher_response_logprobs = teacher_logprobs_all[i][teacher_prompt_length:]

        # Verify lengths match
        assert len(student_response_logprobs) == response_length, (
            f"Student logprobs length mismatch: {len(student_response_logprobs)} vs {response_length}"
        )
        assert len(teacher_response_logprobs) == response_length, (
            f"Teacher logprobs length mismatch: {len(teacher_response_logprobs)} vs {response_length}"
        )

        data["sampling_logprobs"] = student_response_logprobs
        data["teacher_logprobs"] = teacher_response_logprobs

        # Remove intermediate fields
        del data["student_prompt_length"]
        del data["teacher_prompt_length"]

    # Create HuggingFace dataset
    logger.info(f"Creating dataset with {len(processed_data)} examples...")
    dataset = Dataset.from_list(processed_data)

    # Save dataset
    logger.info(f"Saving dataset to: {cfg.output_path}")
    os.makedirs(cfg.output_path, exist_ok=True)
    dataset.save_to_disk(cfg.output_path)

    # Save config for reference
    config_path = os.path.join(cfg.output_path, "prepare_config.json")
    with open(config_path, "w") as f:
        json.dump(chz.asdict(cfg), f, indent=2)
    logger.info(f"Saved config to: {config_path}")

    # Print summary stats
    avg_response_len = sum(len(d["response_tokens"]) for d in processed_data) / len(processed_data)
    logger.info(f"Done! Summary:")
    logger.info(f"  Total examples: {len(processed_data)}")
    logger.info(f"  Average response length: {avg_response_len:.1f} tokens")

    # Print a sample
    sample = processed_data[0]
    logger.info(f"\nSample entry:")
    logger.info(f"  Prompt: {sample['prompt'][:80]}...")
    logger.info(f"  Response tokens: {len(sample['response_tokens'])}")
    logger.info(f"  Sampling logprobs: {sample['sampling_logprobs'][:5]}...")
    logger.info(f"  Teacher logprobs: {sample['teacher_logprobs'][:5]}...")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    asyncio.run(main(cfg))
