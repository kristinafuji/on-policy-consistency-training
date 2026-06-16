#!/usr/bin/env python3
"""
Step 1: Generate BCT Training Data - GPT-OSS 20B

This script generates training data for supervised fine-tuning by:
1. Loading prompts from the BCT-Train dataset (HuggingFace)
2. Generating responses using the base model on UNBIASED prompts
3. Saving as JSONL with (biased_prompt, unbiased_response) pairs

Uses openai/gpt-oss-20b via Tinker API with gpt_oss_no_sysprompt renderer.
Single-phase generation (no thinking tokens for data generation).

The generated data will be used by train_bct_gpt_oss.py to train the model
to give consistent (unbiased) responses even when given biased prompts.
"""

import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from typing import Literal

from dotenv import load_dotenv
load_dotenv()  # Load TINKER_API_KEY from .env file

import chz
import tinker
from tinker import types
from datasets import load_dataset

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer
from utils.renderer_utils import get_renderer_name_with_thinking_mode
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

ALL_ANSWER_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Matches the final answer marker, e.g. "#### D" or "####D"
_ANSWER_RE = re.compile(r'^(.*?)\n*(#{4}\s*[A-H])\s*$', re.DOTALL)

# Channel marker strings as decoded by the GPT-OSS tokenizer
_ANALYSIS_PREFIX    = "<|channel|>analysis<|message|>"
_FINAL_TRANSITION   = "<|end|><|start|>assistant<|channel|>final<|message|>"
_FINAL_PREFIX       = "<|channel|>final<|message|>"


def parse_thinking_and_content(response_text: str) -> tuple[str, str]:
    """Split a GPT-OSS response into (thinking, final_content).

    The base model's decoded response includes literal channel-marker text
    (e.g. '<|channel|>analysis<|message|>...') because those tokens decode
    to their string representations. We strip the markers and re-separate the
    content so build_supervised_example can place it in the correct channels:
      - thinking → <|channel|>analysis<|message|>...
      - content  → <|channel|>final<|message|>...

    If no '#### X' marker is found the whole text goes to content (no thinking).
    """
    text = response_text.strip()

    # Strip leading analysis channel marker if the model included it
    if text.startswith(_ANALYSIS_PREFIX):
        text = text[len(_ANALYSIS_PREFIX):]

    # If there is an explicit analysis→final transition, split on it
    if _FINAL_TRANSITION in text:
        thinking_text, final_text = text.split(_FINAL_TRANSITION, 1)
        # Strip any remaining final-channel prefix inside the final block
        final_text = final_text.lstrip(_FINAL_PREFIX)
        m = _ANSWER_RE.match(final_text.strip())
        content = "\n\n" + (m.group(2).strip() if m else final_text.strip())
        return thinking_text.strip(), content

    # Strip a bare final-channel marker (no preceding analysis block)
    if text.startswith(_FINAL_PREFIX):
        text = text[len(_FINAL_PREFIX):]

    # Fall back: split by the '#### X' answer marker at the end of the text
    m = _ANSWER_RE.match(text)
    if m:
        thinking = m.group(1).strip()
        content = "\n\n" + m.group(2).strip()
        return thinking, content

    return "", text


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

def get_answer_letters(num_choices: int) -> list[str]:
    """Return the list of answer letters for the given number of choices."""
    return ALL_ANSWER_LETTERS[:num_choices]


def get_tokenizer_with_fallback(model_name: str):
    """Get tokenizer with fallback. GPT-OSS uses a Llama-based tokenizer."""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_name}: {e}")
        fallback = "NousResearch/Meta-Llama-3.1-8B-Instruct"
        logger.info(f"Using fallback tokenizer: {fallback}")
        return AutoTokenizer.from_pretrained(fallback, use_fast=True)


@chz.chz
class Config:
    base_model: str = "openai/gpt-oss-20b"

    # Renderer override — bypasses get_renderer_name_with_thinking_mode
    renderer_name_override: str | None = "gpt_oss_medium_reasoning"

    # Thinking mode — not used for GPT-OSS (controlled via renderer variant instead)
    thinking_mode: Literal["enable", "disable"] | None = None

    # Two-phase logit generation — not applicable for GPT-OSS
    two_phase_thinking: bool = False
    thinking_max_tokens: int = 2048

    # Dataset paths
    dataset_name: str = "<hf-username>/Sycophancy_Train"
    output_file: str = "./bct_training_data_gpt_oss.jsonl"

    # Sampling parameters
    temperature: float = 0.0  # Greedy for consistency
    max_tokens: int = 2048  # GPT-OSS reasoning needs room for analysis + final channel
    top_p: float = 0.9
    seed: int = 47

    # Processing
    batch_size: int = 64
    dedupe_unbiased: bool = True

    # Optional: limit number of examples (for testing)
    max_examples: int | None = None


async def generate_unbiased_response(
    prompt_unbiased,
    sampling_client,
    renderer,
    sampling_params,
    two_phase_thinking: bool = False,
    tokenizer=None,
    thinking_max_tokens: int = 2048,
    seed: int = 47,
    num_answer_choices: int = 4,
):
    """Generate a response using the base model on an UNBIASED prompt.

    Single-phase generation for GPT-OSS.

    Returns dict with 'response_text' and 'metadata_extra'.
    """
    model_input = renderer.build_generation_prompt(prompt_unbiased)
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    sequence = response.sequences[0]
    parsed_message, valid_format = renderer.parse_response(sequence.tokens)
    response_text = parsed_message.get("content", "")
    # GPT-OSS reasoning renderer may return content as a list of blocks
    if isinstance(response_text, list):
        response_text = "".join(
            b.get("text", "") or b.get("content", "") if isinstance(b, dict) else str(b)
            for b in response_text
        )
    thinking, content = parse_thinking_and_content(response_text)
    return {
        "thinking": thinking,
        "content": content,
        "metadata_extra": {
            "valid_format": valid_format,
        },
    }


def make_training_example(row, unbiased_result: dict) -> dict:
    """Pair a cached unbiased generation result with a specific biased prompt row.

    The assistant message carries both 'thinking' (for the analysis channel) and
    'content' (for the final channel) so that GptOssRenderer._render_message
    generates the correct <|channel|>analysis / <|channel|>final structure.
    """
    prompt_unbiased = convert_messages(row["prompt_unbiased"])
    prompt_biased = convert_messages(row["prompt_biased"])
    answer = row.get("answer", "")

    metadata = {
        "prompt_unbiased": prompt_unbiased[0]['content'],
        "answer": answer,
    }
    metadata.update(unbiased_result["metadata_extra"])

    assistant_msg = {"role": "assistant", "content": unbiased_result["content"]}
    if unbiased_result["thinking"]:
        assistant_msg["thinking"] = unbiased_result["thinking"]

    return {
        "messages": [
            {"role": "user", "content": prompt_biased[0]['content']},
            assistant_msg,
        ],
        "_metadata": metadata,
    }


async def main(config: Config):
    """Generate BCT training data for GPT-OSS."""

    # Load dataset from HuggingFace
    logger.info(f"Loading dataset from HuggingFace: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split="train")
    logger.info(f"Loaded {len(dataset)} examples")

    # Validate dataset format
    required_columns = ["prompt_unbiased", "prompt_biased", "answer"]
    missing = [col for col in required_columns if col not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Available columns: {dataset.column_names}"
        )

    # Limit if specified
    if config.max_examples:
        dataset = dataset.select(range(min(config.max_examples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} examples for testing")

    # Initialize Tinker client
    logger.info(f"Initializing model: {config.base_model}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=config.base_model,
    )

    # Get tokenizer and renderer
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer_with_fallback(config.base_model)

    if config.renderer_name_override is not None:
        renderer_name = config.renderer_name_override
        logger.info(f"Using renderer override: {renderer_name}")
    else:
        renderer_name = get_renderer_name_with_thinking_mode(config.base_model, config.thinking_mode)

    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"✓ Tokenizer and renderer initialized (using: {renderer_name})")

    # Sampling parameters
    sampling_params = types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        stop=renderer.get_stop_sequences(),
    )

    # Common kwargs for generation
    gen_kwargs = {
        "sampling_client": sampling_client,
        "renderer": renderer,
        "sampling_params": sampling_params,
        "two_phase_thinking": False,
        "tokenizer": tokenizer,
        "thinking_max_tokens": config.thinking_max_tokens,
        "seed": config.seed,
    }

    logger.info(f"Generating responses for {len(dataset)} examples...")
    logger.info(f"Mode: Single-phase (Temperature: {config.temperature}, Max tokens: {config.max_tokens})")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Dedupe unbiased: {config.dedupe_unbiased}")
    logger.info(f"Output file: {config.output_file}")
    logger.info("")
    logger.info("Training approach: Generate responses on UNBIASED prompts,")
    logger.info("                   then train model to produce same responses for BIASED prompts.")

    all_examples = []

    with open(config.output_file, "w") as f:
        if config.dedupe_unbiased:
            # Group rows by prompt_unbiased — generate once per unique question
            groups = defaultdict(list)
            for i in range(len(dataset)):
                row = dataset[i]
                unbiased_key = str(row["prompt_unbiased"])
                groups[unbiased_key].append((i, row))

            group_items = list(groups.items())
            unique_questions = len(group_items)
            logger.info(f"Dedupe: {len(dataset)} rows -> {unique_questions} unique unbiased prompts "
                         f"(~{len(dataset) / unique_questions:.1f}x savings)")

            processed = 0
            for batch_start in range(0, len(group_items), config.batch_size):
                batch_groups = group_items[batch_start:batch_start + config.batch_size]

                unbiased_tasks = []
                for _group_key, group_rows in batch_groups:
                    first_row = group_rows[0][1]
                    prompt_unbiased = convert_messages(first_row["prompt_unbiased"])
                    num_answer_choices = first_row.get("num_answer_choices", 4)
                    unbiased_tasks.append(
                        generate_unbiased_response(
                            prompt_unbiased=prompt_unbiased,
                            num_answer_choices=num_answer_choices,
                            **gen_kwargs,
                        )
                    )

                unbiased_results = await asyncio.gather(*unbiased_tasks)

                batch_examples = []
                for group_idx, (_group_key, group_rows) in enumerate(batch_groups):
                    unbiased_result = unbiased_results[group_idx]
                    for _row_idx, row in group_rows:
                        example = make_training_example(row, unbiased_result)
                        batch_examples.append(example)

                for example in batch_examples:
                    f.write(json.dumps(example) + "\n")
                    all_examples.append(example)

                f.flush()
                processed += sum(len(rows) for _, rows in batch_groups)
                batch_num = batch_start // config.batch_size + 1
                total_batches = (len(group_items) + config.batch_size - 1) // config.batch_size
                logger.info(f"  ✓ Batch {batch_num}/{total_batches} complete "
                            f"({len(batch_groups)} unique prompts -> {len(batch_examples)} examples, "
                            f"{processed}/{len(dataset)} total)")
        else:
            num_batches = (len(dataset) + config.batch_size - 1) // config.batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min(start_idx + config.batch_size, len(dataset))
                batch_rows = [dataset[i] for i in range(start_idx, end_idx)]

                logger.info(f"Processing batch {batch_idx + 1}/{num_batches} ({end_idx}/{len(dataset)} examples)")

                tasks = []
                for row in batch_rows:
                    prompt_unbiased = convert_messages(row["prompt_unbiased"])
                    num_answer_choices = row.get("num_answer_choices", 4)
                    tasks.append(
                        generate_unbiased_response(
                            prompt_unbiased=prompt_unbiased,
                            num_answer_choices=num_answer_choices,
                            **gen_kwargs,
                        )
                    )
                unbiased_results = await asyncio.gather(*tasks)

                for row_result, row in zip(unbiased_results, batch_rows):
                    example = make_training_example(row, row_result)
                    f.write(json.dumps(example) + "\n")
                    all_examples.append(example)

                f.flush()
                logger.info(f"  ✓ Batch {batch_idx + 1} complete ({len(batch_rows)} examples written)")

    logger.info(f"\n{'='*80}")
    logger.info(f"✓ Generation complete!")
    logger.info(f"{'='*80}")
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"Output file: {config.output_file}")

    if all_examples:
        example = all_examples[0]
        logger.info(f"\nFirst example:")
        logger.info(f"Biased prompt: {example['messages'][0]['content'][:100]}...")
        thinking = example['messages'][1].get('thinking', '')
        content = example['messages'][1].get('content', '')
        if thinking:
            logger.info(f"Thinking (analysis): {thinking[:100]}...")
        logger.info(f"Final content: {content[:100]}...")
        logger.info(f"Ground truth: {example['_metadata']['answer'][:100]}...")

    logger.info(f"\nNext step: Train using train_bct_gpt_oss.py")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
