#!/usr/bin/env python3
"""
Step 1: Generate BCT Training Data - Nemotron 3 Nano 30B

This script generates training data for supervised fine-tuning by:
1. Loading prompts from the BCT-Train dataset (HuggingFace)
2. Generating responses using the base model on UNBIASED prompts
3. Saving as JSONL with (biased_prompt, unbiased_response) pairs

Supports two-phase logit generation (enabled by default for Nemotron):
- Phase 1: Generate thinking tokens on unbiased prompts, stopping at ####
- Phase 2: Prefill "#### " and extract logits for answer choices via topk_prompt_logprobs

Nemotron uses the same ChatML + <think>...</think> format as Qwen3 and is now
registered in tinker-cookbook's model_info, so the nemotron3 renderer is used
automatically. The HuggingFace tokenizer is not publicly available, so a
Llama-3 fallback tokenizer is used.

The generated data will be used by train_bct_nemotron.py to train the model
to give consistent (unbiased) responses even when given biased prompts.
"""

import asyncio
import json
import logging
import os
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
    """Get tokenizer with fallback. Nemotron HF tokenizer is not publicly available."""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from {model_name}: {e}")
        fallback = "NousResearch/Meta-Llama-3.1-8B-Instruct"
        logger.info(f"Using fallback tokenizer: {fallback}")
        return AutoTokenizer.from_pretrained(fallback, use_fast=True)


@chz.chz
class Config:
    base_model: str = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    # Leave None — renderer_utils resolves nemotron3 automatically via
    # tinker-cookbook's model_info registry.
    renderer_name_override: str | None = None

    thinking_mode: Literal["enable", "disable"] | None = None

    # Two-phase logit generation — enabled for Nemotron (same as Qwen3)
    two_phase_thinking: bool = True
    thinking_max_tokens: int = 2048

    # Dataset paths
    dataset_name: str = "<hf-username>/Sycophancy_Train"
    output_file: str = "./bct_training_data_nemotron.jsonl"

    # Sampling parameters
    temperature: float = 0.0  # Greedy for consistency
    max_tokens: int = 256
    top_p: float = 0.9
    seed: int = 47

    # Processing
    batch_size: int = 64
    dedupe_unbiased: bool = True

    # Optional: limit number of examples (for testing)
    max_examples: int | None = None


async def run_two_phase_generation(
    prompt_messages,
    sampling_client,
    renderer,
    tokenizer,
    thinking_max_tokens: int = 2048,
    seed: int = 47,
    num_answer_choices: int = 4,
):
    """
    Two-phase generation with logit extraction.

    Phase 1: Generate thinking tokens on the prompt, stopping at #### or stop sequences.
    Phase 2: Prefill "#### " and use topk_prompt_logprobs to extract the model's
             probability distribution over the answer choices at the answer position.

    Returns dict with response text, answer from logits, logits dict, and phase1 token count.
    """
    model_input = renderer.build_generation_prompt(prompt_messages)

    base_stops = renderer.get_stop_sequences()
    hash_token_id = tokenizer.encode("####", add_special_tokens=False)[0]
    stop_token_ids = set(base_stops + [hash_token_id])

    # Phase 1: Generate thinking tokens
    phase1_params = types.SamplingParams(
        max_tokens=thinking_max_tokens,
        temperature=0.0,
        seed=seed,
        stop=base_stops + [hash_token_id],
    )

    response1 = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=phase1_params,
    )

    phase1_tokens_raw = response1.sequences[0].tokens
    phase1_token_count = len(phase1_tokens_raw)

    # Strip trailing stop tokens (#### and <|im_end|>)
    phase1_tokens = list(phase1_tokens_raw)
    while phase1_tokens and phase1_tokens[-1] in stop_token_ids:
        phase1_tokens.pop()

    phase1_text = tokenizer.decode(phase1_tokens, skip_special_tokens=False)

    # Phase 2: Prefill "#### " and extract logits
    if "</think>" in phase1_text:
        suffix_text = "#### "
    else:
        suffix_text = "</think>\n\n#### "

    suffix_tokens = tokenizer.encode(suffix_text, add_special_tokens=False)
    phase2_input_tokens = model_input.to_ints() + phase1_tokens + suffix_tokens
    phase2_input = types.ModelInput.from_ints(phase2_input_tokens)

    phase2_params = types.SamplingParams(
        max_tokens=1,
        temperature=0.0,
        seed=seed,
    )

    response2 = await sampling_client.sample_async(
        prompt=phase2_input,
        num_samples=1,
        sampling_params=phase2_params,
        include_prompt_logprobs=True,
        topk_prompt_logprobs=20,
    )

    # Extract logprobs for answer choices from the last prompt position
    answer_letters = get_answer_letters(num_answer_choices)
    logits_choices = {}
    answer_from_logits = None

    topk = response2.topk_prompt_logprobs
    if topk and len(topk) > 0:
        last_pos_topk = topk[-1]
        if last_pos_topk:
            topk_dict = {token_id: logprob for token_id, logprob in last_pos_topk}

            for letter in answer_letters:
                token_id = tokenizer.encode(f" {letter}", add_special_tokens=False)[0]
                logits_choices[letter] = topk_dict.get(token_id, float("-inf"))

            answer_from_logits = max(logits_choices, key=logits_choices.get)

    if answer_from_logits is None:
        phase2_gen_tokens = response2.sequences[0].tokens
        generated_text = tokenizer.decode(phase2_gen_tokens, skip_special_tokens=False).strip()
        answer_from_logits = generated_text if generated_text in answer_letters else None
        logger.warning("topk_prompt_logprobs not available, using generated token as fallback")

    # Prepend <think>\n so the stored content includes the opening tag.
    # The generation prompt ends with <think>\n, so phase1_text is the continuation
    # after that prefix. SFT training needs the full <think>...\n</think>\n\n answer
    # sequence to match what the model produces at inference time.
    full_response = "<think>\n" + phase1_text + suffix_text + (answer_from_logits or "")

    return {
        "response": full_response,
        "answer_from_logits": answer_from_logits,
        "logits": logits_choices,
        "phase1_tokens": phase1_token_count,
        "num_answer_choices": num_answer_choices,
    }


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

    When two_phase_thinking is True, uses Phase 1 (thinking) + Phase 2 (logit extraction)
    for cleaner answer extraction. Otherwise uses standard single-phase generation.

    Returns dict with 'response_text' and 'metadata_extra'.
    """
    if two_phase_thinking and tokenizer is not None:
        result = await run_two_phase_generation(
            prompt_messages=prompt_unbiased,
            sampling_client=sampling_client,
            renderer=renderer,
            tokenizer=tokenizer,
            thinking_max_tokens=thinking_max_tokens,
            seed=seed,
            num_answer_choices=num_answer_choices,
        )
        return {
            "response_text": result["response"],
            "metadata_extra": {
                "valid_format": True,
                "logits": result["logits"],
                "answer_from_logits": result["answer_from_logits"],
                "phase1_tokens": result["phase1_tokens"],
                "num_answer_choices": num_answer_choices,
            },
        }
    else:
        model_input = renderer.build_generation_prompt(prompt_unbiased)
        response = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )
        sequence = response.sequences[0]
        parsed_message, valid_format = renderer.parse_response(sequence.tokens)
        response_text = parsed_message.get("content", "")
        return {
            "response_text": response_text,
            "metadata_extra": {
                "valid_format": valid_format,
            },
        }


def make_training_example(row, unbiased_result: dict) -> dict:
    """Pair a cached unbiased generation result with a specific biased prompt row."""
    prompt_unbiased = convert_messages(row["prompt_unbiased"])
    prompt_biased = convert_messages(row["prompt_biased"])
    answer = row.get("answer", "")

    metadata = {
        "prompt_unbiased": prompt_unbiased[0]['content'],
        "answer": answer,
    }
    metadata.update(unbiased_result["metadata_extra"])

    return {
        "messages": [
            {"role": "user", "content": prompt_biased[0]['content']},
            {"role": "assistant", "content": unbiased_result["response_text"]},
        ],
        "_metadata": metadata,
    }


async def main(config: Config):
    """Generate BCT training data for Nemotron."""

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
        "two_phase_thinking": config.two_phase_thinking,
        "tokenizer": tokenizer,
        "thinking_max_tokens": config.thinking_max_tokens,
        "seed": config.seed,
    }

    logger.info(f"Generating responses for {len(dataset)} examples...")
    if config.two_phase_thinking:
        logger.info(f"Mode: Two-phase (Phase 1: {config.thinking_max_tokens} thinking tokens, Phase 2: logit extraction)")
    else:
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
        logger.info(f"Unbiased response: {example['messages'][1]['content'][:100]}...")
        logger.info(f"Ground truth: {example['_metadata']['answer'][:100]}...")

    logger.info(f"\nNext step: Train using train_bct_nemotron.py")


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(main))
