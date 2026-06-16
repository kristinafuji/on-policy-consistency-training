#!/usr/bin/env python3
"""Retry vLLM inference on examples that got stuck in the analysis channel.

Reads an existing results.jsonl, identifies responses that start with "analysis"
(meaning the model never produced a final-channel response), reruns just those
with a higher max_tokens, and patches them back into the results file.

Usage:
    python vllm_retry_stuck.py \
        --results-path ./eval_results/vllm-sweep1-opd-gptoss20b-v2-001380/results.jsonl \
        --base-model ./logs/sweep1-opd-gptoss20b-v2/merged_models/001380 \
        --tokenizer openai/gpt-oss-20b \
        --max-tokens 8192
"""

import argparse
import json
import logging
import os

from vllm_inference import (
    DEFAULT_TEACHER_PROMPTS,
    build_prompt_messages,
    parse_gpt_oss_response,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retry stuck vLLM inference examples")
    parser.add_argument("--results-path", type=str, required=True,
                        help="Path to existing results.jsonl")
    parser.add_argument("--base-model", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer override")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens for retry (default: 8192)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Max model context length (must fit prompt + max-tokens)")
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Load existing results
    with open(args.results_path) as f:
        all_results = [json.loads(line) for line in f]

    # Find stuck ones
    stuck_indices = []
    for i, r in enumerate(all_results):
        if r["response"].startswith("analysis"):
            stuck_indices.append(i)

    logger.info(f"Found {len(stuck_indices)}/{len(all_results)} stuck responses")
    if not stuck_indices:
        logger.info("Nothing to retry!")
        return

    # Load config from the same output dir to get teacher settings
    config_path = os.path.join(os.path.dirname(args.results_path), "config.json")
    with open(config_path) as f:
        orig_config = json.load(f)

    teacher_mode = orig_config.get("teacher_mode")
    supervision_key = orig_config.get("supervision_key", "safety_fact")
    teacher_prompt_template = orig_config.get("teacher_prompt")
    if teacher_mode and not teacher_prompt_template:
        teacher_prompt_template = DEFAULT_TEACHER_PROMPTS[teacher_mode]

    # Initialize vLLM
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.base_model,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
        max_num_batched_tokens=8192,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
    )
    if args.tokenizer:
        llm_kwargs["tokenizer"] = args.tokenizer
    llm = LLM(**llm_kwargs)

    tokenizer = llm.get_tokenizer()

    # Stop tokens
    stop_token_ids = []
    for token_str in ["<|return|>", "<|call|>"]:
        token_ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(token_ids) == 1:
            stop_token_ids.append(token_ids[0])

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        stop_token_ids=stop_token_ids if stop_token_ids else None,
        skip_special_tokens=False,
    )

    # Build messages from the stuck results (they contain the original prompts)
    stuck_results = [all_results[i] for i in stuck_indices]
    all_messages = []
    for r in stuck_results:
        messages, _ = build_prompt_messages(
            prompt_content=r["prompt"],
            teacher_mode=teacher_mode,
            teacher_prompt=teacher_prompt_template,
            supervision_value=r.get("safety_fact") if teacher_mode else None,
        )
        all_messages.append(messages)

    logger.info(f"Running inference on {len(all_messages)} stuck examples with max_tokens={args.max_tokens}")
    outputs = llm.chat(messages=all_messages, sampling_params=sampling_params)

    # Patch results
    patched = 0
    still_stuck = 0
    for idx, output in zip(stuck_indices, outputs):
        generated_text = output.outputs[0].text
        response_text, thinking_text = parse_gpt_oss_response(generated_text)

        all_results[idx]["response"] = response_text
        all_results[idx]["thinking"] = thinking_text

        if response_text.startswith("analysis"):
            still_stuck += 1
        else:
            patched += 1

    logger.info(f"Patched {patched}/{len(stuck_indices)} responses")
    logger.info(f"Still stuck: {still_stuck}")

    # Write back
    with open(args.results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Updated {args.results_path}")


if __name__ == "__main__":
    main()
