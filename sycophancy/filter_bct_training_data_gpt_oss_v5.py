#!/usr/bin/env python3
"""
Filter BCT training data to remove degenerate examples where the data generation
pipeline failed to produce a valid thinking block.

These degenerate examples (3.76% of v3 data) occur when parse_thinking_and_content
could not split the model's response into analysis/final channels — typically because
the model hit max_tokens in a repetition loop before reaching "#### X". In those cases
the response was placed entirely in 'content' with no 'thinking' key.

During training, examples without 'thinking' render as:
    <|start|>assistant<|channel|>final<|message|>{content}<|return|>
i.e. NO analysis channel — directly teaching the model to skip reasoning.
This causes generation collapse in the SFT model (it outputs only 7 tokens).

Unlike filter_bct_training_data_gpt_oss.py (v4), this script does NOT filter
for correct/wrong answers — it keeps both, matching the Nemotron SFT approach.

What is removed:
  - Examples with no 'thinking' field (or empty thinking): degenerate responses
  - Examples with no '#### X' answer in content: failed format

What is kept:
  - All examples with valid thinking AND a formatted answer (right or wrong)
"""

import json
import re
import sys
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ANSWER_RE = re.compile(r'####\s*([A-E])')


def main(input_file: str, output_file: str):
    total = 0
    kept = 0
    no_answer = 0
    no_thinking = 0

    with open(input_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            d = json.loads(line)

            msg = d["messages"][1]
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")

            # Skip examples where the model produced no formatted answer
            m = ANSWER_RE.search(content)
            if not m:
                no_answer += 1
                continue

            # Skip examples with no thinking (degenerate: model hit max_tokens or
            # parse_thinking_and_content failed to split analysis/final channels)
            if not thinking:
                no_thinking += 1
                continue

            fout.write(json.dumps(d) + "\n")
            kept += 1

    logger.info(f"Input:  {total} examples")
    logger.info(f"Kept:   {kept} ({100*kept/total:.1f}%)")
    logger.info(f"  Removed no answer:   {no_answer} ({100*no_answer/total:.1f}%)")
    logger.info(f"  Removed no thinking: {no_thinking} ({100*no_thinking/total:.1f}%)")
    logger.info(f"  Note: both correct and wrong answers are kept (like Nemotron SFT)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
