#!/usr/bin/env python3
"""
Prepare GPQA Diamond dataset for evaluation.

Downloads GPQA Diamond from HuggingFace, shuffles answer options,
formats multiple-choice prompts, and saves as a local HuggingFace dataset.

Usage:
    uv run python prepare_gpqa_dataset.py
"""

import os
import random

from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset, load_dataset

OUTPUT_PATH = "./datasets/gpqa-diamond"
SEED = 42
LETTERS = ["A", "B", "C", "D"]


def format_gpqa_prompt(question: str, options: list[str]) -> str:
    """Format a GPQA question as a multiple-choice prompt.

    Uses the openai/simple-evals QUERY_TEMPLATE_MULTICHOICE so that the
    canonical "Answer: $LETTER" extraction regex lands reliably. This is the
    template every frontier paper (DeepSeek-R1, Qwen3, gpt-oss, Claude, Llama-4)
    cites via simple-evals for GPQA numbers.
    """
    choices = "\n".join(f"{letter}) {opt}" for letter, opt in zip(LETTERS, options))
    return (
        "Answer the following multiple choice question. "
        "The last line of your response should be of the following format: "
        "'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. "
        "Think step by step before answering.\n\n"
        f"{question}\n\n"
        f"{choices}"
    )


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}. Delete it to regenerate.")
        return

    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "HF_TOKEN environment variable is required for gated GPQA dataset"

    print("Loading GPQA Diamond from HuggingFace...")
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=hf_token)
    print(f"Loaded {len(ds)} questions")

    rows = []
    rng = random.Random(SEED)

    for idx, example in enumerate(ds):
        question = example["Question"]
        correct = example["Correct Answer"]
        incorrect = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]

        # Shuffle options deterministically
        options = [correct] + incorrect
        rng.shuffle(options)

        correct_letter = LETTERS[options.index(correct)]
        prompt = format_gpqa_prompt(question, options)

        rows.append({
            "prompt": prompt,
            "correct_answer": correct_letter,
            "question": question,
            "subdomain": example.get("Subdomain", example.get("High-level domain", "")),
        })

    output_ds = Dataset.from_list(rows)
    output_ds.save_to_disk(OUTPUT_PATH)
    print(f"Saved {len(output_ds)} examples to {OUTPUT_PATH}")
    print(f"Columns: {output_ds.column_names}")

    # Verify
    answer_distribution = {}
    for row in rows:
        answer_distribution[row["correct_answer"]] = answer_distribution.get(row["correct_answer"], 0) + 1
    print(f"Answer distribution: {answer_distribution}")


if __name__ == "__main__":
    main()
