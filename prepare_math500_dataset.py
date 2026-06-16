#!/usr/bin/env python3
"""
Prepare MATH-500 dataset for evaluation.

Downloads HuggingFaceH4/MATH-500 (the 500-problem stable slice of Hendrycks
MATH from OpenAI's "Let's Verify Step by Step"), formats simple-evals-style
MATH prompts, and saves as a local HuggingFace dataset.

Usage:
    uv run python prepare_math500_dataset.py
"""

import os

from datasets import Dataset, load_dataset

OUTPUT_PATH = "./datasets/math-500"


# Simple-evals MATH query template (openai/simple-evals/math_eval.py).
# Asking for `Answer: $ANSWER` gives math_verify a deterministic hook, but its
# parser is format-agnostic and will also pick up \boxed{...} if the model
# emits that instead.
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form \"Answer: $ANSWER\" "
    "(without quotes) where $ANSWER is the answer to the problem.\n\n"
    "{problem}\n\n"
    "Remember to put your answer on its own line after \"Answer:\", "
    "and you do not need to use a \\boxed command."
)


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}. Delete it to regenerate.")
        return

    print("Loading HuggingFaceH4/MATH-500 from HuggingFace...")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"Loaded {len(ds)} problems")

    rows = []
    for example in ds:
        problem = example["problem"]
        answer = example["answer"]
        assert problem and answer, f"Empty problem or answer: {example}"

        rows.append({
            "prompt": PROMPT_TEMPLATE.format(problem=problem),
            "correct_answer": answer,
            "problem": problem,
            "solution": example.get("solution", ""),
            "subject": example.get("subject", ""),
            "level": example.get("level", -1),
            "unique_id": example.get("unique_id", ""),
        })

    output_ds = Dataset.from_list(rows)
    output_ds.save_to_disk(OUTPUT_PATH)
    print(f"Saved {len(output_ds)} examples to {OUTPUT_PATH}")
    print(f"Columns: {output_ds.column_names}")

    # Verify by subject
    by_subject: dict[str, int] = {}
    for row in rows:
        by_subject[row["subject"]] = by_subject.get(row["subject"], 0) + 1
    print(f"Distribution by subject: {by_subject}")
    print(f"\nFirst example prompt:\n{rows[0]['prompt']}\n")
    print(f"Correct answer: {rows[0]['correct_answer']}")


if __name__ == "__main__":
    main()
