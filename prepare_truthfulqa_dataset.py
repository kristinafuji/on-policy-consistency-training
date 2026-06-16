#!/usr/bin/env python3
"""
Prepare TruthfulQA "new multiple choice" dataset (Jan 2025 version) for evaluation.

Downloads the raw TruthfulQA.csv from the sylinrl/TruthfulQA GitHub repo and
builds a two-option multiple choice dataset using the `Best Answer` + `Best
Incorrect Answer` columns. The order of (A, B) is randomized deterministically
per question (seed=42). Saves as a local HuggingFace dataset at
./datasets/truthfulqa-new-mc/.

See https://github.com/sylinrl/TruthfulQA#update-to-multiple-choice-setting-jan-2025
for the rationale for the new MC version.

Usage:
    uv run python prepare_truthfulqa_dataset.py
"""

import csv
import os
import random
import urllib.request
from io import StringIO

from datasets import Dataset

OUTPUT_PATH = "./datasets/truthfulqa-new-mc"
SOURCE_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
SEED = 42
LETTERS = ["A", "B"]


def format_prompt(question: str, option_a: str, option_b: str) -> str:
    """Format a TruthfulQA question as a two-option multiple-choice prompt."""
    return (
        f"Answer the following multiple choice question. Select one answer.\n\n"
        f"{question}\n\n"
        f"(A) {option_a}\n"
        f"(B) {option_b}\n\n"
        f"Answer with the letter only (A or B)."
    )


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}. Delete it to regenerate.")
        return

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    csv_path = os.path.join(OUTPUT_PATH, "source.csv")

    print(f"Downloading TruthfulQA.csv from {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL) as resp:
        csv_bytes = resp.read()
    assert csv_bytes, "Downloaded CSV is empty"
    with open(csv_path, "wb") as f:
        f.write(csv_bytes)
    print(f"Saved raw CSV to {csv_path} ({len(csv_bytes)} bytes)")

    # Parse the CSV (note: fields contain semicolon-delimited lists, but we
    # only use Best Answer + Best Incorrect Answer which are plain strings).
    reader = csv.DictReader(StringIO(csv_bytes.decode("utf-8")))
    required_cols = {"Type", "Category", "Question", "Best Answer", "Best Incorrect Answer"}
    assert required_cols.issubset(set(reader.fieldnames or [])), (
        f"CSV missing required columns. Found: {reader.fieldnames}. Required: {required_cols}"
    )

    rng = random.Random(SEED)
    rows = []
    for idx, row in enumerate(reader):
        question = row["Question"].strip()
        best_answer = row["Best Answer"].strip()
        best_incorrect = row["Best Incorrect Answer"].strip()
        assert question, f"Empty question at row {idx}"
        assert best_answer, f"Empty Best Answer at row {idx} (question: {question!r})"
        assert best_incorrect, f"Empty Best Incorrect Answer at row {idx} (question: {question!r})"

        # Randomize which option is A vs B
        if rng.random() < 0.5:
            option_a, option_b = best_answer, best_incorrect
            correct_answer = "A"
        else:
            option_a, option_b = best_incorrect, best_answer
            correct_answer = "B"

        rows.append({
            "prompt": format_prompt(question, option_a, option_b),
            "correct_answer": correct_answer,
            "question": question,
            "category": row.get("Category", "").strip(),
            "type": row.get("Type", "").strip(),
            "best_answer": best_answer,
            "best_incorrect_answer": best_incorrect,
            "option_a": option_a,
            "option_b": option_b,
        })

    assert rows, "No rows parsed from CSV — aborting"
    output_ds = Dataset.from_list(rows)
    output_ds.save_to_disk(OUTPUT_PATH)
    print(f"Saved {len(output_ds)} examples to {OUTPUT_PATH}")
    print(f"Columns: {output_ds.column_names}")

    answer_distribution: dict[str, int] = {}
    for row in rows:
        answer_distribution[row["correct_answer"]] = answer_distribution.get(row["correct_answer"], 0) + 1
    print(f"Answer distribution: {answer_distribution}")
    print(f"First example prompt:\n{rows[0]['prompt']}\n")
    print(f"Correct: {rows[0]['correct_answer']}")


if __name__ == "__main__":
    main()
