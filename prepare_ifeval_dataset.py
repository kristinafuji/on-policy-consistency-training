#!/usr/bin/env python3
"""
Prepare IFEval (google/IFEval) dataset for evaluation.

Downloads IFEval from HuggingFace and saves it as a local HuggingFace dataset
at ./datasets/ifeval/. The raw columns (prompt, key, instruction_id_list,
kwargs) are preserved verbatim — the vendored scorer at third_party/ifeval/
needs all of them.

Usage:
    uv run python prepare_ifeval_dataset.py
"""

import os

from datasets import load_dataset

OUTPUT_PATH = "./datasets/ifeval"
EXPECTED_ROWS = 541


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}. Delete it to regenerate.")
        return

    print("Loading google/IFEval from HuggingFace...")
    ds = load_dataset("google/IFEval", split="train")
    assert len(ds) == EXPECTED_ROWS, (
        f"Expected {EXPECTED_ROWS} rows, got {len(ds)}. "
        f"If this differs, upstream dataset changed — update EXPECTED_ROWS."
    )

    required_cols = {"prompt", "key", "instruction_id_list", "kwargs"}
    missing = required_cols - set(ds.column_names)
    assert not missing, f"Missing required columns: {missing}. Found: {ds.column_names}"

    ds.save_to_disk(OUTPUT_PATH)
    print(f"Saved {len(ds)} examples to {OUTPUT_PATH}")
    print(f"Columns: {ds.column_names}")
    print(f"First example keys: {list(ds[0].keys())}")
    print(f"First instruction_id_list: {ds[0]['instruction_id_list']}")


if __name__ == "__main__":
    main()
