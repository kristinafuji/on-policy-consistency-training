#!/usr/bin/env python3
"""
Score GPQA Diamond inference results.

Extracts predicted answer letters from model responses and computes accuracy.

Extraction strategy (thinking-model-aware):
  1. `strip_reasoning()` — drops Qwen3 `<think>...</think>` and gpt-oss Harmony
     analysis parts, leaving only the final answer text.
  2. Simple-evals primary regex (`Answer: $LETTER`). This matches when the
     prompt template instructs the model to emit that sentinel — the
     canonical method used by OpenAI simple-evals, cited by every frontier
     paper (DeepSeek-R1, Qwen3, gpt-oss, Llama-4, Claude).
  3. Fallback: gpt-oss `abcd_grader.extract_abcd()` — a 10-pattern
     priority-ordered cascade that tolerates markdown, \\boxed, (A), etc.
     Its own final fallback returns an arbitrary first char, so we guard
     on `letter in "ABCD"`.

Usage:
    uv run python score_gpqa.py --input-dir ./eval_results/.../gpqa-diamond/my-model
    uv run python score_gpqa.py --input-dirs dir1,dir2
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from utils.abcd_grader import extract_abcd
from utils.response_utils import strip_reasoning


# Copied from openai/simple-evals (MIT) `common.py::ANSWER_PATTERN_MULTICHOICE`.
# Matches "Answer: X", "Answer:$X", "Answer: $X$", case-insensitive.
SIMPLE_EVALS_ANSWER_PATTERN = re.compile(r"(?i)Answer\s*:\s*\$?([A-D])\$?")


def extract_answer(response) -> str | None:
    """Extract A/B/C/D from a model response.

    `response` may be a str, a list of Harmony content-parts, or None.
    """
    text = strip_reasoning(response)
    if not text:
        return None

    m = SIMPLE_EVALS_ANSWER_PATTERN.search(text)
    if m:
        return m.group(1).upper()

    letter = extract_abcd(text)
    if letter and letter in "ABCD":
        return letter
    return None


def score_directory(input_dir: Path, dataset_path: Path) -> dict:
    """Score a single directory's results against the reference dataset."""
    results_path = input_dir / "results.jsonl"
    assert results_path.exists(), f"results.jsonl not found in {input_dir}"

    from datasets import load_from_disk
    dataset = load_from_disk(str(dataset_path))

    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    assert len(results) == len(dataset), (
        f"Result count ({len(results)}) != dataset size ({len(dataset)}). "
        f"Expected n_trials=1."
    )

    correct = 0
    no_answer = 0
    per_question = []
    by_subdomain = defaultdict(lambda: {"correct": 0, "total": 0})
    scored_rows = []

    for idx, (result, ref) in enumerate(zip(results, dataset)):
        response = result.get("response", "")
        expected = ref["correct_answer"]
        subdomain = ref.get("subdomain", "unknown")

        predicted = extract_answer(response)

        is_correct = predicted == expected
        if predicted is None:
            no_answer += 1
        elif is_correct:
            correct += 1

        by_subdomain[subdomain]["total"] += 1
        if is_correct:
            by_subdomain[subdomain]["correct"] += 1

        per_question.append({
            "idx": idx,
            "predicted": predicted,
            "expected": expected,
            "correct": is_correct,
        })

        scored_rows.append({
            **result,
            "extracted_answer": predicted,
            "correct_answer": expected,
            "extraction_correct": is_correct,
            "subdomain": subdomain,
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    subdomain_scores = {}
    for sub, stats in sorted(by_subdomain.items()):
        subdomain_scores[sub] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "n": stats["total"],
        }

    scores = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "no_answer": no_answer,
        "extraction_rate": (total - no_answer) / total if total > 0 else 0.0,
        "by_subdomain": subdomain_scores,
        "per_question": per_question,
    }

    scores_path = input_dir / "gpqa_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved scores to {scores_path}")

    scored_path = input_dir / "results_scored.jsonl"
    with open(scored_path, "w") as f:
        for row in scored_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved scored results to {scored_path}")

    return scores


def print_summary(name: str, scores: dict):
    print(f"\n{'=' * 60}")
    print(f"GPQA Diamond Results: {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {scores['correct']}/{scores['total']} ({scores['accuracy']:.1%})")
    print(f"  No answer extracted: {scores['no_answer']}")
    print(f"  Extraction rate: {scores['extraction_rate']:.1%}")

    if scores["by_subdomain"]:
        print(f"\n  By subdomain:")
        for sub, stats in sorted(scores["by_subdomain"].items()):
            print(f"    {sub}: {stats['correct']}/{stats['n']} ({stats['accuracy']:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Score GPQA Diamond results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=str, help="Single results directory")
    group.add_argument("--input-dirs", type=str, help="Comma-separated results directories")
    parser.add_argument("--dataset-path", type=str, default="./datasets/gpqa-diamond",
                        help="Path to reference GPQA dataset")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    if args.input_dir:
        dirs = [Path(args.input_dir)]
    else:
        dirs = [Path(d.strip()) for d in args.input_dirs.split(",")]

    all_scores = {}
    for d in dirs:
        name = d.name
        scores = score_directory(d, dataset_path)
        all_scores[name] = scores
        print_summary(name, scores)

    if len(all_scores) > 1:
        print(f"\n{'=' * 60}")
        print("Comparison")
        print(f"{'=' * 60}")
        print(f"{'Model':<45} {'Accuracy':>10}")
        print(f"{'-' * 45} {'-' * 10}")
        for name, scores in sorted(all_scores.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"{name:<45} {scores['accuracy']:>9.1%}")


if __name__ == "__main__":
    main()
