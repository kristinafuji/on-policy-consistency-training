#!/usr/bin/env python3
"""
Score MATH-500 inference results using huggingface/Math-Verify.

Math-Verify parses both the gold answer and the model response into SymPy
expressions and checks symbolic / numerical equivalence — no regex fragility,
no LLM judge, deterministic. It natively handles \\boxed{...}, "Answer: X",
and plain expressions in the response.

Usage:
    uv run python score_math500.py --input-dir ./eval_results/.../math-500/my-model
    uv run python score_math500.py --input-dirs dir1,dir2

Requires: math-verify (see pyproject.toml).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from math_verify import parse, verify

from utils.response_utils import strip_reasoning


def _verify_safe(gold_str: str, pred_str: str) -> bool:
    """Parse both sides with math_verify and return True iff they verify equal.

    math_verify raises on some pathological inputs; treat any exception as a
    non-match rather than crashing the whole batch.
    """
    if not pred_str:
        return False
    try:
        # Wrap gold in LaTeX delimiters so the LaTeX extractor fires on plain
        # expressions like `\left( 3, \frac{\pi}{2} \right)`.
        gold = parse(f"${gold_str}$")
        pred = parse(pred_str)
    except Exception:
        return False
    try:
        return bool(verify(gold, pred))
    except Exception:
        return False


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
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    by_level: dict[int, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    scored_rows = []

    for idx, (result, ref) in enumerate(zip(results, dataset)):
        response = result.get("response", "")
        expected = ref["correct_answer"]
        subject = ref.get("subject", "unknown") or "unknown"
        level = int(ref.get("level", -1))

        stripped = strip_reasoning(response)
        is_correct = _verify_safe(expected, stripped)
        if not stripped:
            no_answer += 1
        elif is_correct:
            correct += 1

        by_subject[subject]["total"] += 1
        by_level[level]["total"] += 1
        if is_correct:
            by_subject[subject]["correct"] += 1
            by_level[level]["correct"] += 1

        per_question.append({
            "idx": idx,
            "expected": expected,
            "correct": is_correct,
            "subject": subject,
            "level": level,
        })

        scored_rows.append({
            **result,
            "stripped_response": stripped,
            "correct_answer": expected,
            "extraction_correct": is_correct,
            "subject": subject,
            "level": level,
        })

    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    subject_scores = {}
    for sub, stats in sorted(by_subject.items()):
        subject_scores[sub] = {
            "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
            "correct": stats["correct"],
            "n": stats["total"],
        }
    level_scores = {}
    for lvl, stats in sorted(by_level.items()):
        level_scores[str(lvl)] = {
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
        "by_subject": subject_scores,
        "by_level": level_scores,
        "per_question": per_question,
    }

    scores_path = input_dir / "math500_scores.json"
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
    print(f"MATH-500 Results: {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {scores['correct']}/{scores['total']} ({scores['accuracy']:.1%})")
    print(f"  No answer / empty: {scores['no_answer']}")

    if scores["by_subject"]:
        print(f"\n  By subject:")
        for sub, stats in sorted(scores["by_subject"].items()):
            print(f"    {sub}: {stats['correct']}/{stats['n']} ({stats['accuracy']:.1%})")

    if scores["by_level"]:
        print(f"\n  By level:")
        for lvl, stats in sorted(scores["by_level"].items(), key=lambda x: int(x[0])):
            print(f"    L{lvl}: {stats['correct']}/{stats['n']} ({stats['accuracy']:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Score MATH-500 results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=str, help="Single results directory")
    group.add_argument("--input-dirs", type=str, help="Comma-separated results directories")
    parser.add_argument("--dataset-path", type=str, default="./datasets/math-500",
                        help="Path to reference MATH-500 dataset")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"

    if args.input_dir:
        dirs = [Path(args.input_dir)]
    else:
        dirs = [Path(d.strip()) for d in args.input_dirs.split(",")]

    all_scores = {}
    for d in dirs:
        scores = score_directory(d, dataset_path)
        all_scores[d.name] = scores
        print_summary(d.name, scores)

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
