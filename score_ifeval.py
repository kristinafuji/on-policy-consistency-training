#!/usr/bin/env python3
"""
Score IFEval (google/IFEval) inference results using the vendored
google-research instruction_following_eval scorer at third_party/ifeval/.

Reads `results.jsonl` (one JSON object per line with `prompt` + `response`)
and the reference dataset (for `key`, `instruction_id_list`, `kwargs`), runs
the strict and loose per-instruction verifiers, and writes
`ifeval_scores.json` with the four standard IFEval accuracy metrics plus a
per-instruction-type breakdown.

Usage:
    uv run python score_ifeval.py --input-dir ./eval_results/regression-programmatic/my-model/ifeval
    uv run python score_ifeval.py --input-dirs dir1,dir2
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make the repo root importable so `third_party.ifeval` resolves when this
# script is invoked from the project directory.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The vendored IFEval scorer tokenizes with NLTK punkt_tab. Ensure it's
# available before importing — download once if missing. This runs at
# module-load time so it fails fast and is safe inside the container.
import nltk  # noqa: E402

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    print("NLTK punkt_tab not found; downloading...", flush=True)
    nltk.download("punkt_tab", quiet=True)

from third_party.ifeval import evaluation_lib  # noqa: E402
from utils.response_utils import strip_reasoning  # noqa: E402


def build_inputs(dataset) -> list[evaluation_lib.InputExample]:
    """Convert an HF IFEval dataset into the vendored scorer's InputExample objects.

    HuggingFace normalizes the `kwargs` schema across all rows, filling
    missing per-instruction fields with None. The vendored IFEval scorer
    calls `instruction.build_description(**kwargs_dict)`, which rejects
    unexpected kwargs — so we must strip None-valued entries before
    forwarding.
    """
    inputs = []
    for row in dataset:
        raw_kwargs_list = list(row["kwargs"])
        cleaned_kwargs_list = [
            {k: v for k, v in (d or {}).items() if v is not None}
            for d in raw_kwargs_list
        ]
        inputs.append(
            evaluation_lib.InputExample(
                key=int(row["key"]),
                instruction_id_list=list(row["instruction_id_list"]),
                prompt=row["prompt"],
                kwargs=cleaned_kwargs_list,
            )
        )
    return inputs


def build_prompt_to_response(results: list[dict]) -> dict[str, str]:
    """Build a prompt→response dict from the inference results.

    `strip_reasoning()` drops Qwen3 `<think>...</think>` and gpt-oss Harmony
    analysis-channel parts before the Google verifier sees the response. Without
    this, instruction checks like "respond in fewer than N words" fail because
    thinking alone is thousands of words (see lm-evaluation-harness #3161,
    #3240 — this preprocessing lifted Qwen3-1.7B IFEval by ~26 points).

    Asserts prompts are unique so that dict-based lookup in the vendored
    scorer doesn't silently collapse duplicates.
    """
    prompt_to_response: dict[str, str] = {}
    for idx, result in enumerate(results):
        prompt = result["prompt"]
        assert prompt not in prompt_to_response, (
            f"Duplicate prompt in results.jsonl (row {idx}): {prompt[:100]!r}. "
            f"IFEval expects unique prompts."
        )
        prompt_to_response[prompt] = strip_reasoning(result.get("response"))
    return prompt_to_response


def compute_metrics(outputs: list) -> dict:
    """Compute prompt-level and instruction-level accuracies + per-type breakdown."""
    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total: dict[str, int] = defaultdict(int)
    tier0_correct: dict[str, int] = defaultdict(int)

    tier1_total: dict[str, int] = defaultdict(int)
    tier1_correct: dict[str, int] = defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed in zip(instruction_id_list, follow_instruction_list):
            tier0_key = instruction_id.split(":")[0]
            tier0_total[tier0_key] += 1
            if followed:
                tier0_correct[tier0_key] += 1

            tier1_total[instruction_id] += 1
            if followed:
                tier1_correct[instruction_id] += 1

    return {
        "prompt_level_acc": prompt_correct / prompt_total if prompt_total > 0 else 0.0,
        "inst_level_acc": instruction_correct / instruction_total if instruction_total > 0 else 0.0,
        "prompt_correct": prompt_correct,
        "prompt_total": prompt_total,
        "inst_correct": instruction_correct,
        "inst_total": instruction_total,
        "by_instruction_category": {
            k: {
                "accuracy": tier0_correct[k] / tier0_total[k],
                "correct": tier0_correct[k],
                "n": tier0_total[k],
            }
            for k in sorted(tier0_total.keys())
        },
        "by_instruction_id": {
            k: {
                "accuracy": tier1_correct[k] / tier1_total[k],
                "correct": tier1_correct[k],
                "n": tier1_total[k],
            }
            for k in sorted(tier1_total.keys())
        },
    }


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

    inputs = build_inputs(dataset)
    prompt_to_response = build_prompt_to_response(results)

    # Sanity: every input prompt should appear in prompt_to_response
    missing = [inp.prompt for inp in inputs if inp.prompt not in prompt_to_response]
    assert not missing, (
        f"{len(missing)} input prompts have no response in results.jsonl "
        f"(first missing: {missing[0][:120]!r})"
    )

    # Strict + loose evaluation using the vendored scorer
    strict_outputs = [
        evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        for inp in inputs
    ]
    loose_outputs = [
        evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        for inp in inputs
    ]

    strict_metrics = compute_metrics(strict_outputs)
    loose_metrics = compute_metrics(loose_outputs)

    scores = {
        "prompt_level_strict_acc": strict_metrics["prompt_level_acc"],
        "inst_level_strict_acc": strict_metrics["inst_level_acc"],
        "prompt_level_loose_acc": loose_metrics["prompt_level_acc"],
        "inst_level_loose_acc": loose_metrics["inst_level_acc"],
        "n": len(inputs),
        "strict": strict_metrics,
        "loose": loose_metrics,
    }

    scores_path = input_dir / "ifeval_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved scores to {scores_path}")

    return scores


def print_summary(name: str, scores: dict):
    """Print a summary of IFEval scores."""
    print(f"\n{'=' * 60}")
    print(f"IFEval Results: {name}")
    print(f"{'=' * 60}")
    print(f"  Total prompts: {scores['n']}")
    print(f"  Strict  prompt-level: {scores['prompt_level_strict_acc']:.1%}")
    print(f"  Strict  instruction-level: {scores['inst_level_strict_acc']:.1%}")
    print(f"  Loose   prompt-level: {scores['prompt_level_loose_acc']:.1%}")
    print(f"  Loose   instruction-level: {scores['inst_level_loose_acc']:.1%}")

    print(f"\n  Strict accuracy by instruction category:")
    for cat, stats in sorted(scores["strict"]["by_instruction_category"].items()):
        print(f"    {cat}: {stats['correct']}/{stats['n']} ({stats['accuracy']:.1%})")


def main():
    parser = argparse.ArgumentParser(description="Score IFEval results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=str, help="Single results directory")
    group.add_argument("--input-dirs", type=str, help="Comma-separated results directories")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./datasets/ifeval",
        help="Path to reference IFEval dataset",
    )
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
        print("Comparison (strict prompt-level accuracy)")
        print(f"{'=' * 60}")
        print(f"{'Model':<45} {'Strict':>10} {'Loose':>10}")
        print(f"{'-' * 45} {'-' * 10} {'-' * 10}")
        for name, scores in sorted(
            all_scores.items(), key=lambda x: -x[1]["prompt_level_strict_acc"]
        ):
            print(
                f"{name:<45} "
                f"{scores['prompt_level_strict_acc']:>9.1%} "
                f"{scores['prompt_level_loose_acc']:>9.1%}"
            )


if __name__ == "__main__":
    main()
