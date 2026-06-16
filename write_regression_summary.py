#!/usr/bin/env python3
"""
Aggregate per-eval scores into a single regression_summary.json.

Reads whichever of `gpqa-diamond/gpqa_scores.json`,
`ifeval/ifeval_scores.json`, and `math-500/math500_scores.json` exist
under the given input directory, and writes a combined
`regression_summary.json` at the input-dir root with top-line metrics and
checkpoint metadata. Missing evals are recorded in `missing_evals` rather
than causing a failure — this lets partial runs still produce a summary.

Usage:
    uv run python write_regression_summary.py \\
        --input-dir ./eval_results/regression-programmatic/my-model \\
        --eval-name my-model \\
        --base-model meta-llama/Llama-3.1-8B-Instruct \\
        --checkpoint ./logs/my-run/downloaded_checkpoints/000700/sampler_weights
"""

import argparse
import datetime as dt
import json
from pathlib import Path


def load_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def build_gpqa_section(scores: dict | None) -> dict | None:
    if scores is None:
        return None
    return {
        "accuracy": scores["accuracy"],
        "correct": scores["correct"],
        "total": scores["total"],
        "no_answer": scores.get("no_answer", 0),
        "extraction_rate": scores.get("extraction_rate", 1.0),
        "by_subdomain": scores.get("by_subdomain", {}),
    }


def build_ifeval_section(scores: dict | None) -> dict | None:
    if scores is None:
        return None
    return {
        "prompt_level_strict_acc": scores["prompt_level_strict_acc"],
        "inst_level_strict_acc": scores["inst_level_strict_acc"],
        "prompt_level_loose_acc": scores["prompt_level_loose_acc"],
        "inst_level_loose_acc": scores["inst_level_loose_acc"],
        "n": scores["n"],
        "by_instruction_category_strict": scores.get("strict", {}).get("by_instruction_category", {}),
    }


def build_math500_section(scores: dict | None) -> dict | None:
    if scores is None:
        return None
    return {
        "accuracy": scores["accuracy"],
        "correct": scores["correct"],
        "total": scores["total"],
        "no_answer": scores.get("no_answer", 0),
        "extraction_rate": scores.get("extraction_rate", 1.0),
        "by_subject": scores.get("by_subject", {}),
        "by_level": scores.get("by_level", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="Write regression eval summary JSON")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Root output dir containing per-eval subdirs")
    parser.add_argument("--eval-name", type=str, required=True)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Tinker URI or local path of the checkpoint being evaluated")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    assert input_dir.exists(), f"Input dir not found: {input_dir}"

    gpqa_scores = load_optional_json(input_dir / "gpqa-diamond" / "gpqa_scores.json")
    ifeval_scores = load_optional_json(input_dir / "ifeval" / "ifeval_scores.json")
    math500_scores = load_optional_json(input_dir / "math-500" / "math500_scores.json")

    missing = []
    if gpqa_scores is None:
        missing.append("gpqa")
    if ifeval_scores is None:
        missing.append("ifeval")
    if math500_scores is None:
        missing.append("math500")

    summary = {
        "eval_name": args.eval_name,
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpqa": build_gpqa_section(gpqa_scores),
        "ifeval": build_ifeval_section(ifeval_scores),
        "math500": build_math500_section(math500_scores),
        "missing_evals": missing,
    }

    summary_path = input_dir / "regression_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # Human-readable console summary
    print(f"\n{'=' * 60}")
    print(f"Regression summary: {args.eval_name}")
    print(f"Base model: {args.base_model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'=' * 60}")
    if summary["gpqa"] is not None:
        g = summary["gpqa"]
        print(f"GPQA-Diamond:      {g['correct']}/{g['total']} = {g['accuracy']:.1%}  (extraction {g['extraction_rate']:.1%})")
    else:
        print("GPQA-Diamond:      MISSING")
    if summary["ifeval"] is not None:
        i = summary["ifeval"]
        print(f"IFEval (n={i['n']}):  strict prompt={i['prompt_level_strict_acc']:.1%}  loose prompt={i['prompt_level_loose_acc']:.1%}  strict inst={i['inst_level_strict_acc']:.1%}  loose inst={i['inst_level_loose_acc']:.1%}")
    else:
        print("IFEval:            MISSING")
    if summary["math500"] is not None:
        m = summary["math500"]
        print(f"MATH-500:          {m['correct']}/{m['total']} = {m['accuracy']:.1%}  (extraction {m['extraction_rate']:.1%})")
    else:
        print("MATH-500:          MISSING")


if __name__ == "__main__":
    main()
