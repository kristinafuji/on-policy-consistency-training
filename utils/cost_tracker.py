"""Cost tracking utilities for Tinker API training runs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pricing per million tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "Qwen/Qwen3-8B": {"prefill": 0.13, "sample": 0.40, "train": 0.40},
    "Qwen/Qwen3-4B-Instruct-2507": {"prefill": 0.07, "sample": 0.22, "train": 0.22},
    "meta-llama/Llama-3.1-8B": {"prefill": 0.13, "sample": 0.40, "train": 0.40},
    # Instruct variants use same pricing as base
    "meta-llama/Llama-3.1-8B-Instruct": {"prefill": 0.13, "sample": 0.40, "train": 0.40},
    "openai/gpt-oss-20b": {"prefill": 0.12, "sample": 0.30, "train": 0.36},
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {"prefill": 0.68, "sample": 1.70, "train": 2.04},
}


def get_pricing(model_name: str) -> dict[str, float] | None:
    """Get pricing for a model, handling common naming variations."""
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Try without -Instruct suffix
    base_name = model_name.replace("-Instruct", "")
    if base_name in MODEL_PRICING:
        return MODEL_PRICING[base_name]

    return None


@dataclass
class CostTracker:
    """Tracks token usage and computes costs during training.

    Usage:
        tracker = CostTracker("meta-llama/Llama-3.1-8B-Instruct")

        # During training loop
        tracker.add_prefill(prompt_tokens)
        tracker.add_sample(response_tokens)
        tracker.add_train(training_tokens)

        # Get metrics for logging
        metrics = tracker.get_metrics()
        ml_logger.log_metrics(metrics, step=i_batch)

        # Print final report
        print(tracker.format_report())
    """

    model_name: str
    prefill_tokens: int = 0
    sample_tokens: int = 0
    train_tokens: int = 0
    _pricing: dict[str, float] | None = field(default=None, repr=False)

    def __post_init__(self):
        self._pricing = get_pricing(self.model_name)
        if self._pricing is None:
            logger.warning(
                f"No pricing found for model '{self.model_name}'. "
                f"Cost tracking will show tokens only. "
                f"Known models: {list(MODEL_PRICING.keys())}"
            )

    def add_prefill(self, tokens: int) -> None:
        """Add prefill tokens (prompt processing for sampling or teacher eval)."""
        self.prefill_tokens += tokens

    def add_sample(self, tokens: int) -> None:
        """Add sample tokens (generated response tokens)."""
        self.sample_tokens += tokens

    def add_train(self, tokens: int) -> None:
        """Add train tokens (tokens processed during gradient updates)."""
        self.train_tokens += tokens

    def _compute_cost(self, tokens: int, cost_type: str) -> float:
        """Compute cost for a token count and cost type."""
        if self._pricing is None:
            return 0.0
        return tokens * self._pricing[cost_type] / 1_000_000

    def get_costs(self) -> dict[str, float | int]:
        """Get detailed cost breakdown.

        Returns dict with:
            - prefill_tokens, sample_tokens, train_tokens: token counts
            - prefill_cost, sample_cost, train_cost, total_cost: costs in USD
        """
        prefill_cost = self._compute_cost(self.prefill_tokens, "prefill")
        sample_cost = self._compute_cost(self.sample_tokens, "sample")
        train_cost = self._compute_cost(self.train_tokens, "train")

        return {
            "prefill_tokens": self.prefill_tokens,
            "sample_tokens": self.sample_tokens,
            "train_tokens": self.train_tokens,
            "prefill_cost": prefill_cost,
            "sample_cost": sample_cost,
            "train_cost": train_cost,
            "total_cost": prefill_cost + sample_cost + train_cost,
        }

    def get_metrics(self) -> dict[str, float]:
        """Get metrics dict suitable for ml_logger.log_metrics().

        Returns dict with cost/* prefixed keys.
        """
        costs = self.get_costs()
        return {
            "cost/prefill_tokens": costs["prefill_tokens"],
            "cost/sample_tokens": costs["sample_tokens"],
            "cost/train_tokens": costs["train_tokens"],
            "cost/prefill_usd": costs["prefill_cost"],
            "cost/sample_usd": costs["sample_cost"],
            "cost/train_usd": costs["train_cost"],
            "cost/total_usd": costs["total_cost"],
        }

    def format_report(self) -> str:
        """Format a human-readable cost report."""
        costs = self.get_costs()

        lines = [
            "=" * 50,
            "COST REPORT",
            "=" * 50,
            f"Model: {self.model_name}",
            "",
            "Token Usage:",
            f"  Prefill:  {costs['prefill_tokens']:>12,} tokens",
            f"  Sample:   {costs['sample_tokens']:>12,} tokens",
            f"  Train:    {costs['train_tokens']:>12,} tokens",
            "",
        ]

        if self._pricing is not None:
            lines.extend([
                "Cost Breakdown:",
                f"  Prefill:  ${costs['prefill_cost']:>11.4f}",
                f"  Sample:   ${costs['sample_cost']:>11.4f}",
                f"  Train:    ${costs['train_cost']:>11.4f}",
                "-" * 30,
                f"  TOTAL:    ${costs['total_cost']:>11.4f}",
            ])
        else:
            lines.append("(Pricing not available for this model)")

        lines.append("=" * 50)
        return "\n".join(lines)


def estimate_training_cost(
    model_name: str,
    dataset_size: int,
    batch_size: int,
    samples_per_prompt: int,
    num_epochs: int,
    avg_prompt_tokens: float,
    max_response_tokens: int,
    teacher_suffix_tokens: int = 100,
) -> dict[str, float | int | None]:
    """Estimate total training cost upfront.

    Args:
        model_name: Model identifier for pricing lookup
        dataset_size: Number of unique prompts in dataset
        batch_size: Prompts per batch (batch_size_prompts)
        samples_per_prompt: Trajectories sampled per prompt
        num_epochs: Number of training epochs
        avg_prompt_tokens: Average prompt length in tokens (sample from dataset)
        max_response_tokens: Maximum response length (cfg.max_tokens)
        teacher_suffix_tokens: Estimated tokens added by teacher prompt

    Returns:
        Dict with estimated token counts and costs (None if pricing unavailable)
    """
    pricing = get_pricing(model_name)

    total_samples = dataset_size * samples_per_prompt * num_epochs

    # Prefill: student prompt + teacher prompt (student prompt + suffix + response)
    # Per sample: student sees prompt, teacher sees (prompt + suffix + response)
    student_prefill_per_sample = avg_prompt_tokens
    teacher_prefill_per_sample = avg_prompt_tokens + teacher_suffix_tokens + (max_response_tokens / 2)
    total_prefill = int(total_samples * (student_prefill_per_sample + teacher_prefill_per_sample))

    # Sample: response tokens generated (using max_tokens / 2 as expected average)
    total_sample = int(total_samples * max_response_tokens / 2)

    # Train: all response tokens go through training
    total_train = total_sample

    result: dict[str, float | int | None] = {
        "total_samples": total_samples,
        "estimated_prefill_tokens": total_prefill,
        "estimated_sample_tokens": total_sample,
        "estimated_train_tokens": total_train,
    }

    if pricing is not None:
        prefill_cost = total_prefill * pricing["prefill"] / 1_000_000
        sample_cost = total_sample * pricing["sample"] / 1_000_000
        train_cost = total_train * pricing["train"] / 1_000_000
        result.update({
            "estimated_prefill_cost": prefill_cost,
            "estimated_sample_cost": sample_cost,
            "estimated_train_cost": train_cost,
            "estimated_total_cost": prefill_cost + sample_cost + train_cost,
        })
    else:
        result.update({
            "estimated_prefill_cost": None,
            "estimated_sample_cost": None,
            "estimated_train_cost": None,
            "estimated_total_cost": None,
        })

    return result


def format_cost_estimate(estimate: dict[str, float | int | None], model_name: str) -> str:
    """Format cost estimate as a human-readable string."""
    lines = [
        "=" * 50,
        "ESTIMATED TRAINING COST",
        "=" * 50,
        f"Model: {model_name}",
        f"Total samples: {estimate['total_samples']:,}",
        "",
        "Estimated Token Usage:",
        f"  Prefill:  {estimate['estimated_prefill_tokens']:>12,} tokens",
        f"  Sample:   {estimate['estimated_sample_tokens']:>12,} tokens",
        f"  Train:    {estimate['estimated_train_tokens']:>12,} tokens",
        "",
    ]

    if estimate["estimated_total_cost"] is not None:
        lines.extend([
            "Estimated Cost:",
            f"  Prefill:  ${estimate['estimated_prefill_cost']:>11.4f}",
            f"  Sample:   ${estimate['estimated_sample_cost']:>11.4f}",
            f"  Train:    ${estimate['estimated_train_cost']:>11.4f}",
            "-" * 30,
            f"  TOTAL:    ${estimate['estimated_total_cost']:>11.4f}",
            "",
            "NOTE: This is an estimate. Actual cost may vary based on",
            "      response lengths and early stopping.",
        ])
    else:
        lines.append(f"(Pricing not available for {model_name})")

    lines.append("=" * 50)
    return "\n".join(lines)
