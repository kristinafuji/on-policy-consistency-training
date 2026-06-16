"""
Data preparation for GRPO attacker training.

Handles behavior splitting (train / eval), few-shot PAIR example
selection, and HuggingFace Dataset construction.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset

from .config import GRPOAttackConfig
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def load_behaviors(path: str) -> List[Dict]:
    """Load behavior data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def split_behaviors(
    config: GRPOAttackConfig,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split test behaviors into GRPO training and held-out eval sets.

    Returns:
        (train_behaviors, eval_behaviors)
    """
    test_data = load_behaviors(config.test_data_path)

    rng = random.Random(config.behavior_seed)
    indices = list(range(len(test_data)))
    rng.shuffle(indices)

    train_behaviors = [test_data[i] for i in indices[: config.num_grpo_behaviors]]
    eval_behaviors = [test_data[i] for i in indices[config.num_grpo_behaviors :]]

    print(f"[data] Split {len(test_data)} test behaviors: "
          f"{len(train_behaviors)} GRPO train, {len(eval_behaviors)} held-out eval")

    split_path = Path(config.output_dir) / "behavior_split.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        json.dump({
            "train_behaviors": [b["clean"] for b in train_behaviors],
            "eval_behaviors": [b["clean"] for b in eval_behaviors],
            "seed": config.behavior_seed,
        }, f, indent=2)
    print(f"[data] Behavior split saved to {split_path}")

    return train_behaviors, eval_behaviors


def get_few_shot_pool(config: GRPOAttackConfig) -> List[str]:
    """Get pool of PAIR-wrapped prompts from ACT training set for few-shot examples."""
    train_data = load_behaviors(config.train_data_path)
    return [item["wrapped"] for item in train_data]


def format_prompt(behavior: str, few_shot_examples: List[str]) -> List[Dict]:
    """Format a single behavior into conversational prompt format."""
    user_content = USER_PROMPT_TEMPLATE.format(
        pair_example_1=few_shot_examples[0],
        pair_example_2=few_shot_examples[1],
        pair_example_3=few_shot_examples[2],
        behavior=behavior,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_grpo_dataset(
    train_behaviors: List[Dict],
    config: GRPOAttackConfig,
) -> Dataset:
    """Build HuggingFace Dataset for GRPO training."""
    few_shot_pool = get_few_shot_pool(config)
    rng = random.Random(config.seed)

    rows = []
    for repeat_idx in range(config.num_repeats):
        for item in train_behaviors:
            examples = rng.sample(few_shot_pool, config.num_few_shot)
            prompt = format_prompt(item["clean"], examples)
            rows.append({"prompt": prompt, "behavior": item["clean"]})

    rng.shuffle(rows)

    dataset = Dataset.from_list(rows)
    print(f"[data] Built GRPO dataset: {len(dataset)} samples "
          f"({len(train_behaviors)} behaviors x {config.num_repeats} repeats)")

    return dataset
