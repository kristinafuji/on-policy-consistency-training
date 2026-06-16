#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) pipeline using the Tinker API.

Two phases:
1. Generate teacher completions (Phase 1) - skipped if sft_data_path is provided
2. Train student model via SFT on those completions (Phase 2)

The student sees only the original prompts (no teacher augmentation).
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from typing import Literal

from dotenv import load_dotenv

load_dotenv()  # Load TINKER_API_KEY from .env

import chz
import tinker
from tinker import types as tinker_types

from tinker_cookbook import renderers
from tinker_cookbook.supervised import train as supervised_train
from tinker_cookbook.supervised.data import (
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
    SupervisedDatasetBuilder,
)
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

from generate_sft_data import GenerationConfig, generate_sft_data
from utils.renderer_utils import (
    get_renderer_name_with_thinking_mode,
    should_mask_thinking,
    _get_thinking_markers_for_renderer,
    _matches_at_position,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(filename)s:%(lineno)-4s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


# --- Plain List Dataset (avoids pyarrow type inference issues) ---

class PlainListSupervisedDataset(SupervisedDataset):
    """SupervisedDataset backed by a plain list instead of HF Dataset.

    Avoids pyarrow type inference issues with mixed content types
    (e.g., string vs list content in gpt-oss messages with thinking parts).
    """

    def __init__(self, data: list[dict], batch_size: int, map_fn):
        self.data = data
        self.batch_size = batch_size
        self.map_fn = map_fn
        self._shuffled = data

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        end = start + self.batch_size
        return [self.map_fn(row) for row in self._shuffled[start:end]]

    def set_epoch(self, seed: int = 0):
        self._shuffled = self.data.copy()
        random.Random(seed).shuffle(self._shuffled)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size


@chz.chz
class PlainListConversationFileBuilder(SupervisedDatasetBuilder):
    """Like FromConversationFileBuilder but uses plain lists internally.

    FromConversationFileBuilder goes through datasets.Dataset.from_list()
    which uses pyarrow and fails when message content has mixed types
    (string for user messages, list of parts for assistant messages with thinking).
    This builder avoids that by keeping data as plain Python dicts.
    """

    common_config: ChatDatasetBuilderCommonConfig
    file_path: str
    test_size: int = 0
    shuffle_seed: int = 0

    @property
    def tokenizer(self) -> Tokenizer:
        return get_tokenizer(self.common_config.model_name_for_tokenizer)

    @property
    def renderer(self) -> renderers.Renderer:
        return renderers.get_renderer(self.common_config.renderer_name, self.tokenizer)

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Load conversations from JSONL file
        conversations = []
        with open(self.file_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                assert "messages" in data, (
                    f"Each line must contain a 'messages' field. Got: {list(data.keys())}"
                )
                conversations.append(data)

        # Shuffle
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(conversations)

        # Split into train and test
        if self.test_size > 0 and len(conversations) > self.test_size:
            test_data = conversations[:self.test_size]
            train_data = conversations[self.test_size:]
        else:
            train_data = conversations
            test_data = None

        train_on_what = (
            renderers.TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        renderer = self.renderer

        def map_fn(row: dict) -> tinker.Datum:
            return conversation_to_datum(
                row["messages"], renderer, self.common_config.max_length, train_on_what
            )

        train_dataset = PlainListSupervisedDataset(
            train_data, batch_size=self.common_config.batch_size, map_fn=map_fn
        )

        test_dataset = None
        if test_data is not None:
            test_dataset = PlainListSupervisedDataset(
                test_data, batch_size=len(test_data), map_fn=map_fn
            )

        return train_dataset, test_dataset


# --- Configuration ---

class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


COMMAND_FILENAME = "command.txt"


@chz.chz
class Config:
    # Shared
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    thinking_mode: Literal["enable", "disable"] | None = None
    train_on_thinking: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None

    # Phase 1 (generation) - used only if sft_data_path is None
    source_dataset: str = "./datasets/sageeval-train"
    teacher_type: Literal["cheat", "evc"] = "cheat"
    supervision_key: str = "safety_fact"
    teacher_prompt: str | None = None
    n_samples: int = 2
    gen_temperature: float = 1.0
    gen_max_tokens: int = 1024
    gen_batch_size: int = 512

    # Phase 2 (training)
    sft_data_path: str | None = None  # Skip Phase 1 if provided
    learning_rate: float = 1e-5
    num_epochs: int = 6
    lora_rank: int = 32
    batch_size: int = 64
    save_every: int = 20
    eval_every: int = 20
    max_length: int | None = None
    wandb_project: str | None = "consistency-opd"
    wandb_name: str | None = None
    load_checkpoint_path: str | None = None
    base_url: str | None = None
    enable_trace: bool = False


# --- Validation ---

def validate_config(cfg: Config) -> None:
    """Fail-fast validation. Raises ConfigurationError on any issue."""
    errors: list[str] = []

    # Numeric bounds
    if cfg.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {cfg.learning_rate}")
    if cfg.num_epochs <= 0:
        errors.append(f"num_epochs must be positive, got {cfg.num_epochs}")
    if cfg.lora_rank <= 0:
        errors.append(f"lora_rank must be positive, got {cfg.lora_rank}")
    if cfg.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {cfg.batch_size}")
    if cfg.save_every <= 0:
        errors.append(f"save_every must be positive, got {cfg.save_every}")
    if cfg.eval_every <= 0:
        errors.append(f"eval_every must be positive, got {cfg.eval_every}")

    # log_path is required
    if not cfg.log_path:
        errors.append("log_path must be specified")

    # Phase 1 validation (if generating)
    if cfg.sft_data_path is None:
        if not os.path.exists(cfg.source_dataset):
            errors.append(f"source_dataset path does not exist: {cfg.source_dataset}")
        if cfg.n_samples <= 0:
            errors.append(f"n_samples must be positive, got {cfg.n_samples}")
        if cfg.gen_temperature <= 0:
            errors.append(f"gen_temperature must be positive, got {cfg.gen_temperature}")
        if cfg.gen_max_tokens <= 0:
            errors.append(f"gen_max_tokens must be positive, got {cfg.gen_max_tokens}")
        if cfg.gen_batch_size <= 0:
            errors.append(f"gen_batch_size must be positive, got {cfg.gen_batch_size}")
    else:
        # Pre-generated data validation
        if not os.path.exists(cfg.sft_data_path):
            errors.append(f"sft_data_path does not exist: {cfg.sft_data_path}")
        else:
            # Validate first line has "messages" key
            with open(cfg.sft_data_path, "r") as f:
                first_line = f.readline().strip()
            if first_line:
                first_record = json.loads(first_line)
                if "messages" not in first_record:
                    errors.append(
                        f"sft_data_path first line missing 'messages' key. "
                        f"Got keys: {list(first_record.keys())}"
                    )

    # Thinking mode validation
    is_qwen_model = "qwen" in cfg.model_name.lower()
    is_openai_model = "openai" in cfg.model_name.lower() or "gpt-oss" in cfg.model_name.lower()
    supports_thinking_mode = is_qwen_model or is_openai_model

    if cfg.thinking_mode is not None and not supports_thinking_mode:
        errors.append(
            f"thinking_mode='{cfg.thinking_mode}' is only valid for Qwen or OpenAI models, "
            f"but model_name='{cfg.model_name}'"
        )
    if cfg.train_on_thinking is False and not supports_thinking_mode:
        # For models without thinking mode, train_on_thinking=False is a no-op.
        # Only warn if it was explicitly set (not the default).
        pass

    if cfg.reasoning_effort is not None:
        if not is_openai_model:
            errors.append(
                f"reasoning_effort='{cfg.reasoning_effort}' is only valid for OpenAI models, "
                f"but model_name='{cfg.model_name}'"
            )
        elif cfg.thinking_mode != "enable":
            errors.append(
                f"reasoning_effort='{cfg.reasoning_effort}' requires thinking_mode='enable', "
                f"but thinking_mode='{cfg.thinking_mode}'"
            )

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigurationError(error_msg)


# --- Config Persistence ---

def save_command_line(log_path: str) -> None:
    """Save the full command line used to invoke the script. Appends on resume."""
    os.makedirs(log_path, exist_ok=True)
    command_path = os.path.join(log_path, COMMAND_FILENAME)

    command_line = " ".join(sys.argv)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    is_new_file = not os.path.exists(command_path)
    with open(command_path, "a") as f:
        if not is_new_file:
            f.write("\n" + "=" * 80 + "\n\n")
        f.write(f"# Command executed at: {timestamp}\n")
        f.write(f"# Working directory: {os.getcwd()}\n\n")
        f.write(f"{command_line}\n")

    logger.info(f"Saved command line to {command_path}")


def save_config(cfg: Config, log_path: str) -> None:
    """Save config to log_path for reproducibility."""
    os.makedirs(log_path, exist_ok=True)
    config_path = os.path.join(log_path, "sft_config.json")
    with open(config_path, "w") as f:
        json.dump(chz.asdict(cfg), f, indent=2)
    logger.info(f"Saved config to {config_path}")


# --- Thinking Token Masking for SFT ---

def mask_thinking_tokens_sft(
    data: list[tinker.Datum],
    tokenizer: Tokenizer,
    renderer_name: str,
) -> None:
    """
    Modify datum weights in-place to exclude thinking tokens from SFT loss.

    SFT datums use "weights" (not "mask" like RL datums).
    Sets weights=0 for all tokens in thinking regions (inclusive of markers).

    Weight offset: weights[i] corresponds to predicting token[i+1] (next-token shift).
    So thinking at token positions [start, end) -> zero weights[start-1 : end-1].
    """
    markers = _get_thinking_markers_for_renderer(renderer_name, tokenizer)
    if markers is None:
        logger.warning(f"Renderer '{renderer_name}' does not have thinking tokens to mask")
        return

    think_start_tokens, think_end_tokens = markers

    for datum in data:
        # Get the full token sequence
        # datum.model_input is already right-shifted (missing last token)
        # datum.loss_fn_inputs["target_tokens"] has the target tokens
        input_tokens = datum.model_input.to_ints()
        target_tokens = datum.loss_fn_inputs["target_tokens"].data
        # Reconstruct full sequence: input + last target token
        full_tokens = list(input_tokens) + [target_tokens[-1]]

        weights_td = datum.loss_fn_inputs["weights"]
        weights = list(weights_td.data)

        assert len(weights) == len(input_tokens), (
            f"weights length {len(weights)} != input_tokens length {len(input_tokens)}. "
            f"Expected weights[i] predicts token[i+1]."
        )

        # Verify weights are non-negative
        assert all(w >= 0 for w in weights), (
            f"Found negative weight values. Weights should be non-negative."
        )

        # Find thinking regions in the full token sequence and zero weights
        i = 0
        while i < len(full_tokens):
            if _matches_at_position(full_tokens, i, think_start_tokens):
                start_pos = i
                j = i + len(think_start_tokens)
                while j < len(full_tokens):
                    if _matches_at_position(full_tokens, j, think_end_tokens):
                        end_pos = j + len(think_end_tokens)
                        # Zero weights for thinking region
                        # weights[k] predicts token[k+1], so to mask tokens [start, end):
                        # zero weights[start-1 : end-1]
                        mask_start = max(0, start_pos - 1)
                        mask_end = min(len(weights), end_pos - 1)
                        for k in range(mask_start, mask_end):
                            weights[k] = 0.0
                        i = end_pos
                        break
                    j += 1
                else:
                    # No closing tag - mask to end
                    mask_start = max(0, start_pos - 1)
                    for k in range(mask_start, len(weights)):
                        weights[k] = 0.0
                    logger.warning(
                        f"Unclosed thinking block at position {start_pos}, "
                        f"masking {len(weights) - mask_start} weights to end"
                    )
                    break
            else:
                i += 1

        # Update the weights in the datum
        datum.loss_fn_inputs["weights"] = tinker_types.TensorData(
            data=weights,
            dtype="float32",
            shape=[len(weights)],
        )


# --- Thinking-Masked Dataset Wrapper ---

class ThinkingMaskedDataset(SupervisedDataset):
    """Wraps a SupervisedDataset, applies thinking masking to each batch."""

    def __init__(
        self,
        inner: SupervisedDataset,
        tokenizer: Tokenizer,
        renderer_name: str,
    ):
        self.inner = inner
        self.tokenizer = tokenizer
        self.renderer_name = renderer_name

    def get_batch(self, index: int) -> list[tinker.Datum]:
        data = self.inner.get_batch(index)
        mask_thinking_tokens_sft(data, self.tokenizer, self.renderer_name)
        return data

    def set_epoch(self, seed: int = 0):
        self.inner.set_epoch(seed)

    def __len__(self) -> int:
        return len(self.inner)


@chz.chz
class ThinkingMaskedConversationFileBuilder(SupervisedDatasetBuilder):
    """
    Wraps a FromConversationFileBuilder, applying thinking token masking
    to the returned datasets when masking is needed.
    """
    inner_builder: PlainListConversationFileBuilder
    tokenizer_model_name: str
    renderer_name: str

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train_dataset, test_dataset = self.inner_builder()

        tokenizer = get_tokenizer(self.tokenizer_model_name)

        wrapped_train = ThinkingMaskedDataset(train_dataset, tokenizer, self.renderer_name)

        wrapped_test = None
        if test_dataset is not None:
            wrapped_test = ThinkingMaskedDataset(test_dataset, tokenizer, self.renderer_name)

        return wrapped_train, wrapped_test


# --- Main Pipeline ---

async def main(cfg: Config):
    logger.info("Validating configuration...")
    validate_config(cfg)
    logger.info("Configuration validation passed")

    save_command_line(cfg.log_path)
    save_config(cfg, cfg.log_path)

    renderer_name = get_renderer_name_with_thinking_mode(
        cfg.model_name, cfg.thinking_mode, cfg.reasoning_effort
    )
    logger.info(f"Using renderer: {renderer_name} for model: {cfg.model_name}")

    # ========================
    # Phase 1: Generate (if needed)
    # ========================
    if cfg.sft_data_path is None:
        sft_data_path = os.path.join(cfg.log_path, "sft_data.jsonl")
        logger.info(f"Phase 1: Generating SFT data -> {sft_data_path}")

        gen_config = GenerationConfig(
            model_name=cfg.model_name,
            dataset_path=cfg.source_dataset,
            output_path=sft_data_path,
            teacher_type=cfg.teacher_type,
            supervision_key=cfg.supervision_key,
            teacher_prompt=cfg.teacher_prompt,
            n_samples=cfg.n_samples,
            temperature=cfg.gen_temperature,
            max_tokens=cfg.gen_max_tokens,
            batch_size=cfg.gen_batch_size,
            thinking_mode=cfg.thinking_mode,
            reasoning_effort=cfg.reasoning_effort,
            base_url=cfg.base_url,
        )
        await generate_sft_data(gen_config)
        logger.info("Phase 1 complete.")
    else:
        sft_data_path = cfg.sft_data_path
        logger.info(f"Phase 1 skipped: using pre-generated data from {sft_data_path}")

    # ========================
    # Phase 2: Train via cookbook SFT
    # ========================
    logger.info(f"Phase 2: Training via SFT on {sft_data_path}")

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
    )
    inner_builder = PlainListConversationFileBuilder(
        common_config=common_config,
        file_path=sft_data_path,
    )

    needs_masking = not cfg.train_on_thinking and should_mask_thinking(renderer_name)
    if needs_masking:
        logger.info(f"Thinking token masking enabled (renderer={renderer_name})")
        dataset_builder = ThinkingMaskedConversationFileBuilder(
            inner_builder=inner_builder,
            tokenizer_model_name=cfg.model_name,
            renderer_name=renderer_name,
        )
    else:
        dataset_builder = inner_builder

    train_config = supervised_train.Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        dataset_builder=dataset_builder,
        learning_rate=cfg.learning_rate,
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        save_every=cfg.save_every,
        eval_every=cfg.eval_every,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name or os.path.basename(cfg.log_path.rstrip("/")),
        load_checkpoint_path=cfg.load_checkpoint_path,
        base_url=cfg.base_url,
        enable_trace=cfg.enable_trace,
    )
    await supervised_train.main(train_config)

    logger.info("SFT pipeline completed successfully")


if __name__ == "__main__":
    cfg = chz.entrypoint(Config, allow_hyphens=True)
    asyncio.run(main(cfg))
