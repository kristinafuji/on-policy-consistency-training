#!/usr/bin/env python3

"""
Local PyTorch SFT (Supervised Fine-Tuning) training module.

This module provides two main capabilities:
1. Teacher completion generation: Generate completions using teacher mode (augmented prompts)
2. SFT training: Fine-tune a model on (original_prompt, teacher_completion) pairs

Can be run standalone or imported by train_local_evc.py.

Key features:
- Cross-entropy loss on completion tokens only (prompt tokens masked)
- Incremental JSONL saving for resume support
- Compatible with local_inference.py output format
- Reuses local_ppo/ utilities for LoRA management and checkpointing
"""

# Early progress output (before heavy imports)
import sys
print("[1/6] Starting SFT script, importing libraries...", flush=True)

# IMPORTANT: Import unsloth BEFORE any other ML libraries (transformers, torch, etc.)
# to ensure all Unsloth optimizations are applied correctly.
print("[2/6] Importing Unsloth...", flush=True)
try:
    import unsloth  # noqa: F401
except ImportError:
    print("  (Unsloth not available, will use standard PEFT)", flush=True)

print("[3/6] Importing PyTorch and other libraries...", flush=True)
import json
import logging
import os
import time
from typing import Iterator, Literal

import chz
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
print("[4/6] PyTorch loaded, importing local modules...", flush=True)

# Add tinker-cookbook to path for reusing renderers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tinker-cookbook"))

from tinker_cookbook import renderers
from tinker_cookbook.utils import ml_log

from local_ppo.checkpointing import (
    get_last_checkpoint,
    load_checkpoint,
    save_checkpoint,
    save_command_line,
    save_config,
    validate_resume_config,
    load_config,
)
from local_ppo.gpu_monitor import GPUMonitor, log_gpu_stats_once
from local_ppo.model_manager import LocalModelManager, create_optimizer
from local_ppo.sampling import LocalSampler, pad_token_lists
from utils.renderer_utils import get_renderer_name_with_thinking_mode
print("[5/6] All imports complete, parsing config...", flush=True)

logger = logging.getLogger(__name__)


# --- Default Teacher Prompts ---

DEFAULT_TEACHER_PROMPTS = {
    "user": "\n\nAnswer the question safely and helpfully.",
    "system": """Your job is to produce a safe, policy-aligned response that follows the structure below.

Step 1 — Safety thinking (in tags)
- At the very beginning of your response, include a <safety_thinking> section.
- In <safety_thinking>, explicitly identify the part(s) of the user prompt that contain salient safety risks.
- You MUST include this exact sentence in <safety_thinking>:
  "The user prompt might contain more than one safety risk. I must think hard to consider all the potential safety risks before providing a response."
- Then list any possible potential safety risks as:
  - Possible potential safety risk 1: ...
  - Possible potential safety risk 2: ...
  - Possible potential safety risk 3: ...
  (Include as many as apply.)

Step 2 — Safety warnings (immediately after tags)
- Immediately after </safety_thinking>, start with:
  'Warning: [Safety warning 1], [Safety warning 2], ...'
- Then provide DEATAILED clear, user-facing warnings that correspond to every risk listed in Step 1.

Step 3 — The safe response
- After the warnings, provide the safest possible answer to the user's request.
- You should err on the side of explicitly telling users not to do something whenever there is any potential safety risk, rather than suggesting safer alternatives. THIS IS EXTREMELY IMPORTANT.
- You should err on the side of explicitly telling users not to do something whenever there is any potential safety risk, rather than suggesting safer alternatives. THIS IS EXTREMELY IMPORTANT.
- You should err on the side of explicitly telling users not to do something whenever there is any potential safety risk, rather than suggesting safer alternatives. THIS IS EXTREMELY IMPORTANT.
- Do not output any sentence that could be seen as violating the safety risks you identify, even remotely.""",
}


def get_teacher_prompt(cfg: "SFTConfig") -> str:
    """Resolve the teacher prompt, using mode-dependent default if not specified."""
    if cfg.teacher_prompt is not None:
        return cfg.teacher_prompt
    return DEFAULT_TEACHER_PROMPTS[cfg.teacher_mode]


# --- Configuration ---

@chz.chz
class SFTConfig:
    # Model settings
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507"
    thinking_mode: Literal["enable", "disable"] | None = None

    # Dataset settings
    dataset_path: str = "./datasets/sageeval-train"

    # Teacher prompt configuration (for generation phase)
    teacher_mode: Literal["user", "system"] = "system"
    teacher_prompt: str | None = None

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 512
    sampling_batch_size: int = 32

    # SFT Training hyperparameters
    learning_rate: float = 2e-5
    lora_rank: int = 32
    num_epochs: int = 1
    batch_size: int = 8  # Training batch size (number of examples)
    gradient_accumulation_steps: int = 4

    # Logging and checkpointing
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 100  # Save every N training steps

    # Hardware settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_seq_length: int = 2048

    # GPU monitoring
    gpu_monitor_interval: float = 30.0

    # Optional: skip generation if data already exists
    sft_data_path: str | None = None  # Path to pre-generated JSONL


def validate_sft_config(cfg: SFTConfig) -> None:
    """Validate SFT configuration parameters."""
    errors = []

    if cfg.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {cfg.learning_rate}")
    if cfg.lora_rank <= 0:
        errors.append(f"lora_rank must be positive, got {cfg.lora_rank}")
    if cfg.num_epochs <= 0:
        errors.append(f"num_epochs must be positive, got {cfg.num_epochs}")
    if cfg.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {cfg.batch_size}")
    if cfg.gradient_accumulation_steps <= 0:
        errors.append(f"gradient_accumulation_steps must be positive, got {cfg.gradient_accumulation_steps}")
    if cfg.temperature <= 0:
        errors.append(f"temperature must be positive, got {cfg.temperature}")
    if cfg.max_tokens <= 0:
        errors.append(f"max_tokens must be positive, got {cfg.max_tokens}")
    if cfg.sampling_batch_size <= 0:
        errors.append(f"sampling_batch_size must be positive, got {cfg.sampling_batch_size}")
    if cfg.save_every <= 0:
        errors.append(f"save_every must be positive, got {cfg.save_every}")

    if not os.path.exists(cfg.dataset_path):
        errors.append(f"Dataset path does not exist: {cfg.dataset_path}")

    if not cfg.log_path:
        errors.append("log_path must be specified")

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


# --- Teacher Completion Generation ---

def build_teacher_messages(
    prompt: str,
    teacher_mode: str,
    teacher_prompt: str,
) -> list[dict]:
    """Build teacher conversation messages."""
    if teacher_mode == "system":
        messages = [
            {"role": "system", "content": teacher_prompt},
            {"role": "user", "content": prompt},
        ]
    else:  # "user" mode
        augmented_message = f"{prompt}{teacher_prompt}"
        messages = [{"role": "user", "content": augmented_message}]

    return messages


def generate_teacher_completions(
    model,
    tokenizer,
    dataset: Dataset,
    output_path: str,
    renderer: "renderers.Renderer",
    teacher_mode: str,
    teacher_prompt: str,
    temperature: float,
    max_tokens: float,
    sampling_batch_size: int,
    device: str,
    log_metrics_fn=None,
) -> int:
    """
    Generate teacher completions for all prompts in the dataset.

    Uses teacher mode (augmented system prompt) to generate
    completions, then saves them as (original_prompt, completion) pairs.

    Args:
        model: Model to use for generation.
        tokenizer: Tokenizer.
        dataset: HuggingFace dataset with 'prompt' column.
        output_path: Path to save JSONL output.
        renderer: Renderer for prompt formatting.
        teacher_mode: "system" or "user".
        teacher_prompt: Teacher system prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum generation length.
        sampling_batch_size: Batch size for generation.
        device: Device for generation.
        log_metrics_fn: Optional function to log metrics.

    Returns:
        Number of examples processed.
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check for resume
    existing_count = 0
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_count = sum(1 for _ in f)
        logger.info(f"Found {existing_count} existing results, resuming from there")

    if existing_count >= len(dataset):
        logger.info("All examples already generated, skipping generation phase")
        return existing_count

    # Create sampler (no LoRA for base model generation)
    sampler = LocalSampler(model, tokenizer, device=device)

    # Process in batches
    total_examples = len(dataset)
    start_idx = existing_count
    processed = 0
    total_tokens_generated = 0
    start_time = time.time()

    # Cumulative timing stats
    cumulative_prep_time = 0.0
    cumulative_inference_time = 0.0
    cumulative_parse_time = 0.0

    file_mode = "a" if existing_count > 0 else "w"
    with open(output_path, file_mode) as f:
        for batch_start in range(start_idx, total_examples, sampling_batch_size):
            batch_iter_start = time.time()
            batch_end = min(batch_start + sampling_batch_size, total_examples)
            batch_size = batch_end - batch_start

            # === BATCH PREPARATION ===
            prep_start = time.time()
            all_prompts = []
            all_metadata = []

            for idx in range(batch_start, batch_end):
                row = dataset[idx]
                prompt = row["prompt"]

                # Build teacher messages
                messages = build_teacher_messages(
                    prompt=prompt,
                    teacher_mode=teacher_mode,
                    teacher_prompt=teacher_prompt,
                )

                # Tokenize
                model_input = renderer.build_generation_prompt(messages)
                prompt_tokens = model_input.to_ints()
                all_prompts.append(prompt_tokens)

                all_metadata.append({
                    "dataset_idx": idx,
                    "prompt": prompt,
                    # Include any other metadata from the dataset
                    "safety_fact": row.get("safety_fact", ""),
                    "augmentation_category": row.get("augmentation_category", ""),
                    "safety_category": row.get("safety_category", ""),
                    "prompt_type": row.get("prompt_type", ""),
                    "version": row.get("version", ""),
                })

            # Pad for batched generation
            input_ids, attention_mask = pad_token_lists(
                all_prompts,
                pad_token_id=tokenizer.pad_token_id,
                device=device,
                padding_side="left",
            )
            prep_time = time.time() - prep_start
            cumulative_prep_time += prep_time

            # === MODEL INFERENCE ===
            inference_start = time.time()
            results = sampler.sample(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Required for correct logprobs with left-padding
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
            inference_time = time.time() - inference_start
            cumulative_inference_time += inference_time

            # Count tokens generated
            batch_tokens = sum(len(r.tokens) for r in results)
            total_tokens_generated += batch_tokens
            tokens_per_sec = batch_tokens / inference_time if inference_time > 0 else 0

            # === PARSE AND SAVE ===
            parse_start = time.time()
            for result, metadata in zip(results, all_metadata):
                # Parse response using renderer
                parsed_message, valid_format = renderer.parse_response(result.tokens)

                output = {
                    "dataset_idx": metadata["dataset_idx"],
                    "prompt": metadata["prompt"],
                    "completion": parsed_message["content"],
                    "safety_fact": metadata["safety_fact"],
                    "augmentation_category": metadata["augmentation_category"],
                    "safety_category": metadata["safety_category"],
                    "prompt_type": metadata["prompt_type"],
                    "version": metadata["version"],
                    "valid_format": valid_format,
                    "teacher_mode": teacher_mode,
                }
                f.write(json.dumps(output) + "\n")

            f.flush()
            parse_time = time.time() - parse_start
            cumulative_parse_time += parse_time

            processed += batch_size
            batch_total_time = time.time() - batch_iter_start

            # Log progress with detailed timing
            elapsed = time.time() - start_time
            examples_per_sec = processed / elapsed if elapsed > 0 else 0
            overall_tokens_per_sec = total_tokens_generated / elapsed if elapsed > 0 else 0

            logger.info(
                f"Generated {batch_end}/{total_examples} examples | "
                f"Batch: prep={prep_time:.2f}s, inference={inference_time:.2f}s ({tokens_per_sec:.1f} tok/s), "
                f"parse={parse_time:.2f}s, total={batch_total_time:.2f}s | "
                f"Overall: {examples_per_sec:.1f} ex/s, {overall_tokens_per_sec:.1f} tok/s"
            )

            if log_metrics_fn:
                log_metrics_fn({
                    "sft_gen/examples_processed": batch_end,
                    "sft_gen/examples_per_second": examples_per_sec,
                    "sft_gen/tokens_per_second": overall_tokens_per_sec,
                    "sft_gen/batch_tokens": batch_tokens,
                    "sft_gen/batch_prep_time": prep_time,
                    "sft_gen/batch_inference_time": inference_time,
                    "sft_gen/batch_parse_time": parse_time,
                    "sft_gen/batch_total_time": batch_total_time,
                }, step=batch_end)

            # Clear GPU cache
            torch.cuda.empty_cache()

    # Log final summary
    total_time = time.time() - start_time
    logger.info(f"Generation complete: {total_examples} examples saved to {output_path}")
    logger.info(
        f"Generation timing summary: "
        f"total={total_time:.1f}s, prep={cumulative_prep_time:.1f}s ({cumulative_prep_time/total_time*100:.1f}%), "
        f"inference={cumulative_inference_time:.1f}s ({cumulative_inference_time/total_time*100:.1f}%), "
        f"parse={cumulative_parse_time:.1f}s ({cumulative_parse_time/total_time*100:.1f}%)"
    )
    logger.info(
        f"Generation throughput: {total_tokens_generated} tokens in {total_time:.1f}s = "
        f"{total_tokens_generated/total_time:.1f} tok/s"
    )
    return total_examples


# --- SFT Dataset ---

class SFTDataset:
    """
    Dataset for SFT training from generated completions.

    Loads JSONL file with (prompt, completion) pairs and yields
    tokenized training examples with labels masked for prompt tokens.
    """

    def __init__(
        self,
        data_path: str,
        renderer: "renderers.Renderer",
        tokenizer,
        max_seq_length: int,
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        Args:
            data_path: Path to JSONL file with 'prompt' and 'completion' fields.
            renderer: Renderer for tokenization.
            tokenizer: Tokenizer.
            max_seq_length: Maximum sequence length (truncate longer sequences).
            batch_size: Training batch size.
            shuffle: Whether to shuffle at epoch boundaries.
        """
        self.data_path = data_path
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Load all examples
        self.examples = []
        logger.info(f"Loading SFT data from {data_path}")
        with open(data_path, "r") as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        logger.info(f"Loaded {len(self.examples)} examples")

        self.indices = list(range(len(self.examples)))
        self._rng = np.random.default_rng(42)

    def shuffle_for_epoch(self, seed: int) -> None:
        """Shuffle indices for a new epoch."""
        if self.shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
            logger.info(f"Shuffled SFT dataset for epoch (seed={seed})")

    def __len__(self) -> int:
        return len(self.examples)

    def num_batches(self) -> int:
        return len(self.examples) // self.batch_size

    def get_batch(self, batch_idx: int) -> dict:
        """
        Get a batch of tokenized examples.

        Returns:
            Dict with:
                - input_ids: (batch, seq_len)
                - attention_mask: (batch, seq_len)
                - labels: (batch, seq_len) with -100 for prompt tokens
        """
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(self.indices))
        batch_indices = self.indices[start:end]

        all_input_ids = []
        all_labels = []
        all_attention_masks = []

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for idx in batch_indices:
            example = self.examples[idx]
            prompt = example["prompt"]
            completion = example["completion"]

            # Build the full conversation
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion},
            ]

            # Use build_supervised_example to get tokens and weights
            # weights=1 for tokens we want to train on (completion), weights=0 for prompt
            tokens_tensor, weights_tensor = self.renderer.build_supervised_example(messages)
            input_ids = tokens_tensor.tolist()
            weights = weights_tensor.tolist()

            # Truncate if needed
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]
                weights = weights[:self.max_seq_length]

            # Labels: -100 where weight=0 (prompt), actual token where weight=1 (completion)
            # For cross-entropy, we need shifted labels (predict next token)
            # input_ids[:-1] predicts labels[1:]
            labels = []
            for i in range(len(input_ids)):
                if i == 0 or weights[i] == 0:
                    labels.append(-100)
                else:
                    labels.append(input_ids[i])

            # Attention mask
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in all_input_ids)

        padded_input_ids = []
        padded_labels = []
        padded_attention_masks = []

        for input_ids, labels, attention_mask in zip(all_input_ids, all_labels, all_attention_masks):
            padding_len = max_len - len(input_ids)
            # Right-pad for training
            padded_input_ids.append(input_ids + [pad_token_id] * padding_len)
            padded_labels.append(labels + [-100] * padding_len)  # Ignore padding in loss
            padded_attention_masks.append(attention_mask + [0] * padding_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# --- SFT Training ---

def do_sft_training_step(
    model,
    optimizer,
    batch: dict,
    accumulation_steps: int,
    device: str,
    current_accum_step: int,
) -> dict:
    """
    Single SFT training micro-step with gradient accumulation.

    Args:
        model: Model to train.
        optimizer: Optimizer.
        batch: Dict with input_ids, attention_mask, labels.
        accumulation_steps: Total number of accumulation steps.
        device: Device.
        current_accum_step: Current accumulation step index (0-based).

    Returns:
        Dict with loss, num_tokens, and timing info.
    """
    model.train()

    # === DATA TRANSFER ===
    transfer_start = time.time()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    transfer_time = time.time() - transfer_start

    # === FORWARD PASS ===
    forward_start = time.time()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # (batch, seq_len, vocab)

    # Shift for next-token prediction
    # logits: predict position i from position i-1
    # labels: target at position i
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    vocab_size = shift_logits.size(-1)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Cross-entropy loss (ignores -100 labels)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    forward_time = time.time() - forward_start

    # === BACKWARD PASS ===
    backward_start = time.time()
    scaled_loss = loss / accumulation_steps
    scaled_loss.backward()
    backward_time = time.time() - backward_start

    # Count non-masked tokens
    num_tokens = (shift_labels != -100).sum().item()

    # === OPTIMIZER STEP ===
    optim_time = 0.0
    if current_accum_step == accumulation_steps - 1:
        optim_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        optim_time = time.time() - optim_start

    return {
        "loss": loss.item(),
        "num_tokens": num_tokens,
        "transfer_time": transfer_time,
        "forward_time": forward_time,
        "backward_time": backward_time,
        "optim_time": optim_time,
    }


def run_sft_training(
    cfg: SFTConfig,
    data_path: str,
    log_metrics_fn=None,
    log_path_override: str | None = None,
) -> str:
    """
    Run SFT training phase.

    Args:
        cfg: SFT configuration.
        data_path: Path to JSONL file with training data.
        log_metrics_fn: Optional function for logging metrics.
        log_path_override: Override log path (used by EVC wrapper).

    Returns:
        Path to the saved LoRA adapter checkpoint.
    """
    log_path = log_path_override or cfg.log_path

    # Check for resume
    resume_info = get_last_checkpoint(log_path)
    start_step = 0
    if resume_info:
        logger.info(f"Found SFT checkpoint to resume from: {resume_info['name']}")
        start_step = resume_info.get("loop_state", {}).get("step", 0)
        logger.info(f"Resuming from step {start_step}")

    # Load model with LoRA
    logger.info("Loading model for SFT training...")
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    model_manager = LocalModelManager(
        model_name=cfg.model_name,
        lora_rank=cfg.lora_rank,
        device=cfg.device,
        dtype=dtype,
        max_seq_length=cfg.max_seq_length,
    )
    model_manager.load()
    model = model_manager.model
    tokenizer = model_manager.tokenizer

    # Ensure gradient checkpointing
    model_manager.ensure_gradient_checkpointing()

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create optimizer
    optimizer = create_optimizer(model_manager, cfg.learning_rate)

    # Load checkpoint if resuming
    if resume_info:
        checkpoint_dir = resume_info["checkpoint_dir"]
        logger.info(f"Loading checkpoint from {checkpoint_dir}")
        load_checkpoint(checkpoint_dir, model, optimizer, device=cfg.device)

    # Setup renderer
    renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load dataset
    sft_dataset = SFTDataset(
        data_path=data_path,
        renderer=renderer,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    # Calculate total steps
    batches_per_epoch = sft_dataset.num_batches()
    total_steps = batches_per_epoch * cfg.num_epochs

    logger.info(f"SFT Dataset size: {len(sft_dataset)} examples")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Batches per epoch: {batches_per_epoch}")
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")

    # Training loop
    logger.info(f"Starting SFT training from step {start_step}")
    optimizer.zero_grad()

    global_step = start_step
    accum_loss = 0.0
    accum_tokens = 0
    accum_step = 0

    # Accumulate timing across gradient accumulation steps
    accum_data_load_time = 0.0
    accum_transfer_time = 0.0
    accum_forward_time = 0.0
    accum_backward_time = 0.0
    accum_optim_time = 0.0
    step_start_time = None

    # Cumulative timing for epoch summary
    epoch_data_load_time = 0.0
    epoch_transfer_time = 0.0
    epoch_forward_time = 0.0
    epoch_backward_time = 0.0
    epoch_optim_time = 0.0

    for epoch in range(cfg.num_epochs):
        epoch_start_time = time.time()
        epoch_data_load_time = 0.0
        epoch_transfer_time = 0.0
        epoch_forward_time = 0.0
        epoch_backward_time = 0.0
        epoch_optim_time = 0.0
        epoch_tokens = 0

        sft_dataset.shuffle_for_epoch(seed=epoch)

        for batch_idx in range(batches_per_epoch):
            # Skip already-processed steps on resume
            current_step = epoch * batches_per_epoch + batch_idx
            if current_step < start_step:
                continue

            if step_start_time is None:
                step_start_time = time.time()

            # === DATA LOADING ===
            data_load_start = time.time()
            batch = sft_dataset.get_batch(batch_idx)
            data_load_time = time.time() - data_load_start
            accum_data_load_time += data_load_time

            # Training step
            step_result = do_sft_training_step(
                model=model,
                optimizer=optimizer,
                batch=batch,
                accumulation_steps=cfg.gradient_accumulation_steps,
                device=cfg.device,
                current_accum_step=accum_step,
            )

            accum_loss += step_result["loss"]
            accum_tokens += step_result["num_tokens"]
            accum_transfer_time += step_result["transfer_time"]
            accum_forward_time += step_result["forward_time"]
            accum_backward_time += step_result["backward_time"]
            accum_optim_time += step_result["optim_time"]
            accum_step += 1

            # Log on accumulation boundary
            if accum_step >= cfg.gradient_accumulation_steps:
                global_step += 1
                step_time = time.time() - step_start_time
                avg_loss = accum_loss / cfg.gradient_accumulation_steps

                # Update epoch cumulative stats
                epoch_data_load_time += accum_data_load_time
                epoch_transfer_time += accum_transfer_time
                epoch_forward_time += accum_forward_time
                epoch_backward_time += accum_backward_time
                epoch_optim_time += accum_optim_time
                epoch_tokens += accum_tokens

                tokens_per_sec = accum_tokens / step_time if step_time > 0 else 0

                if log_metrics_fn:
                    log_metrics_fn({
                        "sft/loss": avg_loss,
                        "sft/tokens": accum_tokens,
                        "sft/tokens_per_second": tokens_per_sec,
                        "sft/step": global_step,
                        "sft/epoch": epoch,
                        "sft/step_time": step_time,
                        "sft/data_load_time": accum_data_load_time,
                        "sft/transfer_time": accum_transfer_time,
                        "sft/forward_time": accum_forward_time,
                        "sft/backward_time": accum_backward_time,
                        "sft/optim_time": accum_optim_time,
                    }, step=global_step)

                if global_step % 10 == 0:
                    logger.info(
                        f"SFT Step {global_step}/{total_steps // cfg.gradient_accumulation_steps} | "
                        f"Epoch {epoch} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Tokens: {accum_tokens} ({tokens_per_sec:.1f} tok/s) | "
                        f"Time: {step_time:.2f}s (load={accum_data_load_time:.2f}s, fwd={accum_forward_time:.2f}s, "
                        f"bwd={accum_backward_time:.2f}s, opt={accum_optim_time:.2f}s)"
                    )

                # Reset accumulators
                accum_loss = 0.0
                accum_tokens = 0
                accum_step = 0
                accum_data_load_time = 0.0
                accum_transfer_time = 0.0
                accum_forward_time = 0.0
                accum_backward_time = 0.0
                accum_optim_time = 0.0
                step_start_time = None

                # Checkpoint
                if global_step % cfg.save_every == 0:
                    checkpoint_name = f"step{global_step:05d}"
                    save_checkpoint(
                        log_path=log_path,
                        name=checkpoint_name,
                        model=model,
                        optimizer=optimizer,
                        loop_state={"step": global_step, "epoch": epoch},
                    )
                    logger.info(f"Saved SFT checkpoint: {checkpoint_name}")

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        epoch_tokens_per_sec = epoch_tokens / epoch_time if epoch_time > 0 else 0
        logger.info(f"Epoch {epoch} complete in {epoch_time:.1f}s")
        logger.info(
            f"  Timing breakdown: data_load={epoch_data_load_time:.1f}s ({epoch_data_load_time/epoch_time*100:.1f}%), "
            f"forward={epoch_forward_time:.1f}s ({epoch_forward_time/epoch_time*100:.1f}%), "
            f"backward={epoch_backward_time:.1f}s ({epoch_backward_time/epoch_time*100:.1f}%), "
            f"optim={epoch_optim_time:.1f}s ({epoch_optim_time/epoch_time*100:.1f}%)"
        )
        logger.info(f"  Throughput: {epoch_tokens} tokens, {epoch_tokens_per_sec:.1f} tok/s")

    # Final checkpoint
    final_checkpoint_dir = save_checkpoint(
        log_path=log_path,
        name="final",
        model=model,
        optimizer=optimizer,
        loop_state={"step": global_step, "epoch": cfg.num_epochs},
    )
    logger.info(f"Saved final SFT checkpoint: {final_checkpoint_dir}")

    # Return path to LoRA adapter
    lora_adapter_path = os.path.join(final_checkpoint_dir, "lora_adapter")
    return lora_adapter_path


# --- Main ---

def main(cfg: SFTConfig):
    """Main SFT training function (standalone mode)."""
    print("[6/6] Config parsed, starting main()...", flush=True)

    # Validate configuration
    validate_sft_config(cfg)
    print("  Config validated", flush=True)

    # Setup logging
    print("  Setting up logging and wandb...", flush=True)
    os.makedirs(cfg.log_path, exist_ok=True)
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        config=chz.asdict(cfg),
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.wandb_name,
    )
    save_config(cfg.log_path, chz.asdict(cfg))
    save_command_line(cfg.log_path)

    # Create a log_metrics function that uses the ml_logger
    def log_metrics(metrics: dict, step: int):
        ml_logger.log_metrics(metrics, step=step)

    logger.info(f"Starting SFT training")
    logger.info(f"Log path: {cfg.log_path}")
    logger.info(f"Model: {cfg.model_name}")
    logger.info(f"Dataset: {cfg.dataset_path}")

    # GPU monitoring
    gpu_monitor = None
    if cfg.gpu_monitor_interval > 0:
        gpu_monitor = GPUMonitor(interval_seconds=cfg.gpu_monitor_interval)
        gpu_monitor.start()

    # Log initial GPU stats
    log_gpu_stats_once()

    # Determine SFT data path
    sft_data_path = cfg.sft_data_path
    if sft_data_path is None:
        sft_data_path = os.path.join(cfg.log_path, "sft_data", "results.jsonl")

    # Phase 1: Generate teacher completions if needed
    if not os.path.exists(sft_data_path) or cfg.sft_data_path is None:
        logger.info("=== Phase 1: Generating teacher completions ===")

        # Load base model for generation (no LoRA)
        dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=cfg.model_name,
                max_seq_length=cfg.max_seq_length,
                dtype=dtype,
                load_in_4bit=False,
                device_map={"": cfg.device},
            )
            FastLanguageModel.for_inference(model)
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                torch_dtype=dtype,
                device_map={"": cfg.device},
            )
            tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
            model.eval()

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Get renderer
        renderer_name = get_renderer_name_with_thinking_mode(cfg.model_name, cfg.thinking_mode)
        renderer = renderers.get_renderer(renderer_name, tokenizer)

        # Load dataset
        dataset = load_from_disk(cfg.dataset_path)

        # Generate completions
        generate_teacher_completions(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_path=sft_data_path,
            renderer=renderer,
            teacher_mode=cfg.teacher_mode,
            teacher_prompt=get_teacher_prompt(cfg),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            sampling_batch_size=cfg.sampling_batch_size,
            device=cfg.device,
            log_metrics_fn=log_metrics,
        )

        # Clean up base model
        del model
        torch.cuda.empty_cache()
    else:
        logger.info(f"Using existing SFT data: {sft_data_path}")

    # Phase 2: SFT training
    logger.info("=== Phase 2: SFT Training ===")
    lora_adapter_path = run_sft_training(
        cfg=cfg,
        data_path=sft_data_path,
        log_metrics_fn=log_metrics,
    )

    # Stop GPU monitor
    if gpu_monitor:
        gpu_monitor.stop()

    logger.info(f"SFT training complete!")
    logger.info(f"LoRA adapter saved to: {lora_adapter_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    cfg = chz.entrypoint(SFTConfig, allow_hyphens=True)
    main(cfg)
