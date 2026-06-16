"""Utilities for renderer selection with thinking mode support."""

import logging
from typing import Literal

import tinker
from tinker import types as tinker_types
from tinker_cookbook.model_info import get_model_attributes, get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import Tokenizer

logger = logging.getLogger(__name__)


def _normalize_model_name(model_name: str) -> str:
    """
    Normalize model name by stripping provider prefixes like 'unsloth/'
    or resolving local paths to their canonical HuggingFace model name.

    Maps provider-specific names to their canonical form for renderer lookup.
    E.g., 'unsloth/Qwen3-4B-Instruct-2507' -> 'Qwen/Qwen3-4B-Instruct-2507'
    E.g., './logs/run/merged_models/001400' -> 'openai/gpt-oss-20b' (from config.json)
    """
    import json
    import os

    # If it looks like a local path, try to read _name_or_path from config.json
    if os.path.isdir(model_name):
        config_path = os.path.join(model_name, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
            original_name = config.get("_name_or_path")
            if original_name and "/" in original_name:
                logger.info(f"Resolved local model path to: {original_name}")
                return original_name

    # Strip unsloth prefix and map to canonical organization
    if model_name.startswith("unsloth/"):
        base_name = model_name[len("unsloth/"):]
        # Detect organization from model name
        if base_name.startswith("Qwen"):
            return f"Qwen/{base_name}"
        elif base_name.startswith("Llama"):
            return f"meta-llama/{base_name}"
        elif base_name.startswith("Mistral"):
            return f"mistralai/{base_name}"
        # Return as-is if can't determine org
        return base_name
    return model_name


def get_renderer_name_with_thinking_mode(
    model_name: str,
    thinking_mode: Literal["enable", "disable"] | None,
    reasoning_effort: Literal["low", "medium", "high"] | None = None,
) -> str:
    """
    Get the appropriate renderer name for a model, considering thinking mode.

    Args:
        model_name: The model name (e.g., "Qwen/Qwen3-8B", "meta-llama/Llama-3.1-8B-Instruct")
        thinking_mode:
            - None: Use default renderer for the model
            - "enable": Force thinking enabled (valid for Qwen3 hybrid models and OpenAI models)
            - "disable": Force thinking disabled (valid for Qwen3 hybrid models and OpenAI models)
        reasoning_effort: For OpenAI models only, controls reasoning effort level.
            - None: Use "medium" when thinking_mode="enable"
            - "low", "medium", "high": Explicit reasoning effort level

    Returns:
        The renderer name to use.

    Raises:
        ValueError: If thinking mode is specified for a model that doesn't support it.
    """
    # Normalize model name for renderer lookup
    normalized_name = _normalize_model_name(model_name)
    default_renderer = get_recommended_renderer_name(normalized_name)

    # Get model attributes to determine if thinking mode applies
    try:
        attributes = get_model_attributes(normalized_name)
    except (ValueError, KeyError) as e:
        # If we can't get attributes and no thinking mode specified, use default
        if thinking_mode is None:
            return default_renderer
        raise ValueError(
            f"Cannot determine thinking support for model '{model_name}'. "
            f"Please use a known model or omit the --thinking-mode flag. Error: {e}"
        )

    # Handle OpenAI models
    if attributes.organization == "openai":
        if thinking_mode is None or thinking_mode == "disable":
            return "gpt_oss_no_sysprompt"
        # thinking_mode == "enable"
        effort = reasoning_effort or "medium"  # default to medium
        return f"gpt_oss_{effort}_reasoning"

    # If no thinking mode specified, use the default
    if thinking_mode is None:
        return default_renderer

    # Check if this is a Qwen3 model
    if attributes.organization == "Qwen" and attributes.version_str == "3":
        # Qwen3 Instruct-2507 models don't support thinking
        if "-Instruct" in normalized_name:
            if thinking_mode == "enable":
                raise ValueError(
                    f"Model '{model_name}' is an Instruct-2507 model and does not support thinking mode. "
                    "Remove --thinking-mode enable or use a hybrid model like Qwen/Qwen3-8B."
                )
            # --thinking-mode disable is a no-op for Instruct models
            logger.warning(
                f"--thinking-mode disable has no effect for Instruct-2507 model '{model_name}'. "
                "These models do not have thinking mode."
            )
            return default_renderer

        # Qwen3 hybrid models support thinking mode control
        if thinking_mode == "enable":
            return "qwen3"
        else:  # thinking_mode == "disable"
            return "qwen3_disable_thinking"

    # For non-Qwen3/non-OpenAI models (e.g., Llama), thinking mode doesn't apply
    raise ValueError(
        f"Model '{model_name}' does not support thinking mode control. "
        "The --thinking-mode flag only applies to Qwen3 hybrid models (e.g., Qwen/Qwen3-8B) "
        "or OpenAI models (e.g., openai/gpt-oss-20b)."
    )


def should_mask_thinking(renderer_name: str) -> bool:
    """
    Return True if the renderer produces thinking tokens that should be masked.

    Renderers that produce thinking content:
    - 'qwen3': Uses <think>...</think> XML tags
    - 'gpt_oss_*_reasoning': Uses channel-based format (<|channel|>analysis<|message|>...<|end|>)

    Other renderers like 'qwen3_disable_thinking' force empty thinking blocks,
    'qwen3_instruct' doesn't use thinking at all, and 'gpt_oss_no_sysprompt'
    has no reasoning.
    """
    # Qwen3 with thinking enabled
    if renderer_name == "qwen3":
        return True
    # GPT-OSS with reasoning enabled
    if renderer_name in ("gpt_oss_low_reasoning", "gpt_oss_medium_reasoning", "gpt_oss_high_reasoning"):
        return True
    return False


def _get_thinking_markers_for_renderer(
    renderer_name: str,
    tokenizer: Tokenizer
) -> tuple[list[int], list[int]] | None:
    """
    Get start/end token patterns for thinking regions based on renderer type.

    Args:
        renderer_name: The name of the renderer (e.g., "qwen3", "gpt_oss_medium_reasoning")
        tokenizer: Tokenizer to encode the marker strings

    Returns:
        Tuple of (start_tokens, end_tokens) for the thinking region markers,
        or None if the renderer doesn't have thinking tokens.

    For Qwen3: <think>...</think>
    For GPT-OSS reasoning: <|channel|>analysis<|message|>...<|end|>
    """
    if renderer_name == "qwen3":
        return (
            tokenizer.encode("<think>", add_special_tokens=False),
            tokenizer.encode("</think>", add_special_tokens=False),
        )
    if renderer_name in ("gpt_oss_low_reasoning", "gpt_oss_medium_reasoning", "gpt_oss_high_reasoning"):
        # GPT-OSS uses channel-based format for reasoning
        # The analysis channel starts with <|channel|>analysis<|message|> and ends with <|end|>
        return (
            tokenizer.encode("<|channel|>analysis<|message|>", add_special_tokens=False),
            tokenizer.encode("<|end|>", add_special_tokens=False),
        )
    return None


def mask_thinking_tokens(
    data_D: list[tinker.Datum],
    tokenizer: Tokenizer,
    renderer_name: str = "qwen3",
) -> None:
    """
    Modify datum masks in-place to exclude thinking tokens from training.

    Sets mask=0 for all tokens in thinking regions (inclusive of markers).
    This ensures the model is only trained on the actual response, not the
    chain-of-thought reasoning.

    Supported formats:
    - Qwen3: <think>...</think> XML tags
    - GPT-OSS: <|channel|>analysis<|message|>...<|end|> channel format

    Args:
        data_D: List of training datums to modify in-place.
        tokenizer: Tokenizer to use for encoding the thinking tags.
        renderer_name: The renderer name to determine which markers to use.
    """
    # Get token IDs for thinking tags based on renderer
    markers = _get_thinking_markers_for_renderer(renderer_name, tokenizer)
    if markers is None:
        logger.warning(f"Renderer '{renderer_name}' does not have thinking tokens to mask")
        return

    think_start_tokens, think_end_tokens = markers

    for datum in data_D:
        # Get the token sequence from the datum
        # ModelInput stores tokens in chunks; use to_ints() to extract the flat token list
        tokens = datum.model_input.to_ints()
        if not tokens:
            continue

        # Get current mask
        mask_tensor = datum.loss_fn_inputs["mask"].to_torch()
        mask = mask_tensor.clone()

        # Find all thinking regions and set their mask to 0
        # We need to find start marker followed by end marker and mask everything in between
        i = 0
        while i < len(tokens):
            # Check if we found the start of a thinking block
            if _matches_at_position(tokens, i, think_start_tokens):
                start_pos = i
                # Find the corresponding end marker
                j = i + len(think_start_tokens)
                while j < len(tokens):
                    if _matches_at_position(tokens, j, think_end_tokens):
                        end_pos = j + len(think_end_tokens)
                        # Mask out tokens from start_pos to end_pos (exclusive)
                        # Note: mask is offset by 1 from tokens (target prediction)
                        # mask[t] corresponds to predicting token[t+1]
                        # So to mask predicting tokens in [start_pos, end_pos),
                        # we need to set mask[start_pos-1:end_pos-1] = 0
                        # But we also need to be careful about the offset
                        mask_start = max(0, start_pos - 1)
                        mask_end = min(len(mask), end_pos - 1)
                        if mask_start < mask_end:
                            mask[mask_start:mask_end] = 0.0
                        i = end_pos
                        break
                    j += 1
                else:
                    # No closing tag found (model likely hit token limit mid-thinking)
                    # Mask everything from start marker to end of sequence
                    mask_start = max(0, start_pos - 1)
                    mask_end = len(mask)
                    if mask_start < mask_end:
                        mask[mask_start:mask_end] = 0.0
                    tokens_masked = mask_end - mask_start
                    logger.warning(
                        f"Unclosed thinking block at position {start_pos}, "
                        f"masking {tokens_masked} tokens to end of sequence"
                    )
                    break  # No point searching further
            else:
                i += 1

        # Update the mask in the datum
        datum.loss_fn_inputs["mask"] = tinker_types.TensorData.from_torch(mask)


def _matches_at_position(tokens: list[int], pos: int, pattern: list[int]) -> bool:
    """Check if pattern matches tokens starting at position pos."""
    if pos + len(pattern) > len(tokens):
        return False
    return tokens[pos : pos + len(pattern)] == pattern


def get_output_only_mask(
    tokens: list[int],
    mask: "torch.Tensor",
    tokenizer: Tokenizer,
    renderer_name: str = "qwen3",
) -> "torch.Tensor":
    """
    Compute a mask that excludes thinking tokens from a given mask.

    This is similar to mask_thinking_tokens but:
    - Takes tokens and mask directly (not a datum)
    - Returns a new mask instead of modifying in-place
    - Used for computing metrics on output tokens only

    Supported formats:
    - Qwen3: <think>...</think> XML tags
    - GPT-OSS: <|channel|>analysis<|message|>...<|end|> channel format

    Args:
        tokens: List of token IDs
        mask: Original mask tensor (1 for response tokens, 0 for prompt)
        tokenizer: Tokenizer to encode thinking tags
        renderer_name: The renderer name to determine which markers to use.

    Returns:
        New mask tensor with thinking regions set to 0
    """
    import torch

    if not tokens:
        return mask.clone()

    # Get token IDs for thinking tags based on renderer
    markers = _get_thinking_markers_for_renderer(renderer_name, tokenizer)
    if markers is None:
        # No thinking markers for this renderer, return original mask
        return mask.clone()

    think_start_tokens, think_end_tokens = markers

    output_mask = mask.clone()

    # Find all thinking regions and set their mask to 0
    i = 0
    while i < len(tokens):
        if _matches_at_position(tokens, i, think_start_tokens):
            start_pos = i
            j = i + len(think_start_tokens)
            while j < len(tokens):
                if _matches_at_position(tokens, j, think_end_tokens):
                    end_pos = j + len(think_end_tokens)
                    # Mask out tokens from start_pos to end_pos
                    mask_start = max(0, start_pos - 1)
                    mask_end = min(len(output_mask), end_pos - 1)
                    if mask_start < mask_end:
                        output_mask[mask_start:mask_end] = 0.0
                    i = end_pos
                    break
                j += 1
            else:
                # No closing tag found - mask to end
                mask_start = max(0, start_pos - 1)
                mask_end = len(output_mask)
                if mask_start < mask_end:
                    output_mask[mask_start:mask_end] = 0.0
                break
        else:
            i += 1

    return output_mask
