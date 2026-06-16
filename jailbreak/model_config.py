"""
Model-specific configuration utilities for ACT.

Provides functions to dynamically detect model capabilities and look up
token IDs, avoiding hardcoded values that break with different models.
"""

from typing import Optional, Dict, List
from transformers import PreTrainedTokenizer


def is_harmony_model(model_path: str) -> bool:
    """Detect OpenAI Harmony format models (gpt-oss family)."""
    return any(p in model_path.lower() for p in ["gpt-oss", "gpt_oss"])


def get_reasoning_system_prompt(model_path: str, level: str = "medium") -> Optional[str]:
    """
    Return system prompt for reasoning level control, or None if not applicable.

    For Harmony models (gpt-oss), reasoning is controlled via system prompt:
    "Reasoning: low/medium/high"
    """
    if is_harmony_model(model_path):
        return f"Reasoning: {level}"
    return None


def extract_final_response_harmony(text: str) -> str:
    """
    Extract user-facing response from harmony-format output.

    Harmony format:
        <|channel|>analysis<|message|>...thinking...<|end|>
        <|start|>assistant<|channel|>final<|message|>...answer...<|return|>

    We extract only the final channel content.
    """
    marker = "<|channel|>final<|message|>"
    if marker in text:
        content = text.split(marker, 1)[1]
        for stop in ["<|return|>", "<|end|>"]:
            if stop in content:
                content = content.split(stop, 1)[0]
        return content.strip()
    # Fallback: return as-is (model may not have produced harmony format)
    return text.strip()


def get_token_id(tokenizer: PreTrainedTokenizer, token_str: str) -> Optional[int]:
    """
    Get the token ID for a token string.

    Only returns the ID if the token encodes to a single token.
    Returns None if the token doesn't exist or encodes to multiple tokens.

    Args:
        tokenizer: The tokenizer to use
        token_str: The token string to look up (e.g., "</think>")

    Returns:
        Token ID if found as single token, None otherwise
    """
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]
    return None


def has_thinking_capability(tokenizer: PreTrainedTokenizer, model_path: str) -> bool:
    """
    Check if a model supports <think>...</think> reasoning.

    Qwen3 and similar models output reasoning in <think> tags before
    the actual response. Most other models don't have this capability.

    Note: Qwen3-4B-Instruct-2507 shares the base Qwen3 tokenizer (so </think>
    is in the vocab) but the Instruct-2507 variant is non-thinking.

    Args:
        tokenizer: The tokenizer to check
        model_path: Model path or name (used for pattern matching)

    Returns:
        True if model supports thinking, False otherwise
    """
    lower_path = model_path.lower()

    # Instruct-2507 variants are non-thinking even though their tokenizer
    # still has the <think>/</think> tokens.
    if "instruct-2507" in lower_path or "instruct_2507" in lower_path:
        return False

    # Check if </think> is a single token (strong indicator)
    if get_token_id(tokenizer, "</think>") is not None:
        return True

    # Check model name patterns for known thinking models
    thinking_patterns = ["qwen3", "qwq", "deepseek-r1"]

    return any(pattern in lower_path for pattern in thinking_patterns)


def get_think_end_token_id(tokenizer: PreTrainedTokenizer) -> Optional[int]:
    """
    Get the </think> token ID for thinking models.

    Returns:
        Token ID for </think> or None if not applicable
    """
    return get_token_id(tokenizer, "</think>")


def get_think_start_token_id(tokenizer: PreTrainedTokenizer) -> Optional[int]:
    """
    Get the <think> token ID for thinking models.

    Returns:
        Token ID for <think> or None if not applicable
    """
    return get_token_id(tokenizer, "<think>")


def has_chat_template(tokenizer: PreTrainedTokenizer) -> bool:
    """
    Check if the tokenizer has a chat template defined.

    Args:
        tokenizer: The tokenizer to check

    Returns:
        True if chat template exists, False otherwise
    """
    return hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None


def get_message_boundaries(tokenizer: PreTrainedTokenizer) -> Dict[str, int]:
    """
    Get message boundary token IDs for this tokenizer.

    Different models use different tokens to mark message boundaries:
    - Qwen: <|im_start|>, <|im_end|>
    - Llama: [INST], [/INST]
    - Phi: <|user|>, <|end|>

    Args:
        tokenizer: The tokenizer to check

    Returns:
        Dict mapping token strings to their IDs (only includes tokens that exist)
    """
    boundaries = {}

    # Common boundary token patterns
    candidates = [
        # Qwen-style
        "<|im_end|>",
        "<|im_start|>",
        # Llama-style
        "[/INST]",
        "[INST]",
        # Phi-style
        "<|end|>",
        "<|user|>",
        "<|assistant|>",
        # ChatML-style
        "<|end_of_turn|>",
        "<|eot_id|>",
    ]

    for token in candidates:
        tid = tokenizer.convert_tokens_to_ids(token)
        # Check if token exists (not mapped to unk)
        if tid != tokenizer.unk_token_id:
            boundaries[token] = tid

    # Always include EOS
    boundaries["eos"] = tokenizer.eos_token_id

    return boundaries


def get_eos_variants(tokenizer: PreTrainedTokenizer) -> List[int]:
    """
    Get all EOS-like token IDs for this tokenizer.

    Some models have multiple tokens that can act as end-of-sequence.

    Args:
        tokenizer: The tokenizer to check

    Returns:
        List of token IDs that can act as EOS
    """
    eos_variants = [tokenizer.eos_token_id]

    # Check for additional EOS-like tokens
    eos_candidates = ["<|im_end|>", "<|end|>", "<|eot_id|>", "</s>"]

    for token in eos_candidates:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid != tokenizer.unk_token_id and tid not in eos_variants:
            eos_variants.append(tid)

    return eos_variants


def get_transformer_blocks(model):
    """
    Get transformer blocks for different model architectures.

    Different model families organize their layers differently:
    - Llama/Qwen/Mistral: model.model.layers
    - GPT-2/GPT-J: model.transformer.h
    - GPT-NeoX: model.gpt_neox.layers

    Args:
        model: The loaded model

    Returns:
        The list/module of transformer blocks

    Raises:
        ValueError: If architecture is not recognized
    """
    # Llama, Qwen, Mistral, etc.
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers

    # GPT-2, GPT-J
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h

    # GPT-NeoX
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers

    # Falcon
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h

    raise ValueError(
        f"Unknown model architecture: {type(model)}. "
        f"Cannot find transformer blocks. "
        f"Supported patterns: model.model.layers, transformer.h, gpt_neox.layers"
    )


def get_attention_module(block, layer_idx: int = 0):
    """
    Get the attention module from a transformer block.

    Different architectures name their attention modules differently:
    - Most: self_attn
    - GPT-2: attn
    - Falcon: self_attention

    Args:
        block: A transformer block
        layer_idx: Layer index (for error messages)

    Returns:
        The attention module

    Raises:
        ValueError: If attention module not found
    """
    if hasattr(block, 'self_attn'):
        return block.self_attn
    if hasattr(block, 'attn'):
        return block.attn
    if hasattr(block, 'self_attention'):
        return block.self_attention
    if hasattr(block, 'attention'):
        return block.attention

    raise ValueError(
        f"Cannot find attention module in block {layer_idx}. "
        f"Checked: self_attn, attn, self_attention, attention"
    )


def get_output_projection(attn, layer_idx: int = 0):
    """
    Get the output projection from an attention module.

    Different architectures name their output projections differently:
    - Most: o_proj
    - GPT-2: c_proj
    - Falcon: dense

    Args:
        attn: An attention module
        layer_idx: Layer index (for error messages)

    Returns:
        The output projection module

    Raises:
        ValueError: If output projection not found
    """
    if hasattr(attn, 'o_proj'):
        return attn.o_proj
    if hasattr(attn, 'c_proj'):
        return attn.c_proj
    if hasattr(attn, 'dense'):
        return attn.dense
    if hasattr(attn, 'out_proj'):
        return attn.out_proj

    raise ValueError(
        f"Cannot find output projection in attention {layer_idx}. "
        f"Checked: o_proj, c_proj, dense, out_proj"
    )


def format_chat_message(
    tokenizer: PreTrainedTokenizer,
    messages: list,
    add_generation_prompt: bool = True
) -> str:
    """
    Format messages using the tokenizer's chat template, with fallback.

    Args:
        tokenizer: The tokenizer
        messages: List of message dicts [{"role": "user", "content": "..."}]
        add_generation_prompt: Whether to add assistant prompt at end

    Returns:
        Formatted prompt string
    """
    if has_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
    else:
        # Simple fallback format
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        prompt = "\n".join(parts)
        if add_generation_prompt:
            prompt += "\nAssistant:"

        return prompt
