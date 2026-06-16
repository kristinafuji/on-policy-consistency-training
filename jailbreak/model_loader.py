"""
Standalone model and tokenizer loading utilities for ACT training.

No dependencies on external jailbreaking_defense code.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


def load_model_and_tokenizer(
    model_name_or_path: str,
    device_map: str = "auto",
    attn_implementation: str = None,
    use_4bit: bool = False,
):
    """
    Load a causal language model and its tokenizer.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        device_map: Device map for model loading (default: "auto")
        attn_implementation: Attention implementation ("eager", "sdpa", "flash_attention_2").
                            Use "eager" if you need output_attentions=True.
        use_4bit: Load model in 4-bit (fp4) quantization via bitsandbytes.
                  Required for large models (e.g. gpt-oss-20b) on 46GB GPUs.
                  Train and eval quantization must match for activation consistency.

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load config
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        output_hidden_states=True,
        return_dict_in_generate=True,
        trust_remote_code=True,
    )

    # Workaround for Qwen2 models
    # Setting config=None for Qwen2 allows default config to be used
    if isinstance(config, Qwen2Config):
        config = None

    # Build kwargs for model loading
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "token": os.getenv("HF_TOKEN"),
        "config": config,
        "trust_remote_code": True,
    }

    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Add attention implementation if specified
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs
    ).eval()

    # Freeze all parameters by default
    model.requires_grad_(False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        token=os.getenv("HF_TOKEN"),
        padding_side="left",
        trust_remote_code=True,
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
