"""
Merge Tinker-format LoRA adapter weights into a HuggingFace model and save the merged model.

Standalone reimplementation of tinker-cookbook's merge script with a fix for the gate/up
interleaving bug in gpt-oss (MoE) models, plus support for Mxfp4 dequantization.

Usage:
    uv run python merge_tinker_lora.py \
        --adapter-path ./logs/sweep1-opd-gptoss20b-v2/downloaded_checkpoints/001380/sampler_weights \
        --output-path ./logs/sweep1-opd-gptoss20b-v2/merged_models/001380

    uv run python merge_tinker_lora.py \
        --hf-model openai/gpt-oss-20b \
        --adapter-path <path> \
        --output-path <path> \
        --verify
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config


def log(s: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {s}")


def load_model(model_path: str, cache_dir: str | None = None):
    # Try loading with Mxfp4 dequantization first (for quantized models like GPT-OSS),
    # fall back to standard loading for non-quantized models (Llama, Qwen).
    try:
        log(f"Loading model from {model_path} with Mxfp4 dequantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            quantization_config=Mxfp4Config(dequantize=True),
            cache_dir=cache_dir,
        )
        # Remove quantization_config so the saved model loads cleanly as dense bf16
        if hasattr(model.config, "quantization_config"):
            del model.config.quantization_config
    except Exception as e:
        log(f"Mxfp4 loading failed ({e}), loading as standard bf16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        )
    return model


def load_adapter_weights(adapter_path: str):
    weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"), device="cpu")
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        config = json.load(f)
    return weights, config


def apply_merged_weight(target: torch.Tensor, merged_lora: torch.Tensor):
    assert target.shape == merged_lora.shape, (target.shape, merged_lora.shape)
    new_data = target.float() + merged_lora.float()
    target.copy_(new_data.to(target.dtype))


def merge_adapter_weights(
    base_model: torch.nn.Module, adapter_weights: dict[str, torch.Tensor], config: dict
):
    scaling = config["lora_alpha"] / config["r"]
    log(f"LoRA rank={config['r']}, alpha={config['lora_alpha']}, scaling={scaling}")

    adapter_weight_names = [n.replace(".lora_A", "") for n in adapter_weights if ".lora_A" in n]
    log(f"Found {len(adapter_weight_names)} LoRA weight pairs to merge")

    model_state_dict = base_model.state_dict()
    is_gpt_oss = "GptOss" in str(type(base_model))
    log(f"Model type: {type(base_model).__name__}, is_gpt_oss={is_gpt_oss}")

    for i, n in enumerate(adapter_weight_names):
        target_key = n.replace("base_model.model.", "").replace("model.unembed_tokens", "lm_head")
        lora_A = adapter_weights[n.replace(".weight", ".lora_A.weight")].float()
        lora_B = adapter_weights[n.replace(".weight", ".lora_B.weight")].float() * scaling

        if ".experts" not in n:
            # Non-expert (attention, lm_head) — 2D weights
            if is_gpt_oss:
                target_key = target_key.replace(".attn", ".self_attn")
            assert target_key in model_state_dict, (n, target_key)
            # (lora_rank, in_dim), (out_dim, lora_rank) -> (out_dim, in_dim)
            merged_lora = F.linear(lora_A.T, lora_B).T
            log(f"  [{i+1}/{len(adapter_weight_names)}] {target_key}: {merged_lora.shape}")
            apply_merged_weight(model_state_dict[target_key], merged_lora)
        else:
            # Expert weights — 3D, potentially shared across experts
            assert len(lora_A.shape) == 3 and len(lora_B.shape) == 3, (lora_A.shape, lora_B.shape)
            if lora_A.shape[0] == 1:
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1)
            elif lora_B.shape[0] == 1:
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1)
            # (num_experts, lora_rank, in_dim), (num_experts, out_dim, lora_rank)
            # -> (num_experts, in_dim, out_dim)
            merged_lora = torch.bmm(lora_A.transpose(-1, -2), lora_B.transpose(-1, -2))

            # Rename w1/w2/w3 to gate/down/up
            target_key = target_key.replace(".w1.weight", ".gate_proj.weight")
            target_key = target_key.replace(".w3.weight", ".up_proj.weight")
            target_key = target_key.replace(".w2.weight", ".down_proj.weight")

            if not is_gpt_oss:
                # Separate weight per expert
                merged_lora_t = merged_lora.transpose(-1, -2)
                for exp_idx in range(merged_lora_t.shape[0]):
                    target_key_exp = target_key.replace(".experts", f".experts.{exp_idx}")
                    assert target_key_exp in model_state_dict, (n, target_key_exp)
                    apply_merged_weight(model_state_dict[target_key_exp], merged_lora_t[exp_idx])
            else:
                # gpt-oss: fused gate_up_proj with interleaved gate/up
                # BUG FIX: determine idx BEFORE renaming target_key
                if target_key.endswith(".gate_proj.weight"):
                    idx = 0  # gate = even indices
                    target_key = target_key.replace(".gate_proj.weight", ".gate_up_proj")
                elif target_key.endswith(".up_proj.weight"):
                    idx = 1  # up = odd indices
                    target_key = target_key.replace(".up_proj.weight", ".gate_up_proj")
                else:
                    # down_proj — no interleaving
                    idx = None
                    target_key = target_key.replace(".down_proj.weight", ".down_proj")

                if idx is not None:
                    assert target_key in model_state_dict, (n, target_key)
                    target = model_state_dict[target_key][:, :, idx::2]
                    log(f"  [{i+1}/{len(adapter_weight_names)}] {target_key}[:,:,{idx}::2]: {merged_lora.shape}")
                else:
                    assert target_key in model_state_dict, (n, target_key)
                    target = model_state_dict[target_key]
                    log(f"  [{i+1}/{len(adapter_weight_names)}] {target_key}: {merged_lora.shape}")
                apply_merged_weight(target, merged_lora)


def verify_model(model: torch.nn.Module):
    log("Running post-merge verification...")
    total_params = 0
    nan_params = []
    inf_params = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)

    if nan_params:
        log(f"ERROR: NaN found in {len(nan_params)} parameters: {nan_params[:5]}")
        raise ValueError(f"NaN found in {len(nan_params)} parameters")
    if inf_params:
        log(f"ERROR: Inf found in {len(inf_params)} parameters: {inf_params[:5]}")
        raise ValueError(f"Inf found in {len(inf_params)} parameters")

    log(f"Verification passed: {total_params:,} parameters, no NaN/Inf")


def save_merged_model(model: torch.nn.Module, output_path: str, hf_model: str, cache_dir: str | None):
    """Save merged model manually to bypass gpt-oss's broken reverse_op in save_pretrained."""
    state_dict = model.state_dict()

    # Split into shards of ~5GB each
    max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB
    shards = []
    current_shard = {}
    current_size = 0

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size
    if current_shard:
        shards.append(current_shard)

    # Save shards and build index
    weight_map = {}
    total_size = 0
    for i, shard in enumerate(shards):
        if len(shards) == 1:
            filename = "model.safetensors"
        else:
            filename = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"
        filepath = os.path.join(output_path, filename)
        log(f"  Saving shard {i+1}/{len(shards)}: {filename} ({len(shard)} tensors)")
        save_file(shard, filepath)
        for key, tensor in shard.items():
            weight_map[key] = filename
            total_size += tensor.numel() * tensor.element_size()

    if len(shards) > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with open(os.path.join(output_path, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    # Save config
    config_dict = model.config.to_dict()
    config_dict.pop("quantization_config", None)
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy generation_config if present in source model
    resolved_path = model.config._name_or_path
    if os.path.isdir(resolved_path):
        gen_config_path = os.path.join(resolved_path, "generation_config.json")
        if os.path.isfile(gen_config_path):
            shutil.copy2(gen_config_path, os.path.join(output_path, "generation_config.json"))

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model, cache_dir=cache_dir)
    tokenizer.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(description="Merge Tinker LoRA adapter into HF model")
    parser.add_argument(
        "--hf-model", type=str, default="openai/gpt-oss-20b",
        help="HuggingFace model name or path (default: openai/gpt-oss-20b)",
    )
    parser.add_argument(
        "--adapter-path", type=str, required=True,
        help="Path to directory with adapter_model.safetensors and adapter_config.json",
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Path to save the merged model",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run post-merge verification (NaN/Inf checks)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory for model downloads",
    )
    args = parser.parse_args()

    assert os.path.isdir(args.adapter_path), f"Adapter path not found: {args.adapter_path}"
    assert os.path.isfile(os.path.join(args.adapter_path, "adapter_model.safetensors")), \
        f"adapter_model.safetensors not found in {args.adapter_path}"
    os.makedirs(args.output_path, exist_ok=True)

    log("Loading HF model")
    model = load_model(args.hf_model, cache_dir=args.cache_dir)

    log("Loading adapter weights")
    adapter_weights, adapter_config = load_adapter_weights(args.adapter_path)

    log("Merging adapter weights")
    merge_adapter_weights(model, adapter_weights, adapter_config)

    if args.verify:
        verify_model(model)

    log(f"Saving merged model to {args.output_path}")
    save_merged_model(model, args.output_path, args.hf_model, args.cache_dir)
    log("Done")


if __name__ == "__main__":
    main()
