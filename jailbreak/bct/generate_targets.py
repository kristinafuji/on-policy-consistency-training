"""Generate BCT (Bias-augmented Consistency Training) target completions.

For each (clean, wrapped, [system]) pair in an ACT training JSON, generate the
initial base model's greedy completion on the CLEAN prompt. The result is saved
as an augmented JSON with ``target_full``, ``target_thinking``, ``target_response``
fields per pair. These targets are used as SFT supervision for BCT training,
where the model is taught to reproduce ``target_full`` when given the WRAPPED
prompt.

Protocol matches the Consistency Training paper (arXiv:2510.27062 §3.1.1):
target is generated once from the frozen initial weights and reused across all
BCT training epochs. We only ever generate from CLEAN prompts.

Budget rationale: ``max_new_tokens=8192`` + ``thinking_budget=4096`` fit even
the longest reasoning traces on Phi-4/Qwen3 without truncating the response.

Usage:
    python -m bct.generate_targets \\
        --model_path Qwen/Qwen3-8B \\
        --input_dataset data/Qwen3-8B/pair_train_act.json \\
        --output_dataset data/Qwen3-8B/bct_jailbreak_train.json \\
        --use_vllm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Generate BCT target completions")
    p.add_argument("--model_path", required=True)
    p.add_argument("--input_dataset", required=True,
                   help="JSON with (clean, wrapped, [system]) pairs — the ACT training file")
    p.add_argument("--output_dataset", required=True,
                   help="Output JSON — augmented with target_full/target_thinking/target_response")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--thinking_budget", type=int, default=4096)
    p.add_argument("--reasoning_level", default="medium",
                   help="harmony reasoning level (OSS)")
    p.add_argument("--use_vllm", action="store_true")
    p.add_argument("--batch_size", type=int, default=8,
                   help="HF batch size (ignored for vLLM)")
    p.add_argument("--max_model_len", type=int, default=16384,
                   help="vLLM context window (prompt + max_new_tokens)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--hf_token", default=None)
    p.add_argument("--benign_mode", action="store_true",
                   help="Input is a list of strings (benign_prompts.json) instead of "
                        "clean/wrapped pairs. Synthesizes wrapped=clean.")
    return p.parse_args()


def _load_input(path: str, benign_mode: bool) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    if benign_mode:
        # benign_prompts.json is a list of strings; synthesize (clean, wrapped)
        out = []
        for i, x in enumerate(data):
            if isinstance(x, str):
                out.append({"clean": x, "wrapped": x, "category": "benign", "_idx": i})
            elif isinstance(x, dict):
                # {instruction, input} format (benign_500_v2)
                clean = x.get("input", "") or ""
                system = x.get("instruction", "")
                out.append({
                    "clean": clean,
                    "wrapped": clean,
                    "system": system,
                    "category": "benign",
                    "_idx": i,
                })
        return out
    # Standard ACT training data
    return [dict(p, _idx=i) for i, p in enumerate(data)]


def _extract_fields(full_text: str, adapter) -> Dict[str, str]:
    thinking = adapter.extract_thinking(full_text) if hasattr(adapter, "extract_thinking") else ""
    response = adapter.extract_response(full_text)
    return {
        "target_full": full_text,
        "target_thinking": thinking or "",
        "target_response": response or "",
    }


def _closed_thinking(full_text: str, model_type: str) -> bool:
    if model_type == "harmony":
        return "<|channel|>final<|message|>" in full_text or "<|return|>" in full_text
    if model_type == "gemma4_thinking":
        return "<channel|>" in full_text
    if model_type == "thinking":
        return "</think>" in full_text
    return True  # standard models don't use thinking


def _generate_vllm(model_path, tokenizer, adapter, entries, args):
    from vllm import LLM, SamplingParams

    prompts = [
        adapter.build_prompt(e["clean"], task_instruction=e.get("system"))
        for e in entries
    ]

    vllm_kwargs = dict(
        model=model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(f"[generate] Loading vLLM engine: {model_path}")
    llm = LLM(**vllm_kwargs)

    # Respect per-model sampling needs (Phi-4 requires temperature=0.8 — greedy
    # causes degenerate loops that never emit </think>).
    extra = getattr(adapter, "extra_generate_kwargs", {}) or {}
    if extra.get("do_sample", False):
        sp = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=extra.get("temperature", 0.8),
            top_k=extra.get("top_k", 50),
            top_p=extra.get("top_p", 0.95),
            skip_special_tokens=adapter.skip_special_tokens,
        )
    else:
        sp = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            skip_special_tokens=adapter.skip_special_tokens,
        )

    print(f"[generate] Running {len(prompts)} generations (greedy)")
    outputs = llm.generate(prompts, sampling_params=sp)
    results = [o.outputs[0].text for o in outputs]
    del llm
    import torch; torch.cuda.empty_cache()
    return prompts, results


def _generate_hf(model_path, tokenizer, adapter, entries, args):
    import torch
    from tqdm import tqdm
    from ..model_loader import load_model_and_tokenizer

    print(f"[generate] Loading HF model: {model_path}")
    model, _ = load_model_and_tokenizer(model_path)
    model.eval()
    device = next(model.parameters()).device

    prompts = [
        adapter.build_prompt(e["clean"], task_instruction=e.get("system"))
        for e in entries
    ]

    results = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generate"):
        batch = prompts[i:i + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=False).to(device)
        prompt_lengths = [inputs.input_ids[j].shape[0] for j in range(len(batch))]
        logits_procs = adapter.get_logits_processors(prompt_lengths)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=logits_procs or None,
                **adapter.extra_generate_kwargs,
            )
        if hasattr(outputs, "sequences"):
            outputs = outputs.sequences
        for j, out in enumerate(outputs):
            full_text = tokenizer.decode(
                out[prompt_lengths[j]:].tolist(),
                skip_special_tokens=adapter.skip_special_tokens,
            )
            results.append(full_text)

    del model
    import torch; torch.cuda.empty_cache()
    return prompts, results


def main():
    args = parse_args()

    if os.path.exists(args.output_dataset):
        print(f"[skip] {args.output_dataset} already exists")
        return

    from transformers import AutoTokenizer
    from ..target_adapter import TargetModelAdapter

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        token=os.getenv("HF_TOKEN") or args.hf_token,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter = TargetModelAdapter(
        tokenizer=tokenizer,
        model_path=args.model_path,
        reasoning_level=args.reasoning_level,
        thinking_budget=args.thinking_budget,
    )

    entries = _load_input(args.input_dataset, args.benign_mode)
    print(f"[generate] Loaded {len(entries)} entries from {args.input_dataset}")

    if args.use_vllm:
        prompts, results = _generate_vllm(args.model_path, tokenizer, adapter, entries, args)
    else:
        prompts, results = _generate_hf(args.model_path, tokenizer, adapter, entries, args)

    # Augment and filter
    model_type = adapter._type
    augmented = []
    dropped = 0
    for entry, prompt, full_text in zip(entries, prompts, results):
        if not _closed_thinking(full_text, model_type):
            dropped += 1
            continue
        fields = _extract_fields(full_text, adapter)
        augmented.append({
            "clean": entry["clean"],
            "wrapped": entry["wrapped"],
            "system": entry.get("system", ""),
            "category": entry.get("category", "unknown"),
            "clean_prompt_rendered": prompt,
            **fields,
            "target_length_chars": len(full_text),
        })

    print(f"[generate] Kept {len(augmented)} / {len(entries)} entries "
          f"(dropped {dropped} with unclosed thinking)")

    # Ensure parent dir exists
    Path(args.output_dataset).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_dataset, "w") as f:
        json.dump(augmented, f, indent=2)
    print(f"[generate] Wrote {args.output_dataset}")

    # Sanity samples
    print("\n[generate] Sample entries:")
    for i in (0, len(augmented) // 2, len(augmented) - 1):
        if i < 0 or i >= len(augmented):
            continue
        ex = augmented[i]
        print(f"\n  [{i}] clean        : {ex['clean'][:100]!r}")
        print(f"       wrapped[:80]  : {ex['wrapped'][:80]!r}")
        print(f"       target_think  : {ex['target_thinking'][:150]!r}")
        print(f"       target_response: {ex['target_response'][:200]!r}")


if __name__ == "__main__":
    main()
