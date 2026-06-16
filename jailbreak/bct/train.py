"""BCT (Bias-augmented Consistency Training) — LoRA SFT of (wrapped prompt → teacher's full output).

Usage:
    python -m bct.train_bct \
        --model_path Qwen/Qwen3-8B \
        --model_alias Qwen3-8B \
        --dataset_path data/Qwen3-8B/bct_jailbreak_train.json \
        --benign_dataset_path data/Qwen3-8B/bct_benign_targets.json \
        --track jailbreak \
        --benign_lambda 1.0 \
        --num_epochs 5 --learning_rate 5e-5

For Alpaca track, omit ``--benign_dataset_path`` and pass ``--benign_lambda 0``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="BCT LoRA SFT training")
    # Model + data
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_alias", required=True,
                   help="Used to name checkpoint dir — checkpoints/{alias}/bct/{track}/best")
    p.add_argument("--track", required=True,
                   choices=["jailbreak", "prompt_injection_alpaca_v2"])
    p.add_argument("--dataset_path", required=True,
                   help="Augmented JSON with target_full/target_thinking/target_response")
    p.add_argument("--benign_dataset_path", default=None,
                   help="Augmented benign JSON (same schema). Jailbreak only. Set --benign_lambda 0 for Alpaca.")
    p.add_argument("--benign_lambda", type=float, default=0.0,
                   help="Benign regularization weight. 1.0 for jailbreak, 0 for Alpaca.")
    # Training
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default=None,
                   help="Comma-separated list. If unset, auto-detect per architecture.")
    # Output
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--save_final", action="store_true")
    # Misc
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Locked PEFT imports
    from peft import LoraConfig, get_peft_model
    from ..model_loader import load_model_and_tokenizer
    from ..target_adapter import TargetModelAdapter
    from .dataset import (
        BCTDataset, BenignSFTDataset, collate_bct_batch, train_val_split,
    )
    from .trainer import BCTTrainer

    print("=" * 80)
    print("Bias-augmented Consistency Training (BCT)")
    print("=" * 80)
    print(f"Model:    {args.model_path}")
    print(f"Alias:    {args.model_alias}")
    print(f"Track:    {args.track}")
    print(f"Dataset:  {args.dataset_path}")
    if args.benign_lambda > 0 and args.benign_dataset_path:
        print(f"Benign:   {args.benign_dataset_path} (lambda={args.benign_lambda})")
    else:
        print(f"Benign:   disabled")
    print(f"LR:       {args.learning_rate}")
    print(f"Epochs:   {args.num_epochs}")

    # ---------------- Load base model + tokenizer ----------------
    print("\n[1/5] Loading base model + tokenizer")
    base_model, tokenizer = load_model_and_tokenizer(args.model_path)
    base_model.to(args.device)

    # ---------------- Auto-detect LoRA targets ----------------
    print("\n[2/5] Setting up LoRA")
    if args.lora_target_modules:
        target_modules = args.lora_target_modules.split(",")
    else:
        arch = getattr(base_model.config, "model_type", "")
        if arch in ("phi3", "phi"):
            target_modules = ["qkv_proj"]
            print(f"  Auto-detected Phi architecture: target_modules={target_modules}")
        elif arch == "gemma4":
            target_modules = r".*\.language_model.*\.(q_proj|v_proj)"
            print(f"  Auto-detected Gemma4 architecture: using regex target_modules")
        else:
            target_modules = ["q_proj", "v_proj"]
            print(f"  Default target_modules={target_modules}")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    adapter = TargetModelAdapter(
        tokenizer=tokenizer,
        model_path=args.model_path,
    )

    # ---------------- Datasets ----------------
    print("\n[3/5] Building datasets")
    full_ds = BCTDataset(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        adapter=adapter,
        max_length=args.max_seq_length,
    )
    train_ds, val_ds = train_val_split(full_ds, val_frac=args.val_frac, seed=args.seed)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_bct_batch(b, pad_token_id=pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_bct_batch(b, pad_token_id=pad_id),
    )

    benign_loader = None
    if args.benign_lambda > 0 and args.benign_dataset_path:
        benign_ds = BenignSFTDataset(
            data_path=args.benign_dataset_path,
            tokenizer=tokenizer,
            adapter=adapter,
            max_length=args.max_seq_length,
        )
        benign_loader = DataLoader(
            benign_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda b: collate_bct_batch(b, pad_token_id=pad_id),
        )
        print(f"  benign={len(benign_ds)}")

    # ---------------- Trainer ----------------
    print("\n[4/5] Setting up trainer")
    ckpt_dir = Path(args.checkpoint_dir) / args.model_alias / "bct" / args.track
    trainer = BCTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=ckpt_dir,
        benign_loader=benign_loader,
        benign_lambda=args.benign_lambda,
        device=args.device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_final=args.save_final,
    )

    print("\n[5/5] Training")
    trainer.train()


if __name__ == "__main__":
    main()
