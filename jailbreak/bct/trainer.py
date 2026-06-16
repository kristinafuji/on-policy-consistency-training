"""BCT (Bias-augmented Consistency Training) trainer — LoRA SFT with optional benign regularization.

Structure mirrors ``training/trainer.py::ACTTrainer`` but swaps activation-MSE
for standard next-token CE on the wrapped prompt + target-response sequence.
Validation-best checkpointing; runs up to ``num_epochs`` epochs, keeps the
best LoRA by val loss.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import masked_ce_loss


class BCTTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        warmup_steps: int,
        checkpoint_dir: Path,
        benign_loader: Optional[DataLoader] = None,
        benign_lambda: float = 0.0,
        device: str = "cuda",
        gradient_accumulation_steps: int = 1,
        log_every: int = 10,
        save_final: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.benign_loader = benign_loader
        self.benign_lambda = benign_lambda if benign_loader is not None else 0.0
        self.num_epochs = num_epochs
        self.lr = learning_rate
        self.warmup_steps = warmup_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_every = log_every
        self.save_final = save_final

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.lr,
        )

        # Linear warmup then constant
        def lr_lambda(step: int):
            if step < self.warmup_steps:
                return float(step) / float(max(1, self.warmup_steps))
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.best_val_loss = float("inf")
        self.global_step = 0
        self._benign_iter = None

    def _sample_benign(self):
        """Round-robin sampler that loops the benign loader."""
        if self.benign_loader is None:
            return None
        try:
            batch = next(self._benign_iter)  # type: ignore
        except (StopIteration, TypeError):
            self._benign_iter = iter(self.benign_loader)
            batch = next(self._benign_iter)
        return batch

    def _compute_batch_loss(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        return masked_ce_loss(outputs.logits, batch["labels"])

    def _validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._compute_batch_loss(batch)
                total_loss += float(loss.item())
                n += 1
        self.model.train()
        return total_loss / max(1, n)

    def _save_best(self, tag: str = "best"):
        out_dir = self.checkpoint_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        print(f"  [save] {out_dir}")

    def train(self):
        self.model.train()
        start_time = time.time()

        for epoch in range(self.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")
            running_loss = 0.0
            running_benign = 0.0
            accum_count = 0

            self.optimizer.zero_grad()

            pbar = tqdm(self.train_loader, desc=f"Train ep{epoch+1}")
            for step_idx, batch in enumerate(pbar):
                main_loss = self._compute_batch_loss(batch)
                loss = main_loss

                if self.benign_lambda > 0:
                    benign_batch = self._sample_benign()
                    if benign_batch is not None:
                        benign_loss = self._compute_batch_loss(benign_batch)
                        loss = loss + self.benign_lambda * benign_loss
                        running_benign += float(benign_loss.item())

                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                running_loss += float(main_loss.item())
                accum_count += 1

                if (step_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                if (step_idx + 1) % max(1, self.log_every) == 0:
                    avg_main = running_loss / max(1, accum_count)
                    avg_ben = running_benign / max(1, accum_count)
                    pbar.set_postfix(
                        main=f"{avg_main:.4f}",
                        benign=f"{avg_ben:.4f}" if self.benign_lambda > 0 else "-",
                    )

            # Flush any remaining gradients at epoch end
            if (step_idx + 1) % self.gradient_accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            train_loss = running_loss / max(1, accum_count)
            val_loss = self._validate()
            print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print(f"  [best] val_loss improved to {val_loss:.4f}")
                self._save_best("best")

        if self.save_final:
            self._save_best("final")

        elapsed = time.time() - start_time
        print(f"\n[train] Done. Best val_loss={self.best_val_loss:.4f}  "
              f"elapsed={elapsed/60:.1f} min")
