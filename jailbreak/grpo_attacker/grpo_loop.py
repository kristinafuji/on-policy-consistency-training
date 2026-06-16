"""
Custom GRPO training loop for multi-turn adversarial attacker.

Loss formulation:
  - Advantages computed per-behavior group (GRPO-style normalization)
  - PPO-clipped surrogate loss with KL penalty against frozen base model
  - Loss is token-normalized within each turn, summed across turns
  - Optional gradient regularization (GR) to prevent reward hacking
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import GRPOAttackConfig
from .rollout import AttackerTurn, Episode


def compute_log_probs(
    model, input_ids: Tensor, attention_mask: Tensor, completion_mask: Tensor,
) -> Tensor:
    """Forward pass and compute per-token log-probs for completion tokens."""
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    mask = completion_mask[:, 1:]
    return token_lp * mask


def _collate_turns(turns: List[AttackerTurn], pad_id: int, device: torch.device):
    """Collate AttackerTurn objects into padded tensors."""
    full_seqs = [torch.cat([t.prompt_ids, t.completion_ids]) for t in turns]
    comp_lens = [t.completion_ids.shape[0] for t in turns]
    max_len = max(s.shape[0] for s in full_seqs)
    B = len(turns)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
    completion_mask = torch.zeros(B, max_len, dtype=torch.float, device=device)
    old_log_probs_mat = torch.zeros(B, max_len, dtype=torch.float, device=device)

    for i, (seq, comp_len, turn) in enumerate(zip(full_seqs, comp_lens, turns)):
        seq_len = seq.shape[0]
        input_ids[i, :seq_len] = seq.to(device)
        attention_mask[i, :seq_len] = 1
        comp_start = seq_len - comp_len
        completion_mask[i, comp_start:seq_len] = 1.0
        old_log_probs_mat[i, comp_start:seq_len] = turn.old_log_probs.to(device)

    return input_ids, attention_mask, completion_mask, old_log_probs_mat


def _is_transformer_block_param(name: str) -> bool:
    """Check if a parameter belongs to a transformer block (not embed/lm_head).

    Following the paper: only perturb transformer block parameters,
    skip embedding layer and output head for stability.
    """
    skip = ("embed", "lm_head", "wte", "wpe", "word_embedding", "output.weight")
    return not any(s in name for s in skip)


def _compute_turn_loss(
    model, episodes, turn_idx, advantages, config, pad_id, device,
):
    """Compute the PPO + KL loss for a single turn. Returns (loss, turn_tokens, kl_val)
    or None if no active turns."""
    active_pairs = [(i, ep) for i, ep in enumerate(episodes) if turn_idx < ep.num_turns]
    if not active_pairs:
        return None

    ep_indices = [i for i, _ in active_pairs]
    turn_adv = advantages[ep_indices].to(device)
    turns = [ep.turns[turn_idx] for _, ep in active_pairs]

    input_ids, attn_mask, comp_mask, old_lp_mat = _collate_turns(turns, pad_id, device)
    turn_tokens = int(comp_mask.sum().item())
    if turn_tokens == 0:
        del input_ids, attn_mask, comp_mask, old_lp_mat, turn_adv
        return None

    mask = comp_mask[:, 1:]

    with torch.no_grad(), model.disable_adapter():
        ref_lp = compute_log_probs(model, input_ids, attn_mask, comp_mask).detach()

    new_lp = compute_log_probs(model, input_ids, attn_mask, comp_mask)

    old_lp_shifted = old_lp_mat[:, 1:]
    log_ratio = (new_lp - old_lp_shifted.detach()) * mask
    ratio = log_ratio.exp()

    adv = turn_adv.unsqueeze(1)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
    policy_loss = (-torch.min(surr1, surr2) * mask).sum()

    kl = (new_lp - ref_lp) * mask
    kl_loss = kl.sum()

    turn_loss = (policy_loss + config.beta * kl_loss) / (turn_tokens + 1e-8)
    kl_val = (config.beta * kl_loss / (turn_tokens + 1e-8)).item()

    return turn_loss, turn_tokens, kl_val


def grpo_step(
    model, episodes: List[Episode], optimizer: torch.optim.Optimizer,
    config: GRPOAttackConfig,
) -> dict:
    """One GRPO gradient update over a group of N episodes (same behavior)."""
    device = next(model.parameters()).device
    rewards = torch.tensor([ep.reward for ep in episodes], dtype=torch.float32)
    reward_std = rewards.std().item()

    if reward_std < 1e-6:
        return {"loss": 0.0, "reward_mean": rewards.mean().item(), "reward_std": 0.0, "skipped": True}

    advantages = (rewards - rewards.mean()) / (reward_std + 1e-8)
    pad_id = getattr(model.config, "pad_token_id", None) or 0
    model.train()

    total_loss_val = 0.0
    total_tokens = 0
    kl_sum = 0.0
    gr_norm = 0.0
    max_turns = max(ep.num_turns for ep in episodes)

    # --- Standard forward+backward pass (accumulate gradients across turns) ---
    for turn_idx in range(max_turns):
        result = _compute_turn_loss(
            model, episodes, turn_idx, advantages, config, pad_id, device,
        )
        if result is None:
            continue
        turn_loss, turn_tokens_t, kl_val = result
        turn_loss.backward()
        total_loss_val += turn_loss.item()
        total_tokens += turn_tokens_t
        kl_sum += kl_val

    if total_tokens == 0:
        optimizer.zero_grad()
        return {"loss": 0.0, "reward_mean": rewards.mean().item(), "skipped": True}

    # --- Gradient Regularization (finite-difference, per Ackermann et al. 2026) ---
    if config.gr_gamma > 0:
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]

        # 1. Save original gradients g = ∇L(ϕ)
        saved_grads = {n: p.grad.clone() for n, p in trainable}

        # 2. Perturb parameters: ϕ' = ϕ + ε * ∇L(ϕ)  (transformer blocks only)
        for n, p in trainable:
            if _is_transformer_block_param(n):
                p.data.add_(config.gr_epsilon * saved_grads[n])

        # 3. Zero grads and recompute ∇L(ϕ') at perturbed parameters
        optimizer.zero_grad()
        for turn_idx in range(max_turns):
            result = _compute_turn_loss(
                model, episodes, turn_idx, advantages, config, pad_id, device,
            )
            if result is None:
                continue
            turn_loss, _, _ = result
            turn_loss.backward()

        # 4. Compute GR update: g_final = g + (γ/2) * (∇L(ϕ') - g) / ε
        #    and restore parameters: ϕ = ϕ' - ε * g = ϕ
        for n, p in trainable:
            if p.grad is not None and _is_transformer_block_param(n):
                gr_term = (p.grad - saved_grads[n]) / config.gr_epsilon
                # Clip GR gradient for stability (paper recommendation)
                gr_term.clamp_(-1.0, 1.0)
                gr_norm += gr_term.norm().item() ** 2
                p.grad.copy_(saved_grads[n] + (config.gr_gamma / 2) * gr_term)
            elif p.grad is not None:
                # Non-transformer params: use original gradient (no perturbation)
                p.grad.copy_(saved_grads[n])

            # Restore parameters
            if _is_transformer_block_param(n):
                p.data.sub_(config.gr_epsilon * saved_grads[n])

        gr_norm = gr_norm ** 0.5
        del saved_grads

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    metrics = {
        "loss": total_loss_val, "grad_norm": grad_norm.item(),
        "reward_mean": rewards.mean().item(), "reward_std": reward_std,
        "kl": kl_sum, "total_tokens": total_tokens, "skipped": False,
    }
    if config.gr_gamma > 0:
        metrics["gr_norm"] = gr_norm
    return metrics
