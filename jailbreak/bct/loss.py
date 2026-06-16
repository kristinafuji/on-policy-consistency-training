"""BCT loss: masked next-token cross-entropy."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted cross-entropy with -100 ignore index.

    Args:
        logits: [B, T, V]
        labels: [B, T]  — -100 marks ignored positions
    Returns:
        scalar loss, averaged over non-ignored tokens.
    """
    # Predict labels[t+1] from logits[t]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
