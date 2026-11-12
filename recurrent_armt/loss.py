from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _token_ce(hidden: torch.Tensor, labels: torch.Tensor, lm_head: nn.Module) -> torch.Tensor:
    logits = lm_head(hidden)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return F.cross_entropy(logits, labels, ignore_index=-100, reduction="none")


def lm_token_losses(hidden: torch.Tensor, lm_head: nn.Module, labels: torch.Tensor,
                    labels_mask: torch.Tensor | None = None, chunk_size: int | None = 4096) -> torch.Tensor:
    """Next-token cross-entropy per position, [B, L-1] (0 where ignored).

    Same convention as the original ARMT: ``labels`` are aligned with the inputs, position t predicts
    labels[t + 1], ``labels_mask[t]`` switches the loss of position t off, -100 is ignored.
    With ``chunk_size`` the logits are computed ``chunk_size`` tokens at a time inside a checkpoint, so
    the [B, L, vocab] logits are never stored (they are recomputed chunk by chunk in the backward pass).
    """
    labels = labels[:, 1:]
    if labels_mask is not None:
        labels = labels.masked_fill(~labels_mask[:, :-1].bool(), -100)
    B, L = labels.shape
    h, y = hidden[:, :-1].reshape(B * L, -1), labels.reshape(B * L)
    if chunk_size is None:
        losses = _token_ce(h, y, lm_head)
    else:
        losses = torch.cat([checkpoint(_token_ce, h[s:s + chunk_size], y[s:s + chunk_size], lm_head, use_reentrant=False)
                            for s in range(0, B * L, chunk_size)])
    return losses.view(B, L)


def lm_loss(token_losses: torch.Tensor, labels: torch.Tensor, labels_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over the positions that have a label (the original divides the summed loss by their count)."""
    labels = labels[:, 1:]
    valid = labels != -100
    if labels_mask is not None:
        valid &= labels_mask[:, :-1].bool()
    return token_losses.sum() / valid.sum().clamp(min=1)
