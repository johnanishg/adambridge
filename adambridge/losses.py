"""A causal language-model loss that never holds the full logits tensor.

With a 150k-token vocabulary, the logits of 8 sequences of 512 tokens are 2.3 GiB in
FP32, and training needs them twice (the loss and its gradient). On a 24 GB GPU that is
what stops a 7B model from fitting once the weights are on it. ``causal_lm_loss``
computes the logits for a slice of positions at a time and recomputes each slice in the
backward pass, so the memory for them is one slice's rather than the sequence's. The
value and the gradients are those of the usual loss.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["causal_lm_loss"]


def _chunk_sum(x: torch.Tensor, y: torch.Tensor, lm_head: Callable[[torch.Tensor], torch.Tensor],
               ignore_index: int) -> torch.Tensor:
    return F.cross_entropy(lm_head(x).float(), y, ignore_index=ignore_index, reduction="sum")


def causal_lm_loss(hidden_states: torch.Tensor, lm_head: Callable[[torch.Tensor], torch.Tensor],
                   labels: torch.Tensor, *, chunk_tokens: int = 1024, ignore_index: int = -100) -> torch.Tensor:
    """Next-token cross-entropy, computed ``chunk_tokens`` positions at a time.

    Same value as the Hugging Face causal-LM loss for ``labels``: the token at position
    ``t`` is scored against ``labels[t + 1]``, the logits are upcast to FP32, and the
    mean is over the positions whose label is not ``ignore_index``.

    Args:
        hidden_states: ``[batch, seq, hidden]`` from the model's backbone, for example
            ``model.model(input_ids=ids).last_hidden_state`` for a transformers model.
        lm_head: the output projection, called as ``lm_head(x)`` on ``[n, hidden]``
            (a module such as ``model.lm_head``, or any function).
        labels: ``[batch, seq]`` token ids; usually the input ids themselves.
        chunk_tokens: positions per slice. The peak memory for the logits is about
            ``chunk_tokens * vocab * 10`` bytes.
        ignore_index: label value that is left out of the loss.

    Under ``torch.no_grad()`` the slices are simply computed in turn; with gradients
    enabled each slice runs under ``torch.utils.checkpoint`` and its logits are
    recomputed in the backward pass. Parameters used by ``lm_head`` receive one gradient
    per backward, as with any other checkpointed module.
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [batch, seq, hidden], got shape {tuple(hidden_states.shape)}")
    if tuple(labels.shape) != tuple(hidden_states.shape[:2]):
        raise ValueError(f"labels must be [batch, seq] = {tuple(hidden_states.shape[:2])}, got {tuple(labels.shape)}")
    if not isinstance(chunk_tokens, int) or isinstance(chunk_tokens, bool) or chunk_tokens < 1:
        raise ValueError("chunk_tokens must be a positive int")
    batch, seq, hidden = hidden_states.shape
    flat = hidden_states.reshape(batch * seq, hidden)
    shifted = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].reshape(-1).to(flat.device)
    count = (shifted != ignore_index).sum()
    use_checkpoint = torch.is_grad_enabled() and flat.requires_grad
    total = torch.zeros((), dtype=torch.float32, device=flat.device)
    for lo in range(0, batch * seq, chunk_tokens):
        hi = min(lo + chunk_tokens, batch * seq)
        x, y = flat[lo:hi], shifted[lo:hi]
        if use_checkpoint:
            part = checkpoint(_chunk_sum, x, y, lm_head, ignore_index, use_reentrant=False)
        else:
            part = _chunk_sum(x, y, lm_head, ignore_index)
        total = total + part
    return total / count.clamp(min=1)
