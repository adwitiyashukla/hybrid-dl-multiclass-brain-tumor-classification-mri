from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_balanced_weights(
    samples_per_class: Sequence[int], beta: float = 0.9999
) -> torch.Tensor:
    counts = np.asarray(samples_per_class, dtype=np.float64)
    if (counts <= 0).any():
        raise ValueError(f"every class needs at least one sample, got {samples_per_class}")

    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(counts)

    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedFocalLoss(nn.Module):

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        if weights is not None:
            self.register_buffer("weights", weights)
        else:
            self.weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = self.weights if self.weights is not None else None

        ce = F.cross_entropy(
            logits, targets,
            weight=weights,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        if self.gamma == 0.0:
            loss = ce
        else:
            log_probs = F.log_softmax(logits, dim=1)
            log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            pt = log_pt.detach().exp()
            loss = ((1.0 - pt) ** self.gamma) * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
