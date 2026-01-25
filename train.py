"""Training utilities for D4RT.

This module defines the composite loss used for training and provides
helpers to build the optimizer and learning-rate schedule described in
D4RT_paper.pdf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import math

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from loss_head import LossHead


@dataclass(frozen=True)
class LossWeights:
    """Loss weights for the composite objective."""

    lambda_3d: float = 1.0
    lambda_2d: float = 0.1
    lambda_vis: float = 0.1
    lambda_disp: float = 0.1
    lambda_conf: float = 0.2
    lambda_normal: float = 0.5


class CompositeLoss(Module):
    """Composite loss as described in D4RT_paper.pdf."""

    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        losses: Mapping[str, Tensor],
        confidence: Tensor,
        query_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the composite loss.

        Args:
            losses: Mapping containing per-query task losses with keys:
                "L3D", "L2D", "Lvis", "Ldisp", "Lconf", "Lnormal".
            confidence: Predicted confidence score c for each query.
            query_mask: Boolean mask indicating valid queries.
        """
        weights = self.weights
        conf = confidence.clamp_min(1e-6)

        term_3d = conf * weights.lambda_3d * losses["L3D"]
        term_2d = weights.lambda_2d * losses["L2D"]
        term_vis = weights.lambda_vis * losses["Lvis"]
        term_disp = weights.lambda_disp * losses["Ldisp"]
        term_conf = weights.lambda_conf * losses["Lconf"]
        term_normal = weights.lambda_normal * losses["Lnormal"]
        term_confidence = -weights.lambda_conf * torch.log(conf)

        total = (
            term_3d
            + term_confidence
            + term_2d
            + term_vis
            + term_disp
            + term_conf
            + term_normal
        )
        if query_mask is None:
            return total.mean()
        mask = query_mask.unsqueeze(-1).to(total.dtype)
        masked_total = total * mask
        denom = mask.sum().clamp_min(1.0)
        return masked_total.sum() / denom


def build_optimizer(parameters: Iterable[torch.nn.Parameter], lr: float = 1e-4) -> Optimizer:
    """Build the AdamW optimizer."""

    return AdamW(parameters, lr=lr, weight_decay=0.03)


def build_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 2500,
    total_steps: int = 100000,
    peak_lr: float = 1e-4,
    final_lr: float = 1e-6,
) -> LambdaLR:
    """Create a warmup + cosine annealing schedule.

    The learning rate warms up linearly to ``peak_lr`` for ``warmup_steps``
    and then follows a cosine decay to ``final_lr`` for the remaining steps.
    """

    if total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than warmup_steps")

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = final_lr / peak_lr
        return min_factor + (1.0 - min_factor) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_step(
    model: Module,
    batch: Dict[str, Tensor],
    loss_fn: CompositeLoss,
    loss_head: LossHead | None = None,
    optimizer: Optimizer,
    scheduler: LambdaLR | None = None,
    max_grad_norm: float = 10.0,
) -> Tensor:
    """Run a single training step.

    Expects batch to contain ``meta``, ``images``, ``query``, and a ``targets``
    dict with task-specific ground-truth tensors per query.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)

    predictions = model(batch["meta"], batch["images"], batch["query"])
    if loss_head is None:
        loss_head = LossHead()
    losses, confidence = loss_head(predictions, batch["targets"])
    loss = loss_fn(losses, confidence, batch["targets"].get("query_mask"))
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return loss.detach()
