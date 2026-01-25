"""Loss head for D4RT predictions."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F


class LossHead(Module):
    """Compute task losses from 4D predictions and targets."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions: Tensor, targets: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """Split predictions and compute per-query losses.

        Args:
            predictions: Tensor of shape (B, Nq, 13).
            targets: Dict containing GT tensors with keys "L3D", "L2D", "Lvis",
                "Ldisp", "Lconf", "Lnormal".

        Returns:
            A dict of per-query loss tensors and the predicted confidence.
        """
        if predictions.shape[-1] != 13:
            raise ValueError("Predictions must have 13 dimensions.")

        pred_xyz, pred_uv, pred_vis, pred_disp, pred_normal, pred_conf = torch.split(
            predictions, [3, 2, 1, 3, 3, 1], dim=-1
        )

        tgt_xyz = targets["L3D"].to(predictions.device)
        tgt_uv = targets["L2D"].to(predictions.device)
        tgt_vis = targets["Lvis"].to(predictions.device)
        tgt_disp = targets["Ldisp"].to(predictions.device)
        tgt_conf = targets["Lconf"].to(predictions.device)
        tgt_normal = targets["Lnormal"].to(predictions.device)

        loss_3d = F.smooth_l1_loss(pred_xyz, tgt_xyz, reduction="none").mean(dim=-1, keepdim=True)
        loss_2d = F.smooth_l1_loss(pred_uv, tgt_uv, reduction="none").mean(dim=-1, keepdim=True)
        loss_vis = F.binary_cross_entropy_with_logits(pred_vis, tgt_vis, reduction="none")
        loss_disp = F.smooth_l1_loss(pred_disp, tgt_disp, reduction="none").mean(dim=-1, keepdim=True)
        pred_confidence = torch.sigmoid(pred_conf)
        loss_conf = F.l1_loss(pred_confidence, tgt_conf, reduction="none")
        loss_normal = F.smooth_l1_loss(pred_normal, tgt_normal, reduction="none").mean(dim=-1, keepdim=True)

        losses = {
            "L3D": loss_3d,
            "L2D": loss_2d,
            "Lvis": loss_vis,
            "Ldisp": loss_disp,
            "Lconf": loss_conf,
            "Lnormal": loss_normal,
        }
        return losses, pred_confidence
