# src/neural_astar/utils/geometric_losses.py
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalCentricGeodesicLoss(nn.Module):
    """
    Goal-centric attention supervision.

    Uses goal-to-all geodesic distance (1 x H x W) as teacher and trains
    attention weights to align with softmax(-dist / T) for multiple
    temperatures to capture local/global structure.
    """

    def __init__(
        self,
        temperatures: Optional[List[float]] = None,
        loss_type: str = "kl",
    ) -> None:
        super().__init__()
        self.temperatures = temperatures or [0.5, 1.0, 2.0]
        self.loss_type = loss_type

    def forward(
        self,
        attn_weights: torch.Tensor,  # [B, H, N, N] or [B, N, N]
        geo_dist: torch.Tensor,  # [B, 1, H, W]
        reachable: Optional[torch.Tensor] = None,  # [B, 1, H, W]
    ) -> torch.Tensor:
        if attn_weights.dim() == 4:
            attn_mean = attn_weights.mean(dim=1)
        else:
            attn_mean = attn_weights

        b, n, _ = attn_mean.shape
        # Infer bottleneck spatial size
        side = int(n ** 0.5)
        if side * side != n:
            raise ValueError(f"Attention tokens {n} not square (side={side})")

        geo_ds = F.interpolate(
            geo_dist, size=(side, side), mode="bilinear", align_corners=False
        )
        geo_flat = geo_ds.flatten(2)  # [B,1,N]

        mask_flat = None
        if reachable is not None:
            kernel_size = max(1, reachable.shape[-1] // side)
            reach_ds = F.max_pool2d(reachable, kernel_size=kernel_size)
            mask_flat = reach_ds.flatten(2) > 0.5  # [B,1,N]

        total_loss = 0.0
        for t in self.temperatures:
            target = self._goal_soft_target(geo_flat, mask_flat, temperature=t)
            loss = self._compute_loss(attn_mean, target, mask_flat)
            total_loss = total_loss + loss
        return total_loss / len(self.temperatures)

    def _goal_soft_target(
        self,
        geo_flat: torch.Tensor,  # [B,1,N]
        mask_flat: Optional[torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        neg = -geo_flat / temperature
        if mask_flat is not None:
            neg = neg.masked_fill(~mask_flat, -1e9)
        target_row = F.softmax(neg, dim=-1)  # [B,1,N]
        target = target_row.expand(-1, geo_flat.shape[-1], -1)  # [B,N,N]; row-wise identical
        return target

    def _compute_loss(
        self,
        attn: torch.Tensor,  # [B,N,N]
        target: torch.Tensor,  # [B,N,N]
        mask_flat: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.loss_type == "mse":
            loss = F.mse_loss(attn, target, reduction="none")
        else:
            log_attn = torch.log(attn + 1e-9)
            loss = F.kl_div(log_attn, target, reduction="none", log_target=False)

        if mask_flat is not None:
            mask = mask_flat.squeeze(1)  # [B,N]
            mask_matrix = mask.unsqueeze(1) & mask.unsqueeze(2)
            loss = loss * mask_matrix.float()
            denom = mask_matrix.float().sum() + 1e-9
            return loss.sum() / denom
        return loss.mean()


class VectorFieldLoss(nn.Module):
    """Vector field direction supervision with masking."""

    def __init__(self, loss_type: str = "cosine") -> None:
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        pred_vx: torch.Tensor,  # [B,1,H,W]
        pred_vy: torch.Tensor,  # [B,1,H,W]
        gt_vx: torch.Tensor,  # [B,1,H,W]
        gt_vy: torch.Tensor,  # [B,1,H,W]
        reachable: torch.Tensor,  # [B,1,H,W]
    ) -> torch.Tensor:
        mask = reachable > 0.5

        if self.loss_type == "mse":
            loss = (pred_vx - gt_vx) ** 2 + (pred_vy - gt_vy) ** 2
        else:
            pred_vec = torch.cat([pred_vx, pred_vy], dim=1)
            gt_vec = torch.cat([gt_vx, gt_vy], dim=1)

            dot = (pred_vec * gt_vec).sum(dim=1, keepdim=True)
            loss = 1.0 - dot

        loss = loss * mask.float()
        denom = mask.float().sum() + 1e-8
        return loss.sum() / denom

