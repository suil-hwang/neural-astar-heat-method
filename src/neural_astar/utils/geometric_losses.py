# src/neural_astar/utils/geometric_losses.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectDistanceLoss(nn.Module):
    """Direct supervision for cost map to match geodesic distance."""

    def __init__(self, normalize: bool = True, mask_unreachable: bool = True) -> None:
        super().__init__()
        self.normalize = normalize
        self.mask_unreachable = mask_unreachable

    def forward(
        self,
        pred_cost: torch.Tensor,  # [B,1,H,W]
        gt_distance: torch.Tensor,  # [B,1,H,W]
        reachable: Optional[torch.Tensor] = None,  # [B,1,H,W]
    ) -> torch.Tensor:
        pred = pred_cost
        gt = gt_distance

        if self.normalize:
            if reachable is not None:
                mask = (reachable > 0.5).float()
                denom = (gt * mask).amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            else:
                denom = gt.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            pred = pred / denom
            gt = gt / denom

        diff_sq = (pred - gt) ** 2

        if self.mask_unreachable and reachable is not None:
            mask = (reachable > 0.5).float()
            denom = mask.sum() + 1e-8
            return (diff_sq * mask).sum() / denom

        return diff_sq.mean()


class VectorFieldLoss(nn.Module):
    """Vector field alignment supervision using cosine similarity."""

    def __init__(
        self,
        use_magnitude_weight: bool = True,
        smooth_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_magnitude_weight = use_magnitude_weight
        self.smooth_weight = smooth_weight

        if smooth_weight > 0:
            laplacian = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            self.register_buffer("laplacian_kernel", laplacian)

    def forward(
        self,
        pred_vx: torch.Tensor,  # [B,1,H,W]
        pred_vy: torch.Tensor,  # [B,1,H,W]
        gt_vx: torch.Tensor,  # [B,1,H,W]
        gt_vy: torch.Tensor,  # [B,1,H,W]
        reachable: torch.Tensor,  # [B,1,H,W]
    ) -> torch.Tensor:
        pred_mag = torch.sqrt(pred_vx**2 + pred_vy**2 + 1e-8)
        pred_vx_norm = pred_vx / pred_mag
        pred_vy_norm = pred_vy / pred_mag

        gt_mag = torch.sqrt(gt_vx**2 + gt_vy**2 + 1e-8)
        gt_vx_norm = gt_vx / gt_mag
        gt_vy_norm = gt_vy / gt_mag

        cos_sim = pred_vx_norm * gt_vx_norm + pred_vy_norm * gt_vy_norm

        mask = (reachable > 0.5).float()
        if self.use_magnitude_weight:
            weight = mask * gt_mag
        else:
            weight = mask

        denom = weight.sum() + 1e-8
        loss = ((1.0 - cos_sim) * weight).sum() / denom

        if self.smooth_weight > 0:
            lap_vx = F.conv2d(pred_vx, self.laplacian_kernel, padding=1)
            lap_vy = F.conv2d(pred_vy, self.laplacian_kernel, padding=1)
            smooth_loss = (lap_vx**2 + lap_vy**2).mean()
            loss = loss + self.smooth_weight * smooth_loss

        return loss


class CombinedGeodesicLoss(nn.Module):
    """Combined distance + vector field supervision."""

    def __init__(
        self,
        dist_weight: float = 1.0,
        vec_weight: float = 1.0,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.dist_weight = dist_weight
        self.vec_weight = vec_weight
        self.warmup_epochs = warmup_epochs

        self.dist_loss = DirectDistanceLoss()
        self.vec_loss = VectorFieldLoss()

    def forward(
        self,
        pred_cost: Optional[torch.Tensor],
        pred_vx: Optional[torch.Tensor],
        pred_vy: Optional[torch.Tensor],
        gt_distance: Optional[torch.Tensor],
        gt_vx: Optional[torch.Tensor],
        gt_vy: Optional[torch.Tensor],
        reachable: Optional[torch.Tensor],
        current_epoch: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        device = None
        for t in (pred_cost, pred_vx, pred_vy, gt_distance, reachable):
            if isinstance(t, torch.Tensor):
                device = t.device
                break
        if device is None:
            device = torch.device("cpu")
        zero = torch.tensor(0.0, device=device)

        if self.warmup_epochs > 0:
            warmup_factor = min(1.0, float(current_epoch) / float(self.warmup_epochs))
        else:
            warmup_factor = 1.0

        l_dist = zero
        if pred_cost is not None and gt_distance is not None:
            l_dist = self.dist_loss(pred_cost, gt_distance, reachable)

        l_vec = zero
        if (
            pred_vx is not None
            and pred_vy is not None
            and gt_vx is not None
            and gt_vy is not None
            and reachable is not None
        ):
            l_vec = self.vec_loss(pred_vx, pred_vy, gt_vx, gt_vy, reachable)

        total = (
            self.dist_weight * warmup_factor * l_dist
            + self.vec_weight * warmup_factor * l_vec
        )

        loss_dict = {
            "loss/dist": l_dist.detach(),
            "loss/vec": l_vec.detach(),
            "loss/geo_total": total.detach(),
            "meta/warmup_factor": warmup_factor,
        }

        return total, loss_dict
