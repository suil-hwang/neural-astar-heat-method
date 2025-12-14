# src/neural_astar/utils/geometric_losses.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def _as_float_tensor(x: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, device=device, dtype=torch.float32)


def _softplus_nonneg(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


class LogDistanceLoss(nn.Module):
    """Log-space distance regression loss (SmoothL1 on log1p)."""

    def __init__(self, mask_unreachable: bool = True) -> None:
        super().__init__()
        self.mask_unreachable = mask_unreachable

    def forward(
        self,
        pred_dist_raw: torch.Tensor,      # [B,1,H,W]
        gt_dist: torch.Tensor,            # [B,1,H,W]
        reachable: Optional[torch.Tensor] = None,  # [B,1,H,W]
    ) -> torch.Tensor:
        pred_dist = _softplus_nonneg(pred_dist_raw)
        
        log_pred = torch.log1p(pred_dist)
        log_gt = torch.log1p(gt_dist.clamp_min(0.0))

        per_pix = F.smooth_l1_loss(log_pred, log_gt, reduction="none")

        if self.mask_unreachable and reachable is not None:
            mask = (reachable > 0.5).to(per_pix.dtype)
            denom = mask.sum().clamp_min(1.0)
            return (per_pix * mask).sum() / denom

        return per_pix.mean()


class SobelGrad(nn.Module):
    """Sobel gradients with replicate padding."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(x_pad, self.sobel_x) / 8.0
        gy = F.conv2d(x_pad, self.sobel_y) / 8.0
        return gx, gy


class GradientDirectionConsistencyLoss(nn.Module):
    """Enforce V direction aligns with -∇D."""

    def __init__(self, grad_mag_threshold: float = 1e-3) -> None:
        super().__init__()
        self.grad = SobelGrad()
        self.grad_mag_threshold = float(grad_mag_threshold)

    def forward(
        self,
        pred_dist_raw: torch.Tensor,
        pred_vx: torch.Tensor,
        pred_vy: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_dist = _softplus_nonneg(pred_dist_raw)

        gx, gy = self.grad(pred_dist)
        tvx, tvy = -gx, -gy # Target is negative gradient

        tmag = torch.sqrt(tvx * tvx + tvy * tvy + _EPS)
        pmag = torch.sqrt(pred_vx * pred_vx + pred_vy * pred_vy + _EPS)

        tvx_n, tvy_n = tvx / tmag, tvy / tmag
        pvx_n, pvy_n = pred_vx / pmag, pred_vy / pmag

        cos_sim = (tvx_n * pvx_n + tvy_n * pvy_n).clamp(-1.0, 1.0)
        loss_map = 1.0 - cos_sim

        valid = (tmag > self.grad_mag_threshold).to(loss_map.dtype)
        if reachable is not None:
            valid = valid * (reachable > 0.5).to(loss_map.dtype)

        denom = valid.sum().clamp_min(1.0)
        return (loss_map * valid).sum() / denom


class VectorFieldLoss(nn.Module):
    """Vector field supervision using cosine similarity."""

    def __init__(self, use_magnitude_weight: bool = True, smooth_weight: float = 0.0) -> None:
        super().__init__()
        self.use_magnitude_weight = use_magnitude_weight
        self.smooth_weight = float(smooth_weight)

        if self.smooth_weight > 0.0:
            lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
            self.register_buffer("laplacian_kernel", lap)

    def forward(
        self,
        pred_vx: torch.Tensor,
        pred_vy: torch.Tensor,
        gt_vx: torch.Tensor,
        gt_vy: torch.Tensor,
        reachable: torch.Tensor,
    ) -> torch.Tensor:
        pmag = torch.sqrt(pred_vx * pred_vx + pred_vy * pred_vy + _EPS)
        gmag = torch.sqrt(gt_vx * gt_vx + gt_vy * gt_vy + _EPS)

        pvx_n, pvy_n = pred_vx / pmag, pred_vy / pmag
        gvx_n, gvy_n = gt_vx / gmag, gt_vy / gmag

        cos_sim = (pvx_n * gvx_n + pvy_n * gvy_n).clamp(-1.0, 1.0)

        mask = (reachable > 0.5).to(cos_sim.dtype)
        weight = mask * (gmag if self.use_magnitude_weight else 1.0)
        denom = weight.sum().clamp_min(1.0)
        loss = ((1.0 - cos_sim) * weight).sum() / denom

        if self.smooth_weight > 0.0:
            vx_pad = F.pad(pred_vx, (1, 1, 1, 1), mode="replicate")
            vy_pad = F.pad(pred_vy, (1, 1, 1, 1), mode="replicate")
            lap_vx = F.conv2d(vx_pad, self.laplacian_kernel)
            lap_vy = F.conv2d(vy_pad, self.laplacian_kernel)
            loss = loss + self.smooth_weight * (lap_vx.square() + lap_vy.square()).mean()

        return loss


class EikonalLoss(nn.Module):
    """Penalize |∇D| deviating from 1 on reachable domain."""

    def __init__(self, grad: nn.Module) -> None:
        super().__init__()
        self.grad = grad

    def forward(self, pred_dist_raw: torch.Tensor, reachable: Optional[torch.Tensor]) -> torch.Tensor:
        pred_dist = _softplus_nonneg(pred_dist_raw)
        gx, gy = self.grad(pred_dist)
        gmag = torch.sqrt(gx * gx + gy * gy + _EPS)
        loss_map = (gmag - 1.0).abs()

        if reachable is not None:
            mask = (reachable > 0.5).to(loss_map.dtype)
            denom = mask.sum().clamp_min(1.0)
            return (loss_map * mask).sum() / denom
        return loss_map.mean()


class CombinedGeodesicLoss(nn.Module):
    def __init__(
        self,
        dist_weight: float = 1.0,
        vec_weight: float = 1.0,
        consistency_weight: float = 1.0,
        warmup_epochs: int = 0,
        cons_warmup_epochs: int = 0,
        eikonal_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.dist_weight = dist_weight
        self.vec_weight = vec_weight
        self.consistency_weight = consistency_weight
        self.warmup_epochs = int(warmup_epochs)
        self.cons_warmup_epochs = int(cons_warmup_epochs)
        self.eikonal_weight = float(eikonal_weight)

        self.dist_loss = LogDistanceLoss(mask_unreachable=True)
        self.vec_loss = VectorFieldLoss(use_magnitude_weight=True)
        self.cons_dir_loss = GradientDirectionConsistencyLoss()

        # Share Sobel filters for consistency
        self.eikonal_loss = EikonalLoss(grad=self.cons_dir_loss.grad)

    def _warmup(self, epoch: int, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(warmup_epochs))

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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Alias for clarity: pred_cost comes from dist_head
        pred_dist_raw = pred_cost
        gt_dist = gt_distance

        device = (
            pred_dist_raw.device if isinstance(pred_dist_raw, torch.Tensor)
            else pred_vx.device if isinstance(pred_vx, torch.Tensor)
            else reachable.device if isinstance(reachable, torch.Tensor)
            else torch.device("cpu")
        )
        zero = torch.zeros((), device=device)

        w_geo = self._warmup(current_epoch, self.warmup_epochs)
        w_cons = self._warmup(current_epoch, self.cons_warmup_epochs)

        l_dist = zero
        if pred_dist_raw is not None and gt_dist is not None:
            l_dist = self.dist_loss(pred_dist_raw, gt_dist, reachable)

        l_vec = zero
        if (
            pred_vx is not None and pred_vy is not None and
            gt_vx is not None and gt_vy is not None and
            reachable is not None
        ):
            l_vec = self.vec_loss(pred_vx, pred_vy, gt_vx, gt_vy, reachable)

        l_cons = zero
        if pred_dist_raw is not None and pred_vx is not None and pred_vy is not None:
            l_cons = self.cons_dir_loss(pred_dist_raw, pred_vx, pred_vy, reachable)

        l_eik = zero
        if self.eikonal_weight > 0.0 and pred_dist_raw is not None:
            l_eik = self.eikonal_loss(pred_dist_raw, reachable)

        # Final Loss Combination
        # 1. Distance: Always active (No warmup) to ground the scale
        # 2. Vector: Warmed up by w_geo
        # 3. Consistency: Warmed up by BOTH w_geo and w_cons (conservative)
        # 4. Eikonal: Warmed up by w_geo
        total = (
            self.dist_weight * l_dist +
            w_geo * (self.vec_weight * l_vec) +
            w_geo * (self.consistency_weight * (w_cons * l_cons)) +
            w_geo * (self.eikonal_weight * l_eik)
        )

        log: Dict[str, torch.Tensor] = {
            "loss/dist": l_dist.detach(),
            "loss/vec": l_vec.detach(),
            "loss/cons": l_cons.detach(),
            "loss/eik": l_eik.detach(),
            "loss/geo_total": total.detach(),
            "meta/warmup_geo": _as_float_tensor(w_geo, device),
            "meta/warmup_cons": _as_float_tensor(w_cons, device),
        }
        return total, log
