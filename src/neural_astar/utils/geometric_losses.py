# src/neural_astar/utils/geometric_losses.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS: float = 1e-8


# Utility Functions

def _create_scalar_tensor(value: float, device: torch.device) -> torch.Tensor:
    """Create a scalar tensor on the specified device."""
    return torch.tensor(value, device=device, dtype=torch.float32)


def _compute_vector_magnitude(
    vec_x: torch.Tensor,
    vec_y: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    """Compute magnitude of 2D vector field with numerical stability."""
    return torch.sqrt(vec_x.square() + vec_y.square() + eps)


def _normalize_vector_field(
    vec_x: torch.Tensor,
    vec_y: torch.Tensor,
    eps: float = _EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize 2D vector field. Returns (norm_x, norm_y, magnitude)."""
    magnitude = _compute_vector_magnitude(vec_x, vec_y, eps)
    return vec_x / magnitude, vec_y / magnitude, magnitude


def _compute_masked_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute weighted mean over masked region. """
    if mask is None:
        return values.mean()

    effective_weight = mask if weights is None else mask * weights
    denom = effective_weight.sum().clamp_min(_EPS)
    return (values * effective_weight).sum() / denom


def _get_reachable_mask(
    reachable: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Convert reachable tensor to binary mask."""
    if reachable is None:
        return None
    return (reachable > 0.5).to(dtype)


def _grid_diagonal_from_shape(H: int, W: int) -> float:
    """Return the maximum Euclidean distance on an HxW grid in pixel coordinates. """
    h = max(H - 1, 1)
    w = max(W - 1, 1)
    return math.sqrt(float(h * h + w * w))


# Gradient Computation Modules

class SobelGradient(nn.Module):
    """Compute spatial gradients using 3x3 Sobel operator (replicate padding). """

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0, 0, 0],
             [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
        # /8.0 corresponds to dx=1 scaling for Sobel
        grad_x = F.conv2d(padded, self.sobel_x) / 8.0
        grad_y = F.conv2d(padded, self.sobel_y) / 8.0
        return grad_x, grad_y


class UpwindGradient(nn.Module):
    """Upwind-style (Godunov-inspired) finite difference for Eikonal magnitude."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
        center = padded[..., 1:-1, 1:-1]

        left = padded[..., 1:-1, :-2]
        right = padded[..., 1:-1, 2:]
        up = padded[..., :-2, 1:-1]
        down = padded[..., 2:, 1:-1]

        d_dx_minus = F.relu(center - left)
        d_dx_plus = F.relu(center - right)
        grad_x_abs = torch.maximum(d_dx_minus, d_dx_plus)

        d_dy_minus = F.relu(center - up)
        d_dy_plus = F.relu(center - down)
        grad_y_abs = torch.maximum(d_dy_minus, d_dy_plus)

        return grad_x_abs, grad_y_abs


# Individual Loss Components

class LogDistanceLoss(nn.Module):
    """Log-space distance regression loss: SmoothL1 on log1p(distance)."""

    def __init__(self, mask_unreachable: bool = True) -> None:
        super().__init__()
        self.mask_unreachable = mask_unreachable

    def forward(
        self,
        pred_dist: torch.Tensor,
        gt_dist: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred_dist: [B,1,H,W], expected non-negative (e.g., softplus in encoder)
            gt_dist:   [B,1,H,W]
            reachable: [B,1,H,W] optional
        """
        log_pred = torch.log1p(pred_dist)
        log_gt = torch.log1p(gt_dist.clamp_min(0.0))
        per_pixel = F.smooth_l1_loss(log_pred, log_gt, reduction="none")

        mask: Optional[torch.Tensor] = None
        if self.mask_unreachable and reachable is not None:
            mask = _get_reachable_mask(reachable, per_pixel.dtype)

        return _compute_masked_mean(per_pixel, mask)


class VectorFieldLoss(nn.Module):
    """ Vector field supervision using cosine similarity. """

    def __init__(
        self,
        use_magnitude_weight: bool = True,
        magnitude_weight_clip: Optional[float] = 1.0,
        smooth_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_magnitude_weight = bool(use_magnitude_weight)
        self.magnitude_weight_clip = magnitude_weight_clip
        self.smooth_weight = float(smooth_weight)

        if self.smooth_weight > 0.0:
            laplacian = torch.tensor(
                [[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3)
            self.register_buffer("laplacian_kernel", laplacian)

    def forward(
        self,
        pred_vx: torch.Tensor,
        pred_vy: torch.Tensor,
        gt_vx: torch.Tensor,
        gt_vy: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Normalize to unit vectors for direction-only loss
        pred_nx, pred_ny, _ = _normalize_vector_field(pred_vx, pred_vy)
        gt_nx, gt_ny, gt_mag = _normalize_vector_field(gt_vx, gt_vy)

        cos_sim = (pred_nx * gt_nx + pred_ny * gt_ny).clamp(-1.0, 1.0)
        direction_loss = 1.0 - cos_sim  # in [0,2]

        mask = _get_reachable_mask(reachable, direction_loss.dtype) if reachable is not None else None

        weights: Optional[torch.Tensor] = None
        if self.use_magnitude_weight:
            weights = gt_mag
            if self.magnitude_weight_clip is not None:
                weights = weights.clamp_max(float(self.magnitude_weight_clip))

        loss = _compute_masked_mean(direction_loss, mask, weights)

        if self.smooth_weight > 0.0:
            loss = loss + self._compute_smoothness_penalty(pred_vx, pred_vy)
        return loss

    def _compute_smoothness_penalty(self, pred_vx: torch.Tensor, pred_vy: torch.Tensor) -> torch.Tensor:
        vx_padded = F.pad(pred_vx, (1, 1, 1, 1), mode="replicate")
        vy_padded = F.pad(pred_vy, (1, 1, 1, 1), mode="replicate")
        lap_vx = F.conv2d(vx_padded, self.laplacian_kernel)
        lap_vy = F.conv2d(vy_padded, self.laplacian_kernel)
        return self.smooth_weight * (lap_vx.square() + lap_vy.square()).mean()


class GradientDirectionConsistencyLoss(nn.Module):
    """Enforce consistency between predicted vector field V and negative distance gradient -∇D."""

    def __init__(
        self,
        gradient_magnitude_threshold: float = 1e-3,
        gradient_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.grad = gradient_module if gradient_module is not None else SobelGradient()
        self.gradient_magnitude_threshold = float(gradient_magnitude_threshold)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_vx: torch.Tensor,
        pred_vy: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        grad_x, grad_y = self.grad(pred_dist)
        target_vx, target_vy = -grad_x, -grad_y

        tnx, tny, tmag = _normalize_vector_field(target_vx, target_vy)
        pnx, pny, _ = _normalize_vector_field(pred_vx, pred_vy)

        cos_sim = (tnx * pnx + tny * pny).clamp(-1.0, 1.0)
        loss_map = 1.0 - cos_sim

        valid_mask = (tmag > self.gradient_magnitude_threshold).to(loss_map.dtype)
        if reachable is not None:
            reachable_mask = _get_reachable_mask(reachable, loss_map.dtype)
            if reachable_mask is not None:
                valid_mask = valid_mask * reachable_mask

        return _compute_masked_mean(loss_map, valid_mask)


class EikonalLoss(nn.Module):
    """  Eikonal regularization on normalized distance: """

    def __init__(
        self,
        gradient_module: Optional[nn.Module] = None,
        min_valid_ratio: float = 0.3,
        max_valid_ratio: float = 5.0,
        use_smooth_l1: bool = False,
        smooth_l1_beta: float = 0.01,
    ) -> None:
        super().__init__()
        self.grad = gradient_module if gradient_module is not None else UpwindGradient()

        # Validity band around target magnitude:
        # keep pixels where gmag is within [min_valid_ratio * target, max_valid_ratio * target]
        self.min_valid_ratio = float(min_valid_ratio)
        self.max_valid_ratio = float(max_valid_ratio)

        # Robust penalty option
        self.use_smooth_l1 = bool(use_smooth_l1)
        self.smooth_l1_beta = float(smooth_l1_beta)

    def forward(
        self,
        pred_dist: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gx_abs, gy_abs = self.grad(pred_dist)
        gmag = _compute_vector_magnitude(gx_abs, gy_abs)

        H, W = pred_dist.shape[-2:]
        diagonal = _grid_diagonal_from_shape(H, W)
        target = 1.0 / diagonal

        residual = gmag - target
        if self.use_smooth_l1:
            # SmoothL1 around zero residual (robust near singularities)
            per_pixel = F.smooth_l1_loss(
                residual, torch.zeros_like(residual),
                reduction="none", beta=self.smooth_l1_beta
            )
        else:
            per_pixel = residual.abs()

        # Build mask: reachable AND within valid gradient band
        mask = _get_reachable_mask(reachable, per_pixel.dtype) if reachable is not None else None

        valid = (gmag >= self.min_valid_ratio * target) & (gmag <= self.max_valid_ratio * target)
        valid = valid.to(per_pixel.dtype)

        if mask is None:
            mask = valid
        else:
            mask = mask * valid

        return _compute_masked_mean(per_pixel, mask)



# Combined Loss

class CombinedGeodesicLoss(nn.Module):
    """ Combined geodesic supervision loss with warmup scheduling: """

    def __init__(
        self,
        dist_weight: float = 1.0,
        vec_weight: float = 1.0,
        consistency_weight: float = 1.0,
        eikonal_weight: float = 0.0,
        warmup_epochs: int = 0,
        cons_warmup_epochs: int = 0,
        # Vector weighting knobs
        use_vector_magnitude_weight: bool = True,
        vector_magnitude_weight_clip: Optional[float] = 1.0,
        vector_smooth_weight: float = 0.0,
        # Consistency knobs
        consistency_grad_threshold: float = 1e-3,
        # Eikonal knobs
        eikonal_min_valid_ratio: float = 0.3,
        eikonal_max_valid_ratio: float = 5.0,
        eikonal_use_smooth_l1: bool = False,
        eikonal_smooth_l1_beta: float = 0.01,
    ) -> None:
        super().__init__()

        self.dist_weight = float(dist_weight)
        self.vec_weight = float(vec_weight)
        self.consistency_weight = float(consistency_weight)
        self.eikonal_weight = float(eikonal_weight)

        self.warmup_epochs = int(warmup_epochs)
        self.cons_warmup_epochs = int(cons_warmup_epochs)

        # Gradient modules
        self._sobel = SobelGradient()
        self._upwind = UpwindGradient()

        # Loss components
        self._dist_loss = LogDistanceLoss(mask_unreachable=True)
        self._vec_loss = VectorFieldLoss(
            use_magnitude_weight=use_vector_magnitude_weight,
            magnitude_weight_clip=vector_magnitude_weight_clip,
            smooth_weight=vector_smooth_weight,
        )
        self._cons_loss = GradientDirectionConsistencyLoss(
            gradient_magnitude_threshold=consistency_grad_threshold,
            gradient_module=self._sobel,
        )
        self._eik_loss = EikonalLoss(
            gradient_module=self._upwind,
            min_valid_ratio=eikonal_min_valid_ratio,
            max_valid_ratio=eikonal_max_valid_ratio,
            use_smooth_l1=eikonal_use_smooth_l1,
            smooth_l1_beta=eikonal_smooth_l1_beta,
        )

    @staticmethod
    def _warmup(epoch: int, warmup_epochs: int) -> float:
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(warmup_epochs))

    @staticmethod
    def _get_device(*tensors: Optional[torch.Tensor]) -> torch.device:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                return t.device
        return torch.device("cpu")

    def forward(
        self,
        pred_dist: Optional[torch.Tensor],
        pred_vx: Optional[torch.Tensor],
        pred_vy: Optional[torch.Tensor],
        gt_dist: Optional[torch.Tensor],
        gt_vx: Optional[torch.Tensor],
        gt_vy: Optional[torch.Tensor],
        reachable: Optional[torch.Tensor],
        current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self._get_device(pred_dist, pred_vx, reachable)
        zero = torch.zeros((), device=device)

        w_geo = self._warmup(current_epoch, self.warmup_epochs)
        w_cons = self._warmup(current_epoch, self.cons_warmup_epochs)

        # 1) Distance
        loss_dist = zero
        if pred_dist is not None and gt_dist is not None:
            loss_dist = self._dist_loss(pred_dist, gt_dist, reachable)

        # 2) Vector
        loss_vec = zero
        if pred_vx is not None and pred_vy is not None and gt_vx is not None and gt_vy is not None:
            loss_vec = self._vec_loss(pred_vx, pred_vy, gt_vx, gt_vy, reachable)

        # 3) Consistency
        loss_cons = zero
        if pred_dist is not None and pred_vx is not None and pred_vy is not None:
            loss_cons = self._cons_loss(pred_dist, pred_vx, pred_vy, reachable)

        # 4) Eikonal
        loss_eik = zero
        if self.eikonal_weight > 0.0 and pred_dist is not None:
            loss_eik = self._eik_loss(pred_dist, reachable)

        total = (
            self.dist_weight * loss_dist
            + w_geo * self.vec_weight * loss_vec
            + w_geo * w_cons * self.consistency_weight * loss_cons
            + w_geo * self.eikonal_weight * loss_eik
        )

        log: Dict[str, torch.Tensor] = {
            "loss/dist": loss_dist.detach(),
            "loss/vec": loss_vec.detach(),
            "loss/cons": loss_cons.detach(),
            "loss/eik": loss_eik.detach(),
            "loss/geo_total": total.detach(),
            "meta/warmup_geo": _create_scalar_tensor(w_geo, device),
            "meta/warmup_cons": _create_scalar_tensor(w_cons, device),
        }
        return total, log
