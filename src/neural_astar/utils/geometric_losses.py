# src/neural_astar/utils/geometric_losses.py
from __future__ import annotations

from typing import Dict, Optional, Tuple

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
    eps: float = _EPS
) -> torch.Tensor:
    """Compute magnitude of 2D vector field with numerical stability."""
    return torch.sqrt(vec_x * vec_x + vec_y * vec_y + eps)


def _normalize_vector_field(
    vec_x: torch.Tensor,
    vec_y: torch.Tensor,
    eps: float = _EPS
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize 2D vector field to unit vectors. Returns (norm_x, norm_y, magnitude)."""
    magnitude = _compute_vector_magnitude(vec_x, vec_y, eps)
    return vec_x / magnitude, vec_y / magnitude, magnitude


def _compute_masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute weighted mean over masked region."""
    effective_weight = mask if weights is None else mask * weights
    denominator = effective_weight.sum().clamp_min(1.0)
    return (values * effective_weight).sum() / denominator


def _get_reachable_mask(
    reachable: Optional[torch.Tensor],
    dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Convert reachable tensor to binary mask."""
    if reachable is None:
        return None
    return (reachable > 0.5).to(dtype)


# Gradient Computation

class SobelGradient(nn.Module):
    """Compute spatial gradients using Sobel operator with replicate padding."""

    def __init__(self) -> None:
        super().__init__()
        # Horizontal gradient kernel
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        # Vertical gradient kernel
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (grad_x, grad_y) of the input field."""
        padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(padded, self.sobel_x) / 8.0
        grad_y = F.conv2d(padded, self.sobel_y) / 8.0
        return grad_x, grad_y


# Individual Loss Components

class LogDistanceLoss(nn.Module):
    """Log-space distance regression loss."""

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
            pred_dist: Predicted distance [B,1,H,W], non-negative (softplus applied in encoder)
            gt_dist: Ground truth distance [B,1,H,W]
            reachable: Reachability mask [B,1,H,W], optional
        """
        log_pred = torch.log1p(pred_dist)
        log_gt = torch.log1p(gt_dist.clamp_min(0.0))
        per_pixel_loss = F.smooth_l1_loss(log_pred, log_gt, reduction="none")

        if self.mask_unreachable and reachable is not None:
            mask = _get_reachable_mask(reachable, per_pixel_loss.dtype)
            return _compute_masked_mean(per_pixel_loss, mask)
        return per_pixel_loss.mean()


class VectorFieldLoss(nn.Module):
    """Vector field supervision using cosine similarity. """

    def __init__(
        self, 
        use_magnitude_weight: bool = True, 
        smooth_weight: float = 0.0
    ) -> None:
        super().__init__()
        self.use_magnitude_weight = use_magnitude_weight
        self.smooth_weight = float(smooth_weight)

        if self.smooth_weight > 0.0:
            laplacian = torch.tensor(
                [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            self.register_buffer("laplacian_kernel", laplacian)

    def forward(
        self,
        pred_vx: torch.Tensor,
        pred_vy: torch.Tensor,
        gt_vx: torch.Tensor,
        gt_vy: torch.Tensor,
        reachable: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred_vx, pred_vy: Predicted vector field components [B,1,H,W]
            gt_vx, gt_vy: Ground truth vector field components [B,1,H,W]
            reachable: Reachability mask [B,1,H,W]
        """
        # Normalize vectors to unit length
        pred_x_norm, pred_y_norm, _ = _normalize_vector_field(pred_vx, pred_vy)
        gt_x_norm, gt_y_norm, gt_magnitude = _normalize_vector_field(gt_vx, gt_vy)

        # Cosine similarity loss: 1 - cos(θ) ∈ [0, 2]
        cosine_similarity = (pred_x_norm * gt_x_norm + pred_y_norm * gt_y_norm).clamp(-1.0, 1.0)
        direction_loss = 1.0 - cosine_similarity

        # Compute weighted loss over reachable region
        mask = _get_reachable_mask(reachable, cosine_similarity.dtype)
        weights = gt_magnitude if self.use_magnitude_weight else None
        loss = _compute_masked_mean(direction_loss, mask, weights)

        # Optional: Laplacian smoothness regularization
        if self.smooth_weight > 0.0:
            loss = loss + self._compute_smoothness_penalty(pred_vx, pred_vy)
        return loss

    def _compute_smoothness_penalty(
        self, 
        pred_vx: torch.Tensor, 
        pred_vy: torch.Tensor
    ) -> torch.Tensor:
        """Compute Laplacian smoothness regularization."""
        vx_padded = F.pad(pred_vx, (1, 1, 1, 1), mode="replicate")
        vy_padded = F.pad(pred_vy, (1, 1, 1, 1), mode="replicate")
        laplacian_vx = F.conv2d(vx_padded, self.laplacian_kernel)
        laplacian_vy = F.conv2d(vy_padded, self.laplacian_kernel)
        return self.smooth_weight * (laplacian_vx.square() + laplacian_vy.square()).mean()


class GradientDirectionConsistencyLoss(nn.Module):
    """Enforce consistency between vector field V and negative distance gradient -∇D."""

    def __init__(
        self, 
        gradient_magnitude_threshold: float = 1e-3,
        gradient_module: Optional[SobelGradient] = None
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
        """
        Args:
            pred_dist: Predicted distance [B,1,H,W], non-negative
            pred_vx, pred_vy: Predicted vector field [B,1,H,W]
            reachable: Reachability mask [B,1,H,W], optional
        """
        # Compute distance gradient and target direction (negative gradient)
        grad_x, grad_y = self.grad(pred_dist)
        target_vx, target_vy = -grad_x, -grad_y

        # Normalize both predicted and target vectors
        target_x_norm, target_y_norm, target_magnitude = _normalize_vector_field(target_vx, target_vy)
        pred_x_norm, pred_y_norm, _ = _normalize_vector_field(pred_vx, pred_vy)

        # Cosine similarity loss
        cosine_similarity = (target_x_norm * pred_x_norm + target_y_norm * pred_y_norm).clamp(-1.0, 1.0)
        consistency_loss = 1.0 - cosine_similarity

        # Build validity mask: exclude flat regions (near goal) and unreachable areas
        valid_mask = (target_magnitude > self.gradient_magnitude_threshold).to(consistency_loss.dtype)
        if reachable is not None:
            reachable_mask = _get_reachable_mask(reachable, consistency_loss.dtype)
            valid_mask = valid_mask * reachable_mask

        return _compute_masked_mean(consistency_loss, valid_mask)


class EikonalLoss(nn.Module):
    """Eikonal equation regularization: |∇D| should equal 1."""

    def __init__(self, gradient_module: SobelGradient) -> None:
        super().__init__()
        self.grad = gradient_module

    def forward(
        self, 
        pred_dist: torch.Tensor, 
        reachable: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            pred_dist: Predicted distance [B,1,H,W], non-negative
            reachable: Reachability mask [B,1,H,W], optional
        """
        grad_x, grad_y = self.grad(pred_dist)
        gradient_magnitude = _compute_vector_magnitude(grad_x, grad_y)
        eikonal_violation = (gradient_magnitude - 1.0).abs()

        if reachable is not None:
            mask = _get_reachable_mask(reachable, eikonal_violation.dtype)
            return _compute_masked_mean(eikonal_violation, mask)
        return eikonal_violation.mean()



# Combined Loss

class CombinedGeodesicLoss(nn.Module):
    """Combined geodesic supervision loss with warmup scheduling."""

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
        # Loss weights
        self.dist_weight = dist_weight
        self.vec_weight = vec_weight
        self.consistency_weight = consistency_weight
        self.eikonal_weight = float(eikonal_weight)
        
        # Warmup configuration
        self.warmup_epochs = int(warmup_epochs)
        self.cons_warmup_epochs = int(cons_warmup_epochs)

        # Initialize loss components (shared gradient module for efficiency)
        self._gradient = SobelGradient()
        self._dist_loss = LogDistanceLoss(mask_unreachable=True)
        self._vec_loss = VectorFieldLoss(use_magnitude_weight=True)
        self._consistency_loss = GradientDirectionConsistencyLoss(
            gradient_module=self._gradient
        )
        self._eikonal_loss = EikonalLoss(gradient_module=self._gradient)

    def _compute_warmup_factor(self, epoch: int, warmup_epochs: int) -> float:
        """Compute linear warmup factor [0, 1]."""
        if warmup_epochs <= 0:
            return 1.0
        return min(1.0, float(epoch + 1) / float(warmup_epochs))

    def _get_device(
        self,
        pred_dist: Optional[torch.Tensor],
        pred_vx: Optional[torch.Tensor],
        reachable: Optional[torch.Tensor],
    ) -> torch.device:
        """Determine device from available tensors."""
        for tensor in (pred_dist, pred_vx, reachable):
            if isinstance(tensor, torch.Tensor):
                return tensor.device
        return torch.device("cpu")

    def _compute_distance_loss(
        self,
        pred_dist: Optional[torch.Tensor],
        gt_dist: Optional[torch.Tensor],
        reachable: Optional[torch.Tensor],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance regression loss if inputs available."""
        if pred_dist is None or gt_dist is None:
            return zero
        return self._dist_loss(pred_dist, gt_dist, reachable)

    def _compute_vector_loss(
        self,
        pred_vx: Optional[torch.Tensor],
        pred_vy: Optional[torch.Tensor],
        gt_vx: Optional[torch.Tensor],
        gt_vy: Optional[torch.Tensor],
        reachable: Optional[torch.Tensor],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        """Compute vector field loss if all inputs available."""
        all_available = all(
            t is not None for t in (pred_vx, pred_vy, gt_vx, gt_vy, reachable)
        )
        if not all_available:
            return zero
        return self._vec_loss(pred_vx, pred_vy, gt_vx, gt_vy, reachable)

    def _compute_consistency_loss(
        self,
        pred_dist: Optional[torch.Tensor],
        pred_vx: Optional[torch.Tensor],
        pred_vy: Optional[torch.Tensor],
        reachable: Optional[torch.Tensor],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient-vector consistency loss if inputs available."""
        all_available = all(t is not None for t in (pred_dist, pred_vx, pred_vy))
        if not all_available:
            return zero
        return self._consistency_loss(pred_dist, pred_vx, pred_vy, reachable)

    def _compute_eikonal_loss(
        self,
        pred_dist: Optional[torch.Tensor],
        reachable: Optional[torch.Tensor],
        zero: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Eikonal regularization if enabled and input available."""
        if self.eikonal_weight <= 0.0 or pred_dist is None:
            return zero
        return self._eikonal_loss(pred_dist, reachable)

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
        """ Compute combined geodesic loss."""
        device = self._get_device(pred_dist, pred_vx, reachable)
        zero = torch.zeros((), device=device)

        # Compute warmup factors
        warmup_geo = self._compute_warmup_factor(current_epoch, self.warmup_epochs)
        warmup_cons = self._compute_warmup_factor(current_epoch, self.cons_warmup_epochs)

        # Compute individual losses
        loss_dist = self._compute_distance_loss(pred_dist, gt_dist, reachable, zero)
        loss_vec = self._compute_vector_loss(pred_vx, pred_vy, gt_vx, gt_vy, reachable, zero)
        loss_cons = self._compute_consistency_loss(pred_dist, pred_vx, pred_vy, reachable, zero)
        loss_eik = self._compute_eikonal_loss(pred_dist, reachable, zero)

        # Combine losses with warmup scheduling:
        # - Distance: Always active (anchors the scale)
        # - Vector/Eikonal: Warmed up by warmup_geo
        # - Consistency: Double warmup (warmup_geo * warmup_cons)
        total_loss = (
            self.dist_weight * loss_dist
            + warmup_geo * self.vec_weight * loss_vec
            + warmup_geo * warmup_cons * self.consistency_weight * loss_cons
            + warmup_geo * self.eikonal_weight * loss_eik
        )

        log_dict: Dict[str, torch.Tensor] = {
            "loss/dist": loss_dist.detach(),
            "loss/vec": loss_vec.detach(),
            "loss/cons": loss_cons.detach(),
            "loss/eik": loss_eik.detach(),
            "loss/geo_total": total_loss.detach(),
            "meta/warmup_geo": _create_scalar_tensor(warmup_geo, device),
            "meta/warmup_cons": _create_scalar_tensor(warmup_cons, device),
        }
        return total_loss, log_dict
