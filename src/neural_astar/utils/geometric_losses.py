# src/neural_astar/utils/geometric_losses.py
from __future__ import annotations

import math
from typing import Dict, Final, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Constants
_EPS: Final[float] = 1e-8


# Utility Functions
def _create_scalar_tensor(value: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, device=device, dtype=torch.float32)


def _compute_vector_magnitude(
    vec_x: torch.Tensor,
    vec_y: torch.Tensor,
    eps: float = _EPS,
) -> torch.Tensor:
    return torch.sqrt(vec_x.square() + vec_y.square() + eps)


def _normalize_vector_field(
    vec_x: torch.Tensor,
    vec_y: torch.Tensor,
    eps: float = _EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    magnitude = _compute_vector_magnitude(vec_x, vec_y, eps)
    return vec_x / magnitude, vec_y / magnitude, magnitude


def _compute_masked_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute weighted mean over masked region."""
    # Fast path: no masking or weighting
    if mask is None and weights is None:
        return values.mean()
    
    # Compute effective weight = mask * weights
    if mask is not None and weights is not None:
        effective_weight = mask * weights
    elif mask is not None:
        effective_weight = mask
    else:  # weights is not None
        effective_weight = weights
    
    # Weighted mean with numerical stability
    numerator = (values * effective_weight).sum()
    denominator = effective_weight.sum().clamp_min(_EPS)
    
    return numerator / denominator


def _get_reachable_mask(
    reachable: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if reachable is None:
        return None
    return (reachable > 0.5).to(dtype)


def _grid_diagonal_from_shape(H: int, W: int) -> float:
    h = max(H - 1, 1)
    w = max(W - 1, 1)
    return math.sqrt(float(h * h + w * w))


# Gradient Computation Modules
class SobelGradient(nn.Module):
    """3x3 Sobel operator for smooth gradient estimation."""

    def __init__(self) -> None:
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
        return F.conv2d(padded, self.sobel_x), F.conv2d(padded, self.sobel_y)


class UpwindGradient(nn.Module):
    """1st-order Godunov upwind scheme for Eikonal equations."""

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        padded = F.pad(field, (1, 1, 1, 1), mode="replicate")
        center = padded[..., 1:-1, 1:-1]
        
        # One-sided differences
        Dx_minus = center - padded[..., 1:-1, :-2]   # D⁻ₓ
        Dx_plus = padded[..., 1:-1, 2:] - center     # D⁺ₓ
        Dy_minus = center - padded[..., :-2, 1:-1]   # D⁻ᵧ
        Dy_plus = padded[..., 2:, 1:-1] - center     # D⁺ᵧ
        
        # Godunov flux: max(max(D⁻,0), max(-D⁺,0))
        grad_x = torch.maximum(F.relu(Dx_minus), F.relu(-Dx_plus))
        grad_y = torch.maximum(F.relu(Dy_minus), F.relu(-Dy_plus))
        
        return grad_x, grad_y


class WENO3Gradient(nn.Module):
    """3rd-order WENO scheme for Hamilton-Jacobi equations."""
    
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def _weno3_reconstruct(
        self,
        v_mm: torch.Tensor,
        v_m: torch.Tensor,
        v_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """WENO3 reconstruction for left and right biased derivatives."""
        # Left-biased reconstruction (u⁻)
        # Smoothness indicators: measure oscillation in each stencil
        # β_k = (second derivative approximation)² ≈ smoothness measure
        beta0_m = (v_mm - v_m).square()  # Stencil {i-1, i}
        beta1_m = (v_m - v_p).square()   # Stencil {i, i+1}
        
        # Ideal (linear) weights for 3rd-order accuracy
        # These maximize accuracy when solution is smooth
        d0, d1 = 1.0 / 3.0, 2.0 / 3.0
        
        # Nonlinear weights: reduce contribution from non-smooth stencils
        # α_k = d_k / (ε + β_k)² → small when β_k large (oscillatory)
        alpha0_m = d0 / (self.eps + beta0_m).square()
        alpha1_m = d1 / (self.eps + beta1_m).square()
        w0_m = alpha0_m / (alpha0_m + alpha1_m)
        w1_m = 1.0 - w0_m
        
        # 2nd-order polynomial reconstructions
        # p0: extrapolate from left stencil {i-1, i}
        # p1: interpolate from centered stencil {i, i+1}
        p0_m = 0.5 * (-v_mm + 3.0 * v_m)
        p1_m = 0.5 * (v_m + v_p)
        
        u_minus = w0_m * p0_m + w1_m * p1_m
        
        # Right-biased reconstruction (u⁺)
        # Mirror of left-biased, looking from the other direction
        beta0_p = (v_p - v_m).square()
        beta1_p = (v_m - v_mm).square()
        
        alpha0_p = d0 / (self.eps + beta0_p).square()
        alpha1_p = d1 / (self.eps + beta1_p).square()
        w0_p = alpha0_p / (alpha0_p + alpha1_p)
        w1_p = 1.0 - w0_p
        
        p0_p = 0.5 * (-v_p + 3.0 * v_m)
        p1_p = 0.5 * (v_m + v_mm)
        
        u_plus = w0_p * p0_p + w1_p * p1_p
        
        return u_minus, u_plus

    def forward(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Godunov gradient magnitudes using WENO3 reconstruction. """
        # Pad for 5-point stencil: need φ_{i-2} to φ_{i+2}
        u = F.pad(field, (2, 2, 2, 2), mode="replicate")
        
        # ===== X-direction =====
        # Compute first differences v_k = φ_{k+1} - φ_k
        vx_mm = u[..., 2:-2, 1:-3] - u[..., 2:-2, 0:-4]  # v_{i-2}
        vx_m = u[..., 2:-2, 2:-2] - u[..., 2:-2, 1:-3]   # v_{i-1}
        vx_c = u[..., 2:-2, 3:-1] - u[..., 2:-2, 2:-2]   # v_i (center)
        vx_p = u[..., 2:-2, 4:] - u[..., 2:-2, 3:-1]     # v_{i+1}
        
        # WENO reconstruction at different stencil positions
        ux_minus, _ = self._weno3_reconstruct(vx_mm, vx_m, vx_c)
        _, ux_plus = self._weno3_reconstruct(vx_m, vx_c, vx_p)
        
        # Godunov flux for Eikonal: max(max(u⁻,0), max(-u⁺,0))
        # This selects the "upwind" direction based on characteristic flow
        grad_x = torch.maximum(F.relu(ux_minus), F.relu(-ux_plus))
        
        # ===== Y-direction (analogous) =====
        vy_mm = u[..., 1:-3, 2:-2] - u[..., 0:-4, 2:-2]
        vy_m = u[..., 2:-2, 2:-2] - u[..., 1:-3, 2:-2]
        vy_c = u[..., 3:-1, 2:-2] - u[..., 2:-2, 2:-2]
        vy_p = u[..., 4:, 2:-2] - u[..., 3:-1, 2:-2]
        
        uy_minus, _ = self._weno3_reconstruct(vy_mm, vy_m, vy_c)
        _, uy_plus = self._weno3_reconstruct(vy_m, vy_c, vy_p)
        
        grad_y = torch.maximum(F.relu(uy_minus), F.relu(-uy_plus))
        
        return grad_x, grad_y


# Loss Components
class LogDistanceLoss(nn.Module):
    """Log-space L1 loss for distance regression."""

    def __init__(self, mask_unreachable: bool = True) -> None:
        super().__init__()
        self.mask_unreachable = mask_unreachable

    def forward(
        self,
        pred_dist: torch.Tensor,
        gt_dist: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        log_pred = torch.log1p(pred_dist)
        log_gt = torch.log1p(gt_dist.clamp_min(0.0))
        per_pixel = F.smooth_l1_loss(log_pred, log_gt, reduction="none")
        
        mask = None
        if self.mask_unreachable and reachable is not None:
            mask = _get_reachable_mask(reachable, per_pixel.dtype)
        
        return _compute_masked_mean(per_pixel, mask)


class VectorFieldLoss(nn.Module):
    """Cosine similarity loss for vector field supervision."""

    def __init__(
        self,
        use_magnitude_weight: bool = True,
        magnitude_weight_clip: Optional[float] = 1.0,
    ) -> None:
        super().__init__()
        self.use_magnitude_weight = use_magnitude_weight
        self.magnitude_weight_clip = magnitude_weight_clip

    def forward(
        self,
        pred_vx: torch.Tensor,
        pred_vy: torch.Tensor,
        gt_vx: torch.Tensor,
        gt_vy: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_nx, pred_ny, _ = _normalize_vector_field(pred_vx, pred_vy)
        gt_nx, gt_ny, gt_mag = _normalize_vector_field(gt_vx, gt_vy)
        
        cos_sim = (pred_nx * gt_nx + pred_ny * gt_ny).clamp(-1.0, 1.0)
        loss_map = 1.0 - cos_sim
        
        mask = _get_reachable_mask(reachable, loss_map.dtype)
        weights = None
        if self.use_magnitude_weight:
            weights = gt_mag
            if self.magnitude_weight_clip is not None:
                weights = weights.clamp_max(self.magnitude_weight_clip)
        
        return _compute_masked_mean(loss_map, mask, weights)


class GradientDirectionConsistencyLoss(nn.Module):
    """Enforce V ≈ -∇D consistency between predicted vector field and distance gradient. """

    def __init__(
        self,
        gradient_threshold_ratio: float = 0.1,
        gradient_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.grad = gradient_module or SobelGradient()
        self.threshold_ratio = gradient_threshold_ratio

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_vx: torch.Tensor,
        pred_vy: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute gradient of predicted distance
        grad_x, grad_y = self.grad(pred_dist)
        target_vx, target_vy = -grad_x, -grad_y
        
        # Normalize both vector fields
        tnx, tny, tmag = _normalize_vector_field(target_vx, target_vy)
        pnx, pny, _ = _normalize_vector_field(pred_vx, pred_vy)
        
        # Cosine similarity loss: L = 1 - cos(θ)
        cos_sim = (tnx * pnx + tny * pny).clamp(-1.0, 1.0)
        loss_map = 1.0 - cos_sim
        
        # Scale-aware threshold
        # Expected gradient magnitude for normalized distance: 1/diagonal
        # Mask regions where gradient is too small (singularities, boundaries)
        H, W = pred_dist.shape[-2:]
        expected_grad_mag = 1.0 / _grid_diagonal_from_shape(H, W)
        adaptive_threshold = self.threshold_ratio * expected_grad_mag
        
        valid_mask = (tmag > adaptive_threshold).to(loss_map.dtype)
        
        if reachable is not None:
            reach_mask = _get_reachable_mask(reachable, loss_map.dtype)
            if reach_mask is not None:
                valid_mask = valid_mask * reach_mask
        
        return _compute_masked_mean(loss_map, valid_mask)

class EikonalLoss(nn.Module):
    """Eikonal regularization: |∇D| ≈ 1/diagonal."""

    def __init__(
        self,
        gradient_module: Optional[nn.Module] = None,
        min_valid_ratio: float = 0.3,
        max_valid_ratio: float = 5.0,
        use_smooth_l1: bool = True,
        smooth_l1_beta: float = 0.01,
    ) -> None:
        super().__init__()
        self.grad = gradient_module or WENO3Gradient()
        self.min_valid_ratio = min_valid_ratio
        self.max_valid_ratio = max_valid_ratio
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta

    def forward(
        self,
        pred_dist: torch.Tensor,
        reachable: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gx, gy = self.grad(pred_dist)
        gmag = _compute_vector_magnitude(gx, gy)
        
        H, W = pred_dist.shape[-2:]
        target = 1.0 / _grid_diagonal_from_shape(H, W)
        
        residual = gmag - target
        if self.use_smooth_l1:
            per_pixel = F.smooth_l1_loss(
                residual, torch.zeros_like(residual),
                reduction="none", beta=self.smooth_l1_beta
            )
        else:
            per_pixel = residual.abs()
        
        # Validity mask
        mask = _get_reachable_mask(reachable, per_pixel.dtype)
        valid = (gmag >= self.min_valid_ratio * target) & (gmag <= self.max_valid_ratio * target)
        valid = valid.to(per_pixel.dtype)
        
        final_mask = valid if mask is None else mask * valid
        return _compute_masked_mean(per_pixel, final_mask)


# Uncertainty Weighting (Kendall et al., CVPR 2018)
class UncertaintyWeightedLoss(nn.Module):
    """ Homoscedastic uncertainty weighting for multi-task learning. """
    
    def __init__(self, num_tasks: int) -> None:
        super().__init__()
        # log_var = log(σ²), initialized to 0 → σ = 1
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(
        self, 
        losses: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            losses: List of scalar loss tensors
            
        Returns:
            total_loss: Weighted sum with learned weights
            sigmas: Dictionary of σ values for logging
        """
        assert len(losses) == len(self.log_vars)
        
        total = torch.tensor(0.0, device=losses[0].device)
        sigmas = {}
        
        for i, loss in enumerate(losses):
            # precision = 1/σ² = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])
            # L_i = (1/2σ²) * loss + log(σ) = 0.5 * precision * loss + 0.5 * log_var
            weighted = 0.5 * precision * loss + 0.5 * self.log_vars[i]
            total = total + weighted
            # σ = exp(0.5 * log_var)
            sigmas[f"sigma_{i}"] = math.exp(0.5 * self.log_vars[i].item())
        
        return total, sigmas


# Combined Loss
class CombinedGeodesicLoss(nn.Module):
    """Combined geodesic supervision loss with optional uncertainty weighting."""

    def __init__(
        self,
        # Base weights (used when uncertainty weighting is disabled)
        dist_weight: float = 1.0,
        vec_weight: float = 1.0,
        consistency_weight: float = 0.5,
        eikonal_weight: float = 0.01,
        # Feature flags
        use_uncertainty_weighting: bool = False,
        use_weno_gradient: bool = True,
        # Component configs (scale-aware)
        consistency_threshold_ratio: float = 0.1,
        eikonal_min_valid_ratio: float = 0.3,
        eikonal_max_valid_ratio: float = 5.0,
    ) -> None:
        super().__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Fixed weights (fallback)
        self.dist_weight = dist_weight
        self.vec_weight = vec_weight
        self.consistency_weight = consistency_weight
        self.eikonal_weight = eikonal_weight
        
        # Gradient modules
        self._sobel = SobelGradient()
        self._eikonal_grad = WENO3Gradient() if use_weno_gradient else UpwindGradient()
        
        # Loss components
        self._dist_loss = LogDistanceLoss()
        self._vec_loss = VectorFieldLoss()
        self._cons_loss = GradientDirectionConsistencyLoss(
            gradient_threshold_ratio=consistency_threshold_ratio,
            gradient_module=self._sobel,
        )
        self._eik_loss = EikonalLoss(
            gradient_module=self._eikonal_grad,
            min_valid_ratio=eikonal_min_valid_ratio,
            max_valid_ratio=eikonal_max_valid_ratio,
        )
        
        # Uncertainty weighting
        if self.use_uncertainty_weighting:
            self._uncertainty = UncertaintyWeightedLoss(num_tasks=4)

    def forward(
        self,
        pred_dist: Optional[torch.Tensor],
        pred_vx: Optional[torch.Tensor],
        pred_vy: Optional[torch.Tensor],
        gt_dist: Optional[torch.Tensor],
        gt_vx: Optional[torch.Tensor],
        gt_vy: Optional[torch.Tensor],
        reachable: Optional[torch.Tensor],
        current_epoch: int = 0,  # Kept for API compatibility
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        device = pred_dist.device if pred_dist is not None else torch.device("cpu")
        zero = torch.tensor(0.0, device=device)
        
        # Compute individual losses
        l_dist = self._dist_loss(pred_dist, gt_dist, reachable) \
            if pred_dist is not None and gt_dist is not None else zero
        
        l_vec = self._vec_loss(pred_vx, pred_vy, gt_vx, gt_vy, reachable) \
            if all(t is not None for t in [pred_vx, pred_vy, gt_vx, gt_vy]) else zero
        
        l_cons = self._cons_loss(pred_dist, pred_vx, pred_vy, reachable) \
            if all(t is not None for t in [pred_dist, pred_vx, pred_vy]) else zero
        
        l_eik = self._eik_loss(pred_dist, reachable) \
            if pred_dist is not None else zero
        
        # Aggregate
        log: Dict[str, torch.Tensor] = {
            "loss/dist": l_dist.detach(),
            "loss/vec": l_vec.detach(),
            "loss/cons": l_cons.detach(),
            "loss/eik": l_eik.detach(),
        }
        
        if self.use_uncertainty_weighting:
            total, sigmas = self._uncertainty([l_dist, l_vec, l_cons, l_eik])
            log["loss/geo_total"] = total.detach()
            for name, sigma in sigmas.items():
                log[f"meta/{name}"] = _create_scalar_tensor(sigma, device)
        else:
            total = (
                self.dist_weight * l_dist
                + self.vec_weight * l_vec
                + self.consistency_weight * l_cons
                + self.eikonal_weight * l_eik
            )
            log["loss/geo_total"] = total.detach()
        
        return total, log
