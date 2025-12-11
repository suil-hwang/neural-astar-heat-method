# src/neural_astar/planner/geodesic_attention.py
"""
Geodesic-aware attention with navigation geodesic priors.

Improvements:
- Navigation geodesic positional encoding from multi-anchor embedding phi
- Pairwise geodesic kernel bias inside attention logits
- Learnable beta per head to balance semantic vs geodesic cues
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NavigationGeodesicPE(nn.Module):
    """Project multi-anchor geodesic embedding into feature space."""

    def __init__(
        self,
        num_anchors: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Linear(num_anchors, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        phi: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map
            phi: [B, M, H, W] multi-anchor embedding
        Returns:
            [B, C, H, W] with geodesic positional encoding applied
        """
        batch, _, height, width = x.shape
        phi_flat = phi.flatten(2).transpose(1, 2)
        geo_pe = self.proj(phi_flat)
        geo_pe = geo_pe.transpose(1, 2).reshape(batch, self.embed_dim, height, width)
        return x + geo_pe


class PairwiseGeodesicKernel(nn.Module):
    """Compute log geodesic kernel matrix from embeddings."""

    def __init__(
        self,
        num_anchors: int,
        init_alpha: float = 1.0,
        learnable_alpha: bool = True,
    ) -> None:
        super().__init__()
        self.num_anchors = num_anchors
        self.normalizer = math.sqrt(2 * num_anchors)

        if learnable_alpha:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        else:
            self.register_buffer("log_alpha", torch.tensor(math.log(init_alpha)))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi: [B, M, H, W] or [B, N, M]
        Returns:
            log_k_geo: [B, N, N]
        """
        if phi.dim() == 4:
            batch, num_anchors, height, width = phi.shape
            phi = phi.flatten(2).transpose(1, 2)

        batch, num_nodes, _ = phi.shape
        phi_sq = (phi ** 2).sum(dim=-1)
        pairwise_sq = (
            phi_sq.unsqueeze(2)
            + phi_sq.unsqueeze(1)
            - 2 * torch.bmm(phi, phi.transpose(1, 2))
        ).clamp(min=0)
        log_k_geo = -self.alpha * pairwise_sq / self.normalizer
        return log_k_geo


class GeodesicAwareAttention(nn.Module):
    """
    Self-attention that combines semantic similarity with geodesic proximity.

    Attention logits:
        L_ij = (q_i^T k_j)/sqrt(d) + beta * log(k_geo(i,j)) + mask
    """

    def __init__(
        self,
        embed_dim: int,
        num_anchors: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        qkv_bias: bool = True,
        init_beta: float = 0.5,
        learnable_beta: bool = True,
        use_geodesic_pe: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.num_anchors = num_anchors

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.use_geodesic_pe = use_geodesic_pe
        if use_geodesic_pe:
            self.geo_pe = NavigationGeodesicPE(num_anchors, embed_dim, dropout)

        self.geo_kernel = PairwiseGeodesicKernel(
            num_anchors, init_alpha=1.0, learnable_alpha=True
        )

        if learnable_beta:
            self.beta = nn.Parameter(torch.full((num_heads,), init_beta))
        else:
            self.register_buffer("beta", torch.full((num_heads,), init_beta))

        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights: Optional[torch.Tensor] = None
        self._last_geo_bias: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            phi: [B, M, H, W] geodesic embedding
            spatial_mask: [B, N, N] or [B, 1, H, W] (True = keep)
        """
        batch, channels, height, width = x.shape
        num_tokens = height * width

        if self.use_geodesic_pe and phi is not None:
            x = self.geo_pe(x, phi)

        x_flat = x.flatten(2).transpose(1, 2)
        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        q = q.reshape(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if phi is not None:
            if phi.shape[-2:] != (height, width):
                phi = F.interpolate(phi, size=(height, width), mode="bilinear", align_corners=False)
            log_k_geo = self.geo_kernel(phi)
            self._last_geo_bias = log_k_geo

            beta = self.beta.view(1, self.num_heads, 1, 1)
            geo_bias = beta * log_k_geo.unsqueeze(1)
            attn_logits = attn_logits + geo_bias

        if spatial_mask is not None:
            if spatial_mask.dim() == 4:
                mask_flat = spatial_mask.flatten(2).squeeze(1)
                mask_matrix = mask_flat.unsqueeze(1) & mask_flat.unsqueeze(2)
                spatial_mask = mask_matrix
            mask = spatial_mask.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self._last_attn_weights = attn_weights

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch, num_tokens, channels)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(batch, channels, height, width)
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self._last_attn_weights

    def get_mean_attention_weights(self) -> Optional[torch.Tensor]:
        if self._last_attn_weights is None:
            return None
        return self._last_attn_weights.mean(dim=1)

    def get_geodesic_bias(self) -> Optional[torch.Tensor]:
        return self._last_geo_bias

    def get_beta(self) -> torch.Tensor:
        return self.beta.detach()


class GeodesicAttentionBlock(nn.Module):
    """Transformer-style block with geodesic-aware attention."""

    def __init__(
        self,
        embed_dim: int,
        num_anchors: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        init_beta: float = 0.5,
        use_geodesic_pe: bool = True,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = GeodesicAwareAttention(
            embed_dim=embed_dim,
            num_anchors=num_anchors,
            num_heads=num_heads,
            dropout=dropout,
            init_beta=init_beta,
            use_geodesic_pe=use_geodesic_pe,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, channels, height, width = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(batch, channels, height, width)

        attn_out = self.attn(x_norm, phi, spatial_mask)
        x = x + attn_out

        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        x = x_flat.transpose(1, 2).reshape(batch, channels, height, width)
        return x

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self.attn.get_attention_weights()

    def get_beta(self) -> torch.Tensor:
        return self.attn.get_beta()


class LocalGeodesicAttention(nn.Module):
    """
    Local window attention with geodesic bias (NATTEN-style).
    """

    def __init__(
        self,
        embed_dim: int,
        num_anchors: int,
        num_heads: int = 4,
        window_size: int = 7,
        dropout: float = 0.1,
        init_beta: float = 0.5,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.num_anchors = num_anchors

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.geo_kernel = PairwiseGeodesicKernel(num_anchors)
        self.beta = nn.Parameter(torch.full((num_heads,), init_beta))

        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, channels, height, width = x.shape
        window_size = self.window_size
        pad = window_size // 2

        x_pad = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0)

        if phi is not None:
            if phi.shape[-2:] != (height, width):
                phi = F.interpolate(phi, size=(height, width), mode="bilinear", align_corners=False)
            phi_pad = F.pad(phi, (pad, pad, pad, pad), mode="constant", value=1.0)

        x_flat = x.flatten(2).transpose(1, 2)
        num_tokens = height * width

        q = self.q_proj(x_flat).reshape(batch, num_tokens, self.num_heads, self.head_dim)

        x_unfold = x_pad.unfold(2, window_size, 1).unfold(3, window_size, 1)
        x_unfold = x_unfold.permute(0, 2, 3, 4, 5, 1).reshape(
            batch, num_tokens, window_size * window_size, channels
        )

        k = self.k_proj(x_unfold).reshape(
            batch, num_tokens, window_size * window_size, self.num_heads, self.head_dim
        )
        v = self.v_proj(x_unfold).reshape(
            batch, num_tokens, window_size * window_size, self.num_heads, self.head_dim
        )

        q = q.unsqueeze(2)
        attn_logits = (q * k).sum(dim=-1) * self.scale
        attn_logits = attn_logits.permute(0, 3, 1, 2)

        if phi is not None:
            phi_unfold = phi_pad.unfold(2, window_size, 1).unfold(3, window_size, 1)
            phi_unfold = phi_unfold.permute(0, 2, 3, 4, 5, 1).reshape(
                batch, num_tokens, window_size * window_size, self.num_anchors
            )
            phi_center = phi.flatten(2).transpose(1, 2).unsqueeze(2)
            diff_sq = ((phi_center - phi_unfold) ** 2).sum(dim=-1)
            log_k_geo = -self.geo_kernel.alpha * diff_sq / self.geo_kernel.normalizer

            beta = self.beta.view(1, self.num_heads, 1, 1)
            geo_bias = beta * log_k_geo.unsqueeze(1)
            attn_logits = attn_logits + geo_bias

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self._last_attn_weights = attn_weights

        v = v.permute(0, 3, 1, 2, 4)
        out = (attn_weights.unsqueeze(-1) * v).sum(dim=-2)
        out = out.permute(0, 2, 1, 3).reshape(batch, num_tokens, channels)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(batch, channels, height, width)
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self._last_attn_weights

