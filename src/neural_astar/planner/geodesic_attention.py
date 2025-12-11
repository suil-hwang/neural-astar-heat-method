# src/neural_astar/planner/geodesic_attention.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeodesicAwareAttention(nn.Module):
    """Self-attention supervised to learn geodesic-like relationships."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.dropout = nn.Dropout(dropout)
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            spatial_mask: [B, N, N] or None (True = keep, False = mask)
        Returns:
            [B, C, H, W]
        """
        b, c, h, w = x.shape
        n = h * w

        x_flat = x.flatten(2).transpose(1, 2)  # [B, N, C]

        q = self.q_proj(x_flat)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        q = q.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,N,N]

        if spatial_mask is not None:
            mask = spatial_mask.unsqueeze(1)  # [B,1,N,N]
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        self._last_attn_weights = attn_weights

        out = torch.matmul(attn_weights, v)  # [B,H,N,D]
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self._last_attn_weights

    def get_mean_attention_weights(self) -> Optional[torch.Tensor]:
        if self._last_attn_weights is None:
            return None
        return self._last_attn_weights.mean(dim=1)


class GeodesicAttentionBlock(nn.Module):
    """Attention block with LayerNorm + FFN + residual."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = GeodesicAwareAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
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
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat)
        x_norm = x_norm.transpose(1, 2).reshape(b, c, h, w)

        attn_out = self.attn(x_norm, spatial_mask)
        x = x + attn_out

        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        x = x_flat.transpose(1, 2).reshape(b, c, h, w)
        return x

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        return self.attn.get_attention_weights()

