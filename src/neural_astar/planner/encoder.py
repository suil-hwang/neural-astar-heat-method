# src/neural_astar/planner/encoder.py
from __future__ import annotations

import inspect
from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBase(nn.Module):
    """Base encoder class for Neural A*."""
    
    def __init__(
        self, 
        input_dim: int, 
        encoder_depth: int = 4, 
        const: float = None
    ):
        """Initialize base encoder."""
        super().__init__()
        self.output_channels = 1
        self.model = self.construct_encoder(input_dim, encoder_depth)
        if const is not None:
            self.const = nn.Parameter(torch.tensor([const], dtype=torch.float32))
        else:
            self.register_buffer("const", torch.tensor(1.0, dtype=torch.float32))

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        """Override in subclass."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: output cost map."""
        raw_output = self.model(x)
        return torch.sigmoid(raw_output) * self.const
    
    def get_gate_map(self):
        """Return gate map (override in Gated encoders)."""
        return None


class Unet(EncoderBase):
    """U-Net based encoder."""
    
    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        """Build U-Net structure."""
        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        return smp.Unet(
            encoder_name="vgg16_bn",
            encoder_weights=None,
            classes=self.output_channels,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )


class CNN(EncoderBase):
    """Simple CNN encoder."""
    
    CHANNELS = [32, 64, 128, 256]

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        """Build CNN structure."""
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [self.output_channels]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
        return nn.Sequential(*blocks[:-1])


class CNNDownSize(CNN):
    """Simple CNN encoder with downsampling."""
    
    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        """Build downsampling CNN structure."""
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [self.output_channels]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d((2, 2)))
        return nn.Sequential(*blocks[:-2])



class MultiHeadGeoUnet(nn.Module):
    """
    Multi-head U-Net encoder for Neural A* with auxiliary geodesic supervision.

    Key idea:
      - Cost head (main task): produces the guidance cost map used by A*
      - Geo heads (aux task): predict geodesic distance + vector field for supervision

    This avoids conflicting gradients where the A* cost map is forced to match
    geodesic distance directly.
    """

    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def __init__(
        self,
        input_dim: int = 3,
        encoder_depth: int = 4,
        const: float = None,
        backbone: str = "vgg16_bn",
        predict_vector_field: bool = True,
    ) -> None:
        super().__init__()
        encoder_depth = int(encoder_depth)
        if encoder_depth < 1 or encoder_depth > len(self.DECODER_CHANNELS):
            raise ValueError(
                f"MultiHeadGeoUnet requires 1 <= encoder_depth <= {len(self.DECODER_CHANNELS)}, "
                f"got {encoder_depth}."
            )

        self.predict_vector_field = predict_vector_field
        self._last_geo_predictions: dict[str, torch.Tensor] | None = None

        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights=None,
            classes=1,  # not used (we use decoder features + custom heads)
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

        decoder_params = list(inspect.signature(self.unet.decoder.forward).parameters.values())
        self._decoder_accepts_varargs = bool(
            decoder_params and decoder_params[0].kind == inspect.Parameter.VAR_POSITIONAL
        )

        last_dec_ch = decoder_channels[-1] if decoder_channels else 16

        # Main head: cost map for A*
        self.cost_head = nn.Sequential(
            nn.Conv2d(last_dec_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Auxiliary head: geodesic distance (not used by A*)
        self.dist_head = nn.Sequential(
            nn.Conv2d(last_dec_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Auxiliary head: vector field (unit vectors toward goal)
        if self.predict_vector_field:
            self.vec_head = nn.Sequential(
                nn.Conv2d(last_dec_ch, last_dec_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(last_dec_ch, 2, 1),
            )
            nn.init.zeros_(self.vec_head[-1].weight)
            nn.init.zeros_(self.vec_head[-1].bias)

        if const is not None:
            self.const = nn.Parameter(torch.tensor([const], dtype=torch.float32))
        else:
            self.register_buffer("const", torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del spatial_mask  # reserved for compatibility
        input_spatial = x.shape[-2:]

        encoder_feats = self.unet.encoder(x)
        if self._decoder_accepts_varargs:
            decoder_output = self.unet.decoder(*encoder_feats)
        else:
            decoder_output = self.unet.decoder(encoder_feats)

        # Auxiliary predictions (cached for geo loss)
        dist_logits = self.dist_head(decoder_output)
        if dist_logits.shape[-2:] != input_spatial:
            dist_logits = F.interpolate(
                dist_logits, size=input_spatial, mode="bilinear", align_corners=False
            )
        pred_dist = F.softplus(dist_logits)

        pred_vx = pred_vy = None
        if self.predict_vector_field:
            vec_field = self.vec_head(decoder_output)
            vec_norm = torch.norm(vec_field, dim=1, keepdim=True).clamp_min(1e-8)
            vec_unit = vec_field / vec_norm
            if vec_unit.shape[-2:] != input_spatial:
                vec_unit = F.interpolate(
                    vec_unit, size=input_spatial, mode="bilinear", align_corners=False
                )
            pred_vx, pred_vy = vec_unit[:, 0:1], vec_unit[:, 1:2]

        self._last_geo_predictions = {"dist": pred_dist}
        if pred_vx is not None and pred_vy is not None:
            self._last_geo_predictions.update({"vx": pred_vx, "vy": pred_vy})

        # Main prediction: cost map for A* search
        cost_logits = self.cost_head(decoder_output)
        if cost_logits.shape[-2:] != input_spatial:
            cost_logits = F.interpolate(
                cost_logits, size=input_spatial, mode="bilinear", align_corners=False
            )
        return torch.sigmoid(cost_logits) * self.const

    def get_geo_predictions(self) -> dict[str, torch.Tensor] | None:
        return self._last_geo_predictions

    def get_vector_field(self):
        if not self._last_geo_predictions:
            return None
        vx = self._last_geo_predictions.get("vx")
        vy = self._last_geo_predictions.get("vy")
        if vx is None or vy is None:
            return None
        return vx, vy
