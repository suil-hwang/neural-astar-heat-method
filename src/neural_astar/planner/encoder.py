# src/neural_astar/planner/encoder.py
from __future__ import annotations

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




class DirectGeoUnet(nn.Module):
    """
    U-Net encoder without attention for direct geodesic supervision.

    Predicts:
      - cost map: geodesic distance approximation
      - vector field: direction toward goal (unit vectors)
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
                f"DirectGeoUnet requires 1 <= encoder_depth <= {len(self.DECODER_CHANNELS)}, "
                f"got {encoder_depth}."
            )
        self.output_channels = 1
        self.predict_vector_field = predict_vector_field
        self._last_vector_field = None

        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights=None,
            classes=self.output_channels,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

        last_dec_ch = decoder_channels[-1] if decoder_channels else 16
        self.cost_head = nn.Sequential(
            nn.Conv2d(last_dec_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

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
        input_spatial = x.shape[-2:]
        self._last_vector_field = None

        encoder_feats = list(self.unet.encoder(x))
        decoder_output = self.unet.decoder(*encoder_feats)

        if self.predict_vector_field:
            vec_field = self.vec_head(decoder_output)
            vec_norm = torch.norm(vec_field, dim=1, keepdim=True) + 1e-8
            vec_unit = vec_field / vec_norm
            if vec_unit.shape[-2:] != input_spatial:
                vec_unit = F.interpolate(
                    vec_unit, size=input_spatial, mode="bilinear", align_corners=False
                )
            self._last_vector_field = (vec_unit[:, 0:1], vec_unit[:, 1:2])

        cost_logits = self.cost_head(decoder_output)
        if cost_logits.shape[-2:] != input_spatial:
            cost_logits = F.interpolate(
                cost_logits, size=input_spatial, mode="bilinear", align_corners=False
            )
        return torch.sigmoid(cost_logits) * self.const

    def get_vector_field(self):
        return self._last_vector_field


class DirectGeoCNN(nn.Module):
    """
    Simple CNN encoder without attention for direct geodesic supervision.
    """

    CHANNELS = [32, 64, 128, 256]

    def __init__(
        self,
        input_dim: int = 3,
        encoder_depth: int = 4,
        const: float = None,
        predict_vector_field: bool = True,
    ) -> None:
        super().__init__()
        self.output_channels = 1
        self.predict_vector_field = predict_vector_field
        self._last_vector_field = None

        channels = [input_dim] + self.CHANNELS[:encoder_depth]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
            if i < len(channels) - 2:
                blocks.append(nn.MaxPool2d(kernel_size=2))
        self.stem = nn.Sequential(*blocks)

        bottleneck_channels = channels[-1] if channels else 32
        self.cost_head = nn.Conv2d(
            bottleneck_channels, self.output_channels, 3, 1, 1
        )

        if self.predict_vector_field:
            self.vec_head = nn.Sequential(
                nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channels, 2, 1),
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
        input_spatial = x.shape[-2:]
        self._last_vector_field = None
        feats = self.stem(x)

        if self.predict_vector_field:
            vec_field = self.vec_head(feats)
            vec_norm = torch.norm(vec_field, dim=1, keepdim=True) + 1e-8
            vec_unit = vec_field / vec_norm
            if vec_unit.shape[-2:] != input_spatial:
                vec_unit = F.interpolate(
                    vec_unit, size=input_spatial, mode="bilinear", align_corners=False
                )
            self._last_vector_field = (vec_unit[:, 0:1], vec_unit[:, 1:2])

        logits = self.cost_head(feats)
        if logits.shape[-2:] != input_spatial:
            logits = F.interpolate(
                logits, size=input_spatial, mode="bilinear", align_corners=False
            )
        return torch.sigmoid(logits) * self.const

    def get_vector_field(self):
        return self._last_vector_field
