# src/neural_astar/planner/encoder.py
from __future__ import annotations

from typing import Optional, Dict, Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .geodesic_attention import GeodesicAttentionBlock, LocalGeodesicAttention


class EncoderBase(nn.Module):
    """Base encoder class for Neural A*."""

    def __init__(
        self,
        input_dim: int,
        encoder_depth: int = 4,
        const: float = None,
        **_: Any,
    ):
        super().__init__()
        self.output_channels = 1
        self.model = self.construct_encoder(input_dim, encoder_depth)
        if const is not None:
            self.const = nn.Parameter(torch.tensor([const], dtype=torch.float32))
        else:
            self.register_buffer("const", torch.tensor(1.0, dtype=torch.float32))

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        raw_output = self.model(x)
        return torch.sigmoid(raw_output) * self.const

    def get_gate_map(self):
        return None


class Unet(EncoderBase):
    """U-Net based encoder."""

    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def construct_encoder(self, input_dim: int, encoder_depth: int) -> nn.Module:
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
        channels = [input_dim] + self.CHANNELS[:encoder_depth] + [self.output_channels]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool2d((2, 2)))
        return nn.Sequential(*blocks[:-2])


class GeoAttentionUnet(nn.Module):
    """
    U-Net with geodesic attention on skips and bottleneck.
    """

    DECODER_CHANNELS = [256, 128, 64, 32, 16]

    def __init__(
        self,
        input_dim: int = 3,
        encoder_depth: int = 4,
        const: float = None,
        num_attention_blocks: int = 2,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        backbone: str = "vgg16_bn",
        predict_vector_field: bool = True,
        num_anchors: int = 8,
        init_beta: float = 0.5,
        use_geodesic_pe: bool = True,
        use_local_attention: bool = False,
        local_window_size: int = 7,
        **_: Any,
    ) -> None:
        super().__init__()

        self.output_channels = 1
        self.num_anchors = num_anchors
        self.predict_vector_field = predict_vector_field
        self._last_attn_weights = []
        self._last_vector_field: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._last_betas = []

        decoder_channels = self.DECODER_CHANNELS[:encoder_depth]
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights=None,
            classes=self.output_channels,
            in_channels=input_dim,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
        )

        encoder_channels = list(self.unet.encoder.out_channels)
        skip_channels = encoder_channels[1:-1]
        self.skip_attn_blocks = nn.ModuleList(
            [
                GeodesicAttentionBlock(
                    embed_dim=ch,
                    num_anchors=num_anchors,
                    num_heads=max(1, ch // 64),
                    dropout=attention_dropout,
                    init_beta=init_beta,
                    use_geodesic_pe=use_geodesic_pe,
                )
                for ch in skip_channels
            ]
        )

        bottleneck_channels = encoder_channels[-1]
        if use_local_attention:
            self.attn_blocks = nn.ModuleList(
                [
                    LocalGeodesicAttention(
                        embed_dim=bottleneck_channels,
                        num_anchors=num_anchors,
                        num_heads=attention_heads,
                        window_size=local_window_size,
                        dropout=attention_dropout,
                        init_beta=init_beta,
                    )
                    for _ in range(num_attention_blocks)
                ]
            )
        else:
            self.attn_blocks = nn.ModuleList(
                [
                    GeodesicAttentionBlock(
                        embed_dim=bottleneck_channels,
                        num_anchors=num_anchors,
                        num_heads=attention_heads,
                        dropout=attention_dropout,
                        init_beta=init_beta,
                        use_geodesic_pe=use_geodesic_pe,
                    )
                    for _ in range(num_attention_blocks)
                ]
            )

        if self.predict_vector_field:
            final_channels = decoder_channels[-1]
            self.vector_head = nn.Sequential(
                nn.Conv2d(final_channels, final_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_channels, 2, kernel_size=1),
            )
            nn.init.zeros_(self.vector_head[-1].weight)
            nn.init.zeros_(self.vector_head[-1].bias)

        if const is not None:
            self.const = nn.Parameter(torch.tensor([const], dtype=torch.float32))
        else:
            self.register_buffer("const", torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self._last_attn_weights = []
        self._last_vector_field = None
        self._last_betas = []

        input_spatial = x.shape[-2:]
        encoder_feats = list(self.unet.encoder(x))

        for idx, block in enumerate(self.skip_attn_blocks, start=1):
            feat = encoder_feats[idx]
            feat_h, feat_w = feat.shape[-2:]
            phi_ds = None
            if phi is not None:
                phi_ds = F.interpolate(
                    phi, size=(feat_h, feat_w), mode="bilinear", align_corners=False
                )
            mask_ds = None
            if spatial_mask is not None:
                mask_ds = (
                    F.interpolate(
                        spatial_mask.float(),
                        size=(feat_h, feat_w),
                        mode="nearest",
                    )
                    > 0.5
                )
            encoder_feats[idx] = block(feat, phi_ds, mask_ds)
            attn = block.get_attention_weights()
            if attn is not None:
                self._last_attn_weights.append(attn)
            if hasattr(block, "get_beta"):
                self._last_betas.append(block.get_beta())

        bottleneck = encoder_feats[-1]
        bn_h, bn_w = bottleneck.shape[-2:]
        phi_bn = (
            F.interpolate(phi, size=(bn_h, bn_w), mode="bilinear", align_corners=False)
            if phi is not None
            else None
        )
        mask_bn = None
        if spatial_mask is not None:
            mask_bn = (
                F.interpolate(
                    spatial_mask.float(), size=(bn_h, bn_w), mode="nearest"
                )
                > 0.5
            )

        for block in self.attn_blocks:
            bottleneck = block(bottleneck, phi_bn, mask_bn)
            attn = block.get_attention_weights()
            if attn is not None:
                self._last_attn_weights.append(attn)
            if hasattr(block, "get_beta"):
                self._last_betas.append(block.get_beta())

        encoder_feats[-1] = bottleneck
        decoder_output = self.unet.decoder(*encoder_feats)

        if self.predict_vector_field:
            vec_field = self.vector_head(decoder_output)
            vec_norm = torch.norm(vec_field, dim=1, keepdim=True) + 1e-8
            vec_unit = vec_field / vec_norm
            if vec_unit.shape[-2:] != input_spatial:
                vec_unit = F.interpolate(
                    vec_unit, size=input_spatial, mode="bilinear", align_corners=False
                )
            self._last_vector_field = (vec_unit[:, 0:1], vec_unit[:, 1:2])

        logits = self.unet.segmentation_head(decoder_output)
        if logits.shape[-2:] != input_spatial:
            logits = F.interpolate(
                logits, size=input_spatial, mode="bilinear", align_corners=False
            )

        return torch.sigmoid(logits) * self.const

    def get_attention_weights(self):
        return self._last_attn_weights

    def get_vector_field(self):
        return self._last_vector_field

    def get_betas(self):
        return self._last_betas


class GeoAttentionCNN(nn.Module):
    """CNN backbone with geodesic attention."""

    CHANNELS = [32, 64, 128, 256]

    def __init__(
        self,
        input_dim: int = 3,
        encoder_depth: int = 4,
        const: float = None,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        predict_vector_field: bool = True,
        num_anchors: int = 8,
        init_beta: float = 0.5,
        use_geodesic_pe: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        self.output_channels = 1
        self.num_anchors = num_anchors
        self.predict_vector_field = predict_vector_field
        self._last_attn_weights = []
        self._last_vector_field: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        channels = [input_dim] + self.CHANNELS[:encoder_depth]
        blocks = []
        for i in range(len(channels) - 1):
            blocks.append(nn.Conv2d(channels[i], channels[i + 1], 3, 1, 1))
            blocks.append(nn.BatchNorm2d(channels[i + 1]))
            blocks.append(nn.ReLU())
            if i < len(channels) - 2:
                blocks.append(nn.MaxPool2d(kernel_size=2))
        self.stem = nn.Sequential(*blocks)

        bottleneck_channels = channels[-1]
        self.attn_block = GeodesicAttentionBlock(
            embed_dim=bottleneck_channels,
            num_anchors=num_anchors,
            num_heads=attention_heads,
            dropout=attention_dropout,
            init_beta=init_beta,
            use_geodesic_pe=use_geodesic_pe,
        )
        self.head = nn.Conv2d(bottleneck_channels, self.output_channels, 3, 1, 1)
        if self.predict_vector_field:
            self.vector_head = nn.Sequential(
                nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channels, 2, 1),
            )
            nn.init.zeros_(self.vector_head[-1].weight)
            nn.init.zeros_(self.vector_head[-1].bias)

        if const is not None:
            self.const = nn.Parameter(torch.tensor([const], dtype=torch.float32))
        else:
            self.register_buffer("const", torch.tensor(1.0, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        phi: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_spatial = x.shape[-2:]
        self._last_attn_weights = []
        self._last_vector_field = None

        feats = self.stem(x)
        feat_h, feat_w = feats.shape[-2:]

        phi_ds = (
            F.interpolate(phi, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
            if phi is not None
            else None
        )
        mask_ds = None
        if spatial_mask is not None:
            mask_ds = (
                F.interpolate(
                    spatial_mask.float(), size=(feat_h, feat_w), mode="nearest"
                )
                > 0.5
            )

        feats = self.attn_block(feats, phi_ds, mask_ds)
        attn = self.attn_block.get_attention_weights()
        if attn is not None:
            self._last_attn_weights.append(attn)

        if self.predict_vector_field:
            vec_field = self.vector_head(feats)
            vec_norm = torch.norm(vec_field, dim=1, keepdim=True) + 1e-8
            vec_unit = vec_field / vec_norm
            if vec_unit.shape[-2:] != input_spatial:
                vec_unit = F.interpolate(
                    vec_unit, size=input_spatial, mode="bilinear", align_corners=False
                )
            self._last_vector_field = (vec_unit[:, 0:1], vec_unit[:, 1:2])

        logits = self.head(feats)
        if logits.shape[-2:] != input_spatial:
            logits = F.interpolate(
                logits, size=input_spatial, mode="bilinear", align_corners=False
            )
        return torch.sigmoid(logits) * self.const

    def get_attention_weights(self):
        return self._last_attn_weights

    def get_vector_field(self):
        return self._last_vector_field

    def get_betas(self):
        if hasattr(self.attn_block, "get_beta"):
            return [self.attn_block.get_beta()]
        return []


class GeoAttentionEncoderWrapper(nn.Module):
    """
    Wrapper to handle concatenated phi channels in a single tensor.
    """

    def __init__(self, encoder: nn.Module, num_anchors: int = 8) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_anchors = num_anchors

    def forward(
        self,
        x: torch.Tensor,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_input = x[:, :3]
        phi = x[:, 3 : 3 + self.num_anchors]
        return self.encoder(base_input, phi, spatial_mask)

    def get_attention_weights(self):
        return self.encoder.get_attention_weights()

    def get_vector_field(self):
        return self.encoder.get_vector_field()

    def get_betas(self):
        if hasattr(self.encoder, "get_betas"):
            return self.encoder.get_betas()
        return []
