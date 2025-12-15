from __future__ import annotations

import re
from dataclasses import dataclass
from glob import glob

import torch

# Map detected input channel count to NeuralAstar encoder_input strings.
# - len("m+") == 2 : map + (start + goal)
# - len("msg") == 3: map, start, goal
# - "msgdr"/"mgsvxyr" are kept for backward compatibility with older checkpoints.
ENCODER_INPUT_MAP: dict[int, str] = {
    2: "m+",
    3: "msg",
    5: "msgdr",
    7: "mgsvxyr",
}


@dataclass
class EncoderInfo:
    """Encoder configuration detected from a PyTorch Lightning checkpoint dir."""

    encoder_arch: str
    input_channels: int
    is_gated: bool
    encoder_depth: int


def load_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load the latest PL checkpoint state_dict and normalize key prefixes."""
    ckpt_files = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

    ckpt = torch.load(ckpt_files[-1], map_location="cpu", weights_only=True)
    raw_state = ckpt.get("state_dict", ckpt)

    # Remove leading "planner." (PyTorch Lightning module wrapper) for analysis
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        normalized_key = key.split("planner.", 1)[-1] if key.startswith("planner.") else key
        state[normalized_key] = value

    return state


def _detect_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """Detect encoder input channel count from state dict weights."""
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 4:
            continue
        if key.endswith("encoder.model.0.weight") or key.endswith("encoder.stem.0.weight"):
            return int(value.shape[1])

    # U-Net variants (segmentation_models_pytorch): scan encoder conv weights and take the minimum in_channels.
    # This works reliably for our supported encoders (Unet / MultiHeadGeoUnet).
    in_channels = []
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 4:
            continue
        if key.startswith("encoder.unet.encoder"):
            in_channels.append(int(value.shape[1]))
    if in_channels:
        return min(in_channels)

    return 3


def _detect_encoder_depth(state_dict: dict[str, torch.Tensor]) -> int:
    """Detect encoder depth from decoder block count (U-Net) or conv stage count (CNN)."""
    block_indices = []
    for key in state_dict.keys():
        # Match both patterns:
        # - Our model: encoder.unet.decoder.blocks.N
        # - Neural A* original: encoder.model.decoder.blocks.N
        match = re.search(r"decoder\.blocks\.(\d+)", key)
        if match:
            try:
                block_indices.append(int(match.group(1)))
            except ValueError:
                continue
    if block_indices:
        return max(block_indices) + 1

    # CNN: count conv weights in EncoderBase.model (depth == conv_count - 1).
    model_convs = [
        value
        for key, value in state_dict.items()
        if key.startswith("encoder.model.") and key.endswith(".weight") and getattr(value, "ndim", 0) >= 4
    ]
    if model_convs:
        return max(len(model_convs) - 1, 1)

    # Legacy CNN variants with an explicit stem.
    stem_convs = [
        value
        for key, value in state_dict.items()
        if key.startswith("encoder.stem.") and key.endswith(".weight") and getattr(value, "ndim", 0) >= 4
    ]
    if stem_convs:
        return len(stem_convs)

    return 4


def _detect_architecture(state_dict: dict[str, torch.Tensor], has_gated: bool) -> str:
    """Detect encoder architecture type from state dict."""
    del has_gated  # reserved for legacy checkpoints
    has_decoder = any("decoder" in key for key in state_dict.keys())
    has_geoattention = any(
        ("attn_block" in key) or ("attn_blocks" in key) for key in state_dict.keys()
    )
    has_cost_head = any("cost_head" in key for key in state_dict.keys())
    has_dist_head = any("dist_head" in key for key in state_dict.keys())
    has_vec_head = any("vec_head" in key for key in state_dict.keys())

    # Our geo-supervised model (MultiHeadGeoUnet). Legacy DirectGeoUnet checkpoints
    # can still be loaded into MultiHeadGeoUnet with strict=False (dist_head will be
    # randomly initialized in that case).
    if has_cost_head and (has_dist_head or has_vec_head):
        if has_decoder:
            return "MultiHeadGeoUnet"
        raise ValueError(
            "Legacy CNN-based geo encoders are no longer supported. "
            "Please retrain with encoder.arch=MultiHeadGeoUnet."
        )

    if has_geoattention:
        raise ValueError(
            "GeoAttention models are no longer supported in this project. "
            "This checkpoint contains GeoAttention keys (attn_block/attn_blocks)."
        )

    return "Unet" if has_decoder else "CNN"


def detect_encoder_type(checkpoint_path: str) -> EncoderInfo:
    """Detect encoder type, input channels, and depth from checkpoint dir."""
    state_dict = load_checkpoint_state_dict(checkpoint_path)

    # Detect Gated Fusion: encoder.fusion.feat_norm or encoder.fusion.feat_geo
    has_gated_fusion = any(
        "encoder.fusion.feat_norm" in key or "encoder.fusion.feat_geo" in key
        for key in state_dict.keys()
    )

    return EncoderInfo(
        encoder_arch=_detect_architecture(state_dict, has_gated_fusion),
        input_channels=_detect_input_channels(state_dict),
        is_gated=has_gated_fusion,
        encoder_depth=_detect_encoder_depth(state_dict),
    )

