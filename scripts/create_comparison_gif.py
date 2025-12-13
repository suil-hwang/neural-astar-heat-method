# scripts/create_comparison_gif.py
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from glob import glob
from typing import Any

import hydra
import numpy as np
import torch
from moviepy import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

# Prefer local src/ over any globally installed neural_astar
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_from_ptl_checkpoint

# Filter specific warnings instead of suppressing all
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_float32_matmul_precision("high")



# Constants
MODEL_DISPLAY_NAMES = {
    "vanilla": "A*",
    "neural_astar": "Neural A*",
    "ours": "Ours (Multi-Head)",
}
ENCODER_INPUT_MAP = {3: "msg", 2: "m+"}
DEFAULT_FPS = 15
PAUSE_FRAMES = 15
INCLUDE_INFER_TIME = False


# Data Classes
@dataclass
class EncoderInfo:
    """Encoder configuration detected from checkpoint."""
    encoder_arch: str
    input_channels: int
    is_gated: bool
    encoder_depth: int


@dataclass
class ModelOutput:
    """Container for model inference results."""
    outputs: Any
    time_ms: float
    display_name: str


def parse_include_infer_time_flag(argv: list[str]) -> bool:
    """Parse CLI flag to control inference time display (default: hidden)."""
    parser = argparse.ArgumentParser(
        add_help=False, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--include-infer-time",
        dest="include_infer_time",
        action="store_true",
        help="Show inference time in GIF headers (default: hidden)",
    )
    parser.add_argument(
        "--no-include-infer-time",
        dest="include_infer_time",
        action="store_false",
        help="Hide inference time in GIF headers",
    )
    parser.set_defaults(include_infer_time=False)
    args, remaining = parser.parse_known_args(argv)
    sys.argv = [sys.argv[0]] + remaining
    return args.include_infer_time


# Checkpoint Analysis Functions
def _load_checkpoint_state_dict(checkpoint_path: str) -> dict[str, torch.Tensor]:
    """Load state dict from the latest checkpoint in the directory."""
    ckpt_files = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

    ckpt = torch.load(ckpt_files[-1], map_location="cpu", weights_only=True)
    raw_state = ckpt.get("state_dict", ckpt)

    # Remove leading "planner." prefix for analysis
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        normalized_key = key.split("planner.", 1)[-1] if key.startswith("planner.") else key
        state[normalized_key] = value

    return state


def _detect_input_channels(state_dict: dict[str, torch.Tensor]) -> int:
    """Detect input channel count from state dict weights."""
    for key, value in state_dict.items():
        # Standard encoder
        if "planner.encoder.model.0.weight" in key or "encoder.model.0.weight" in key:
            return value.shape[1]
        # CNN variants with stem
        if "encoder.stem.0.weight" in key:
            return value.shape[1]
        # U-Net variants
        if "encoder.unet.encoder" in key and value.ndim >= 4:
            return value.shape[1]
    return 3


def _detect_encoder_depth(state_dict: dict[str, torch.Tensor]) -> int:
    """Detect encoder depth from decoder block count."""
    block_indices = []
    for key, value in state_dict.items():
        match = re.search(r"decoder\.blocks\.(\d+)", key)
        if match:
            try:
                block_indices.append(int(match.group(1)))
            except ValueError:
                continue

    if block_indices:
        return max(block_indices) + 1

    stem_convs = [
        key
        for key, value in state_dict.items()
        if "stem" in key and key.endswith("weight") and getattr(value, "ndim", 0) >= 4
    ]
    if stem_convs:
        return len(stem_convs)

    return 4


def _detect_architecture(state_dict: dict[str, torch.Tensor], has_gated: bool) -> str:
    """Detect encoder architecture type from state dict."""
    has_unet = any("decoder" in key for key in state_dict.keys())
    has_geoattention = any(
        ("attn_block" in key) or ("attn_blocks" in key) for key in state_dict.keys()
    )
    has_cost_head = any("cost_head" in key for key in state_dict.keys())
    has_dist_head = any("dist_head" in key for key in state_dict.keys())
    has_vec_head = any("vec_head" in key for key in state_dict.keys())

    if has_cost_head and (has_dist_head or has_vec_head):
        if has_unet:
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
    return "Unet" if has_unet else "CNN"


def detect_encoder_type(checkpoint_path: str) -> EncoderInfo:
    """Detect encoder type and input channels from checkpoint."""
    state_dict = _load_checkpoint_state_dict(checkpoint_path)

    # Detect Gated Fusion: feat_norm or feat_geo keys exist
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


# Font Utilities
def get_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a font for drawing text, with fallback to default."""
    font_candidates = ["arial.ttf", "DejaVuSans.ttf"]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# Frame Processing Functions
def add_title_to_frame(
    frame: np.ndarray,
    title: str,
    time_ms: float,
    steps: int,
    header_height: int = 40,
    show_infer_time: bool = True,
) -> np.ndarray:
    """Add title, time, and steps information to the top of a frame."""
    height, width = frame.shape[:2]

    # Create white header
    header = np.ones((header_height, width, 3), dtype=np.uint8) * 255
    header_img = Image.fromarray(header)
    draw = ImageDraw.Draw(header_img)

    # Draw title (centered)
    title_font = get_font(14)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 5), title, fill=(0, 0, 0), font=title_font)

    # Draw metrics (centered below title)
    metrics_font = get_font(11)
    metrics_parts = [f"Steps: {steps}"]
    if show_infer_time:
        metrics_parts.insert(0, f"Time: {time_ms:.1f}ms")
    metrics_text = " | ".join(metrics_parts)
    metrics_bbox = draw.textbbox((0, 0), metrics_text, font=metrics_font)
    metrics_width = metrics_bbox[2] - metrics_bbox[0]
    metrics_x = (width - metrics_width) // 2
    draw.text((metrics_x, 22), metrics_text, fill=(0, 0, 0), font=metrics_font)

    # Convert back to numpy and combine
    header = np.array(header_img)
    return np.vstack([header, frame])


def create_side_by_side_frame(
    frames_list: list[np.ndarray], separator_width: int = 8
) -> np.ndarray:
    """Create a side-by-side comparison frame with separators."""
    if not frames_list:
        return np.array([])

    height = frames_list[0].shape[0]
    separator = np.ones((height, separator_width, 3), dtype=np.uint8) * 255

    result = []
    for i, frame in enumerate(frames_list):
        if i > 0:
            result.append(separator)
        result.append(frame)

    return np.hstack(result)


# Model Loading Functions
def load_vanilla_planner() -> VanillaAstar:
    """Initialize Vanilla A* planner."""
    return VanillaAstar()


def load_neural_planner(
    checkpoint_path: str, fallback_encoder_depth: int
) -> tuple[NeuralAstar, EncoderInfo] | None:
    """Load Neural A* or Ours model from checkpoint."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        return None

    try:
        # Auto-detect encoder type and input channels from checkpoint
        info = detect_encoder_type(checkpoint_path)
        encoder_depth = info.encoder_depth or fallback_encoder_depth
        print(
            f"  [INFO] Detected: arch={info.encoder_arch}, "
            f"channels={info.input_channels}, depth={encoder_depth}, "
            f"gated={info.is_gated}"
        )

        # Create encoder_input string based on detected channels
        encoder_input = ENCODER_INPUT_MAP.get(
            info.input_channels, "x" * info.input_channels
        )

        # Create model with detected architecture
        planner = NeuralAstar(
            encoder_input=encoder_input,
            encoder_arch=info.encoder_arch,
            encoder_depth=encoder_depth,
            learn_obstacles=False,
            Tmax=1.0,
        )

        # Load checkpoint (strict=False for partial loading)
        state_dict = load_from_ptl_checkpoint(checkpoint_path)
        missing_keys, unexpected_keys = planner.load_state_dict(state_dict, strict=False)

        if missing_keys:
            preview = missing_keys[:3]
            suffix = "..." if len(missing_keys) > 3 else ""
            print(f"  [WARN] Missing keys: {preview}{suffix}")
        if unexpected_keys:
            preview = unexpected_keys[:3]
            suffix = "..." if len(unexpected_keys) > 3 else ""
            print(f"  [WARN] Unexpected keys: {preview}{suffix}")

        print(f"  [OK] Loaded checkpoint: {checkpoint_path}")
        return planner, info

    except Exception as e:
        print(f"  [FAIL] Failed to load checkpoint: {e}")
        return None


# Inference Functions
def run_inference(
    model_name: str,
    planner: VanillaAstar | NeuralAstar,
    map_design: torch.Tensor,
    start_map: torch.Tensor,
    goal_map: torch.Tensor,
    input_channels: int,
    dataset_path: str,
    problem_id: int,
    device: str = "cuda",
) -> ModelOutput | None:
    """Run inference for a single model and return results with timing.
    
    Measures pure inference time separately (without intermediate results storage).
    """
    display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)

    try:
        # Prepare input based on model type
        model_input = map_design

        # Measure pure inference time (without intermediate results)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        _ = planner(model_input, start_map, goal_map, store_intermediate_results=False)

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000

        # Generate intermediate results for visualization (not timed)
        outputs = planner(
            model_input, start_map, goal_map, store_intermediate_results=True
        )

        print(
            f"  [OK] Pure inference: {inference_time_ms:.2f}ms, "
            f"Frames: {len(outputs.intermediate_results)}"
        )

        return ModelOutput(
            outputs=outputs, time_ms=inference_time_ms, display_name=display_name
        )

    except Exception as e:
        print(f"  [FAIL] Inference failed: {e}")
        traceback.print_exc()
        return None


# GIF Creation Functions
def create_individual_gif(
    model_output: ModelOutput,
    map_design: torch.Tensor,
    output_path: str,
    scale: int = 4,
    show_infer_time: bool = False,
) -> None:
    """Create and save an individual model's GIF."""
    frames = []
    total_steps = len(model_output.outputs.intermediate_results)
    for ir in model_output.outputs.intermediate_results:
        frame = visualize_results(map_design, ir, scale=scale)
        frame_with_title = add_title_to_frame(
            frame,
            model_output.display_name,
            model_output.time_ms,
            total_steps,
            show_infer_time=show_infer_time,
        )
        frames.append(frame_with_title)

    # Add pause at the end
    frames_with_pause = frames + [frames[-1]] * PAUSE_FRAMES

    clip = ImageSequenceClip(frames_with_pause, fps=DEFAULT_FPS)
    clip.write_gif(output_path)
    print(f"  [OK] {output_path}")


def create_comparison_gif(
    all_outputs: dict[str, ModelOutput],
    map_design: torch.Tensor,
    output_path: str,
    scale: int = 4,
    show_infer_time: bool = False,
) -> None:
    """Create and save a side-by-side comparison GIF."""
    max_frames = max(
        len(output.outputs.intermediate_results) for output in all_outputs.values()
    )

    combined_frames = []
    for frame_idx in range(max_frames):
        frame_list = []
        for model_output in all_outputs.values():
            idx = min(frame_idx, len(model_output.outputs.intermediate_results) - 1)
            total_steps = len(model_output.outputs.intermediate_results)
            frame = visualize_results(
                map_design, model_output.outputs.intermediate_results[idx], scale=scale
            )
            frame_with_title = add_title_to_frame(
                frame,
                model_output.display_name,
                model_output.time_ms,
                total_steps,
                show_infer_time=show_infer_time,
            )
            frame_list.append(frame_with_title)

        combined_frame = create_side_by_side_frame(frame_list)
        combined_frames.append(combined_frame)

    # Add pause at the end
    combined_frames_with_pause = combined_frames + [combined_frames[-1]] * PAUSE_FRAMES

    clip = ImageSequenceClip(combined_frames_with_pause, fps=DEFAULT_FPS)
    clip.write_gif(output_path)


# Main Entry Point
@hydra.main(config_path="config", config_name="create_gif", version_base="1.3")
def main(config) -> None:
    """Main entry point for GIF creation."""
    global INCLUDE_INFER_TIME
    show_infer_time = INCLUDE_INFER_TIME
    dataname = os.path.basename(config.dataset)
    problem_id = config.get("problem_id", 1)
    model_dir = config.get("modeldir", "model")

    print(f"\n{'=' * 60}")
    print(f"Creating Comparison GIF for Problem {problem_id}")
    print(f"Dataset: {dataname}")
    print(f"{'=' * 60}\n")

    # Load test data
    dataloader = create_dataloader(
        config.dataset + ".npz", "test", 100, shuffle=False, num_starts=1
    )
    map_designs, start_maps, goal_maps, _ = next(iter(dataloader))

    # Get single problem
    map_design = map_designs[problem_id : problem_id + 1]
    start_map = start_maps[problem_id : problem_id + 1]
    goal_map = goal_maps[problem_id : problem_id + 1]

    # Define models to compare
    models_config = {
        "vanilla": None,
        "neural_astar": os.path.join(model_dir, "neural_astar", dataname),
        "ours": os.path.join(model_dir, "ours", dataname),
    }

    # Process all models
    all_outputs: dict[str, ModelOutput] = {}

    for model_name, checkpoint_path in models_config.items():
        print(f"Processing {model_name.upper()} model...")

        input_channels = 3  # default for vanilla and neural_astar

        if model_name == "vanilla":
            planner = load_vanilla_planner()
        else:
            result = load_neural_planner(checkpoint_path, config.encoder.depth)
            if result is None:
                continue
            planner, info = result
            input_channels = info.input_channels

        # Run inference
        model_output = run_inference(
            model_name,
            planner,
            map_design,
            start_map,
            goal_map,
            input_channels,
            config.dataset,
            problem_id,
        )
        if model_output is not None:
            all_outputs[model_name] = model_output

    if not all_outputs:
        print("\nError: No models were successfully loaded!")
        return

    # Create output directory
    savedir = f"{config.resultdir}/comparison"
    os.makedirs(savedir, exist_ok=True)

    # Save individual GIFs
    print("\nSaving individual GIFs...")
    for model_name, model_output in all_outputs.items():
        gif_path = f"{savedir}/{model_name}_{dataname}_{problem_id:04d}.gif"
        create_individual_gif(
            model_output, map_design, gif_path, show_infer_time=show_infer_time
        )

    # Create comparison GIF
    print("\nCreating comparison GIF...")
    comparison_path = f"{savedir}/comparison_{dataname}_{problem_id:04d}.gif"
    create_comparison_gif(
        all_outputs, map_design, comparison_path, show_infer_time=show_infer_time
    )

    print(f"\n{'=' * 60}")
    print(f"[OK] Comparison GIF saved: {comparison_path}")
    print(f"  Models compared: {', '.join(all_outputs.keys())}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    INCLUDE_INFER_TIME = parse_include_infer_time_flag(sys.argv[1:])
    main()

# Usage:
# python scripts/create_comparison_gif.py dataset=data/maze_preprocessed/mazes_032_moore_c8_ours modeldir=model problem_id=0 resultdir=results 
# python scripts/create_comparison_gif.py dataset=data/maze_preprocessed/mixed_064_moore_c16_ours modeldir=model problem_id=15 resultdir=results 
#
# Note:
# - Time: Inference time in milliseconds (excluding visualization).
# - Steps: Number of A* search iterations.
# - Ours may have fewer steps but similar time due to Neural Network overhead.
