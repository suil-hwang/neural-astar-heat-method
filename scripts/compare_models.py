# scripts/compare_models.py
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Prefer local src/ over any globally installed neural_astar
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

# Allow importing shared script utilities when executed as a module (e.g. `import scripts.compare_models`).
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from common import ENCODER_INPUT_MAP, EncoderInfo, detect_encoder_type

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import load_from_ptl_checkpoint

# Filter specific warnings instead of suppressing all
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constants
PLOT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
GUIDANCE_KEYS = ("vx", "vy", "dist", "reachable")


@dataclass
class ModelMetrics:
    """Aggregated metrics for a single model."""

    is_optimal: list[bool] = field(default_factory=list)
    exp: list[float] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    explorations: list[float] = field(default_factory=list)


@dataclass
class PlotMetrics:
    """Metrics prepared for visualization."""

    names: list[str] = field(default_factory=list)
    opt: list[float] = field(default_factory=list)
    exp: list[float] = field(default_factory=list)
    hmean: list[float] = field(default_factory=list)
    time: list[float] = field(default_factory=list)


# Model Loading Functions
def load_vanilla_astar(device: str) -> VanillaAstar:
    """Initialize Vanilla A* planner."""
    logger.info("Initializing Vanilla A*...")
    model = VanillaAstar(use_differentiable_astar=True).to(device)
    model.eval()
    return model


def load_neural_astar(model_path: str, device: str) -> NeuralAstar | None:
    """Load Neural A* model from checkpoint."""
    if not os.path.exists(model_path):
        logger.info(f"[SKIP] Neural A* checkpoint not found: {model_path}")
        return None

    logger.info(f"Loading Neural A* from {model_path}...")
    try:
        model = NeuralAstar(
            encoder_input="m+",
            encoder_arch="CNN",
            encoder_depth=4,
            Tmax=1.0,
        )
        model.load_state_dict(load_from_ptl_checkpoint(model_path), strict=False)
        model.to(device).eval()
        logger.info("  [OK] Neural A* loaded")
        return model
    except Exception as e:
        logger.error(f"  [FAIL] Error loading Neural A*: {e}")
        return None


def load_ours_model(model_path: str, device: str) -> NeuralAstar | None:
    """Load our Geodesic-guided Neural A* with Gated Fusion."""
    if not os.path.exists(model_path):
        logger.info(f"[SKIP] Ours checkpoint not found: {model_path}")
        return None

    logger.info(f"Loading Ours model from {model_path}...")
    try:
        # Auto-detect encoder type from checkpoint
        info = detect_encoder_type(model_path)
        logger.info(f"  Encoder: {info.encoder_arch} (Gated: {info.is_gated})")
        logger.info(f"  Input channels: {info.input_channels}")
        logger.info(f"  Encoder depth: {info.encoder_depth}")

        # Set encoder_input based on channel count
        encoder_input = ENCODER_INPUT_MAP.get(
            info.input_channels, "x" * info.input_channels
        )

        model = NeuralAstar(
            encoder_input=encoder_input,
            encoder_arch=info.encoder_arch,
            encoder_depth=info.encoder_depth,
            Tmax=1.0,
        )
        model.load_state_dict(load_from_ptl_checkpoint(model_path), strict=False)
        model.to(device).eval()
        logger.info("  [OK] Ours model loaded")
        return model
    except Exception as e:
        logger.exception(f"  [FAIL] Error loading Ours model: {e}")
        return None


def load_all_models(
    model_dir: str, dataset_name: str, device: str
) -> dict[str, torch.nn.Module]:
    """Load all available models for comparison."""
    models: dict[str, torch.nn.Module] = {}

    # Vanilla A* (always available)
    models["Vanilla"] = load_vanilla_astar(device)

    # Neural A*
    na_path = f"{model_dir}/neural_astar/{dataset_name}"
    na_model = load_neural_astar(na_path, device)
    if na_model is not None:
        models["Neural A*"] = na_model

    # Ours (Geo-supervised)
    ours_path = f"{model_dir}/ours/{dataset_name}"
    ours_model = load_ours_model(ours_path, device)
    if ours_model is not None:
        models["Ours"] = ours_model

    return models


# Evaluation Functions
def compute_metrics(
    pred_paths: torch.Tensor,
    pred_histories: torch.Tensor,
    opt_lengths: torch.Tensor,
    goal_maps: torch.Tensor,
    vanilla_explorations: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute evaluation metrics for a batch."""
    path_lengths = pred_paths.sum((1, 2, 3)).detach().cpu().numpy()
    explorations = pred_histories.sum((1, 2, 3)).detach().cpu().numpy()
    opt_lengths_np = opt_lengths.sum((1, 2, 3)).detach().cpu().numpy()

    reached_goal = (pred_paths * goal_maps).sum((1, 2, 3)).detach().cpu().numpy() > 0.5
    path_steps = path_lengths - 1
    length_is_optimal = (path_steps <= opt_lengths_np * 1.001) | (
        path_steps <= opt_lengths_np + 0.5
    )
    is_optimal = reached_goal & length_is_optimal

    if vanilla_explorations is not None:
        reduction = (vanilla_explorations - explorations) / (
            vanilla_explorations + 1e-8
        )
        exp_ratios = np.maximum(100 * reduction, 0)
    else:
        exp_ratios = np.zeros_like(explorations)

    return is_optimal, exp_ratios, path_lengths, explorations


def _evaluate_single_model(
    name: str,
    planner: torch.nn.Module,
    map_designs: torch.Tensor,
    start_maps: torch.Tensor,
    goal_maps: torch.Tensor,
    opt_trajs: torch.Tensor,
    batch_vanilla_exp: np.ndarray | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Evaluate a single model on one batch and return metrics."""
    if device == "cuda":
        torch.cuda.synchronize()
    start_t = time.time()

    outputs = planner(map_designs, start_maps, goal_maps)

    if device == "cuda":
        torch.cuda.synchronize()
    runtime = time.time() - start_t

    is_opt, exp, _, expls = compute_metrics(
        outputs.paths,
        outputs.histories,
        opt_trajs,
        goal_maps,
        vanilla_explorations=batch_vanilla_exp,
    )
    return is_opt, exp, runtime, expls


def run_comparison(
    models: dict[str, torch.nn.Module],
    test_loader: DataLoader,
    device: str = "cuda",
) -> dict[str, ModelMetrics]:
    """Evaluate all models and collect metrics."""
    results = {name: ModelMetrics() for name in models}
    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Models"):
            map_designs, start_maps, goal_maps, opt_trajs = batch
            map_designs = map_designs.to(device)
            start_maps = start_maps.to(device)
            goal_maps = goal_maps.to(device)
            current_batch_size = map_designs.size(0)

            batch_vanilla_exp = None
            for name, planner in models.items():
                is_opt, exp, runtime, expls = _evaluate_single_model(
                    name,
                    planner,
                    map_designs,
                    start_maps,
                    goal_maps,
                    opt_trajs,
                    batch_vanilla_exp,
                    device,
                )

                results[name].is_optimal.extend(is_opt)
                results[name].exp.extend(exp)
                results[name].times.append(runtime / current_batch_size)
                results[name].explorations.extend(expls)

                if name == "Vanilla":
                    batch_vanilla_exp = expls

            sample_idx += current_batch_size

    return results


# Results Visualization Functions
def compute_plot_metrics(raw_results: dict[str, ModelMetrics]) -> PlotMetrics:
    """Compute aggregated metrics for plotting."""
    plot_metrics = PlotMetrics()

    for name, res in raw_results.items():
        opt_mean = np.mean(res.is_optimal) * 100
        exp_mean = np.mean(res.exp)
        time_mean = np.mean(res.times) * 1000

        if opt_mean + exp_mean > 0:
            hmean = 2 * (opt_mean * exp_mean) / (opt_mean + exp_mean)
        else:
            hmean = 0.0

        plot_metrics.names.append(name)
        plot_metrics.opt.append(opt_mean)
        plot_metrics.exp.append(exp_mean)
        plot_metrics.hmean.append(hmean)
        plot_metrics.time.append(time_mean)

    return plot_metrics


def print_results_table(plot_metrics: PlotMetrics) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 60)
    print(
        f"{'Model':<15} | {'Opt (%)':<10} | {'Exp (%)':<10} | "
        f"{'Hmean':<10} | {'Time (ms)':<10}"
    )
    print("-" * 60)

    for i, name in enumerate(plot_metrics.names):
        print(
            f"{name:<15} | {plot_metrics.opt[i]:<10.2f} | "
            f"{plot_metrics.exp[i]:<10.2f} | {plot_metrics.hmean[i]:<10.2f} | "
            f"{plot_metrics.time[i]:<10.2f}"
        )
    print("=" * 60)


def plot_metrics(plot_metrics: PlotMetrics, output_path: str) -> None:
    """Generate and save comparison bar charts."""
    n_models = len(plot_metrics.names)
    colors = PLOT_COLORS[:n_models]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    chart_configs = [
        (plot_metrics.opt, "Path Optimality (Opt)\n(Higher is better)", "%", (0, 105)),
        (plot_metrics.exp, "Exploration Reduction (Exp)\n(Higher is better)", "%", None),
        (plot_metrics.hmean, "Harmonic Mean\n(Higher is better)", None, None),
        (plot_metrics.time, "Average Runtime\n(Lower is better)", "ms", None),
    ]

    for ax, (data, title, ylabel, ylim) in zip(axes, chart_configs):
        ax.bar(plot_metrics.names, data, color=colors, alpha=0.8)
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"\nMetrics plot saved to: {output_path}")
    plt.show()


def main() -> None:
    """Main entry point for model comparison."""
    parser = argparse.ArgumentParser(
        description="Compare Vanilla/Neural A*/Ours models"
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to .npz dataset (without .npz)",
    )
    parser.add_argument(
        "--model-dir", type=str, default="model",
        help="Directory containing model subfolders",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = args.device
    dataset_name = os.path.basename(args.dataset)

    # Load data
    logger.info(f"\nLoading dataset from {args.dataset}...")
    test_loader = create_dataloader(
        args.dataset + ".npz", "test", batch_size=args.batch_size, shuffle=False
    )

    # Initialize models
    models = load_all_models(args.model_dir, dataset_name, device)
    if not models:
        logger.error("\nNo models to evaluate!")
        return
    logger.info(f"\nModels to compare: {list(models.keys())}")

    # Run evaluation
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)
    raw_results = run_comparison(
        models,
        test_loader,
        device,
    )

    # Display results
    metrics = compute_plot_metrics(raw_results)
    print_results_table(metrics)
    # plot_metrics(metrics, "comparison_metrics.png")


if __name__ == "__main__":
    main()

# Usage:
# python scripts/compare_models.py --model-dir model --dataset data/maze_preprocessed/mazes_032_moore_c8_ours
# python scripts/compare_models.py --model-dir model --dataset data/maze_preprocessed/mixed_064_moore_c16_ours
# python scripts/compare_models.py --model-dir model --dataset data/maze_preprocessed/all_064_moore_c16_ours
