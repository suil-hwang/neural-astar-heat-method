# heat_method/visualization.py
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Ensure parent directory is in path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heat_method.solver import GeodesicHeatSolver
from heat_method.validation import check_reachability, trace_path


# Visualization Functions
def visualize_result(
    map_design: NDArray,
    goal_pos: Tuple[int, int],
    vec_x: NDArray,
    vec_y: NDArray,
    dist_grid: NDArray,
    reachable_mask: Optional[NDArray] = None,
    sample_idx: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_paths: bool = False,
    num_paths: int = 5,
) -> None:
    """Visualize Geodesic Guidance Field results."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    H, W = map_design.shape
    g_r, g_c = goal_pos
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    suptitle = title if title is not None else f"Sample {sample_idx}"
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    

    # 1. Map with Goal
    ax = axes[0, 0]
    ax.set_title("Map & Goal", fontsize=12)
    ax.imshow(map_design, cmap="gray", interpolation="nearest")
    ax.scatter(g_c, g_r, c="red", s=150, marker="*", 
               label="Goal", edgecolors="white", linewidths=1.5, zorder=10)
    ax.legend(loc="upper right")
    ax.axis("off")
    
    # 2. Geodesic Distance Field
    ax = axes[0, 1]
    ax.set_title("Geodesic Distance (Heat Method)", fontsize=12)
    
    # Compute valid distance range
    valid_dist = dist_grid[np.isfinite(dist_grid)]
    if len(valid_dist) > 0:
        vmin, vmax = np.min(valid_dist), np.max(valid_dist)
    else:
        vmin, vmax = 0, 1
    
    # Create RGB image with special handling for unreachable/obstacles
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_geo = plt.cm.viridis
    dist_rgb = cmap_geo(norm(np.clip(dist_grid, vmin, vmax)))[:, :, :3]
    
    # Mark unreachable passable cells (gray) and obstacles (black)
    unreachable_mask = (map_design == 1) & ~np.isfinite(dist_grid)
    obstacle_mask = map_design == 0
    
    dist_rgb[unreachable_mask] = [0.5, 0.5, 0.5]  # Gray for unreachable
    dist_rgb[obstacle_mask] = [0.0, 0.0, 0.0]     # Black for obstacles
    
    ax.imshow(dist_rgb, interpolation="nearest")
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_geo, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Distance", fontsize=10)
    ax.axis("off")
    
    # 3. Vector Guidance Field
    ax = axes[1, 0]
    ax.set_title("Vector Guidance Field", fontsize=12)
    ax.imshow(map_design, cmap="gray", interpolation="nearest")
    
    # Quiver plot with adaptive stride
    stride = max(1, H // 16)
    Y, X = np.mgrid[0:H:stride, 0:W:stride]
    U = vec_x[::stride, ::stride]
    V = vec_y[::stride, ::stride]
    
    # Filter out zero vectors for cleaner visualization
    magnitude = np.sqrt(U**2 + V**2)
    mask = magnitude > 0.1
    
    ax.quiver(
        X[mask], Y[mask], U[mask], -V[mask],  # Negate V for image coordinates
        color="red", scale=25, width=0.003, headwidth=4, headlength=5
    )
    
    # Optionally show traced paths
    if show_paths and reachable_mask is not None:
        valid_starts = np.argwhere((reachable_mask > 0) & (map_design == 1))
        if len(valid_starts) > 0:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(valid_starts), min(num_paths, len(valid_starts)), replace=False)
            
            colors = plt.cm.rainbow(np.linspace(0, 1, len(indices)))
            for idx, color in zip(indices, colors):
                start = valid_starts[idx]
                path, reached = trace_path(
                    vec_x, vec_y,
                    (float(start[0]), float(start[1])),
                    goal_pos,
                    method="rk4"
                )
                style = "-" if reached else "--"
                ax.plot(path[:, 1], path[:, 0], style, color=color, 
                       linewidth=1.5, alpha=0.7)
    
    ax.scatter(g_c, g_r, c="red", s=100, marker="*", 
               edgecolors="white", linewidths=1.5, zorder=10)
    ax.axis("off")
    
    # 4. Reachable Mask
    ax = axes[1, 1]
    ax.set_title("Reachable Mask", fontsize=12)
    
    if reachable_mask is not None:
        # Custom colormap: dark gray (unreachable) → teal (reachable)
        reachable_cmap = ListedColormap([
            (0.25, 0.25, 0.25, 1.0),  # Unreachable: dark gray
            (0.0, 0.8, 0.45, 1.0),    # Reachable: teal
        ])
        
        im = ax.imshow(
            reachable_mask,
            cmap=reachable_cmap,
            vmin=0, vmax=1,
            interpolation="nearest",
        )
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1])
        cbar.ax.set_yticklabels(["Blocked", "Reachable"])
        
        ax.scatter(g_c, g_r, c="red", s=100, marker="*", 
                   edgecolors="white", linewidths=1.5, zorder=10)
    else:
        ax.text(0.5, 0.5, "No reachable mask", 
                ha="center", va="center", fontsize=12,
                transform=ax.transAxes)
    ax.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def visualize_sample(
    data_path: str,
    split: str,
    sample_idx: int,
    save_path: Optional[str] = None,
    show_paths: bool = False,
    t_coef: float = 1.0,
) -> None:
    """Load and visualize a single sample from a dataset."""
    print(f"Loading dataset: {data_path}")
    data = np.load(data_path)
    
    # Map split to array indices
    split_to_idx = {"train": 0, "valid": 4, "test": 8}
    if split not in split_to_idx:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'valid', or 'test'.")
    
    base_idx = split_to_idx[split]
    
    maps = data[f"arr_{base_idx}"]
    goals = data[f"arr_{base_idx + 1}"]
    
    # Handle channel dimension
    if maps.ndim == 4:
        maps = maps.squeeze(1)
    if goals.ndim == 4:
        goals = goals.squeeze(1)
    
    if sample_idx >= len(maps):
        raise ValueError(f"Sample index {sample_idx} out of range (max: {len(maps) - 1})")
    
    map_design = maps[sample_idx]
    goal_map = goals[sample_idx]
    
    # Extract goal position
    g_coords = np.argwhere(goal_map == 1)
    if len(g_coords) > 0:
        g_r, g_c = int(g_coords[0, 0]), int(g_coords[0, 1])
    else:
        valid_indices = np.argwhere(map_design == 1)
        if len(valid_indices) > 0:
            g_r, g_c = int(valid_indices[0, 0]), int(valid_indices[0, 1])
        else:
            g_r, g_c = 0, 0
    
    print(f"Processing sample {sample_idx} with goal at ({g_r}, {g_c})...")
    
    # Compute guidance
    result = GeodesicHeatSolver.compute_guidance(
        map_design, (g_r, g_c), t_coef=t_coef
    )
    
    # Print statistics
    print(f"  vec_x range: [{result.vec_x.min():.4f}, {result.vec_x.max():.4f}]")
    print(f"  vec_y range: [{result.vec_y.min():.4f}, {result.vec_y.max():.4f}]")
    print(f"  dist range: [{result.dist_normalized.min():.4f}, {result.dist_normalized.max():.4f}]")
    print(f"  reachable ratio: {result.reachable.mean():.2%}")
    
    # Reachability check
    valid_area = (result.reachable > 0) & (map_design == 1)
    reachability = check_reachability(
        result.vec_x, result.vec_y, valid_area, (g_r, g_c),
        method="rk4"
    )
    print(f"  Reachability Score: {reachability:.2%}")
    
    # Prepare distance for display (unnormalize)
    H, W = map_design.shape
    diag = np.sqrt(H**2 + W**2)
    dist_display = result.dist_normalized * diag
    dist_display[result.reachable == 0] = np.nan
    
    # Generate title
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    title = f"{dataset_name} / {split} / idx={sample_idx} / reach={reachability:.0%}"
    
    visualize_result(
        map_design,
        (g_r, g_c),
        result.vec_x,
        result.vec_y,
        dist_display,
        result.reachable,
        sample_idx,
        title,
        save_path,
        show_paths=show_paths,
    )


def main() -> None:
    """Command-line interface for visualization."""
    parser = argparse.ArgumentParser(
        description="Heat Method Visualization - Visualize geodesic guidance fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python -m heat_method.visualization \\
            --data_path data/maze/mazes_032_moore_c8.npz \\
            --split test --sample_idx 0
        
        python -m heat_method.visualization \\
            --data_path data/maze/mixed_064_moore_c16.npz \\
            --split train --sample_idx 42 --show_paths \\
            --save_path figures/sample_42.png
                """,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the .npz dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Data split to visualize (default: train)",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of sample to visualize (default: 0)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the figure (optional)",
    )
    parser.add_argument(
        "--show_paths",
        action="store_true",
        help="Overlay traced paths on vector field",
    )
    parser.add_argument(
        "--t_coef",
        type=float,
        default=1.0,
        help="Heat method time coefficient (default: 1.0)",
    )
    
    args = parser.parse_args()
    
    visualize_sample(
        args.data_path,
        args.split,
        args.sample_idx,
        args.save_path,
        args.show_paths,
        args.t_coef,
    )


if __name__ == "__main__":
    main()

# Example usage:
# python heat_method/visualization.py --data_path data/maze/mazes_032_moore_c8.npz --split test --sample_idx 0
# python heat_method/visualization.py --data_path data/maze/mixed_064_moore_c16.npz --split test --sample_idx 0
# python heat_method/visualization.py --data_path data/maze/all_064_moore_c16.npz --split test --sample_idx 0
