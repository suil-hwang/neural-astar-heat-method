# heat_method/visualization.py
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
from typing import Tuple, Optional

# Ensure parent directory is in path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heat_method.solver import GeodesicHeatSolver
from heat_method.validation import check_reachability


def visualize_result(
    map_design: np.ndarray,
    goal_pos: Tuple[int, int],
    vec_x: np.ndarray,
    vec_y: np.ndarray,
    dist_grid: np.ndarray,
    reachable_mask: Optional[np.ndarray] = None,
    sample_idx: int = 0,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize Geodesic Guidance Field and Vector Guidance Field.
    
    Creates 4 subplots in a 2x2 grid:
        1. Map with Goal marker
        2. Geodesic Distance Field 
        3. Vector Guidance Field 
        4. Reachable Mask 
    """
    import matplotlib.pyplot as plt
    
    H, W = map_design.shape
    g_r, g_c = goal_pos
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    suptitle = title if title is not None else f"Sample {sample_idx}"
    fig.suptitle(suptitle, fontsize=14)
    
    # 1.  Map with Goal
    axes[0, 0].set_title("Map & Goal")
    axes[0, 0].imshow(map_design, cmap='gray')
    axes[0, 0].scatter(
        g_c, g_r, c='red', s=100, label='Goal', edgecolors='white'
    )
    axes[0, 0].legend()
    axes[0, 0].axis('off')
    
    # 2. Geodesic Distance Field
    valid_dist = dist_grid[np.isfinite(dist_grid)]
    if len(valid_dist) > 0:
        vmin, vmax = np.min(valid_dist), np.max(valid_dist)
    else:
        vmin, vmax = 0, 1
    
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_geo = plt.cm.viridis
    dist_rgb = cmap_geo(norm(np.clip(dist_grid, vmin, vmax)))[:, :, :3]
    
    unreachable_mask = (map_design == 1) & np.isnan(dist_grid)
    obstacle_mask = map_design == 0
    
    dist_rgb[unreachable_mask] = [0.5, 0.5, 0.5]
    dist_rgb[obstacle_mask] = [0.0, 0.0, 0.0]
    
    axes[0, 1].imshow(dist_rgb)
    axes[0, 1].set_title("Geodesic Distance (Heat Method)")
    
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0, 1])
    axes[0, 1].axis('off')
    
    # 3. Vector Guidance Field 
    axes[1, 0].set_title("Vector Guidance Field")
    axes[1, 0].imshow(map_design, cmap='gray')
    
    stride = max(1, H // 16)
    X, Y = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    U = vec_x[0::stride, 0::stride]
    V = vec_y[0::stride, 0::stride]
    
    axes[1, 0].quiver(X, Y, U, -V, color='red', scale=20)
    axes[1, 0].scatter(g_c, g_r, c='red', s=80, edgecolors='white')
    axes[1, 0].axis('off')

    # 4. Reachable Mask
    axes[1, 1].set_title("Reachable Mask")
    if reachable_mask is not None:
        from matplotlib.colors import ListedColormap

        # Two-color mask: unreachable = dark gray, reachable = vivid teal
        reachable_cmap = ListedColormap(
            [(0.25, 0.25, 0.25, 1.0), (0.0, 0.8, 0.45, 1.0)]
        )

        im = axes[1, 1].imshow(
            reachable_mask,
            cmap=reachable_cmap,
            vmin=0,
            vmax=1,
            interpolation='nearest',
        )
        cbar = plt.colorbar(
            im,
            ax=axes[1, 1],
            fraction=0.046,
            pad=0.04,
            ticks=[0, 1],
        )
        cbar.ax.set_yticklabels(["Blocked/Unreachable", "Reachable"])

        axes[1, 1].scatter(g_c, g_r, c='red', s=80, edgecolors='white')
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No reachable mask",
            ha='center',
            va='center',
            fontsize=12,
        )
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()


def visualize_sample(
    data_path: str,
    split: str,
    sample_idx: int,
    save_path: Optional[str] = None,
) -> None:
    """Load and visualize a single sample from the dataset."""
    print(f"Loading dataset: {data_path}")
    data = np.load(data_path)
    
    split_to_idx = {"train": 0, "valid": 4, "test": 8}
    base_idx = split_to_idx[split]
    
    maps = data[f'arr_{base_idx}']
    goals = data[f'arr_{base_idx + 1}']
    
    if maps.ndim == 4:
        maps = maps.squeeze(1)
    if goals.ndim == 4:
        goals = goals.squeeze(1)
    
    map_design = maps[sample_idx]
    goal_map = goals[sample_idx]
    
    g_coords = np.argwhere(goal_map == 1)
    if len(g_coords) > 0:
        g_r, g_c = g_coords[0]
    else:
        valid_indices = np.argwhere(map_design == 1)
        g_r, g_c = valid_indices[0] if len(valid_indices) > 0 else (0, 0)
    
    print(f"Processing sample {sample_idx} with goal at ({g_r}, {g_c})...")
    
    result = GeodesicHeatSolver.compute_guidance(map_design, (g_r, g_c))
    
    print(f"  vec_x range: [{result.vec_x.min():.3f}, {result.vec_x.max():.3f}]")
    print(f"  vec_y range: [{result.vec_y.min():.3f}, {result.vec_y.max():.3f}]")
    print(f"  dist range: [{result.dist_normalized.min():.3f}, {result.dist_normalized.max():.3f}]")
    print(f"  reachable ratio: {result.reachable.mean():.2%}")
    
    # Reachability check
    valid_area = (result.reachable > 0) & (map_design == 1)
    reachability = check_reachability(
        result.vec_x, result.vec_y, valid_area, (g_r, g_c)
    )
    print(f"  Reachability Score: {reachability:.2%}")
    
    # Prepare distance for display (unnormalize)
    H, W = map_design.shape
    diag = np.sqrt(H**2 + W**2)
    dist_display = result.dist_normalized * diag
    dist_display[result.reachable == 0] = np.nan
    
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    title = f"{dataset_name} / {split} / {sample_idx}"

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
    )


def main():
    """CLI entry point for visualization."""
    parser = argparse.ArgumentParser(
        description="Heat Method Visualization - Visualize geodesic guidance fields"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to the .npz dataset"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Data split to visualize"
    )
    parser.add_argument(
        "--sample_idx", 
        type=int, 
        default=0,
        help="Index of the sample to visualize"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the figure (optional)"
    )
    args = parser.parse_args()
    
    visualize_sample(
        args.data_path,
        args.split,
        args.sample_idx,
        args.save_path,
    )


if __name__ == "__main__":
    main()


# Example usage:
# python heat_method/visualization.py --data_path data/maze/mazes_032_moore_c8.npz --split test --sample_idx 0
# python heat_method/visualization.py --data_path data/maze_preprocessed/mazes_032_moore_c8_ours.npz --split test --sample_idx 0 

# python heat_method/visualization.py --data_path data/maze/mixed_064_moore_c16.npz --split test --sample_idx 0
# python heat_method/visualization.py --data_path data/maze_preprocessed/mixed_064_moore_c16_ours.npz --split test --sample_idx 0


