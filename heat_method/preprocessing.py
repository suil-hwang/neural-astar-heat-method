# heat_method/preprocessing.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from numpy.typing import NDArray

# Ensure parent directory is in path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heat_method.solver import GeodesicHeatSolver
from heat_method.dtypes import GeodesicGuidance


# Dataset Split Configuration
SPLIT_CONFIG: Dict[str, Dict[str, str]] = {
    "train": {"map_key": "arr_0", "goal_key": "arr_1"},
    "valid": {"map_key": "arr_4", "goal_key": "arr_5"},
    "test": {"map_key": "arr_8", "goal_key": "arr_9"},
}


def _extract_goal_position(
    goal_map: NDArray,
    map_design: NDArray,
) -> tuple[int, int]:
    """Extract goal position from goal map with fallback. """
    # Primary: find marked goal
    g_coords = np.argwhere(goal_map == 1)
    if len(g_coords) > 0:
        return int(g_coords[0, 0]), int(g_coords[0, 1])
    
    # Fallback: first passable cell
    valid_indices = np.argwhere(map_design == 1)
    if len(valid_indices) > 0:
        return int(valid_indices[0, 0]), int(valid_indices[0, 1])
    
    # No valid position found
    return 0, 0


def _process_single_sample(
    map_design: NDArray,
    goal_map: NDArray,
    t_coef: float = 1.0,
    use_robust: bool = True,
) -> GeodesicGuidance:
    """Process a single map sample."""
    g_r, g_c = _extract_goal_position(goal_map, map_design)
    
    return GeodesicHeatSolver.compute_guidance(
        map_design,
        (g_r, g_c),
        t_coef=t_coef,
        use_robust=use_robust,
    )


def process_and_save(
    data_path: str,
    output_path: str,
    t_coef: float = 1.0,
    use_robust: bool = True,
    show_progress: bool = True,
) -> None:
    """Compute Geodesic Guidance Fields for all splits and save as extended NPZ."""
    # Progress bar setup
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
            print("Note: Install tqdm for progress bars (pip install tqdm)")
    else:
        tqdm = None
    
    print(f"Loading dataset: {data_path}")
    data = np.load(data_path)
    
    # Preserve original data
    save_dict: Dict[str, Any] = dict(data)
    
    for split_name, keys in SPLIT_CONFIG.items():
        print(f"\n{'='*50}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'='*50}")
        
        if keys["map_key"] not in data:
            print(f"  [SKIP] {keys['map_key']} not found in dataset")
            continue
        
        maps = data[keys["map_key"]]
        goals = data[keys["goal_key"]]
        
        # Handle channel dimension
        if maps.ndim == 4:
            maps = maps.squeeze(1)
        if goals.ndim == 4:
            goals = goals.squeeze(1)
        
        N, H, W = maps.shape
        print(f"  Samples: {N}, Size: {H}×{W}")
        
        # Pre-allocate output arrays
        vx_array = np.zeros((N, H, W), dtype=np.float32)
        vy_array = np.zeros((N, H, W), dtype=np.float32)
        dist_array = np.ones((N, H, W), dtype=np.float32)
        reachable_array = np.zeros((N, H, W), dtype=np.float32)
        
        # Process samples
        iterator = range(N)
        if tqdm is not None:
            iterator = tqdm(iterator, desc=f"  {split_name}")
        
        for i in iterator:
            result = _process_single_sample(
                maps[i], goals[i],
                t_coef=t_coef,
                use_robust=use_robust,
            )
            
            vx_array[i] = result.vec_x
            vy_array[i] = result.vec_y
            dist_array[i] = result.dist_normalized
            reachable_array[i] = result.reachable
        
        # Add channel dimension and save
        save_dict[f"{split_name}_vec_x"] = np.expand_dims(vx_array, 1)
        save_dict[f"{split_name}_vec_y"] = np.expand_dims(vy_array, 1)
        save_dict[f"{split_name}_dist"] = np.expand_dims(dist_array, 1)
        save_dict[f"{split_name}_reachable"] = np.expand_dims(reachable_array, 1)
        
        print(f"  Added: {split_name}_vec_x, {split_name}_vec_y, "
              f"{split_name}_dist, {split_name}_reachable")
    
    # Save extended dataset
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving extended dataset to: {output_path}")
    np.savez_compressed(output_path, **save_dict)
    
    print(f"\nExtended dataset saved successfully!")
    print(f"Keys: {list(save_dict.keys())}")


def main() -> None:
    """Command-line interface for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Heat Method Preprocessing - Extend dataset with guidance fields",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python -m heat_method.preprocessing \\
            --data_path data/maze/mazes_032_moore_c8.npz
        
        python -m heat_method.preprocessing \\
            --data_path data/maze/mixed_064_moore_c16.npz \\
            --output_path data/preprocessed/mixed_064_ours.npz \\
            --t_coef 2.0
                """,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the .npz dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for extended dataset (default: {data_path}_ours.npz)",
    )
    parser.add_argument(
        "--t_coef",
        type=float,
        default=1.0,
        help="Heat method time coefficient (default: 1.0)",
    )
    parser.add_argument(
        "--no_robust",
        action="store_true",
        help="Disable robust intrinsic triangulation",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar",
    )
    
    args = parser.parse_args()
    
    # Default output path
    output_path = args.output_path
    if output_path is None:
        base, ext = os.path.splitext(args.data_path)
        output_path = f"{base}_ours{ext}"
    
    process_and_save(
        args.data_path,
        output_path,
        t_coef=args.t_coef,
        use_robust=not args.no_robust,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()

# Example usage:
# python heat_method/preprocessing.py --data_path data/maze/mazes_032_moore_c8.npz --output_path data/maze_preprocessed/mazes_032_moore_c8_ours.npz
# python heat_method/preprocessing.py --data_path data/maze/mixed_064_moore_c16.npz --output_path data/maze_preprocessed/mixed_064_moore_c16_ours.npz
# python heat_method/preprocessing.py --data_path data/maze/all_064_moore_c16.npz --output_path data/maze_preprocessed/all_064_moore_c16_ours.npz 