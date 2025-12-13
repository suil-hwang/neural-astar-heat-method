# heat_method/preprocessing.py
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Ensure parent directory is in path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heat_method.solver import GeodesicHeatSolver



def process_and_save(
    data_path: str,
    output_path: str,
) -> None:
    """
    Compute Geodesic Guidance Field for all data splits and save as extended NPZ.
    
    Original NPZ structure (preserved):
        - Train (arr_0~arr_3): 800 maps for model training
        - Valid (arr_4~arr_7): 100 maps for validation during training
        - Test  (arr_8~arr_11): 100 maps for final evaluation
    
    New keys added per split:
        - {split}_vec_x: (N, 1, H, W) X direction vectors
        - {split}_vec_y: (N, 1, H, W) Y direction vectors
        - {split}_dist: (N, 1, H, W) Geodesic distance (normalized)
        - {split}_reachable: (N, 1, H, W) Reachable mask
    """
    from tqdm import tqdm
    
    print(f"Loading dataset: {data_path}")
    data = np.load(data_path)
    
    save_dict = dict(data)
    
    splits = {
        'train': {'map_key': 'arr_0', 'goal_key': 'arr_1'},
        'valid': {'map_key': 'arr_4', 'goal_key': 'arr_5'},
        'test':  {'map_key': 'arr_8', 'goal_key': 'arr_9'},
    }
    
    for split_name, keys in splits.items():
        print(f"\n{'='*50}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'='*50}")
        
        if keys['map_key'] not in data:
            print(f"  [SKIP] {keys['map_key']} not found in dataset")
            continue
            
        maps = data[keys['map_key']]
        goals = data[keys['goal_key']]
        
        if maps.ndim == 4:
            maps = maps.squeeze(1)
        if goals.ndim == 4:
            goals = goals.squeeze(1)
        
        N, H, W = maps.shape
        print(f"  Samples: {N}, Size: {H}x{W}")
        
        vx_list, vy_list, dist_list, reachable_list = [], [], [], []
        
        for i in tqdm(range(N), desc=f"  {split_name}"):
            map_design = maps[i]
            goal_map = goals[i]
            
            g_coords = np.argwhere(goal_map == 1)
            if len(g_coords) > 0:
                g_r, g_c = g_coords[0]
            else:
                valid_indices = np.argwhere(map_design == 1)
                if len(valid_indices) > 0:
                    g_r, g_c = valid_indices[0]
                else:
                    vx_list.append(np.zeros((H, W), dtype=np.float32))
                    vy_list.append(np.zeros((H, W), dtype=np.float32))
                    dist_list.append(np.ones((H, W), dtype=np.float32))
                    reachable_list.append(np.zeros((H, W), dtype=np.float32))
                    continue
            
            result = GeodesicHeatSolver.compute_guidance(
                map_design, (g_r, g_c)
            )
            
            vx_list.append(result.vec_x)
            vy_list.append(result.vec_y)
            dist_list.append(result.dist_normalized)
            reachable_list.append(result.reachable)
        
        save_dict[f"{split_name}_vec_x"] = np.expand_dims(np.array(vx_list), 1)
        save_dict[f"{split_name}_vec_y"] = np.expand_dims(np.array(vy_list), 1)
        save_dict[f"{split_name}_dist"] = np.expand_dims(np.array(dist_list), 1)
        save_dict[f"{split_name}_reachable"] = np.expand_dims(np.array(reachable_list), 1)
        
        print(
            f"  Added: {split_name}_vec_x, {split_name}_vec_y, "
            f"{split_name}_dist, {split_name}_reachable"
        )
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    print(f"\nSaving extended dataset to: {output_path}")
    np.savez_compressed(output_path, **save_dict)
    
    print(f"\nExtended dataset saved successfully!")
    print(f"Keys: {list(save_dict.keys())}")


def main():
    """CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(
        description="Heat Method Preprocessing - Extend dataset with guidance fields"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="Path to the .npz dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for extended dataset (default: {data_path}_ours.npz)"
    )
    args = parser.parse_args()
    
    output_path = args.output_path
    if output_path is None:
        base, ext = os.path.splitext(args.data_path)
        output_path = f"{base}_ours{ext}"
    
    process_and_save(
        args.data_path, 
        output_path,
    )


if __name__ == "__main__":
    main()


# Example usage:
# python heat_method/preprocessing.py --data_path data/maze/mazes_032_moore_c8.npz --output_path data/maze_preprocessed/mazes_032_moore_c8_ours.npz
# python heat_method/preprocessing.py --data_path data/maze/mixed_064_moore_c16.npz --output_path data/maze_preprocessed/mixed_064_moore_c16_ours.npz