# heat_method/preprocessing.py
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

# Ensure parent directory is in path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heat_method.solver import GeodesicHeatSolver
from heat_method.anchor_embedding import MultiAnchorGeodesicEmbedding


def process_and_save(
    data_path: str,
    output_path: str,
    num_anchors: int = 8,
    anchor_method: str = "fps",
    compute_phi: bool = True,
) -> None:
    """
    Compute geodesic guidance fields and multi-anchor embeddings.

    Design note: start 위치는 episode마다 달라 dataset 단계에서 알 수 없으므로
    anchors에는 goal + FPS/grid 포인트만 포함한다. start 정보를 포함하고 싶다면
    온라인으로 phi를 다시 계산하거나 별도 앵커 소스를 제공해야 한다.
    """
    from tqdm import tqdm

    print(f"Loading dataset: {data_path}")
    print(f"  num_anchors: {num_anchors}")
    print(f"  anchor_method: {anchor_method}")
    print(f"  compute_phi: {compute_phi}")

    data = np.load(data_path)
    save_dict = dict(data)

    splits = {
        "train": {"map_key": "arr_0", "goal_key": "arr_1"},
        "valid": {"map_key": "arr_4", "goal_key": "arr_5"},
        "test": {"map_key": "arr_8", "goal_key": "arr_9"},
    }

    for split_name, keys in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split...")
        print(f"{'='*60}")

        if keys["map_key"] not in data:
            print(f"  [SKIP] {keys['map_key']} not found in dataset")
            continue

        maps = data[keys["map_key"]]
        goals = data[keys["goal_key"]]

        if maps.ndim == 4:
            maps = maps.squeeze(1)
        if goals.ndim == 4:
            goals = goals.squeeze(1)

        num_samples, height, width = maps.shape
        print(f"  Samples: {num_samples}, Size: {height}x{width}")

        vx_list, vy_list, dist_list, reachable_list = [], [], [], []
        phi_list, anchor_coords_list = [], []

        for i in tqdm(range(num_samples), desc=f"  {split_name}"):
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
                    vx_list.append(np.zeros((height, width), dtype=np.float32))
                    vy_list.append(np.zeros((height, width), dtype=np.float32))
                    dist_list.append(np.ones((height, width), dtype=np.float32))
                    reachable_list.append(np.zeros((height, width), dtype=np.float32))
                    if compute_phi:
                        phi_list.append(
                            np.ones((num_anchors, height, width), dtype=np.float32)
                        )
                        anchor_coords_list.append(
                            np.zeros((num_anchors, 2), dtype=np.int32)
                        )
                    continue

            goal_pos = (int(g_r), int(g_c))

            result = GeodesicHeatSolver.compute_guidance(map_design, goal_pos)
            vx_list.append(result.vec_x)
            vy_list.append(result.vec_y)
            dist_list.append(result.dist_normalized)
            reachable_list.append(result.reachable)

            if compute_phi:
                embedding = MultiAnchorGeodesicEmbedding.compute_embedding(
                    map_design,
                    goal_pos,
                    start_pos=None,
                    num_anchors=num_anchors,
                    anchor_method=anchor_method,
                    normalize=True,
                )
                phi = embedding.phi.transpose(2, 0, 1)
                phi_list.append(phi)
                anchor_coords_list.append(embedding.anchor_coords)

        save_dict[f"{split_name}_vx"] = np.expand_dims(np.array(vx_list), 1)
        save_dict[f"{split_name}_vy"] = np.expand_dims(np.array(vy_list), 1)
        save_dict[f"{split_name}_dist"] = np.expand_dims(np.array(dist_list), 1)
        save_dict[f"{split_name}_reachable"] = np.expand_dims(
            np.array(reachable_list), 1
        )

        print(
            f"  Added: {split_name}_vx, {split_name}_vy, "
            f"{split_name}_dist, {split_name}_reachable"
        )

        if compute_phi:
            save_dict[f"{split_name}_phi"] = np.array(phi_list)

            max_anchors = max(ac.shape[0] for ac in anchor_coords_list)
            padded_coords = []
            for ac in anchor_coords_list:
                if ac.shape[0] < max_anchors:
                    padding = np.zeros((max_anchors - ac.shape[0], 2), dtype=ac.dtype)
                    ac = np.vstack([ac, padding])
                padded_coords.append(ac)
            save_dict[f"{split_name}_anchor_coords"] = np.array(padded_coords)

            print(
                f"  Added: {split_name}_phi "
                f"(shape: {save_dict[f'{split_name}_phi'].shape})"
            )
            print(f"  Added: {split_name}_anchor_coords")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"\nSaving extended dataset to: {output_path}")
    np.savez_compressed(output_path, **save_dict)

    print(f"\n{'='*60}")
    print("Preprocessing complete!")
    print(f"{'='*60}")
    print(f"Output keys: {list(save_dict.keys())}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Output size: {size_mb:.1f} MB")


def main():
    """CLI entry point for preprocessing."""
    parser = argparse.ArgumentParser(
        description=(
            "Heat Method Preprocessing - "
            "Extend dataset with guidance fields and multi-anchor phi"
        )
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
        help="Output path (default: {data_path}_ours.npz)",
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        default=8,
        help="Number of anchors for phi embedding",
    )
    parser.add_argument(
        "--anchor_method",
        type=str,
        default="fps",
        choices=["fps", "grid"],
        help="Anchor selection method",
    )
    parser.add_argument(
        "--no_phi",
        action="store_true",
        help="Skip phi computation (only compute goal-centric fields)",
    )
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        base, ext = os.path.splitext(args.data_path)
        output_path = f"{base}_ours{ext}"

    process_and_save(
        args.data_path,
        output_path,
        num_anchors=args.num_anchors,
        anchor_method=args.anchor_method,
        compute_phi=not args.no_phi,
    )


if __name__ == "__main__":
    main()


# Example usage:
# python heat_method/preprocessing.py \
#     --data_path data/maze/mazes_032_moore_c8.npz \
#     --output_path data/maze_preprocessed/mazes_032_moore_c8_ours.npz \
#     --num_anchors 8