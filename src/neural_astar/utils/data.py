# src/neural_astar/utils/data.py
from __future__ import annotations, print_function

import numpy as np
import torch
import torch.utils.data as data
from typing import Tuple, Optional
from PIL import Image
from torchvision.utils import make_grid

from neural_astar.planner.differentiable_astar import AstarOutput


def visualize_results(
    map_designs: torch.Tensor,
    planner_outputs: AstarOutput,
    scale: int = 1,
) -> np.ndarray:
    """Create a visualization of search results."""
    if isinstance(planner_outputs, dict):
        histories = planner_outputs["histories"]
        paths = planner_outputs["paths"]
    else:
        histories = planner_outputs.histories
        paths = planner_outputs.paths

    results = make_grid(map_designs).permute(1, 2, 0)
    h = make_grid(histories).permute(1, 2, 0)
    p = make_grid(paths).permute(1, 2, 0).float()
    results[h[..., 0] == 1] = torch.tensor([0.2, 0.8, 0])
    results[p[..., 0] == 1] = torch.tensor([1.0, 0.0, 0])

    results = ((results.numpy()) * 255.0).astype("uint8")

    if scale > 1:
        results = Image.fromarray(results).resize(
            [x * scale for x in results.shape[:2]], resample=Image.NEAREST
        )
        results = np.asarray(results)

    return results


def create_dataloader(
    filename: str,
    split: str,
    batch_size: int,
    num_starts: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    geo_supervision: bool = False,
    use_phi: bool = False,
    num_anchors: int = 8,
) -> data.DataLoader:
    """
    Create dataloader from npz file.

    Args:
        filename: path to dataset npz
        split: train/valid/test
        batch_size: batch size
        num_starts: random starts per sample
        shuffle: shuffle dataset
        num_workers: DataLoader workers
        geo_supervision: load geodesic supervision targets
        use_phi: load multi-anchor phi embedding
        num_anchors: number of anchors in phi
    """
    dataset = MazeDataset(
        filename,
        split,
        num_starts=num_starts,
        geo_supervision=geo_supervision,
        use_phi=use_phi,
        num_anchors=num_anchors,
    )
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )


class MazeDataset(data.Dataset):
    """
    Dataset for shortest path problems with optional multi-anchor phi embedding.
    """

    def __init__(
        self,
        filename: str,
        split: str,
        pct1: float = 0.55,
        pct2: float = 0.70,
        pct3: float = 0.85,
        num_starts: int = 1,
        geo_supervision: bool = False,
        use_phi: bool = False,
        num_anchors: int = 8,
    ):
        assert filename.endswith("npz"), "Must be .npz format"

        self.filename = filename
        self.dataset_type = split
        self.pcts = np.array([pct1, pct2, pct3, 1.0])
        self.num_starts = num_starts
        self.geo_supervision = geo_supervision
        self.use_phi = use_phi
        self.num_anchors = num_anchors

        self._load_data(filename)

        self.num_actions = self.opt_policies.shape[1]
        self.num_orient = self.opt_policies.shape[2]

    def _load_data(self, filename: str) -> None:
        """Load all required data from npz file."""
        with np.load(filename) as f:
            dataset2idx = {"train": 0, "valid": 4, "test": 8}
            idx = dataset2idx[self.dataset_type]

            self.map_designs = f[f"arr_{idx}"].astype(np.float32)
            self.goal_maps = f[f"arr_{idx + 1}"].astype(np.float32)
            self.opt_policies = f[f"arr_{idx + 2}"].astype(np.float32)
            self.opt_dists = f[f"arr_{idx + 3}"].astype(np.float32)

            if self.geo_supervision:
                prefix = self.dataset_type
                required_keys = [
                    f"{prefix}_dist",
                    f"{prefix}_reachable",
                    f"{prefix}_vx",
                    f"{prefix}_vy",
                ]
                missing = [k for k in required_keys if k not in f]

                if missing:
                    raise ValueError(
                        "Geodesic supervision data missing. "
                        f"Required keys not found: {missing}"
                    )

                self.guidance_dist = f[f"{prefix}_dist"].astype(np.float32)
                self.guidance_reachable = f[f"{prefix}_reachable"].astype(np.float32)
                self.guidance_vx = f[f"{prefix}_vx"].astype(np.float32)
                self.guidance_vy = f[f"{prefix}_vy"].astype(np.float32)
                print(
                    f"Loaded geodesic targets for {prefix}: "
                    "dist, reachable, vx, vy"
                )

            if self.use_phi:
                phi_key = f"{self.dataset_type}_phi"
                if phi_key in f:
                    self.phi = f[phi_key].astype(np.float32)
                    print(f"Loaded phi embedding: shape {self.phi.shape}")
                    anchor_key = f"{self.dataset_type}_anchor_coords"
                    if anchor_key in f:
                        self.anchor_coords = f[anchor_key]
                    else:
                        self.anchor_coords = None
                else:
                    print(
                        f"WARNING: phi key '{phi_key}' not found. "
                        "Will compute phi on-the-fly (slower)."
                    )
                    self.phi = None
                    self.anchor_coords = None

        split_name = self.dataset_type.capitalize()
        print(f"Number of {split_name} Samples: {self.map_designs.shape[0]}")
        print(f"\tSize: {self.map_designs.shape[-2]}x{self.map_designs.shape[-1]}")

    def __len__(self) -> int:
        return self.map_designs.shape[0]

    def __getitem__(self, index: int) -> Tuple:
        """
        Return a single sample; tuple length depends on supervision flags.
        """
        map_design = self.map_designs[index][np.newaxis]
        goal_map = self.goal_maps[index]
        opt_policy = self.opt_policies[index]
        opt_dist = self.opt_dists[index]

        start_maps, opt_trajs = [], []
        for _ in range(self.num_starts):
            start_map = self.get_random_start_map(opt_dist)
            opt_traj = self.get_opt_traj(start_map, goal_map, opt_policy)
            start_maps.append(start_map)
            opt_trajs.append(opt_traj)

        start_map = np.concatenate(start_maps)
        opt_traj = np.concatenate(opt_trajs)

        result = [map_design, start_map, goal_map, opt_traj]

        if self.geo_supervision:
            dist = self._ensure_channel_dim(self.guidance_dist[index])
            reachable = self._ensure_channel_dim(self.guidance_reachable[index])
            vx = self._ensure_channel_dim(self.guidance_vx[index])
            vy = self._ensure_channel_dim(self.guidance_vy[index])
            result.extend([dist, reachable, vx, vy])

        if self.use_phi:
            if self.phi is not None:
                phi = self.phi[index]
            else:
                phi = self._compute_phi_online(index, map_design[0], goal_map)
            result.append(phi)

        return tuple(result)

    def _ensure_channel_dim(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr[np.newaxis, ...]
        return arr

    def _compute_phi_online(
        self,
        index: int,
        map_design: np.ndarray,
        goal_map: np.ndarray,
    ) -> np.ndarray:
        """
        Compute phi on-the-fly when not precomputed (slow fallback).
        """
        import sys
        import os

        sys.path.insert(
            0,
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ),
        )

        from heat_method.anchor_embedding import MultiAnchorGeodesicEmbedding

        g_coords = np.argwhere(goal_map == 1)
        if len(g_coords) > 0:
            goal_pos = (int(g_coords[0, 0]), int(g_coords[0, 1]))
        else:
            height, width = map_design.shape
            goal_pos = (height // 2, width // 2)

        embedding = MultiAnchorGeodesicEmbedding.compute_embedding(
            map_design,
            goal_pos,
            start_pos=None,
            num_anchors=self.num_anchors,
            anchor_method="fps",
            normalize=True,
        )

        phi = embedding.phi.transpose(2, 0, 1)
        return phi.astype(np.float32)

    def get_opt_traj(
        self,
        start_map: np.ndarray,
        goal_map: np.ndarray,
        opt_policy: np.ndarray,
    ) -> np.ndarray:
        """Get optimal path from start to goal using pre-computed policy."""
        opt_traj = np.zeros_like(start_map)
        opt_policy = opt_policy.transpose((1, 2, 3, 0))
        current_loc = tuple(np.array(np.nonzero(start_map)).squeeze())
        goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())

        while goal_loc != current_loc:
            opt_traj[current_loc] = 1.0
            next_loc = self.next_loc(current_loc, opt_policy[current_loc])
            assert opt_traj[next_loc] == 0.0, (
                "Revisiting position while following optimal policy"
            )
            current_loc = next_loc

        return opt_traj

    def get_random_start_map(self, opt_dist: np.ndarray) -> np.ndarray:
        """Sample random start map using distance percentiles."""
        od_vct = opt_dist.flatten()
        od_vals = od_vct[od_vct > od_vct.min()]
        od_th = np.percentile(od_vals, 100.0 * (1 - self.pcts))
        r = np.random.randint(0, len(od_th) - 1)
        start_candidate = (od_vct >= od_th[r + 1]) & (od_vct <= od_th[r])
        start_idx = np.random.choice(np.where(start_candidate)[0])
        start_map = np.zeros_like(opt_dist)
        start_map.ravel()[start_idx] = 1.0
        return start_map

    def next_loc(self, current_loc: tuple, one_hot_action: np.ndarray) -> tuple:
        """Choose next location based on action."""
        action_to_move = [
            (0, -1, 0),
            (0, 0, +1),
            (0, 0, -1),
            (0, +1, 0),
            (0, -1, +1),
            (0, -1, -1),
            (0, +1, +1),
            (0, +1, -1),
        ]
        move = action_to_move[np.argmax(one_hot_action)]
        return tuple(np.add(current_loc, move))


def collate_fn_with_phi(batch):
    """
    Collate batches with variable tuple lengths (optional phi/geodesic fields).
    """
    batch_len = len(batch[0])
    result = []
    for i in range(batch_len):
        tensors = [torch.from_numpy(item[i]) for item in batch]
        result.append(torch.stack(tensors))
    return tuple(result)
