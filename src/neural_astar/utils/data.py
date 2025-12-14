# src/neural_astar/utils/data.py
from __future__ import annotations, print_function

import numpy as np
import torch
import torch.utils.data as data
from neural_astar.planner.differentiable_astar import AstarOutput
from PIL import Image
from torchvision.utils import make_grid


def visualize_results(
    map_designs: torch.Tensor, planner_outputs: AstarOutput, scale: int = 1
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
) -> data.DataLoader:
    """Create dataloader from npz file"""

    dataset = MazeDataset(
        filename,
        split,
        num_starts=num_starts,
        geo_supervision=geo_supervision,
    )
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, persistent_workers=(num_workers > 0),
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None
    )


class MazeDataset(data.Dataset):
    def __init__(
        self,
        filename: str,
        split: str,
        pct1: float = 0.55,
        pct2: float = 0.70,
        pct3: float = 0.85,
        num_starts: int = 1,
        geo_supervision: bool = False,
    ):
        """
        Custom dataset for shortest path problems
        See planning-datasets repository for how to create original file.
        """
        assert filename.endswith("npz")  # Must be .npz format
        self.filename = filename
        self.dataset_type = split  # train, valid, test
        self.pcts = np.array([pct1, pct2, pct3, 1.0])
        self.num_starts = num_starts
        self.geo_supervision = geo_supervision

        (
            self.map_designs,
            self.goal_maps,
            self.opt_policies,
            self.opt_dists,
        ) = self._process(filename)

        self.num_actions = self.opt_policies.shape[1]
        self.num_orient = self.opt_policies.shape[2]

    def _process(self, filename: str):
        with np.load(filename) as f:
            dataset2idx = {"train": 0, "valid": 4, "test": 8}
            idx = dataset2idx[self.dataset_type]
            map_designs = f["arr_" + str(idx)]
            goal_maps = f["arr_" + str(idx + 1)]
            opt_policies = f["arr_" + str(idx + 2)]
            opt_dists = f["arr_" + str(idx + 3)]
            
            if self.geo_supervision:
                guidance_prefix = self.dataset_type
                dist_key = f"{guidance_prefix}_dist"
                reachable_key = f"{guidance_prefix}_reachable"
                # Preferred (current) naming convention
                vx_key = f"{guidance_prefix}_vec_x"
                vy_key = f"{guidance_prefix}_vec_y"

                # Backward-compatible fallback (older preprocessing outputs)
                vx_key_fallback = f"{guidance_prefix}_vx"
                vy_key_fallback = f"{guidance_prefix}_vy"

                required_base = [dist_key, reachable_key]
                missing_base = [k for k in required_base if k not in f]
                if missing_base:
                    raise ValueError(
                        f"Geodesic supervision data not found in {filename}. "
                        f"Missing keys: {missing_base}"
                    )

                vec_keys_present = (vx_key in f) and (vy_key in f)
                vec_keys_fallback_present = (
                    (vx_key_fallback in f) and (vy_key_fallback in f)
                )
                if not vec_keys_present and vec_keys_fallback_present:
                    print(
                        "WARNING: Using legacy geodesic vector field keys: "
                        f"{vx_key_fallback}, {vy_key_fallback}. "
                        f"Prefer: {vx_key}, {vy_key}"
                    )
                    vx_key, vy_key = vx_key_fallback, vy_key_fallback

                required = [dist_key, reachable_key, vx_key, vy_key]
                missing = [k for k in required if k not in f]
                if missing:
                    raise ValueError(
                        f"Geodesic supervision data not found in {filename}. "
                        f"Missing keys: {missing}. Tried vector keys: "
                        f"[{guidance_prefix}_vec_x/{guidance_prefix}_vec_y] and "
                        f"[{guidance_prefix}_vx/{guidance_prefix}_vy]."
                    )

                self.guidance_dist = f[dist_key].astype(np.float32)
                self.guidance_reachable = f[reachable_key].astype(np.float32)
                self.guidance_vx = f[vx_key].astype(np.float32)
                self.guidance_vy = f[vy_key].astype(np.float32)
                print(
                    f"Loaded geodesic targets: {dist_key}, {reachable_key}, "
                    f"{vx_key}, {vy_key}"
                )

        # Set proper datatypes
        map_designs = map_designs.astype(np.float32)
        goal_maps = goal_maps.astype(np.float32)
        opt_policies = opt_policies.astype(np.float32)
        opt_dists = opt_dists.astype(np.float32)

        # Print number of samples
        if self.dataset_type == "train":
            print("Number of Train Samples: {0}".format(map_designs.shape[0]))
        elif self.dataset_type == "valid":
            print("Number of Validation Samples: {0}".format(map_designs.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(map_designs.shape[0]))
        print("\tSize: {}x{}".format(map_designs.shape[1], map_designs.shape[2]))
        return map_designs, goal_maps, opt_policies, opt_dists

    def __getitem__(self, index: int):
        map_design = self.map_designs[index][np.newaxis]
        goal_map = self.goal_maps[index]
        opt_policy = self.opt_policies[index]
        opt_dist = self.opt_dists[index]
        start_maps, opt_trajs = [], []
        for i in range(self.num_starts):
            start_map = self.get_random_start_map(opt_dist)
            opt_traj = self.get_opt_traj(start_map, goal_map, opt_policy)
            start_maps.append(start_map)
            opt_trajs.append(opt_traj)
        start_map = np.concatenate(start_maps)
        opt_traj = np.concatenate(opt_trajs)
        
        if self.geo_supervision:
            dist = self.guidance_dist[index]  # [1, H, W]
            reachable = self.guidance_reachable[index]  # [1, H, W]
            vx = self.guidance_vx[index]  # [1, H, W]
            vy = self.guidance_vy[index]  # [1, H, W]
            if dist.ndim == 2:
                dist = dist[np.newaxis, ...]
            if reachable.ndim == 2:
                reachable = reachable[np.newaxis, ...]
            if vx.ndim == 2:
                vx = vx[np.newaxis, ...]
            if vy.ndim == 2:
                vy = vy[np.newaxis, ...]
            return map_design, start_map, goal_map, opt_traj, dist, reachable, vx, vy

        return map_design, start_map, goal_map, opt_traj

    def __len__(self):
        return self.map_designs.shape[0]

    def get_opt_traj(
        self, start_map: np.ndarray, goal_map: np.ndarray, opt_policy: np.ndarray
    ) -> np.ndarray:
        """Get optimal path from start to goal using pre-computed optimal policy."""

        opt_traj = np.zeros_like(start_map)
        opt_policy = opt_policy.transpose((1, 2, 3, 0))
        current_loc = tuple(np.array(np.nonzero(start_map)).squeeze())
        goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())
        while goal_loc != current_loc:
            opt_traj[current_loc] = 1.0
            next_loc = self.next_loc(current_loc, opt_policy[current_loc])
            assert (
                opt_traj[next_loc] == 0.0
            ), "Revisiting the same position while following the optimal policy"
            current_loc = next_loc

        return opt_traj

    def get_random_start_map(self, opt_dist: np.ndarray) -> np.ndarray:
        """
        Get random start map
        This function first chooses one of 55-70, 70-85, and 85-100 percentile intervals.
        Then it picks out a random single point from the region in the selected interval.
        """
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
        """Choose next location based on the selected action."""
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
