from __future__ import annotations

"""
Multi-anchor geodesic embedding for 2D navigation grids.

Based on NGD-Transformer (Zhuang et al., 2022):
- Select M anchor points (goal/start + farthest point sampling or grid)
- Compute geodesic distance from each anchor to all cells
- Stack distances into phi(i) in R^M for each cell i
"""

import numpy as np
from typing import Tuple, List, Optional, NamedTuple

from .solver import GeodesicHeatSolver
from .dtypes import GridMap


class AnchorEmbedding(NamedTuple):
    """Container for multi-anchor geodesic embedding outputs."""

    phi: np.ndarray
    anchor_coords: np.ndarray
    anchor_types: List[str]


class MultiAnchorGeodesicEmbedding:
    """Compute navigation geodesic embedding phi(i) for 2D grids."""

    @staticmethod
    def select_anchors_fps(
        grid_map: GridMap,
        goal_pos: Tuple[int, int],
        start_pos: Optional[Tuple[int, int]] = None,
        num_additional: int = 6,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Select anchor points using farthest point sampling.
        """
        np.random.seed(seed)
        height, width = grid_map.shape

        free_cells = np.argwhere(grid_map == 1)
        if len(free_cells) == 0:
            return np.array([[goal_pos[0], goal_pos[1]]])

        anchors = [np.array(goal_pos)]
        if start_pos is not None:
            anchors.append(np.array(start_pos))

        remaining = num_additional

        while remaining > 0 and len(free_cells) > len(anchors):
            anchor_array = np.array(anchors)
            dists = np.sqrt(
                ((free_cells[:, None, :] - anchor_array[None, :, :]) ** 2).sum(axis=2)
            )
            min_dists = dists.min(axis=1)

            farthest_idx = np.argmax(min_dists)
            new_anchor = free_cells[farthest_idx]

            if min_dists[farthest_idx] < 2.0:
                break

            anchors.append(new_anchor)
            remaining -= 1

        return np.array(anchors)

    @staticmethod
    def select_anchors_grid(
        grid_map: GridMap,
        goal_pos: Tuple[int, int],
        start_pos: Optional[Tuple[int, int]] = None,
        grid_divisions: int = 3,
    ) -> np.ndarray:
        """
        Select anchors on a regular grid pattern (simpler alternative to FPS).
        """
        height, width = grid_map.shape

        anchors = [np.array(goal_pos)]
        if start_pos is not None:
            anchors.append(np.array(start_pos))

        rows = np.linspace(0, height - 1, grid_divisions + 2)[1:-1].astype(int)
        cols = np.linspace(0, width - 1, grid_divisions + 2)[1:-1].astype(int)

        for r in rows:
            for c in cols:
                if grid_map[r, c] == 1:
                    anchors.append(np.array([r, c]))
                else:
                    for dr in range(-3, 4):
                        for dc in range(-3, 4):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < height and 0 <= nc < width and grid_map[nr, nc] == 1:
                                anchors.append(np.array([nr, nc]))
                                break
                        else:
                            continue
                        break

        unique_anchors = []
        for anchor in anchors:
            is_dup = False
            for u_anchor in unique_anchors:
                if np.allclose(anchor, u_anchor):
                    is_dup = True
                    break
            if not is_dup:
                unique_anchors.append(anchor)

        return np.array(unique_anchors)

    @classmethod
    def compute_embedding(
        cls,
        grid_map: GridMap,
        goal_pos: Tuple[int, int],
        start_pos: Optional[Tuple[int, int]] = None,
        num_anchors: int = 8,
        anchor_method: str = "fps",
        normalize: bool = True,
    ) -> AnchorEmbedding:
        """
        Compute multi-anchor geodesic embedding phi(i) for all cells.
        """
        height, width = grid_map.shape

        num_additional = num_anchors - (2 if start_pos else 1)
        num_additional = max(0, num_additional)

        if anchor_method == "fps":
            anchors = cls.select_anchors_fps(
                grid_map, goal_pos, start_pos, num_additional
            )
        else:
            anchors = cls.select_anchors_grid(
                grid_map, goal_pos, start_pos, grid_divisions=int(np.sqrt(num_additional))
            )

        num_selected = len(anchors)
        phi = np.zeros((height, width, num_selected), dtype=np.float32)
        anchor_types: List[str] = []

        for idx, anchor in enumerate(anchors):
            anchor_row, anchor_col = int(anchor[0]), int(anchor[1])
            result = GeodesicHeatSolver.compute_guidance(
                grid_map, (anchor_row, anchor_col)
            )
            if normalize:
                phi[:, :, idx] = result.dist_normalized
            else:
                diagonal = np.sqrt(height**2 + width**2)
                phi[:, :, idx] = result.dist_normalized * diagonal

            if idx == 0:
                anchor_types.append("goal")
            elif idx == 1 and start_pos is not None:
                anchor_types.append("start")
            else:
                anchor_types.append(f"fps_{idx}")

        return AnchorEmbedding(
            phi=phi,
            anchor_coords=anchors,
            anchor_types=anchor_types,
        )

    @staticmethod
    def compute_pairwise_kernel(
        phi: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        """
        Compute pairwise geodesic kernel matrix k_geo(i,j).
        """
        height, width, num_anchors = phi.shape
        num_nodes = height * width

        phi_flat = phi.reshape(num_nodes, num_anchors)
        phi_sq = (phi_flat ** 2).sum(axis=1)
        pairwise_sq = phi_sq[:, None] + phi_sq[None, :] - 2 * (phi_flat @ phi_flat.T)
        pairwise_sq = np.maximum(pairwise_sq, 0)

        k_geo = np.exp(-alpha * pairwise_sq / np.sqrt(2 * num_anchors))
        return k_geo.astype(np.float32)


def compute_and_cache_embeddings(
    grid_map: GridMap,
    goal_pos: Tuple[int, int],
    start_pos: Optional[Tuple[int, int]] = None,
    num_anchors: int = 8,
) -> dict:
    """
    Convenience function to compute geodesic supervision targets.
    """
    guidance = GeodesicHeatSolver.compute_guidance(grid_map, goal_pos)

    embedding = MultiAnchorGeodesicEmbedding.compute_embedding(
        grid_map, goal_pos, start_pos, num_anchors
    )

    return {
        "phi": embedding.phi,
        "anchor_coords": embedding.anchor_coords,
        "anchor_types": embedding.anchor_types,
        "vec_x": guidance.vec_x,
        "vec_y": guidance.vec_y,
        "dist": guidance.dist_normalized,
        "reachable": guidance.reachable,
    }

