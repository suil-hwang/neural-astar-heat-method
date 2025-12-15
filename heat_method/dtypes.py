# heat_method/types.py
from __future__ import annotations

from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray


# Type Aliases
GridMap: TypeAlias = NDArray[np.float32]
"""Binary occupancy grid. Shape: (H, W). Values: 1.0=passable, 0.0=obstacle."""

VectorField: TypeAlias = NDArray[np.float32]
"""2D scalar field (one component of vector field). Shape: (H, W)."""

DistanceMap: TypeAlias = NDArray[np.float32]
"""Geodesic distance field. Shape: (H, W). Values: normalized [0, 1]."""

ReachableMap: TypeAlias = NDArray[np.float32]
"""Binary reachability mask. Shape: (H, W). Values: 1.0=reachable, 0.0=blocked."""



# Data Containers
class GeodesicGuidance(NamedTuple):
    """Container for geodesic guidance field outputs."""
    vec_x: VectorField
    vec_y: VectorField
    dist_normalized: DistanceMap
    reachable: ReachableMap


class MeshData(NamedTuple):
    """Container for triangle mesh data with grid-mesh correspondence."""
    vertices: NDArray[np.float64]       # (N, 3) float64
    faces: NDArray[np.int32]            # (M, 3) int32
    vertex_ids: NDArray[np.int32]       # (H, W) int32, grid→vertex mapping
    face_pixels: NDArray[np.int32]      # (M, 2) int32, face→pixel (row, col) of TL corner
