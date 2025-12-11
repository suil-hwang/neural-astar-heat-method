# heat_method/types.py
from __future__ import annotations

import numpy as np
from typing import Tuple, NamedTuple


# Type Aliases

GridMap = np.ndarray        # (H, W) float32, 1=passable, 0=obstacle
VectorField = np.ndarray    # (H, W) float32
DistanceMap = np.ndarray    # (H, W) float32
ReachableMap = np.ndarray   # (H, W) float32



# NamedTuples

class GeodesicGuidance(NamedTuple):
    """Container for geodesic guidance field outputs."""
    vec_x: VectorField
    vec_y: VectorField
    dist_normalized: DistanceMap
    reachable: ReachableMap


class MeshData(NamedTuple):
    """Container for mesh conversion outputs with face-to-pixel mapping."""
    vertices: np.ndarray      # (N, 3) float64
    faces: np.ndarray         # (M, 3) int32
    vertex_ids: np.ndarray    # (H, W) int32, grid→vertex mapping
    face_pixels: np.ndarray   # (M, 2) int32, face→pixel (row, col) of TL corner
