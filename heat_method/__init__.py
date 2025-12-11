# heat_method/__init__.py

from .dtypes import (
    GridMap,
    VectorField,
    DistanceMap,
    ReachableMap,
    GeodesicGuidance,
    MeshData,
)
from .solver import GeodesicHeatSolver
from .validation import check_reachability
from .visualization import visualize_result


__all__ = [
    # Classes
    "GeodesicHeatSolver",
    # Type aliases
    "GridMap",
    "VectorField",
    "DistanceMap",
    "ReachableMap",
    # NamedTuples
    "GeodesicGuidance",
    "MeshData",
    # Functions
    "check_reachability",
    "visualize_result",
]

__version__ = "1.0.0"
