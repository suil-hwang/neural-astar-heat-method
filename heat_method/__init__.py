# heat_method/__init__.py

from heat_method.dtypes import (
    GridMap,
    VectorField,
    DistanceMap,
    ReachableMap,
    GeodesicGuidance,
    MeshData,
)
from heat_method.solver import GeodesicHeatSolver
from heat_method.validation import check_reachability, trace_path


__version__ = "1.0.0"
__author__ = "Heat Method Team"

__all__ = [
    # Type aliases
    "GridMap",
    "VectorField", 
    "DistanceMap",
    "ReachableMap",
    # Data containers
    "GeodesicGuidance",
    "MeshData",
    # Main solver
    "GeodesicHeatSolver",
    # Validation utilities
    "check_reachability",
    "trace_path",
]