"""Pure Python structured airfoil mesh generator.

It needs no external meshing executable or custom compiled kernel.
"""

from .exporters import write_cgns, write_fluent_msh, write_vtu
from .generator import GridGenerator, generate_mesh
from .models import AirfoilMesh, AirfoilProfile, DualMetrics, MeshMetrics, MeshSettings
from .optimization import (
    MeshOptimizationTrial,
    OptimizedMeshResult,
    generate_optimized_mesh,
)

__all__ = [
    "AirfoilMesh",
    "GridGenerator",
    "AirfoilProfile",
    "DualMetrics",
    "MeshMetrics",
    "MeshSettings",
    "MeshOptimizationTrial",
    "OptimizedMeshResult",
    "generate_mesh",
    "generate_optimized_mesh",
    "write_cgns",
    "write_fluent_msh",
    "write_vtu",
]

__version__ = "1.0.0"
