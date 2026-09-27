"""High-level O-grid generation workflow."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .exporters import write_fluent_msh, write_su2
from .geometry import fit_farfield, prepare_surface_with_diagnostics, wall_distance
from .models import (
    AirfoilMesh,
    AirfoilProfile,
    GeometryConditioningMode,
    MeshSettings,
    SpacingMethod,
    SurfacePointMode,
)
from .quality import compute_quality
from .solvers import (
    HyperbolicMeshError,
    _reported_algebraic_fallback,
    apply_normal_spacing,
    improve_near_wall_orthogonality,
    march_hyperbolic,
)


class GridGenerator:
    """Generate one periodic O-grid without compiled code."""

    def __init__(
        self, airfoil: AirfoilProfile | str | Path, settings: MeshSettings | None = None
    ):
        self.airfoil = (
            airfoil
            if isinstance(airfoil, AirfoilProfile)
            else AirfoilProfile.read(airfoil)
        )
        self.settings = (settings or MeshSettings()).validated(self.airfoil)
        self.mesh: AirfoilMesh | None = None
        self.geometry_diagnostics: dict[str, object] | None = None

    @property
    def first_layer_wall_distance(self) -> float:
        """Full first-layer height in chord units, not wall-to-cell-centre distance."""
        return wall_distance(
            self.settings.wall_y_plus_target,
            self.settings.flow_reynolds_number,
            self.settings.wall_reference_length_chords,
        )

    def run(self) -> AirfoilMesh:
        self.mesh = None
        wall, corners, diagnostics = prepare_surface_with_diagnostics(
            self.airfoil, self.settings
        )
        self.geometry_diagnostics = diagnostics
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                nodes = march_hyperbolic(
                    wall,
                    self.settings,
                    corner_indices=corners
                    if self.airfoil.has_trailing_edge_gap
                    else (),
                )
            for warning in caught:
                warnings.warn(warning.message, warning.category, stacklevel=2)
            if any("fallback mesh" in str(warning.message) for warning in caught):
                diagnostics["failure_classification"] = "HYPERBOLIC_MARCHING_FAILED"
                diagnostics["fallback_reason"] = "internal marching acceptance failure"
        except (ValueError, np.linalg.LinAlgError) as error:
            warnings.warn(
                f"Hyperbolic calculation failed: {error}", RuntimeWarning, stacklevel=2
            )
            diagnostics["failure_classification"] = "HYPERBOLIC_MARCHING_FAILED"
            diagnostics["fallback_reason"] = str(error)
            nodes = _reported_algebraic_fallback(wall, self.settings)
        nodes = fit_farfield(nodes, self.settings)
        nodes = apply_normal_spacing(nodes, self.settings)
        nodes = improve_near_wall_orthogonality(nodes)
        nodes[:, -1] = nodes[:, 0]
        if not np.all(np.isfinite(nodes)):
            raise HyperbolicMeshError(
                "cannot export a mesh with non-finite coordinates"
            )
        quality = compute_quality(nodes)
        if not np.any(np.abs(quality.signed_cell_area) > 0.0):
            raise HyperbolicMeshError("cannot export a completely collapsed mesh")
        issues = quality.issues()
        if issues:
            warnings.warn(
                "Mesh retained for export: " + "; ".join(issues),
                RuntimeWarning,
                stacklevel=2,
            )
        result = AirfoilMesh(nodes, quality, diagnostics)
        self.mesh = result
        return result

    def write(
        self,
        directory: str | Path,
        project_name: str | None = None,
        *,
        su2: bool = True,
        cgns: bool = False,
        vtu: bool = False,
        msh: bool = False,
    ) -> dict[str, Path]:
        if self.mesh is None:
            self.run()
        assert self.mesh is not None
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        name = project_name or self.airfoil.name
        if Path(name).name != name:
            raise ValueError("project_name must be a file name, not a path")
        files: dict[str, Path] = {}
        if su2:
            files["su2"] = output / f"{name}.su2"
            write_su2(files["su2"], self.mesh)
        if cgns:
            from .exporters import write_cgns

            files["cgns"] = output / f"{name}.cgns"
            write_cgns(files["cgns"], self.mesh)
        if vtu:
            from .exporters import write_vtu

            files["vtu"] = output / f"{name}_quality.vtu"
            write_vtu(files["vtu"], self.mesh)
        if msh:
            files["msh"] = output / f"{name}.msh"
            write_fluent_msh(files["msh"], self.mesh)
        return files


def generate_mesh(
    airfoil: AirfoilProfile | str | Path,
    *,
    output_directory: str | Path | None = None,
    project_name: str | None = None,
    write_su2_output: bool = True,
    write_cgns_output: bool = False,
    write_vtu_output: bool = False,
    write_msh_output: bool = False,
    circumferential_node_count: int = 401,
    leading_edge_cell_length: float | None = 1.130001e-3,
    trailing_edge_cell_length: float | None = 5.075471e-4,
    trailing_edge_face_cell_count: int = 19,
    farfield_radius_chords: float = 50.0,
    farfield_angular_bias: float = 1.0,
    wall_normal_node_count: int = 251,
    surface_point_mode: SurfacePointMode = "REDISTRIBUTE",
    spacing_method: SpacingMethod = "BERNSTEIN3",
    geometry_conditioning_mode: GeometryConditioningMode = "AUTO",
    geometry_deviation_tolerance_chords: float = 2.0e-4,
    geometry_max_refinement_depth: int = 10,
    wall_y_plus_target: float = 1.0,
    flow_reynolds_number: float = 9.0e6,
    wall_reference_length_chords: float = 1.0,
    hyperbolic_normal_coupling: float = 4.22,
    hyperbolic_implicit_smoothing: float = 35.60,
    hyperbolic_explicit_smoothing: float = 0.69,
    farfield_uniformity_weight: float = 0.14,
    hyperbolic_area_smoothing_passes: int = 46,
    hyperbolic_max_pseudo_aspect_ratio: float = 8.0,
) -> tuple[AirfoilMesh, dict[str, Path]]:
    """Generate one O-grid and optionally write its output files.

    flow_reynolds_number is U_inf*L/nu, with L/chord supplied by
    wall_reference_length_chords (default 1: full-chord Reynolds number).
    wall_y_plus_target is the desired wall-adjacent cell-centre y-plus.
    These inputs set the full first-layer height using the turbulent
    skin-friction correlation in geometry.wall_distance; the actual solved
    local y-plus must still be checked against the resulting wall shear.
    """
    settings = MeshSettings(
        circumferential_node_count=circumferential_node_count,
        leading_edge_cell_length=leading_edge_cell_length,
        trailing_edge_cell_length=trailing_edge_cell_length,
        trailing_edge_face_cell_count=trailing_edge_face_cell_count,
        farfield_radius_chords=farfield_radius_chords,
        farfield_angular_bias=farfield_angular_bias,
        wall_normal_node_count=wall_normal_node_count,
        surface_point_mode=surface_point_mode,
        spacing_method=spacing_method,
        geometry_conditioning_mode=geometry_conditioning_mode,
        geometry_deviation_tolerance_chords=geometry_deviation_tolerance_chords,
        geometry_max_refinement_depth=geometry_max_refinement_depth,
        wall_y_plus_target=wall_y_plus_target,
        flow_reynolds_number=flow_reynolds_number,
        wall_reference_length_chords=wall_reference_length_chords,
        hyperbolic_normal_coupling=hyperbolic_normal_coupling,
        hyperbolic_implicit_smoothing=hyperbolic_implicit_smoothing,
        hyperbolic_explicit_smoothing=hyperbolic_explicit_smoothing,
        farfield_uniformity_weight=farfield_uniformity_weight,
        hyperbolic_area_smoothing_passes=hyperbolic_area_smoothing_passes,
        hyperbolic_max_pseudo_aspect_ratio=hyperbolic_max_pseudo_aspect_ratio,
    )
    generator = GridGenerator(airfoil, settings)
    mesh = generator.run()
    files = (
        generator.write(
            output_directory,
            project_name,
            su2=write_su2_output,
            cgns=write_cgns_output,
            vtu=write_vtu_output,
            msh=write_msh_output,
        )
        if output_directory is not None
        else {}
    )
    return mesh, files
