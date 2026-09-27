"""Validated data models for O-grid airfoil meshes."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Literal

import numpy as np

from .geometry import AirfoilProfile, wall_distance
from .utils import finite_report, node_signature

NEIGHBOR_DIRECTIONS = (
    "circumferential_previous",
    "circumferential_next",
    "wallward",
    "outward",
)
ISSUE_NAMES = (
    "inverted",
    "degenerate",
    "invalid_corner",
    "nonfinite",
    "high_skew",
    "overlap",
    "overlap_unchecked",
)

MINIMUM_CELL_ORTHOGONAL_QUALITY = 0.01
HARD_INVALID_ISSUES = (
    "inverted",
    "degenerate",
    "invalid_corner",
    "nonfinite",
    "overlap",
)


@dataclass(frozen=True)
class CellMetrics:
    """One row per zero-based cell ID; arrays remain in memory and read-only."""

    ids: np.ndarray
    node_ids: np.ndarray
    indices: np.ndarray
    centers: np.ndarray
    signed_area: np.ndarray
    corner_jacobians: np.ndarray
    face_orthogonality: np.ndarray
    neighbor_orthogonality: np.ndarray
    cell_orthogonal_quality: np.ndarray
    skew_degrees: np.ndarray
    edge_aspect_ratio: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_area_ratios: np.ndarray
    issue_flags: np.ndarray
    overlap_pairs: np.ndarray

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            getattr(self, descriptor.name).setflags(write=False)

    @property
    def poor_quality(self) -> np.ndarray:
        """Flag invalid cells or orthogonal quality at or below 0.01."""
        hard_columns = [ISSUE_NAMES.index(name) for name in HARD_INVALID_ISSUES]
        return (
            np.any(self.issue_flags[:, hard_columns], axis=1)
            | ~np.isfinite(self.cell_orthogonal_quality)
            | (self.cell_orthogonal_quality <= MINIMUM_CELL_ORTHOGONAL_QUALITY)
        )

    @property
    def problem_cell_ids(self) -> np.ndarray:
        return self.ids[self.poor_quality]

    def report(self, cell_id: int) -> dict[str, object]:
        """Look up a cell's location, metrics, connectivity, and issue labels."""
        if isinstance(cell_id, bool) or not isinstance(cell_id, int | np.integer):
            raise TypeError("cell_id must be an integer")
        if not 0 <= cell_id < len(self.ids):
            raise IndexError("cell_id is outside this mesh")
        report = {
            field.name: getattr(self, field.name)[cell_id].tolist()
            for field in fields(self)
            if field.name != "overlap_pairs"
        }
        report["cell_id"] = report.pop("ids")
        report["center"] = report.pop("centers")
        report["poor_quality"] = bool(self.poor_quality[cell_id])
        report["issues"] = [
            name
            for name, flagged in zip(
                ISSUE_NAMES, self.issue_flags[cell_id], strict=True
            )
            if flagged
        ]
        report["neighbor_directions"] = NEIGHBOR_DIRECTIONS
        pairs = self.overlap_pairs[np.any(self.overlap_pairs == cell_id, axis=1)]
        report["overlapping_cell_ids"] = pairs[pairs != cell_id].tolist()
        return finite_report(report)

    def summary(self) -> dict[str, float | int | None]:
        """Count the same per-cell flags used in reports and VTU output."""
        counts = dict(
            zip(ISSUE_NAMES, np.count_nonzero(self.issue_flags, axis=0), strict=True)
        )
        finite_areas = np.abs(self.signed_area[np.isfinite(self.signed_area)])
        finite_quality = self.cell_orthogonal_quality[
            np.isfinite(self.cell_orthogonal_quality)
        ]
        finite_face_orthogonality = self.face_orthogonality[
            np.isfinite(self.face_orthogonality)
        ]
        finite_neighbor_orthogonality = self.neighbor_orthogonality[
            np.isfinite(self.neighbor_orthogonality)
        ]
        worst_quality_id = (
            int(self.ids[np.nanargmin(self.cell_orthogonal_quality)])
            if finite_quality.size
            else None
        )
        return {
            "cell_count": len(self.ids),
            "inverted_cell_count": int(counts["inverted"]),
            "degenerate_cell_count": int(counts["degenerate"]),
            "invalid_corner_cell_count": int(counts["invalid_corner"]),
            "nonfinite_cell_count": int(counts["nonfinite"]),
            "high_skew_cell_count": int(counts["high_skew"]),
            "overlapping_cell_count": int(counts["overlap"]),
            "overlap_pair_count": len(self.overlap_pairs),
            "overlap_unchecked_cell_count": int(counts["overlap_unchecked"]),
            "poor_quality_cell_count": int(np.count_nonzero(self.poor_quality)),
            "minimum_absolute_cell_area": float(finite_areas.min())
            if finite_areas.size
            else 0.0,
            "maximum_skew_degrees": float(np.nanmax(self.skew_degrees)),
            "minimum_cell_orthogonal_quality": (
                float(finite_quality.min()) if finite_quality.size else None
            ),
            "minimum_cell_orthogonal_quality_cell_id": worst_quality_id,
            "mean_cell_orthogonal_quality": (
                float(finite_quality.mean()) if finite_quality.size else None
            ),
            "median_cell_orthogonal_quality": (
                float(np.median(finite_quality)) if finite_quality.size else None
            ),
            "first_percentile_cell_orthogonal_quality": (
                float(np.percentile(finite_quality, 1.0))
                if finite_quality.size
                else None
            ),
            "fifth_percentile_cell_orthogonal_quality": (
                float(np.percentile(finite_quality, 5.0))
                if finite_quality.size
                else None
            ),
            "minimum_face_orthogonality": (
                float(finite_face_orthogonality.min())
                if finite_face_orthogonality.size
                else None
            ),
            "minimum_neighbor_orthogonality": (
                float(finite_neighbor_orthogonality.min())
                if finite_neighbor_orthogonality.size
                else None
            ),
            "maximum_neighbor_skewness_degrees": (
                float(
                    np.degrees(
                        np.arccos(
                            np.clip(finite_neighbor_orthogonality.min(), -1.0, 1.0)
                        )
                    )
                )
                if finite_neighbor_orthogonality.size
                else None
            ),
        }

    def issues(self) -> list[str]:
        counts = np.count_nonzero(self.issue_flags, axis=0)
        result = [
            f"{int(count)} cells: {name.replace('_', ' ')}"
            for name, count in zip(ISSUE_NAMES, counts, strict=True)
            if count
        ]
        if count := np.count_nonzero(self.poor_quality):
            result.append(
                f"{int(count)} cells: orthogonal quality at or below "
                f"{MINIMUM_CELL_ORTHOGONAL_QUALITY:g} or hard invalid"
            )
        return result


SurfacePointMode = Literal["REDISTRIBUTE", "PRESERVE"]
SpacingMethod = Literal["BERNSTEIN3", "SIGMOID5", "HYBRID"]
GeometryConditioningMode = Literal[
    "AUTO", "SEGMENTED_HERMITE", "POLYLINE", "NATURAL_CUBIC_LEGACY"
]


@dataclass(frozen=True)
class MeshSettings:
    """Control O-grid surface spacing, numerical solution, and file output."""

    circumferential_node_count: int = 401
    leading_edge_cell_length: float | None = 1.130001e-3
    trailing_edge_cell_length: float | None = 5.075471e-4
    trailing_edge_face_cell_count: int = 19
    farfield_radius_chords: float = 50.0
    farfield_angular_bias: float = 1.0
    wall_normal_node_count: int = 251
    surface_point_mode: SurfacePointMode = "REDISTRIBUTE"
    spacing_method: SpacingMethod = "BERNSTEIN3"
    geometry_conditioning_mode: GeometryConditioningMode = "AUTO"
    geometry_deviation_tolerance_chords: float = 2.0e-4
    geometry_max_refinement_depth: int = 10
    wall_y_plus_target: float = 1.0
    flow_reynolds_number: float = 9.0e6
    wall_reference_length_chords: float = 1.0
    hyperbolic_normal_coupling: float = 4.22
    hyperbolic_implicit_smoothing: float = 35.60
    hyperbolic_explicit_smoothing: float = 0.69
    farfield_uniformity_weight: float = 0.14
    hyperbolic_area_smoothing_passes: int = 46
    hyperbolic_max_pseudo_aspect_ratio: float = 8.0

    def validated(self, airfoil: AirfoilProfile | None = None) -> MeshSettings:
        point_mode = self.surface_point_mode.upper()
        if point_mode not in {"REDISTRIBUTE", "PRESERVE"}:
            raise ValueError("surface_point_mode must be REDISTRIBUTE or PRESERVE")
        spacing_method = self.spacing_method.upper()
        if spacing_method not in {"BERNSTEIN3", "SIGMOID5", "HYBRID"}:
            raise ValueError("spacing_method must be BERNSTEIN3, SIGMOID5, or HYBRID")
        conditioning_mode = self.geometry_conditioning_mode.upper()
        conditioning_modes = {
            "AUTO",
            "SEGMENTED_HERMITE",
            "POLYLINE",
            "NATURAL_CUBIC_LEGACY",
        }
        if conditioning_mode not in conditioning_modes:
            raise ValueError(
                "geometry_conditioning_mode must be AUTO, SEGMENTED_HERMITE, "
                "POLYLINE, or NATURAL_CUBIC_LEGACY"
            )
        integer_values = (
            self.circumferential_node_count,
            self.trailing_edge_face_cell_count,
            self.wall_normal_node_count,
            self.hyperbolic_area_smoothing_passes,
            self.geometry_max_refinement_depth,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int | np.integer)
            for value in integer_values
        ):
            raise TypeError("node, cell, iteration, and output counts must be integers")
        if self.circumferential_node_count < 20 or self.wall_normal_node_count < 3:
            raise ValueError(
                "at least 20 circumferential and 3 wall-normal nodes are required"
            )
        if self.trailing_edge_face_cell_count < 1:
            raise ValueError("trailing_edge_face_cell_count must be at least 1")
        if self.geometry_max_refinement_depth < 1:
            raise ValueError("geometry_max_refinement_depth must be at least 1")
        if self.geometry_deviation_tolerance_chords <= 0.0:
            raise ValueError("geometry_deviation_tolerance_chords must be positive")
        if (
            airfoil is not None
            and airfoil.has_trailing_edge_gap
            and self.trailing_edge_face_cell_count < 16
        ):
            raise ValueError(
                "trailing_edge_face_cell_count must be at least 16 for a blunt "
                "trailing edge"
            )
        if self.circumferential_node_count < self.trailing_edge_face_cell_count + 6:
            raise ValueError("circumferential_node_count is too small for the TE face")
        numeric_values = (
            self.farfield_radius_chords,
            self.farfield_angular_bias,
            self.wall_y_plus_target,
            self.flow_reynolds_number,
            self.wall_reference_length_chords,
            self.hyperbolic_normal_coupling,
            self.hyperbolic_implicit_smoothing,
            self.hyperbolic_explicit_smoothing,
            self.farfield_uniformity_weight,
            self.hyperbolic_max_pseudo_aspect_ratio,
            self.geometry_deviation_tolerance_chords,
        )
        if not all(np.isfinite(value) for value in numeric_values):
            raise ValueError("all numerical settings must be finite")
        if self.farfield_radius_chords < 20.0:
            raise ValueError("farfield_radius_chords must be at least 20")
        positive_values = (
            self.farfield_angular_bias,
            self.wall_y_plus_target,
            self.wall_reference_length_chords,
        )
        if min(positive_values) <= 0.0:
            raise ValueError(
                "farfield bias, wall y-plus, and reference length must be positive"
            )
        wall_distance(
            self.wall_y_plus_target,
            self.flow_reynolds_number,
            self.wall_reference_length_chords,
        )
        if self.hyperbolic_normal_coupling < 0.5:
            raise ValueError("hyperbolic_normal_coupling must be at least 0.5")
        if (
            min(self.hyperbolic_implicit_smoothing, self.hyperbolic_explicit_smoothing)
            < 0.0
        ):
            raise ValueError("hyperbolic smoothing values cannot be negative")
        if not 0.0 <= self.farfield_uniformity_weight <= 1.0:
            raise ValueError("farfield_uniformity_weight must be in [0, 1]")
        if self.hyperbolic_area_smoothing_passes < 0:
            raise ValueError("hyperbolic_area_smoothing_passes cannot be negative")
        if self.hyperbolic_max_pseudo_aspect_ratio <= 0.0:
            raise ValueError("hyperbolic_max_pseudo_aspect_ratio must be positive")
        cell_lengths = (self.leading_edge_cell_length, self.trailing_edge_cell_length)
        if any(
            value is not None and (not np.isfinite(value) or value <= 0.0)
            for value in cell_lengths
        ):
            raise ValueError(
                "leading-edge and trailing-edge cell lengths must be finite and "
                "positive"
            )
        return replace(
            self,
            surface_point_mode=point_mode,
            spacing_method=spacing_method,
            geometry_conditioning_mode=conditioning_mode,
        )


@dataclass(frozen=True)
class DualMetrics:
    """Node-based dual-volume metrics, indexed by the exported node IDs."""

    orthogonality_degrees: np.ndarray
    face_area_ratio: np.ndarray
    subvolume_ratio: np.ndarray

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            getattr(self, descriptor.name).setflags(write=False)

    @property
    def undefined(self) -> np.ndarray:
        return ~np.isfinite(
            np.column_stack(
                (self.orthogonality_degrees, self.face_area_ratio, self.subvolume_ratio)
            )
        ).all(axis=1)

    def report(self, node_id: int) -> dict[str, object]:
        """Look up metrics for one unique mesh node."""
        if isinstance(node_id, bool) or not isinstance(node_id, int | np.integer):
            raise TypeError("node_id must be an integer")
        if not 0 <= node_id < len(self.orthogonality_degrees):
            raise IndexError("node_id is outside this mesh")
        return finite_report(
            {
                "node_id": node_id,
                **{
                    item.name: getattr(self, item.name)[node_id]
                    for item in fields(self)
                },
            }
        )

    def summary(self) -> dict[str, float | int | None]:
        """Report finite extrema and count nodes with undefined metrics."""
        result: dict[str, float | int | None] = {
            "undefined_dual_node_count": int(np.count_nonzero(self.undefined))
        }
        for descriptor in fields(self):
            values = getattr(self, descriptor.name)
            values = values[np.isfinite(values)]
            result[f"minimum_dual_{descriptor.name}"] = (
                float(values.min()) if values.size else None
            )
            result[f"maximum_dual_{descriptor.name}"] = (
                float(values.max()) if values.size else None
            )
        return result


@dataclass(frozen=True)
class MeshMetrics:
    skew_degrees: np.ndarray
    tangential_growth: np.ndarray
    normal_growth: np.ndarray
    signed_cell_area: np.ndarray
    corner_jacobians: np.ndarray
    cells: CellMetrics
    _node_signature: bytes | None = field(default=None, repr=False, compare=False)
    dual: DualMetrics | None = None

    def __post_init__(self) -> None:
        for descriptor in fields(self):
            value = getattr(self, descriptor.name)
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def issues(self) -> list[str]:
        """Describe quality concerns without rejecting the mesh."""
        issues = self.cells.issues()
        if self.dual is not None and (count := np.count_nonzero(self.dual.undefined)):
            issues.append(f"{count} nodes: undefined dual-volume quality")
        return issues

    def summary(self) -> dict[str, float | int | None]:
        return {
            **self.cells.summary(),
            "maximum_absolute_tangential_growth": float(
                np.nanmax(np.abs(self.tangential_growth))
            ),
            "maximum_absolute_normal_growth": float(
                np.nanmax(np.abs(self.normal_growth))
            ),
            **(self.dual.summary() if self.dual is not None else {}),
        }

    def compact_summary(self) -> dict[str, float | int | None]:
        """Return the compact CFD quality summary."""

        def extrema(values: np.ndarray) -> tuple[float | None, float | None]:
            finite = np.asarray(values)[np.isfinite(values)]
            if not finite.size:
                return None, None
            return float(finite.min()), float(finite.max())

        hard_columns = [ISSUE_NAMES.index(name) for name in HARD_INVALID_ISSUES]
        invalid = np.any(self.cells.issue_flags[:, hard_columns], axis=1)
        significant_areas = self.cells.signed_area[
            np.isfinite(self.cells.signed_area) & (self.cells.signed_area != 0.0)
        ]
        orientation = (
            1.0
            if np.count_nonzero(significant_areas > 0.0)
            >= np.count_nonzero(significant_areas < 0.0)
            else -1.0
        )
        minimum_jacobian, maximum_jacobian = extrema(
            orientation * self.cells.corner_jacobians
        )
        minimum_quality, maximum_quality = extrema(self.cells.cell_orthogonal_quality)
        minimum_area_ratio, maximum_area_ratio = extrema(
            self.cells.neighbor_area_ratios
        )
        minimum_aspect_ratio, maximum_aspect_ratio = extrema(
            self.cells.edge_aspect_ratio
        )
        dual_orthogonality = (
            self.dual.orthogonality_degrees
            if self.dual is not None
            else np.asarray([], dtype=float)
        )
        minimum_dual, maximum_dual = extrema(dual_orthogonality)
        return {
            "cell_count": len(self.cells.ids),
            "invalid_cell_count": int(np.count_nonzero(invalid)),
            "minimum_scaled_corner_jacobian": minimum_jacobian,
            "maximum_scaled_corner_jacobian": maximum_jacobian,
            "minimum_cell_orthogonal_quality": minimum_quality,
            "maximum_cell_orthogonal_quality": maximum_quality,
            "minimum_dual_orthogonality_degrees": minimum_dual,
            "maximum_dual_orthogonality_degrees": maximum_dual,
            "minimum_neighbor_area_ratio": minimum_area_ratio,
            "maximum_neighbor_area_ratio": maximum_area_ratio,
            "minimum_edge_aspect_ratio": minimum_aspect_ratio,
            "maximum_edge_aspect_ratio": maximum_aspect_ratio,
        }


@dataclass(frozen=True)
class AirfoilMesh:
    """Store one periodic O-grid as wall-normal, circumferential coordinates."""

    nodes: np.ndarray
    quality: MeshMetrics | None = None
    geometry_diagnostics: dict[str, object] | None = None
    cells: CellMetrics = field(init=False)

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float).copy()
        if nodes.ndim != 3 or nodes.shape[2] != 2 or min(nodes.shape[:2]) < 3:
            raise ValueError(
                "grid nodes must have shape (normal, circumferential, coordinate)"
            )
        if not np.all(np.isfinite(nodes)):
            raise ValueError("grid contains non-finite coordinates")
        if not np.array_equal(nodes[:, 0], nodes[:, -1]):
            raise ValueError("O-grid seam nodes must match exactly")
        nodes.setflags(write=False)
        object.__setattr__(self, "nodes", nodes)
        from .quality import compute_quality

        quality = self.quality
        # Reuse quality data only for the same coordinates.
        if (
            quality is None
            or quality.dual is None
            or quality._node_signature != node_signature(nodes)
        ):
            quality = compute_quality(nodes)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "cells", quality.cells)
        if self.geometry_diagnostics is not None:
            object.__setattr__(
                self, "geometry_diagnostics", dict(self.geometry_diagnostics)
            )

    @property
    def circumferential_node_count(self) -> int:
        return int(self.nodes.shape[1])

    @property
    def wall_normal_node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def wall(self) -> np.ndarray:
        return self.nodes[0]
