from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from gridfoil import GridGenerator, MeshSettings


def test_optimized_naca0012_matches_or_improves_pushed_quality() -> None:
    settings = MeshSettings(
        leading_edge_cell_length=0.0020841394958560056,
        trailing_edge_cell_length=0.0002490216626885873,
        trailing_edge_face_cell_count=16,
        farfield_angular_bias=1.1390696813749193,
        hyperbolic_normal_coupling=5.2028,
        hyperbolic_implicit_smoothing=28.314320000000002,
        hyperbolic_explicit_smoothing=0.4333199999999999,
        farfield_uniformity_weight=0.20264,
        hyperbolic_area_smoothing_passes=58,
        hyperbolic_max_pseudo_aspect_ratio=7.1,
    )

    mesh = GridGenerator(Path("examples/naca0012/naca0012.dat"), settings).run()
    quality = mesh.quality.compact_summary()

    assert quality["cell_count"] == 100_000
    assert quality["invalid_cell_count"] == 0
    assert quality["minimum_scaled_corner_jacobian"] >= 0.7218702247
    assert quality["minimum_cell_orthogonal_quality"] >= 0.4185086819
    assert quality["minimum_dual_orthogonality_degrees"] >= 70.1755494341
    assert quality["maximum_neighbor_area_ratio"] <= 1.6265792004
    assert quality["maximum_edge_aspect_ratio"] <= 1408.4207583


def test_sharp_trailing_edge_march_avoids_algebraic_fallback() -> None:
    settings = MeshSettings(
        circumferential_node_count=151,
        wall_normal_node_count=251,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = GridGenerator(Path("examples/naca4412/naca4412.dat"), settings).run()

    assert mesh.geometry_diagnostics is not None
    assert mesh.geometry_diagnostics["trailing_edge_kind"] == "SHARP"
    assert not any("fallback" in str(item.message).lower() for item in caught)
    assert mesh.quality.compact_summary()["invalid_cell_count"] == 0


def test_naca0012_stabilized_march_does_not_bend_outward() -> None:
    settings = MeshSettings(
        circumferential_node_count=105,
        wall_normal_node_count=210,
        farfield_radius_chords=50.0,
        surface_point_mode="REDISTRIBUTE",
        wall_y_plus_target=1.0,
        flow_reynolds_number=9.0e6,
        wall_reference_length_chords=1.0,
    )
    nodes = GridGenerator(Path("examples/naca0012/naca0012.dat"), settings).run().nodes
    wall = nodes[0]
    aft_surface = (wall[:, 0] > 0.55) & (wall[:, 0] < 0.9)
    segments = np.diff(nodes[:, aft_surface], axis=0)
    directions = segments / np.linalg.norm(segments, axis=2, keepdims=True)
    turns = np.degrees(
        np.arccos(np.clip(np.sum(directions[1:] * directions[:-1], axis=2), -1.0, 1.0))
    )
    outer_angles = np.unwrap(np.arctan2(nodes[-1, :, 1], nodes[-1, :, 0] - 0.5))
    outer_spacing = np.abs(np.diff(outer_angles))

    assert np.max(turns[:150]) < 1.0
    assert np.max(outer_spacing) / np.min(outer_spacing) < 2.0
