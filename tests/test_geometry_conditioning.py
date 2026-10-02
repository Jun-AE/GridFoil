from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gridfoil.geometry import (
    AirfoilProfile,
    prepare_surface,
    prepare_surface_with_diagnostics,
    transform_airfoil,
    wall_corner_indices,
)
from gridfoil.models import MeshSettings


def test_reader_accepts_headers_commas_and_blank_lines(tmp_path: Path) -> None:
    source = tmp_path / "profile.dat"
    source.write_text(
        "Example profile\n\n1.0, 0.0\n0.5, 0.08\n0.0, 0.0\n0.5, -0.08\n1.0, 0.0\n",
        encoding="utf-8",
    )

    profile = AirfoilProfile.read(source)

    assert profile.points.shape == (5, 2)
    assert profile.name == "profile"


def test_reader_rejects_comments_after_coordinates_start(tmp_path: Path) -> None:
    source = tmp_path / "profile.dat"
    source.write_text(
        "Example profile\n1.0 0.0\n# Inline source note\n0.5 0.08\n"
        "0.0 0.0\n0.5 -0.08\n1.0 0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid coordinate on line 3"):
        AirfoilProfile.read(source)


def test_sharp_trailing_edge_remains_exactly_closed() -> None:
    points = np.array(((1.0, 0.0), (0.55, 0.08), (0.0, 0.0), (0.55, -0.08), (1.0, 0.0)))
    profile = AirfoilProfile(points)
    expected = transform_airfoil(profile.points)[0]

    surface, _ = prepare_surface(
        profile,
        MeshSettings(circumferential_node_count=50).validated(profile),
    )

    assert np.array_equal(surface[0], expected)
    assert np.array_equal(surface[-1], expected)


def test_blunt_trailing_edge_endpoints_and_gap_remain_exact() -> None:
    points = np.array(
        ((1.0, 0.01), (0.55, 0.08), (0.0, 0.0), (0.55, -0.08), (1.0, -0.01))
    )
    profile = AirfoilProfile(points)
    expected = transform_airfoil(profile.points)

    surface, corners = prepare_surface(
        profile,
        MeshSettings(
            circumferential_node_count=50,
            trailing_edge_face_cell_count=16,
        ).validated(profile),
    )

    upper, lower = corners
    assert np.array_equal(surface[upper], expected[0])
    assert np.array_equal(surface[lower], expected[-1])
    assert np.array_equal(surface[-1], expected[0])
    assert np.linalg.norm(surface[upper] - surface[lower]) == np.linalg.norm(
        expected[0] - expected[-1]
    )


def test_mean_camber_split_handles_both_branches_above_zero() -> None:
    points = np.array(
        ((1.0, 0.11), (0.55, 0.18), (0.0, 0.10), (0.55, 0.04), (1.0, 0.09))
    )
    profile = AirfoilProfile(points)
    _, _, diagnostics = prepare_surface_with_diagnostics(
        profile,
        MeshSettings(circumferential_node_count=50).validated(profile),
    )

    mean_camber = diagnostics["mean_camber"]
    assert mean_camber["split_method"] == "ORDERED_CONTOUR_VALIDATED_BY_MEAN_CAMBER"
    assert mean_camber["negative_thickness_station_count"] == 0


def test_conditioning_modes_preserve_blunt_te_endpoints() -> None:
    points = np.array(
        ((1.0, 0.01), (0.55, 0.08), (0.0, 0.0), (0.55, -0.08), (1.0, -0.01))
    )
    profile = AirfoilProfile(points)
    expected = transform_airfoil(profile.points)

    modes = ("AUTO", "SEGMENTED_HERMITE", "POLYLINE", "NATURAL_CUBIC_LEGACY")
    for mode in modes:
        surface, corners, diagnostics = prepare_surface_with_diagnostics(
            profile,
            MeshSettings(
                circumferential_node_count=50,
                trailing_edge_face_cell_count=16,
                geometry_conditioning_mode=mode,
            ).validated(profile),
        )
        upper, lower = corners
        assert np.array_equal(surface[upper], expected[0])
        assert np.array_equal(surface[lower], expected[-1])
        assert diagnostics["trailing_edge_kind"] == "BLUNT"


def test_wall_corner_indices_flags_only_sharp_turns() -> None:
    square = np.array(((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))
    assert wall_corner_indices(square) == (0, 1, 2, 3)

    angles = np.linspace(0.0, 2.0 * np.pi, 33)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    circle[-1] = circle[0]
    assert wall_corner_indices(circle) == ()


def test_duplicate_point_is_retained_as_conditioning_break_metadata() -> None:
    points = np.array(
        (
            (1.0, 0.0),
            (0.7, 0.06),
            (0.7, 0.06),
            (0.0, 0.0),
            (0.7, -0.06),
            (1.0, 0.0),
        )
    )
    profile = AirfoilProfile(points)
    _, _, diagnostics = prepare_surface_with_diagnostics(
        profile,
        MeshSettings(circumferential_node_count=50).validated(profile),
    )

    assert diagnostics["corner_indices"] == [1]
    assert len(diagnostics["upper"]) == 2
