from __future__ import annotations

from dataclasses import replace

import numpy as np

from gridfoil.models import MeshSettings
from gridfoil.solvers import (
    _assemble_system,
    _known_metrics,
    _smooth_target_area,
    _target_areas,
)


def test_target_area_uses_local_dual_edge_width() -> None:
    unique = np.array(((0.0, 0.0), (0.8, -0.2), (1.4, 0.1), (0.9, 0.8), (0.1, 0.6)))
    front = np.vstack((unique, unique[0]))
    settings = replace(
        MeshSettings(),
        hyperbolic_area_smoothing_passes=0,
        farfield_uniformity_weight=0.0,
    )
    height = 0.03

    area = _target_areas(front, height, 0.0, settings)

    forward = np.roll(unique, -1, axis=0) - unique
    backward = unique - np.roll(unique, 1, axis=0)
    expected = (
        -0.5
        * height
        * (np.linalg.norm(forward, axis=1) + np.linalg.norm(backward, axis=1))
    )
    assert np.allclose(area[:-1], expected)
    assert area[-1] == area[0]


def test_log_diffusion_preserves_orientation_and_geometric_mean() -> None:
    area = -np.array((1.0, 8.0, 2.0, 4.0, 1.0))

    smoothed = _smooth_target_area(area, passes=12)

    assert np.all(smoothed < 0.0)
    assert smoothed[-1] == smoothed[0]
    assert np.isclose(
        np.mean(np.log(-smoothed[:-1])),
        np.mean(np.log(-area[:-1])),
    )


def test_implicit_blocks_preserve_constant_modes() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 17)
    front = np.column_stack((np.cos(angles), np.sin(angles)))
    target_area = _target_areas(front, 0.02, 0.3, MeshSettings())
    tangent, normal_metric = _known_metrics(front, target_area)

    lower, diagonal, upper, rhs = _assemble_system(
        front,
        tangent,
        normal_metric,
        level=3,
        normal_count=21,
        options=MeshSettings(),
    )

    identity = np.broadcast_to(np.eye(2), diagonal.shape)
    assert np.allclose(lower + diagonal + upper, identity)
    assert np.all(np.isfinite(rhs))
