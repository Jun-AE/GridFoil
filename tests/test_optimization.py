from __future__ import annotations

from pathlib import Path

from gridfoil import generate_optimized_mesh


def test_optimizer_keeps_geometry_policy_controls_fixed() -> None:
    result = generate_optimized_mesh(
        Path("examples/naca0012/naca0012.dat"),
        optimize=True,
        optimization_budget=1,
        circumferential_node_count=101,
        wall_normal_node_count=21,
        wall_y_plus_target=1.0,
        flow_reynolds_number=9.0e6,
        wall_reference_length_chords=1.0,
    )

    expected = {
        "spacing_method",
        "geometry_conditioning_mode",
        "geometry_deviation_tolerance_chords",
        "geometry_max_refinement_depth",
    }
    assert expected <= set(result.inactive_controls)
    assert expected.isdisjoint(result.free_controls)
