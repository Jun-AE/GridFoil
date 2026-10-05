from __future__ import annotations

from pathlib import Path

from gridfoil import MeshSettings, generate_optimized_mesh
from gridfoil.optimization import _bounds


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


def test_optimizer_bounds_scale_with_resolution() -> None:
    coarse = _bounds(MeshSettings(circumferential_node_count=101))
    fine = _bounds(MeshSettings(circumferential_node_count=401))

    assert fine["leading_edge_cell_length"][1] < coarse["leading_edge_cell_length"][1]
    assert fine["trailing_edge_cell_length"][0] < coarse["trailing_edge_cell_length"][0]


def test_optimizer_is_deterministic_and_keeps_baseline_safe() -> None:
    kwargs = dict(
        optimize=True,
        optimization_budget=8,
        circumferential_node_count=101,
        wall_normal_node_count=41,
    )
    airfoil = Path("examples/naca0012/naca0012.dat")

    first = generate_optimized_mesh(airfoil, **kwargs)
    second = generate_optimized_mesh(airfoil, **kwargs)

    assert vars(first.settings) == vars(second.settings)
    assert first.selected_valid
    assert (
        first.selected_metrics["objective_total"]
        <= first.baseline_metrics["objective_total"] + 1.0e-9
    )
