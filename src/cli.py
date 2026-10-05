"""Command-line interface for the GridFoil."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .geometry import wall_distance
from .optimization import generate_optimized_mesh


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=f"GridFoil {__version__}: 2D quadrilateral airfoil O-grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples (from the repository root):
  gridfoil examples/naca0012/naca0012.dat --output mesh
  gridfoil examples/naca0012/naca0012.dat --output mesh/naca0012 --cgns --msh --vtu
  gridfoil examples/naca0012/naca0012.dat --output mesh/naca0012 --no-su2 --msh

Input: Selig-style two-column x y contour, upper trailing edge around the leading edge
through lower trailing edge, with optional headings before the coordinates. Lednicer
format and comments after the first coordinate are not supported. Airfoil files are
available from the UIUC Airfoil Coordinates Database and AirfoilTools.
Quote paths with spaces. Re is chord-based by default; y-plus sets the calculated
full first-layer height using a turbulent skin-friction correlation.
MSH is Fluent legacy ASCII 2D, not Gmsh. Fluent must use 2D Solver mode.
Revalidate regenerated meshes in Fluent or STAR-CCM+ before production use.
See README.md for import steps, mesh counts, and quality metrics.
""",
    )
    result.add_argument(
        "--version", action="version", version=f"GridFoil {__version__}"
    )
    result.add_argument(
        "airfoil",
        type=Path,
        help="airfoil .dat coordinate file (unzip the archive first)",
    )
    result.add_argument(
        "--output", type=Path, default=Path.cwd(), help="output directory"
    )
    result.add_argument("--name", help="output project name")
    result.add_argument("--surface-point-mode", choices=("REDISTRIBUTE", "PRESERVE"))
    result.add_argument(
        "--spacing-method",
        choices=("BERNSTEIN3", "SIGMOID5", "HYBRID"),
        help="surface spacing law (default: BERNSTEIN3)",
    )
    result.add_argument(
        "--geometry-conditioning-mode",
        choices=("AUTO", "SEGMENTED_HERMITE", "POLYLINE", "NATURAL_CUBIC_LEGACY"),
        help="airfoil reference-curve conditioner (default: AUTO)",
    )
    result.add_argument(
        "--geometry-deviation-tolerance-chords",
        type=float,
        help="maximum conditioner deviation in chord units (default: 2e-4)",
    )
    result.add_argument(
        "--geometry-max-refinement-depth",
        type=int,
        help="maximum adaptive geometry refinement depth (default: 10)",
    )
    result.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the local mesh-quality optimizer (default: off)",
    )
    result.add_argument(
        "--optimization-budget",
        type=int,
        default=16,
        help="maximum optimizer evaluations (default: 16)",
    )
    result.add_argument(
        "--su2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write SU2 (default: on)",
    )
    result.add_argument(
        "--cgns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write CGNS (default: off)",
    )
    result.add_argument(
        "--msh",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write Fluent legacy ASCII 2D MSH (default: off)",
    )
    result.add_argument(
        "--vtu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write diagnostic VTU (default: off)",
    )
    result.add_argument(
        "--circumferential-node-count",
        type=int,
        help="nodes per ring including repeated seam (source default: 401)",
    )
    result.add_argument(
        "--wall-normal-node-count",
        type=int,
        help="nodes from wall to farfield (source default: 251)",
    )
    result.add_argument(
        "--farfield-radius-chords",
        type=float,
        help="farfield radius in chord lengths (source default: 50)",
    )
    result.add_argument(
        "--leading-edge-cell-length",
        type=float,
        help=(
            "first wall panel length at both leading-edge corners in chord "
            "units (source default: 1.130001e-3)"
        ),
    )
    result.add_argument(
        "--trailing-edge-cell-length",
        type=float,
        help=(
            "first wall panel length at both trailing-edge corners in chord "
            "units; smaller values add more cells at the trailing edge "
            "(source default: 5.075471e-4)"
        ),
    )
    result.add_argument(
        "--flow-reynolds-number",
        type=float,
        default=9.0e6,
        help="Re based on U_inf and reference length (default: 9e6)",
    )
    result.add_argument(
        "--wall-y-plus-target",
        type=float,
        default=1.0,
        help="target cell-centre y-plus for first-layer sizing (default: 1)",
    )
    result.add_argument(
        "--wall-reference-length-chords",
        type=float,
        default=1.0,
        help="reference length L/c used in the supplied Re (default: 1, full chord)",
    )
    result.add_argument(
        "--hyperbolic-max-pseudo-aspect-ratio",
        type=float,
        help=(
            "largest radial-step/local-width ratio before temporary substeps "
            "(default: 8)"
        ),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.airfoil.suffix.lower() == ".zip":
        argument_parser.error(
            "unzip the airfoil archive first, then pass an extracted .dat file"
        )
    if not args.airfoil.is_file():
        argument_parser.error(f"airfoil file does not exist: {args.airfoil}")
    controls = {
        key: value
        for key, value in {
            "surface_point_mode": args.surface_point_mode,
            "spacing_method": args.spacing_method,
            "geometry_conditioning_mode": args.geometry_conditioning_mode,
            "geometry_deviation_tolerance_chords": (
                args.geometry_deviation_tolerance_chords
            ),
            "geometry_max_refinement_depth": args.geometry_max_refinement_depth,
            "flow_reynolds_number": args.flow_reynolds_number,
            "wall_y_plus_target": args.wall_y_plus_target,
            "wall_reference_length_chords": args.wall_reference_length_chords,
            "circumferential_node_count": args.circumferential_node_count,
            "wall_normal_node_count": args.wall_normal_node_count,
            "farfield_radius_chords": args.farfield_radius_chords,
            "leading_edge_cell_length": args.leading_edge_cell_length,
            "trailing_edge_cell_length": args.trailing_edge_cell_length,
            "hyperbolic_max_pseudo_aspect_ratio": (
                args.hyperbolic_max_pseudo_aspect_ratio
            ),
        }.items()
        if value is not None
    }
    result = generate_optimized_mesh(
        args.airfoil,
        optimize=args.optimize,
        optimization_budget=args.optimization_budget,
        output_directory=args.output,
        project_name=args.name,
        write_su2_output=args.su2,
        write_cgns_output=args.cgns,
        write_vtu_output=args.vtu,
        write_msh_output=args.msh,
        **controls,
    )
    mesh = result.mesh
    settings = result.settings
    first_layer_height = wall_distance(
        settings.wall_y_plus_target,
        settings.flow_reynolds_number,
        settings.wall_reference_length_chords,
    )
    report = {
        "version": __version__,
        "circumferential_node_count": mesh.circumferential_node_count,
        "wall_normal_node_count": mesh.wall_normal_node_count,
        # Backward-compatible alias.
        "first_layer_wall_distance": first_layer_height,
        "first_layer_height_chords": first_layer_height,
        "estimated_wall_cell_center_distance_chords": first_layer_height / 2.0,
        "flow_reynolds_number": settings.flow_reynolds_number,
        "wall_y_plus_target": settings.wall_y_plus_target,
        "wall_reference_length_chords": settings.wall_reference_length_chords,
        "optimization_enabled": args.optimize,
        "optimization_budget": args.optimization_budget,
        "optimization_improved": result.improved,
        "optimization_trial_count": len(result.trials),
        "explicit_controls": list(result.explicit_controls),
        "free_controls": list(result.free_controls),
        "quality": mesh.quality.summary() if mesh.quality else {},
        "quality_issues": mesh.quality.issues() if mesh.quality else [],
        "geometry": mesh.geometry_diagnostics or {},
        "files": {name: str(path.resolve()) for name, path in result.files.items()},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
