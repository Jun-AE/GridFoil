"""Generate the optimized NACA4412 sharp-trailing-edge example mesh."""

from pathlib import Path

from gridfoil import generate_optimized_mesh


def main() -> None:
    output = Path(__file__).resolve().parent / "naca4412"
    airfoil = output / "naca4412.dat"

    result = generate_optimized_mesh(
        airfoil,
        optimize=True,
        optimization_budget=8,
        circumferential_node_count=210,
        wall_normal_node_count=210,
        farfield_radius_chords=50.0,
        surface_point_mode="REDISTRIBUTE",
        wall_y_plus_target=1.0,
        flow_reynolds_number=9.0e6,
        wall_reference_length_chords=1.0,
        output_directory=output,
        project_name=airfoil.stem,
        write_cgns_output=True,
        write_vtu_output=True,
        write_msh_output=True,
    )

    print(f"Improved: {result.improved}")
    print(f"Default quality: {result.baseline_metrics}")
    print(f"Optimized quality: {result.selected_metrics}")
    for name, path in sorted(result.files.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
