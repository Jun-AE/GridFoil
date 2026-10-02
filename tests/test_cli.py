from __future__ import annotations

import json
from pathlib import Path

from gridfoil.cli import main


def test_cli_generates_su2_mesh(tmp_path: Path, capsys) -> None:
    airfoil = Path("examples/naca0012/naca0012.dat")

    exit_code = main(
        [
            str(airfoil),
            "--output",
            str(tmp_path),
            "--circumferential-node-count",
            "101",
            "--wall-normal-node-count",
            "21",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["circumferential_node_count"] == 101
    assert report["wall_normal_node_count"] == 21
    assert Path(report["files"]["su2"]).is_file()


def test_cli_accepts_surface_spacing_controls(tmp_path: Path, capsys) -> None:
    airfoil = Path("examples/naca4412i.dat")

    exit_code = main(
        [
            str(airfoil),
            "--output",
            str(tmp_path),
            "--circumferential-node-count",
            "101",
            "--wall-normal-node-count",
            "21",
            "--leading-edge-cell-length",
            "1.0e-3",
            "--trailing-edge-cell-length",
            "1.0e-4",
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "leading_edge_cell_length" in report["explicit_controls"]
    assert "trailing_edge_cell_length" in report["explicit_controls"]
    assert Path(report["files"]["su2"]).is_file()
