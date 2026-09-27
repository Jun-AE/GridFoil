"""SU2, Fluent MSH, CGNS/HDF5, and diagnostic VTU mesh writers."""

from __future__ import annotations

from base64 import b64encode
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, ElementTree, SubElement

import numpy as np

from .models import HARD_INVALID_ISSUES, ISSUE_NAMES, AirfoilMesh
from .utils import atomic_output, dominant_orientation

if TYPE_CHECKING:
    import h5py


def _write_lines(path: str | Path, lines) -> None:
    """Write text atomically without retaining the complete file in memory."""
    with atomic_output(path) as temporary:
        with temporary.open("w", encoding="ascii", newline="\n") as stream:
            for line in lines:
                stream.write(f"{line}\n")


def write_su2(
    path: str | Path,
    grid: AirfoilMesh,
    wall_tag: str = "airfoil",
    farfield_tag: str = "farfield",
) -> None:
    nodes = grid.nodes
    nr, nt, _ = nodes.shape
    ni = nt - 1

    def node(j: int, i: int) -> int:
        return j * ni + i % ni

    elements = grid.cells.node_ids

    def lines():
        yield "NDIME= 2"
        yield f"NELEM= {len(elements)}"
        for cell_id, element in zip(grid.cells.ids, elements, strict=True):
            yield f"9 {' '.join(map(str, element))} {cell_id}"
        yield f"NPOIN= {nr * ni}"
        for j, row in enumerate(nodes):
            for i, point in enumerate(row[:ni]):
                yield f"{point[0]:.16e} {point[1]:.16e} {node(j, i)}"
        yield "NMARK= 2"
        for name, j in ((wall_tag, 0), (farfield_tag, nr - 1)):
            yield f"MARKER_TAG= {name}"
            yield f"MARKER_ELEMS= {ni}"
            for i in range(ni):
                a, b = (i, i + 1) if j == 0 else (i + 1, i)
                yield f"3 {node(j, a)} {node(j, b)}"

    _write_lines(path, lines())


_ATTRIBUTE_SIZES = {"name": 33, "label": 33, "type": 3}


def _bytes(value: str, *, null_terminated: bool = False) -> np.ndarray:
    """Return ASCII data for a CGNS character array."""
    encoded = value.encode("ascii")
    if null_terminated:
        encoded += b"\0"
    return np.frombuffer(encoded, dtype=np.int8).copy()


def _string_attribute(group: h5py.Group, key: str, value: str, size: int) -> None:
    """Write the fixed-length scalar string that the CGNS library uses."""
    encoded = value.encode("ascii")
    if len(encoded) >= size:
        raise ValueError(f"CGNS text is too long: {value!r}")
    group.attrs.create(key, np.bytes_(encoded), dtype=f"S{size}")


def _set_node_attributes(
    group: h5py.Group,
    name: str,
    label: str,
    data_type: str,
    *,
    flags: bool = True,
) -> None:
    for key, value in (("name", name), ("label", label), ("type", data_type)):
        _string_attribute(group, key, value, _ATTRIBUTE_SIZES[key])
    if flags:
        group.attrs.create("flags", np.zeros(1, dtype=np.int32))


def _node(
    parent: h5py.Group,
    name: str,
    label: str,
    data_type: str = "MT",
    data: np.ndarray | None = None,
) -> h5py.Group:
    group = parent.create_group(name, track_order=True)
    _set_node_attributes(group, name, label, data_type)
    if data is not None:
        group.create_dataset(" data", data=data)
    return group


def _range(parent: h5py.Group, name: str, values: tuple[tuple[int, int], ...]) -> None:
    # HDF5 reverses CGNS's Fortran-ordered dimensions.
    _node(parent, name, "IndexRange_t", "I4", np.asarray(values, dtype=np.int32))


def _boundary(
    zone_bc: h5py.Group,
    name: str,
    boundary_type: str,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    boundary = _node(zone_bc, name, "BC_t", "C1", _bytes(boundary_type))
    _range(boundary, "PointRange", (start, end))


def _write_boundaries(zone: h5py.Group, grid: AirfoilMesh) -> None:
    normal_count, surface_count = grid.nodes.shape[:2]
    zone_bc = _node(zone, "ZoneBC", "ZoneBC_t")
    _boundary(zone_bc, "Airfoil", "BCWall", (1, 1), (surface_count, 1))
    _boundary(
        zone_bc,
        "Farfield",
        "BCFarfield",
        (1, normal_count),
        (surface_count, normal_count),
    )


def _write_o_grid_seam(zone: h5py.Group, grid: AirfoilMesh) -> None:
    normal_count, surface_count = grid.nodes.shape[:2]
    connections = _node(zone, "ZoneGridConnectivity", "ZoneGridConnectivity_t")
    seam = _node(
        connections,
        "PeriodicSeam",
        "GridConnectivity1to1_t",
        "C1",
        _bytes("Zone1"),
    )
    _range(seam, "PointRange", ((1, 1), (1, normal_count)))
    _range(
        seam,
        "PointRangeDonor",
        ((surface_count, 1), (surface_count, normal_count)),
    )
    _node(seam, "Transform", "DataArray_t", "I4", np.asarray([1, 2], dtype=np.int32))


def write_cgns(path: str | Path, grid: AirfoilMesh) -> None:
    """Write one two-dimensional structured zone in CGNS/HDF5 format."""
    import h5py

    normal_count, surface_count = grid.nodes.shape[:2]
    zone_size = np.asarray(
        (
            (surface_count, normal_count),
            (surface_count - 1, normal_count - 1),
            (0, 0),
        ),
        dtype=np.int32,
    )
    with atomic_output(path) as temporary:
        with h5py.File(temporary, "w", track_order=True) as cgns:
            _set_node_attributes(
                cgns,
                "HDF5 MotherNode",
                "Root Node of HDF5 File",
                "MT",
                flags=False,
            )
            cgns.create_dataset(
                " format", data=_bytes("IEEE_LITTLE_64", null_terminated=True)
            )
            version = f"HDF5 Version {h5py.version.hdf5_version}"
            cgns.create_dataset(
                " hdf5version", data=_bytes(version, null_terminated=True)
            )

            _node(
                cgns,
                "CGNSLibraryVersion",
                "CGNSLibraryVersion_t",
                "R4",
                np.asarray([4.5], dtype=np.float32),
            )
            base = _node(
                cgns,
                "Base",
                "CGNSBase_t",
                "I4",
                np.asarray([2, 2], dtype=np.int32),
            )
            zone = _node(base, "Zone1", "Zone_t", "I4", zone_size)
            _node(zone, "ZoneType", "ZoneType_t", "C1", _bytes("Structured"))
            coordinates = _node(zone, "GridCoordinates", "GridCoordinates_t")
            _node(coordinates, "CoordinateX", "DataArray_t", "R8", grid.nodes[..., 0])
            _node(coordinates, "CoordinateY", "DataArray_t", "R8", grid.nodes[..., 1])
            _write_boundaries(zone, grid)
            _write_o_grid_seam(zone, grid)


def _array(parent: Element, name: str, values: np.ndarray) -> None:
    """Store an uncompressed, little-endian VTK binary array."""
    values = np.asarray(values)
    dtype, vtk_type = (
        ("<f8", "Float64")
        if values.dtype.kind == "f"
        else ("u1", "UInt8")
        if values.dtype.kind == "b" or values.dtype == np.dtype("uint8")
        else ("<i8", "Int64")
    )
    data = values.astype(dtype, copy=False).tobytes()
    header = np.array([len(data)], dtype="<u8").tobytes()
    element = SubElement(
        parent,
        "DataArray",
        type=vtk_type,
        Name=name,
        format="binary",
        NumberOfComponents=str(values.shape[1] if values.ndim == 2 else 1),
    )
    element.text = b64encode(header + data).decode("ascii")


def write_vtu(path: str | Path, mesh: AirfoilMesh) -> None:
    """Write the mesh and compact quality data to VTU."""
    points = mesh.nodes[:, :-1].reshape(-1, 2)
    cells = mesh.cells
    root = Element(
        "VTKFile",
        type="UnstructuredGrid",
        version="1.0",
        byte_order="LittleEndian",
        header_type="UInt64",
    )
    grid = SubElement(root, "UnstructuredGrid")
    field_data = SubElement(grid, "FieldData")
    for name, value in mesh.quality.compact_summary().items():
        if value is not None:
            _array(field_data, name, np.asarray([value]))
    piece = SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(len(points)),
        NumberOfCells=str(len(cells.ids)),
    )
    _array(
        SubElement(piece, "Points"),
        "coordinates",
        np.column_stack((points, np.zeros(len(points)))),
    )
    topology = SubElement(piece, "Cells")
    _array(topology, "connectivity", cells.node_ids.ravel())
    _array(topology, "offsets", 4 * np.arange(1, len(cells.ids) + 1))
    _array(topology, "types", np.full(len(cells.ids), 9, dtype=np.uint8))
    point_data = SubElement(piece, "PointData")
    _array(point_data, "node_id", np.arange(len(points)))
    if mesh.quality.dual is not None:
        dual = mesh.quality.dual
        for name, values in (
            ("dual_orthogonality_degrees", dual.orthogonality_degrees),
            ("undefined_dual_quality", dual.undefined),
        ):
            _array(point_data, name, values)
    data = SubElement(piece, "CellData", Scalars="invalid_cell")
    hard_columns = [ISSUE_NAMES.index(name) for name in HARD_INVALID_ISSUES]
    invalid = np.any(cells.issue_flags[:, hard_columns], axis=1)
    orientation = dominant_orientation(mesh.nodes) or 1
    oriented_jacobians = orientation * cells.corner_jacobians
    finite_ratios = np.where(
        np.isfinite(cells.neighbor_area_ratios),
        cells.neighbor_area_ratios,
        np.nan,
    )
    for name, values in (
        ("cell_id", cells.ids),
        ("invalid_cell", invalid),
        ("minimum_scaled_corner_jacobian", np.nanmin(oriented_jacobians, axis=1)),
        ("maximum_scaled_corner_jacobian", np.nanmax(oriented_jacobians, axis=1)),
        ("cell_orthogonal_quality", cells.cell_orthogonal_quality),
        ("minimum_neighbor_area_ratio", np.nanmin(finite_ratios, axis=1)),
        ("maximum_neighbor_area_ratio", np.nanmax(finite_ratios, axis=1)),
        ("edge_aspect_ratio", cells.edge_aspect_ratio),
    ):
        _array(data, name, values)
    with atomic_output(path) as temporary:
        ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)


def write_fluent_msh(path: str | Path, mesh: AirfoilMesh) -> None:
    """Write a 2D Fluent legacy ASCII computational mesh without remeshing.

    c0 lies to the left of each directed edge, c1 to its right. Only the
    repeated O-grid seam column is removed. Invalid orientation is rejected;
    advisory quality flags never cause cells to be removed or repaired.
    """
    nodes = mesh.nodes
    if not np.all(np.isfinite(nodes)) or not np.array_equal(nodes[:, 0], nodes[:, -1]):
        raise ValueError(
            "Fluent MSH requires finite coordinates and an exactly closed seam"
        )
    rings, columns = nodes.shape[:2]
    n = columns - 1
    points = nodes[:, :-1].reshape(-1, 2)
    cells = mesh.cells
    connectivity = cells.node_ids
    if connectivity.shape != ((rings - 1) * n, 4):
        raise ValueError("Fluent MSH requires quadrilateral O-grid connectivity")
    if not np.array_equal(cells.ids, np.arange(len(connectivity))):
        raise ValueError("Fluent MSH requires contiguous source cell IDs")
    if np.any(connectivity < 0) or np.any(connectivity >= len(points)):
        raise ValueError("Fluent MSH cell references an invalid node")
    repeated = np.any(np.diff(np.sort(connectivity, axis=1), axis=1) == 0, axis=1)
    if np.any(repeated):
        raise ValueError(
            f"Fluent MSH repeated nodes in cell {np.flatnonzero(repeated)[0]}"
        )

    # Validate the exported coordinates directly.
    vertices = points[connectivity]
    edge_vectors = np.roll(vertices, -1, axis=1) - vertices
    following = np.roll(edge_vectors, -1, axis=1)
    corners = (
        edge_vectors[..., 0] * following[..., 1]
        - edge_vectors[..., 1] * following[..., 0]
    )
    orientation = (
        1 if np.count_nonzero(corners > 0) >= np.count_nonzero(corners < 0) else -1
    )
    invalid = np.any(~np.isfinite(corners) | (orientation * corners <= 0), axis=1)
    if np.any(invalid):
        invalid_cell = np.flatnonzero(invalid)[0]
        raise ValueError(
            f"Fluent MSH cannot orient invalid/degenerate cell {invalid_cell}; "
            "use VTU diagnostics; no cells were repaired or removed"
        )
    ordered = connectivity if orientation > 0 else connectivity[:, [0, 3, 2, 1]]
    faces: dict[tuple[int, int], list[int]] = {}
    for cell_id, quad in enumerate(ordered, start=1):
        for a, b in zip(quad, np.roll(quad, -1), strict=True):
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b))
            if key not in faces:
                faces[key] = [a, b, cell_id, 0]
            else:
                face = faces[key]
                if face[3] or face[:2] != [b, a]:
                    raise ValueError(f"Fluent MSH inconsistent/non-manifold edge {key}")
                face[3] = cell_id

    groups: list[list[list[int]]] = [[], [], []]
    neighbors = [set() for _ in connectivity]
    for face in faces.values():
        a, b, c0, c1 = face
        if c1:
            groups[0].append(face)
            neighbors[c0 - 1].add(c1 - 1)
            neighbors[c1 - 1].add(c0 - 1)
        elif a < n and b < n:
            groups[1].append(face)
        elif a >= (rings - 1) * n and b >= (rings - 1) * n:
            groups[2].append(face)
        else:
            raise ValueError(f"Fluent MSH unexplained boundary edge {(a, b)}")
    if [len(group) for group in groups] != [(2 * rings - 3) * n, n, n]:
        raise ValueError("Fluent MSH O-grid face counts do not match")
    for cell_id, adjacent in enumerate(cells.neighbor_ids):
        if neighbors[cell_id] != {int(value) for value in adjacent if value >= 0}:
            raise ValueError(f"Fluent MSH adjacency mismatch in cell {cell_id}")

    def lines():
        yield '(0 "GridFoil Fluent legacy ASCII; normalized chord; no solution data")'
        yield "(2 2)"
        yield f"(10 (0 1 {len(points):x} 0 2))"
        yield f"(12 (0 1 {len(connectivity):x} 0))"
        yield f"(13 (0 1 {len(faces):x} 0))"
        yield f"(10 (1 1 {len(points):x} 1 2)("
        for x, y in points:
            yield f"{x:.16e} {y:.16e}"
        yield "))"
        yield f"(12 (2 1 {len(connectivity):x} 1 3))"
        first = 1
        for zone, bc_type, group in zip((3, 4, 5), (2, 3, 9), groups, strict=True):
            last = first + len(group) - 1
            yield f"(13 ({zone:x} {first:x} {last:x} {bc_type:x} 2)("
            for a, b, c0, c1 in group:
                yield f"{a + 1:x} {b + 1:x} {c0:x} {c1:x}"
            yield "))"
            first = last + 1
        for zone, kind, name in (
            (2, "fluid", "fluid"),
            (3, "interior", "interior"),
            (4, "wall", "airfoil"),
            (5, "pressure-far-field", "farfield"),
        ):
            yield f"(39 ({zone} {kind} {name} 1)())"

    _write_lines(path, lines())
