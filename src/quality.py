"""Per-cell quality, global overlap checks, and in-memory nodal metrics."""

import numpy as np

from .models import ISSUE_NAMES, CellMetrics, DualMetrics, MeshMetrics
from .utils import (
    _growth,
    dominant_orientation,
    node_signature,
    scaled_corner_jacobians,
    signed_cell_areas,
)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _triangulate(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split simple quadrilaterals along an internal diagonal."""
    first = vertices[:, [[0, 1, 2], [0, 2, 3]]]
    second = vertices[:, [[0, 1, 3], [1, 2, 3]]]
    scale = np.max(np.ptp(vertices, axis=1), axis=1)
    floor = 32.0 * np.finfo(float).eps * scale**2

    def valid(triangles: np.ndarray) -> np.ndarray:
        areas = _cross(
            triangles[:, :, 1] - triangles[:, :, 0],
            triangles[:, :, 2] - triangles[:, :, 0],
        )
        return np.all(areas > floor[:, None], axis=1) | np.all(
            areas < -floor[:, None], axis=1
        )

    use_first, use_second = valid(first), valid(second)
    triangles = np.where(use_first[:, None, None, None], first, second)
    checked = (use_first | use_second) & np.all(np.isfinite(vertices), axis=(1, 2))
    return triangles, checked


def _convex_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Test convex polygons on their separating axes; touching is not overlap."""
    origin = a[:, :1]
    a, b = a - origin, b - origin
    edges = np.concatenate(
        (np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b), axis=1
    )
    axes = np.stack((-edges[..., 1], edges[..., 0]), axis=-1)
    axes /= np.linalg.norm(axes, axis=-1)[..., None]
    projection_a = np.einsum("nvd,nad->nav", a, axes)
    projection_b = np.einsum("nvd,nad->nav", b, axes)
    depth = np.minimum(projection_a.max(axis=-1), projection_b.max(axis=-1))
    depth -= np.maximum(projection_a.min(axis=-1), projection_b.min(axis=-1))
    scale = np.max(np.abs(np.concatenate((a, b), axis=1)), axis=(1, 2))
    return np.all(depth > (64.0 * np.finfo(float).eps * scale)[:, None], axis=1)


def overlapping_cell_pairs(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find positive-area overlaps, including containment, without an all-pairs matrix.

    Returns cell-ID pairs and a per-cell checked mask. Degenerate or self-crossing
    polygons with no valid diagonal are unchecked, not declared overlap-free.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles, checked = _triangulate(vertices)
    lower, upper = vertices.min(axis=1), vertices.max(axis=1)
    edges = np.roll(vertices, -1, axis=1) - vertices
    corners = _cross(edges, np.roll(edges, -1, axis=1))
    floor = 32.0 * np.finfo(float).eps * np.max(upper - lower, axis=1) ** 2
    convex = np.all(corners > floor[:, None], axis=1) | np.all(
        corners < -floor[:, None], axis=1
    )
    order = np.flatnonzero(checked)
    order = order[np.argsort(lower[order, 0], kind="stable")]
    active = np.empty(0, dtype=np.int64)
    pending: list[np.ndarray] = []
    overlaps: list[np.ndarray] = []
    pending_count = 0

    def flush() -> None:
        if not pending:
            return
        pairs = np.concatenate(pending)
        for start in range(0, len(pairs), 4096):
            batch = pairs[start : start + 4096]
            found = np.zeros(len(batch), dtype=bool)
            direct = convex[batch[:, 0]] & convex[batch[:, 1]]
            # Convex quads need one polygon test.
            found[direct] = _convex_overlap(
                vertices[batch[direct, 0]], vertices[batch[direct, 1]]
            )
            for first in range(2):
                for second in range(2):
                    remaining = np.flatnonzero(~direct & ~found)
                    if not remaining.size:
                        break
                    found[remaining] |= _convex_overlap(
                        triangles[batch[remaining, 0], first],
                        triangles[batch[remaining, 1], second],
                    )
            overlaps.append(batch[found])
        pending.clear()

    for cell_id in order:
        active = active[upper[active, 0] > lower[cell_id, 0]]
        candidates = active[
            (upper[active, 1] > lower[cell_id, 1])
            & (lower[active, 1] < upper[cell_id, 1])
        ]
        if candidates.size:
            pending.append(
                np.column_stack((candidates, np.full(len(candidates), cell_id)))
            )
            pending_count += len(candidates)
        if pending_count >= 4096:
            flush()
            pending_count = 0
        active = np.append(active, cell_id)
    flush()
    pairs = np.concatenate(overlaps) if overlaps else np.empty((0, 2), dtype=np.int64)
    return np.sort(pairs, axis=1), checked


def assess_cells(nodes: np.ndarray) -> CellMetrics:
    """Assess every cell; periodic seam nodes use the same IDs as SU2 output.

    Centers are vertex averages (bilinear cell centers), not area centroids.
    Aspect ratio is longest edge / shortest edge. Neighbor area ratios are
    max(abs(area)) / min(abs(area)); absent neighbors have ID -1 and ratio NaN.
    """
    radial, circumferential = np.array(nodes.shape[:2]) - 1
    periodic = np.array_equal(nodes[:, 0], nodes[:, -1])
    node_columns = circumferential if periodic else circumferential + 1
    j, i = np.indices((radial, circumferential))
    ids = np.arange(radial * circumferential, dtype=np.int64).reshape(j.shape)
    next_i = (i + 1) % node_columns
    node_ids = np.stack(
        (
            j * node_columns + i,
            j * node_columns + next_i,
            (j + 1) * node_columns + next_i,
            (j + 1) * node_columns + i,
        ),
        axis=-1,
    )
    vertices = np.stack(
        (nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, 1:], nodes[1:, :-1]), axis=-2
    )
    edges = np.roll(vertices, -1, axis=-2) - vertices
    lengths = np.linalg.norm(edges, axis=-1)
    following = np.roll(edges, -1, axis=-2)
    products = lengths * np.roll(lengths, -1, axis=-1)
    cosine = np.divide(
        -np.sum(edges * following, axis=-1),
        products,
        out=np.ones_like(products),
        where=products > 0.0,
    )
    skew = np.max(
        np.abs(90.0 - np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))), axis=-1
    )
    shortest, longest = lengths.min(axis=-1), lengths.max(axis=-1)
    aspect = np.divide(
        longest, shortest, out=np.full_like(longest, np.inf), where=shortest > 0.0
    )
    area = signed_cell_areas(nodes)
    corners = scaled_corner_jacobians(nodes)
    orientation = dominant_orientation(nodes)
    tolerance = 32.0 * np.finfo(float).eps
    # Use the finite-volume cell centroid.
    following_vertices = np.roll(vertices, -1, axis=-2)
    twice_triangle_areas = _cross(vertices, following_vertices)
    twice_area = np.sum(twice_triangle_areas, axis=-1)
    centroid_numerator = np.sum(
        (vertices + following_vertices) * twice_triangle_areas[..., None], axis=-2
    )
    centers = np.divide(
        centroid_numerator,
        3.0 * twice_area[..., None],
        out=vertices.mean(axis=-2),
        where=np.abs(twice_area[..., None]) > (2.0 * tolerance * longest**2)[..., None],
    )
    overlap_pairs, overlap_checked = overlapping_cell_pairs(vertices.reshape(-1, 4, 2))
    overlapping = np.zeros(area.size, dtype=bool)
    overlapping[overlap_pairs.ravel()] = True
    flags = np.stack(
        (
            orientation * area < 0.0,
            (np.abs(area) <= tolerance * longest**2) | (shortest == 0.0),
            np.any(orientation * corners <= tolerance, axis=-1),
            ~np.all(np.isfinite(vertices), axis=(-1, -2)) | ~np.isfinite(area),
            skew > 75.0,
            overlapping.reshape(area.shape),
            ~overlap_checked.reshape(area.shape),
        ),
        axis=-1,
    )
    neighbors = np.stack(
        (
            np.roll(ids, 1, axis=1),
            np.roll(ids, -1, axis=1),
            ids - circumferential,
            ids + circumferential,
        ),
        axis=-1,
    )
    neighbors[0, :, 2] = -1
    neighbors[-1, :, 3] = -1
    if not periodic:
        neighbors[:, 0, 0] = -1
        neighbors[:, -1, 1] = -1
    # Fluent-style quality checks face-to-cell and face-to-neighbor alignment.
    # Boundary faces only use the face-to-cell term.
    face_midpoints = 0.5 * (vertices + following_vertices)
    outward_normals = orientation * np.stack((edges[..., 1], -edges[..., 0]), axis=-1)

    def normalized_dot(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        scale = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
        return np.divide(
            np.sum(first * second, axis=-1),
            scale,
            out=np.full(scale.shape, np.nan),
            where=scale > 0.0,
        )

    face_orthogonality = normalized_dot(
        outward_normals, face_midpoints - centers[..., None, :]
    )
    # Face order: wallward, next, outward, previous.
    face_neighbor_ids = neighbors[..., [2, 1, 3, 0]]
    flat_centers = centers.reshape(-1, 2)
    neighbor_centers = flat_centers[np.maximum(face_neighbor_ids, 0)]
    neighbor_orthogonality = normalized_dot(
        outward_normals, neighbor_centers - centers[..., None, :]
    )
    neighbor_orthogonality[face_neighbor_ids < 0] = np.nan
    per_face_quality = np.where(
        face_neighbor_ids >= 0,
        np.minimum(face_orthogonality, neighbor_orthogonality),
        face_orthogonality,
    )
    minimum_orthogonality = np.min(per_face_quality, axis=-1)
    cell_orthogonal_quality = np.where(
        np.isfinite(minimum_orthogonality),
        np.clip(minimum_orthogonality, 0.0, 1.0),
        0.0,
    )
    adjacent_area = np.abs(area).ravel()[np.maximum(neighbors, 0)]
    smaller = np.minimum(np.abs(area)[..., None], adjacent_area)
    larger = np.maximum(np.abs(area)[..., None], adjacent_area)
    ratios = np.divide(
        larger, smaller, out=np.full_like(larger, np.inf), where=smaller > 0.0
    )
    ratios[neighbors < 0] = np.nan
    return CellMetrics(
        ids.ravel(),
        node_ids.reshape(-1, 4),
        np.stack((j, i), axis=-1).reshape(-1, 2),
        centers.reshape(-1, 2),
        area.ravel(),
        corners.reshape(-1, 4),
        face_orthogonality.reshape(-1, 4),
        neighbor_orthogonality.reshape(-1, 4),
        cell_orthogonal_quality.ravel(),
        skew.ravel(),
        aspect.ravel(),
        neighbors.reshape(-1, 4),
        ratios.reshape(-1, 4),
        flags.reshape(-1, len(ISSUE_NAMES)),
        overlap_pairs,
    )


def compute_dual_quality(points: np.ndarray, node_ids: np.ndarray) -> DualMetrics:
    """Assess median-dual volumes of a 2D quad mesh with physical boundaries.

    Shared edges contribute one summed face normal. Boundary half-edges form
    an additional face at each boundary node. Undefined metrics remain NaN;
    neither coordinates nor exported cell ordering are changed. The physical
    boundary loops must not share nodes, as with an O-grid wall and farfield.
    """
    vertices = points[node_ids]
    first = _cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    second = _cross(vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0])
    # Orient a local copy of the connectivity.
    clockwise = (first < 0.0) & (second < 0.0)
    cells = np.where(clockwise[:, None], node_ids[:, [0, 3, 2, 1]], node_ids)
    vertices = points[cells]
    centers = vertices.mean(axis=1)
    side_centers = np.repeat(centers, 4, axis=0)
    sides = np.stack((cells, np.roll(cells, -1, axis=1)), axis=-1).reshape(-1, 2)
    endpoints = points[sides]
    midpoints = endpoints.mean(axis=1)
    offsets = side_centers - midpoints
    normals = np.stack((offsets[:, 1], -offsets[:, 0]), axis=-1)
    normals[sides[:, 0] > sides[:, 1]] *= -1.0
    edges, inverse, counts = np.unique(
        np.sort(sides, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    face_normals = np.zeros((len(edges), 2))
    np.add.at(face_normals, inverse, normals)
    face_areas = np.linalg.norm(face_normals, axis=1)
    directions = points[edges[:, 1]] - points[edges[:, 0]]
    lengths = np.linalg.norm(directions, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.sum(
            (face_normals / face_areas[:, None]) * (directions / lengths[:, None]),
            axis=1,
        )
    angles = 90.0 - np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    # Only mesh-edge dual faces count here.
    weighted_angles = np.bincount(
        edges.ravel(), np.repeat(face_areas * angles, 2), minlength=len(points)
    )
    total_areas = np.bincount(
        edges.ravel(), np.repeat(face_areas, 2), minlength=len(points)
    )
    smallest_face, largest_face = np.full(len(points), np.inf), np.zeros(len(points))
    np.minimum.at(smallest_face, edges.ravel(), np.repeat(face_areas, 2))
    np.maximum.at(largest_face, edges.ravel(), np.repeat(face_areas, 2))

    boundary = counts[inverse] == 1
    boundary_nodes = sides[boundary]
    half_edges = np.stack(
        (
            midpoints[boundary] - endpoints[boundary, 0],
            endpoints[boundary, 1] - midpoints[boundary],
        ),
        axis=1,
    )
    boundary_vectors = np.zeros((len(points), 2))
    np.add.at(boundary_vectors, boundary_nodes.ravel(), half_edges.reshape(-1, 2))
    boundary_ids = np.unique(boundary_nodes)
    # Sum boundary vectors before taking their magnitude.
    boundary_areas = np.linalg.norm(boundary_vectors[boundary_ids], axis=1)
    smallest_face[boundary_ids] = np.minimum(
        smallest_face[boundary_ids], boundary_areas
    )
    largest_face[boundary_ids] = np.maximum(largest_face[boundary_ids], boundary_areas)

    relative = endpoints - side_centers[:, None]
    subareas = 0.5 * np.abs(_cross(-offsets[:, None], relative))
    smallest_volume, largest_volume = (
        np.full(len(points), np.inf),
        np.zeros(len(points)),
    )
    np.minimum.at(smallest_volume, sides.ravel(), subareas.ravel())
    np.maximum.at(largest_volume, sides.ravel(), subareas.ravel())

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.full(len(points), np.nan),
            where=denominator > 0.0,
        )

    return DualMetrics(
        ratio(weighted_angles, total_areas),
        ratio(largest_face, smallest_face),
        ratio(largest_volume, smallest_volume),
    )


def compute_quality(nodes: np.ndarray) -> MeshMetrics:
    cells = assess_cells(nodes)
    radial_count, surface_count, _ = nodes.shape
    skew = np.zeros((radial_count, surface_count))
    cell_skew = cells.skew_degrees.reshape(radial_count - 1, surface_count - 1)
    for row, column in ((0, 0), (0, 1), (1, 0), (1, 1)):
        target = skew[row : row + radial_count - 1, column : column + surface_count - 1]
        np.maximum(target, cell_skew, out=target)
    tangential_growth = np.zeros_like(skew)
    normal_growth = np.zeros_like(skew)
    if surface_count > 2:
        tangential_growth[:, 1:-1] = _growth(
            nodes[:, 2:] - nodes[:, 1:-1], nodes[:, 1:-1] - nodes[:, :-2]
        )
    if radial_count > 2:
        normal_growth[1:-1] = _growth(nodes[2:] - nodes[1:-1], nodes[1:-1] - nodes[:-2])
    shape = (radial_count - 1, surface_count - 1)
    unique = nodes[:, :-1] if np.array_equal(nodes[:, 0], nodes[:, -1]) else nodes
    return MeshMetrics(
        skew,
        tangential_growth,
        normal_growth,
        cells.signed_area.reshape(shape),
        cells.corner_jacobians.reshape(*shape, 4),
        cells,
        node_signature(nodes),
        compute_dual_quality(unique.reshape(-1, 2), cells.node_ids),
    )
