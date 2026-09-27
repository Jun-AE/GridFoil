"""Linear systems, hyperbolic marching, stabilization, and recovery."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .geometry import (
    cumulative_distance,
    interpolate_polyline,
    surface_normals,
    wall_distance,
)
from .models import MeshSettings
from .utils import (
    dominant_orientation,
    geometric_distances,
    growth_ratio,
    growth_ratios,
    has_consistent_orientation,
    periodic_first_derivative,
    scaled_corner_jacobians,
)

_IMPLICIT_TRANSITION_LAYERS = 40


def _solve_banded_corners(
    bands: np.ndarray, rhs: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Solve a band matrix plus corner updates; retain all right-hand sides.

    The fast low-rank solve requires an invertible band part. If that split
    fails or loses accuracy, factor the complete sparse matrix instead.
    """
    width = len(bands) // 2
    size = bands.shape[1]
    rhs = np.asarray(rhs, dtype=float)
    vector = rhs.ndim == 1
    values = rhs[:, None] if vector else rhs
    count = values.shape[1]
    try:
        combined = solve_banded(
            (width, width), bands, np.column_stack((values, left)), check_finite=False
        )
        base, responses = combined[:, :count], combined[:, count:]
        correction = np.eye(left.shape[1]) + right @ responses
        solution = base - responses @ np.linalg.solve(correction, right @ base)

        # Check the full-system residual.
        product = left @ (right @ solution)
        scale = np.abs(left) @ (np.abs(right) @ np.abs(solution)) + np.abs(values)
        for offset in range(-width, width + 1):
            columns = np.arange(max(0, -offset), min(size, size - offset))
            terms = bands[width + offset, columns, None] * solution[columns]
            product[columns + offset] += terms
            scale[columns + offset] += np.abs(terms)
        if np.all(np.isfinite(solution)) and np.all(
            np.abs(product - values) <= 1.0e-12 * scale
        ):
            return solution[:, 0] if vector else solution
    except np.linalg.LinAlgError:
        pass

    band_rows, columns = np.indices(bands.shape)
    rows = columns + band_rows - width
    valid = (rows >= 0) & (rows < bands.shape[1]) & (bands != 0.0)
    matrix = csc_matrix(
        (bands[valid], (rows[valid], columns[valid])), shape=(len(values), len(values))
    )
    matrix += csc_matrix(left) @ csc_matrix(right)
    try:
        solution = splu(matrix).solve(values)
    except RuntimeError as error:
        raise np.linalg.LinAlgError("the complete cyclic system is singular") from error
    return solution[:, 0] if vector else solution


def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    *,
    cyclic: bool = False,
) -> np.ndarray:
    """Solve a scalar tridiagonal system, with optional corner entries."""
    count = len(diagonal)
    bands = np.zeros((3, count), dtype=float)
    bands[0, 1:] = upper[:-1]
    bands[1] = diagonal
    bands[2, :-1] = lower[1:]
    if not cyclic:
        return solve_banded(
            (1, 1),
            bands,
            rhs,
            overwrite_ab=True,
            check_finite=False,
        )

    left = np.zeros((count, 2), dtype=float)
    left[0, 0] = 1.0
    left[-1, 1] = 1.0
    right = np.zeros((2, count), dtype=float)
    right[0, -1] = lower[0]
    right[1, 0] = upper[-1]
    return _solve_banded_corners(bands, rhs, left, right)


def _pack_block_bands(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Pack a two-by-two block-tridiagonal matrix for LAPACK."""
    block_count = len(diagonal)
    bands = np.zeros((7, 2 * block_count), dtype=float)
    blocks = np.arange(block_count)
    for row_offset in range(2):
        for column_offset in range(2):
            rows = 2 * blocks + row_offset
            columns = 2 * blocks + column_offset
            bands[3 + rows - columns, columns] = diagonal[:, row_offset, column_offset]

            rows = 2 * blocks[1:] + row_offset
            columns = 2 * blocks[:-1] + column_offset
            bands[3 + rows - columns, columns] = lower[1:, row_offset, column_offset]

            rows = 2 * blocks[:-1] + row_offset
            columns = 2 * blocks[1:] + column_offset
            bands[3 + rows - columns, columns] = upper[:-1, row_offset, column_offset]
    return bands


def solve_block_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    corner_blocks: Sequence[tuple[int, int, np.ndarray]] = (),
) -> np.ndarray:
    """Solve a two-by-two block system with optional off-band corners."""
    block_count = len(diagonal)
    scalar_count = 2 * block_count
    bands = _pack_block_bands(lower, diagonal, upper)
    if not corner_blocks:
        return solve_banded(
            (3, 3),
            bands,
            rhs,
            overwrite_ab=True,
            check_finite=False,
        )

    rank = 2 * len(corner_blocks)
    left = np.zeros((scalar_count, rank), dtype=float)
    right = np.zeros((rank, scalar_count), dtype=float)
    identity = np.eye(2)
    for index, (row, column, block) in enumerate(corner_blocks):
        correction = slice(2 * index, 2 * index + 2)
        left[2 * row : 2 * row + 2, correction] = identity
        right[correction, 2 * column : 2 * column + 2] = block

    return _solve_banded_corners(bands, rhs, left, right)


class HyperbolicMeshError(ValueError):
    """Raised when hyperbolic marching cannot create a valid next layer."""


@lru_cache(maxsize=32)
def _periodic_diffusion_gain(size: int, passes: int) -> np.ndarray:
    """Return the exact spectral gain for periodic discrete diffusion."""
    wave_number = 2.0 * np.pi * np.fft.rfftfreq(size)
    laplacian_eigenvalue = 4.0 * np.sin(0.5 * wave_number) ** 2
    diffusion_time = passes / 8.0
    return np.exp(-diffusion_time * laplacian_eigenvalue)


def _smooth_target_area(area: np.ndarray, passes: int) -> np.ndarray:
    if passes == 0:
        return area.copy()
    unique = area[:-1]
    orientation = np.sign(np.median(unique))
    if orientation == 0.0 or np.any(orientation * unique <= 0.0):
        raise ValueError("target cell areas must have one nonzero orientation")
    log_magnitude = np.log(orientation * unique)
    spectrum = np.fft.rfft(log_magnitude)
    smoothed_log = np.fft.irfft(
        spectrum * _periodic_diffusion_gain(len(unique), passes), n=len(unique)
    )
    smoothed = orientation * np.exp(smoothed_log)
    return np.concatenate((smoothed, smoothed[:1]))


def _target_areas(
    current: np.ndarray,
    cell_size: float,
    march_fraction: float,
    options: MeshSettings,
) -> np.ndarray:
    unique = current[:-1]
    forward = np.roll(unique, -1, axis=0) - unique
    backward = unique - np.roll(unique, 1, axis=0)
    dual_width = 0.5 * (
        np.linalg.norm(forward, axis=1) + np.linalg.norm(backward, axis=1)
    )
    if np.any(dual_width <= 1.0e-14):
        raise ValueError("target-area construction found a zero surface width")
    area = -cell_size * dual_width
    area = np.concatenate((area, area[:1]))
    smoothed = _smooth_target_area(
        area,
        options.hyperbolic_area_smoothing_passes,
    )
    progress = np.clip(march_fraction, 0.0, 1.0)
    transition = progress * progress * (3.0 - 2.0 * progress)
    blend = options.farfield_uniformity_weight * transition
    uniform = np.sign(np.mean(smoothed)) * np.exp(
        np.mean(np.log(np.abs(smoothed[:-1])))
    )
    result = (1.0 - blend) * smoothed + blend * uniform
    result[-1] = result[0]
    return result


def _known_metrics(
    current: np.ndarray, area: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tangent = periodic_first_derivative(current)
    denominator = np.sum(tangent * tangent, axis=1)
    count = len(current) - 1
    if np.any(denominator[:count] <= 1.0e-24):
        raise ValueError("The hyperbolic march found a degenerate surface metric.")
    normal_metric = np.column_stack(
        (-tangent[:, 1] * area / denominator, tangent[:, 0] * area / denominator)
    )
    return tangent, normal_metric


def _solve_block_system(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve one periodic surface line."""
    block_count = len(diagonal)
    corners: list[tuple[int, int, np.ndarray]] = []
    corners.extend(
        (
            (0, block_count - 1, lower[0]),
            (block_count - 1, 0, upper[-1]),
        )
    )
    return solve_block_tridiagonal(lower, diagonal, upper, rhs, corners)


def _assemble_system(
    current: np.ndarray,
    tangent: np.ndarray,
    normal_metric: np.ndarray,
    level: int,
    normal_count: int,
    options: MeshSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    surface_count = len(current)
    count = surface_count - 1
    epsilon_scale = (level - 1.0) / (normal_count - 2.0)
    implicit = epsilon_scale * options.hyperbolic_implicit_smoothing
    explicit = epsilon_scale * options.hyperbolic_explicit_smoothing
    local_tangent = tangent[:count]
    local_normal = normal_metric[:count]
    coordinate_map = np.empty((count, 2, 2), dtype=float)
    coordinate_map[:, 0] = local_tangent
    coordinate_map[:, 1, 0] = -local_tangent[:, 1]
    coordinate_map[:, 1, 1] = local_tangent[:, 0]
    normal_transport = np.empty_like(coordinate_map)
    normal_transport[:, 0] = local_normal
    normal_transport[:, 1, 0] = local_normal[:, 1]
    normal_transport[:, 1, 1] = -local_normal[:, 0]
    transport = np.linalg.solve(coordinate_map, normal_transport)
    coupling = 0.5 * options.hyperbolic_normal_coupling * transport

    identity = np.eye(2)
    lower = -implicit * identity - coupling
    diagonal = np.broadcast_to((1.0 + 2.0 * implicit) * identity, coupling.shape).copy()
    upper = -implicit * identity + coupling

    minus = np.roll(current[:count], 1, axis=0)
    plus = np.roll(current[:count], -1, axis=0)

    second = plus - 2.0 * current[:count] + minus
    first = plus - minus
    rhs = current[:count] - (implicit + explicit) * second
    rhs += np.einsum("nij,nj->ni", coupling, first)
    # This is B inverse times [0, area].
    rhs += normal_metric[:count]
    return lower, diagonal, upper, rhs.ravel()


def apply_normal_spacing(grid: np.ndarray, options: MeshSettings) -> np.ndarray:
    result = grid.copy()
    first = wall_distance(
        options.wall_y_plus_target,
        options.flow_reynolds_number,
        options.wall_reference_length_chords,
    )
    segment_lengths = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    arc_lengths = np.vstack(
        (np.zeros(grid.shape[1]), np.cumsum(segment_lengths, axis=0))
    )
    ratios = growth_ratios(arc_lengths[-1], first, result.shape[0] - 1)
    widths = first * ratios[:, None] ** np.arange(result.shape[0] - 1)
    targets = np.column_stack((np.zeros(result.shape[1]), np.cumsum(widths, axis=1)))
    targets[:, -1] = arc_lengths[-1]
    for index in range(result.shape[1]):
        line_arc = arc_lengths[:, index]
        result[:, index, 0] = np.interp(targets[index], line_arc, grid[:, index, 0])
        result[:, index, 1] = np.interp(targets[index], line_arc, grid[:, index, 1])
    return result


def improve_near_wall_orthogonality(
    grid: np.ndarray,
    *,
    layer_count: int = 10,
    strength: float = 0.25,
) -> np.ndarray:
    """Gently align the first layers with wall normals without changing spacing.

    The correction decays linearly away from the wall.  Each corrected point
    retains its original arc distance along its radial grid line, while a
    positivity-preserving line search prevents the correction from folding
    cells on strongly concave or blunt geometries.
    """
    result = np.asarray(grid, dtype=float).copy()
    corrected_layers = min(layer_count, len(result) - 2)
    if corrected_layers <= 0 or strength <= 0.0:
        return result
    expected_orientation = dominant_orientation(result)
    normals = surface_normals(result[0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    radial_lengths = np.linalg.norm(np.diff(result, axis=0), axis=2)
    radial_distance = np.vstack(
        (np.zeros(result.shape[1]), np.cumsum(radial_lengths, axis=0))
    )
    proposed = result.copy()
    for level in range(1, corrected_layers + 1):
        weight = strength * (1.0 - (level - 1) / corrected_layers)
        normal_offset = result[0] + radial_distance[level, :, None] * normals
        blended = (1.0 - weight) * result[level] + weight * normal_offset
        if level == 1:
            direction = blended - result[0]
            magnitude = np.linalg.norm(direction, axis=1)
            if np.any(magnitude <= 1.0e-14):
                return result
            blended = result[0] + radial_distance[level, :, None] * (
                direction / magnitude[:, None]
            )
        proposed[level] = blended
    proposed[:, -1] = proposed[:, 0]
    fraction = 1.0
    for _ in range(12):
        candidate = result + fraction * (proposed - result)
        candidate[:, -1] = candidate[:, 0]
        if has_consistent_orientation(candidate, expected_orientation):
            return candidate
        fraction *= 0.5
    return result


def _corner_ray_constraints(
    wall: np.ndarray, corner_indices: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Balance the two wall normals at each blunt TE endpoint.

    Unit edge tangents remove the bias from unequal surface/TE panel lengths.
    Their outward normal bisector minimizes the worst wall-corner angle error.
    Retain this direction through the march so the two corner lines do not turn
    back toward the surface normal as successive layers smooth the TE corner.
    A ray parallel to either wall edge would instead collapse its adjacent cell.
    """
    indices = np.asarray(corner_indices, dtype=int)
    count = len(wall) - 1
    if indices.ndim != 1 or np.any((indices < 0) | (indices >= count)):
        raise ValueError("corner indices must reference unique wall nodes")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("corner indices must not repeat")
    before = wall[indices] - wall[(indices - 1) % count]
    after = wall[(indices + 1) % count] - wall[indices]
    before_length = np.linalg.norm(before, axis=1)
    after_length = np.linalg.norm(after, axis=1)
    if np.any(before_length <= 1.0e-14) or np.any(after_length <= 1.0e-14):
        raise ValueError("undefined TE corner direction at a zero-length wall edge")
    tangent = before / before_length[:, None] + after / after_length[:, None]
    magnitude = np.linalg.norm(tangent, axis=1)
    if np.any(magnitude <= 1.0e-14):
        raise ValueError("undefined TE corner bisector at a reversing wall edge")
    directions = np.column_stack((tangent[:, 1], -tangent[:, 0])) / magnitude[:, None]
    return indices, directions


def _local_corner_constraints(
    wall: np.ndarray,
    normal_metric: np.ndarray,
    distance: float,
    constraints: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Blend TE bisectors back into the unconstrained march over local panel scales."""
    if constraints is None:
        return None
    indices, bisectors = constraints
    count = len(wall) - 1
    before = np.linalg.norm(wall[indices] - wall[(indices - 1) % count], axis=1)
    after = np.linalg.norm(wall[(indices + 1) % count] - wall[indices], axis=1)
    transition = 4.0 * float(np.max(np.maximum(before, after)))
    start = 0.25 * transition
    if distance >= transition:
        return None
    fraction = np.clip((distance - start) / (transition - start), 0.0, 1.0)
    weight = 1.0 - fraction * fraction * (3.0 - 2.0 * fraction)
    natural = normal_metric[indices]
    natural /= np.linalg.norm(natural, axis=1)[:, None]
    directions = weight * bisectors + (1.0 - weight) * natural
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return indices, directions


def _stabilized_normal_step(
    current: np.ndarray,
    spacing: float,
    level: int,
    normal_count: int,
    options: MeshSettings,
    corner_constraints: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Continue the march when the implicit linear system becomes ill-conditioned.

    The implicit system can become singular on coarse outer levels. This step
    uses smoothed normals to prevent a folded mesh.
    """
    normals = surface_normals(current)
    unique = len(current) - 1
    strength = max(
        0.0,
        options.hyperbolic_implicit_smoothing * level / max(normal_count - 1, 1) * 0.15,
    )
    if strength > 0.0:
        lower = np.full(unique, -strength)
        diagonal = np.full(unique, 1.0 + 2.0 * strength)
        upper = np.full(unique, -strength)
        smoothed = solve_tridiagonal(
            lower,
            diagonal,
            upper,
            normals[:unique],
            cyclic=True,
        )
        magnitudes = np.linalg.norm(smoothed, axis=1)
        if np.any(magnitudes <= 1.0e-14):
            raise ValueError("Normal smoothing produced a zero-length direction.")
        smoothed /= magnitudes[:, None]
    else:
        smoothed = normals[:unique]
    candidate = current.copy()
    candidate[:unique] = current[:unique] + spacing * smoothed
    if corner_constraints is not None:
        indices, directions = corner_constraints
        candidate[indices] = current[indices] + spacing * directions
    candidate[-1] = candidate[0]
    return candidate


def _close_layer(candidate: np.ndarray) -> None:
    """Restore the duplicate O-grid seam node."""
    candidate[-1] = candidate[0]


def _has_jacobian_margin(
    strip: np.ndarray,
    expected_orientation: int,
    minimum_scaled_jacobian: float = 1.0e-6,
) -> bool:
    """Require a finite positive Jacobian margin, not orientation alone."""
    corners = scaled_corner_jacobians(strip)
    return bool(
        np.all(np.isfinite(corners))
        and np.all(expected_orientation * corners > minimum_scaled_jacobian)
    )


def _relax_layer_displacement(
    current: np.ndarray,
    proposed: np.ndarray,
    expected_orientation: int,
) -> np.ndarray | None:
    """Reduce displacement only beside cells that would reverse orientation."""
    if not np.all(np.isfinite(proposed)):
        return None
    strip = np.stack((current, proposed))
    if _has_jacobian_margin(strip, expected_orientation):
        return proposed

    displacement = proposed - current
    factors = np.ones(len(current), dtype=float)
    for _ in range(20):
        corners = scaled_corner_jacobians(strip)[0]
        if not np.all(np.isfinite(corners)):
            return None
        invalid = np.flatnonzero(
            np.any(
                expected_orientation * corners <= 32.0 * np.finfo(float).eps, axis=-1
            )
        )
        if invalid.size == 0:
            return strip[1]
        affected = np.unique(np.concatenate((invalid, invalid + 1)))
        factors[affected] *= 0.5
        factors[[0, -1]] = min(factors[0], factors[-1])
        strip[1] = current + factors[:, None] * displacement
        _close_layer(strip[1])
        if _has_jacobian_margin(strip, expected_orientation):
            return strip[1].copy()
    return None


def _adaptive_layer(
    current: np.ndarray,
    proposed: np.ndarray | None,
    spacing: float,
    level: int,
    normal_count: int,
    options: MeshSettings,
    expected_orientation: int,
    implicit_weight: float,
    corner_constraints: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray | None:
    """Retry a failing layer with local relaxation and smaller normal steps."""
    if proposed is not None:
        proposed = proposed.copy()
        if implicit_weight < 1.0:
            stabilized = _stabilized_normal_step(
                current,
                spacing,
                level,
                normal_count,
                options,
                corner_constraints,
            )
            proposed = stabilized + implicit_weight * (proposed - stabilized)
            _close_layer(proposed)
        accepted = _relax_layer_displacement(
            current,
            proposed,
            expected_orientation,
        )
        if accepted is not None:
            return accepted

    for exponent in range(13):
        reduced_spacing = spacing * 2.0**-exponent
        stabilized = _stabilized_normal_step(
            current,
            reduced_spacing,
            level,
            normal_count,
            options,
            corner_constraints,
        )
        accepted = _relax_layer_displacement(
            current,
            stabilized,
            expected_orientation,
        )
        if accepted is not None:
            return accepted
    return None


def _pseudo_step_count(
    current: np.ndarray,
    spacing: float,
    maximum_aspect_ratio: float,
) -> int:
    """Choose temporary substeps from radial height / local surface width.

    The temporary fronts are numerical marching aids and are not retained in
    the final grid, so this does not change mesh topology or radial spacing.
    """
    unique = current[:-1]
    forward = np.roll(unique, -1, axis=0) - unique
    backward = unique - np.roll(unique, 1, axis=0)
    edge_widths = np.linalg.norm(forward, axis=1)
    if np.any(edge_widths <= 1.0e-14):
        raise ValueError("pseudo-marching requires positive tangential widths")
    widths = 0.5 * (edge_widths + np.linalg.norm(backward, axis=1))
    worst_aspect_ratio = spacing / float(np.min(widths))
    return max(1, int(np.ceil(worst_aspect_ratio / maximum_aspect_ratio)))


def _algebraic_fallback(inner: np.ndarray, options: MeshSettings) -> np.ndarray:
    """Construct a non-folded normal-tangent grid for pathological coarse inputs."""
    center = np.array((0.5, 0.0))
    raw_angles = np.unwrap(np.arctan2(inner[:, 1] - center[1], inner[:, 0] - center[0]))
    angles = np.maximum.accumulate(raw_angles)
    for index in range(1, len(angles)):
        angles[index] = max(angles[index], angles[index - 1] + 1.0e-10)
    angles = 2.0 * np.pi * (angles - angles[0]) / (angles[-1] - angles[0])
    outer = center + options.farfield_radius_chords * np.column_stack(
        (np.cos(angles), np.sin(angles))
    )
    outer[-1] = outer[0]
    grid = np.empty((options.wall_normal_node_count, len(inner), 2), dtype=float)
    grid[0] = inner
    dense_t = np.linspace(0.0, 1.0, max(400, options.wall_normal_node_count * 8))
    first = wall_distance(
        options.wall_y_plus_target,
        options.flow_reynolds_number,
        options.wall_reference_length_chords,
    )
    for index in range(len(inner)):
        p0 = inner[index]
        p2 = outer[index]
        curve = (1.0 - dense_t[:, None]) * p0 + dense_t[:, None] * p2
        arc_lengths = cumulative_distance(curve)
        targets = geometric_distances(
            arc_lengths[-1], first, options.wall_normal_node_count - 1
        )
        grid[:, index] = interpolate_polyline(curve, targets)
    grid[:, -1] = grid[:, 0]
    return grid


def _reported_algebraic_fallback(
    inner: np.ndarray,
    options: MeshSettings,
) -> np.ndarray:
    """Build an algebraic recovery mesh and report that recovery was used."""
    grid = _algebraic_fallback(inner, options)
    warnings.warn(
        "Hyperbolic marching failed; an algebraic fallback mesh was used.",
        RuntimeWarning,
        stacklevel=2,
    )
    return grid


def march_hyperbolic(
    inner_edge: np.ndarray,
    options: MeshSettings,
    *,
    corner_indices: Sequence[int] = (),
) -> np.ndarray:
    """Generate each normal layer with an implicit hyperbolic system."""
    normal_count = options.wall_normal_node_count
    surface_count = len(inner_edge)
    grid = np.empty((normal_count, surface_count, 2), dtype=float)
    grid[0] = inner_edge
    corner_constraints = (
        _corner_ray_constraints(inner_edge, corner_indices)
        if len(corner_indices)
        else None
    )
    first = wall_distance(
        options.wall_y_plus_target,
        options.flow_reynolds_number,
        options.wall_reference_length_chords,
    )
    ratio = growth_ratio(options.farfield_radius_chords, first, normal_count - 1)
    cell_sizes = first * ratio ** np.arange(normal_count - 1, dtype=float)
    expected_orientation = 0
    implicit_weight = 1.0
    for level in range(normal_count - 1):
        substeps = _pseudo_step_count(
            grid[level],
            cell_sizes[level],
            options.hyperbolic_max_pseudo_aspect_ratio,
        )
        substep_size = cell_sizes[level] / substeps
        current = grid[level]
        distance = float(np.sum(cell_sizes[:level]))
        for substep in range(substeps):
            march_fraction = (level - 1.0) / (normal_count - 2.0)
            area = _target_areas(
                current,
                substep_size,
                march_fraction,
                options,
            )
            tangent, normal_metric = _known_metrics(current, area)
            lower, diagonal, upper, rhs = _assemble_system(
                current, tangent, normal_metric, level, normal_count, options
            )
            local_corner_constraints = _local_corner_constraints(
                inner_edge,
                normal_metric,
                distance + substep * substep_size,
                corner_constraints,
            )
            if local_corner_constraints is not None:
                indices, directions = local_corner_constraints
                lower[indices] = 0.0
                upper[indices] = 0.0
                diagonal[indices] = np.eye(2)
                rhs.reshape(-1, 2)[indices] = (
                    current[indices] + substep_size * directions
                )
            proposed: np.ndarray | None = None
            try:
                solution = _solve_block_system(lower, diagonal, upper, rhs)
            except np.linalg.LinAlgError:
                pass
            else:
                next_level = solution.reshape(-1, 2)
                proposed = current.copy()
                proposed[:-1] = next_level
                proposed[-1] = next_level[0]
                displacement = np.linalg.norm(proposed - current, axis=1)
                unstable = (
                    not np.all(np.isfinite(proposed))
                    or float(np.max(displacement)) > 2.0 * substep_size
                )
                if unstable:
                    proposed = None
            weight_step = 1.0 / (_IMPLICIT_TRANSITION_LAYERS * substeps)
            if proposed is None:
                implicit_weight = max(0.0, implicit_weight - weight_step)
            else:
                implicit_weight = min(1.0, implicit_weight + weight_step)
            reference = proposed
            effective_level = level + (substep + 1.0) / substeps
            if reference is None:
                reference = _stabilized_normal_step(
                    current,
                    substep_size,
                    effective_level,
                    normal_count,
                    options,
                    local_corner_constraints,
                )
            strip_orientation = dominant_orientation(np.stack((current, reference)))
            if expected_orientation == 0:
                expected_orientation = strip_orientation
            candidate = _adaptive_layer(
                current,
                proposed,
                substep_size,
                effective_level,
                normal_count,
                options,
                expected_orientation,
                implicit_weight,
                local_corner_constraints,
            )
            if candidate is None:
                return _reported_algebraic_fallback(inner_edge, options)
            current = candidate
        grid[level + 1] = current
        if not np.all(np.isfinite(grid[level + 1])):
            raise ValueError(f"hyperbolic solve failed at level {level + 2}")
    result = apply_normal_spacing(grid, options)
    if not has_consistent_orientation(result, expected_orientation):
        result = _reported_algebraic_fallback(inner_edge, options)
    return result
