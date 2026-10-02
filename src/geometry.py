"""Airfoil loading, cleanup, surface preparation, and farfield geometry."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize

from .utils import periodic_first_derivative

if TYPE_CHECKING:
    from .models import MeshSettings


def _segments_intersect(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> bool:
    """Detect crossings, overlaps, and contacts between two segments."""

    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    scale = max(np.linalg.norm(b - a), np.linalg.norm(d - c))
    tolerance = 32.0 * np.finfo(float).eps * scale
    determinants = (
        cross(b - a, c - a),
        cross(b - a, d - a),
        cross(d - c, a - c),
        cross(d - c, b - c),
    )
    if (
        determinants[0] * determinants[1] < 0.0
        and determinants[2] * determinants[3] < 0.0
    ):
        return True
    for value, point, start, end in zip(
        determinants, (c, d, a, b), (a, a, c, c), (b, b, d, d), strict=True
    ):
        if abs(value) <= tolerance * scale and np.all(
            (point >= np.minimum(start, end) - tolerance)
            & (point <= np.maximum(start, end) + tolerance)
        ):
            return True
    return False


def has_open_trailing_edge(points: np.ndarray) -> bool:
    """Use one chord-relative gap tolerance for all trailing-edge decisions."""
    midpoint = 0.5 * (points[0] + points[-1])
    chord = float(np.max(np.linalg.norm(points - midpoint, axis=1)))
    return bool(np.linalg.norm(points[0] - points[-1]) > chord * 1.0e-12)


@dataclass(frozen=True)
class AirfoilProfile:
    """An airfoil ordered from the upper TE around the LE to the lower TE."""

    points: np.ndarray
    name: str = "airfoil"
    corner_points: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
            raise ValueError("airfoil points must have shape (n, 2), n >= 5")
        if not np.all(np.isfinite(points)):
            raise ValueError("airfoil contains non-finite coordinates")
        scale = float(np.max(np.ptp(points, axis=0)))
        if scale <= 1.0e-14:
            raise ValueError("airfoil chord is zero")
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        duplicate_tolerance = max(scale * 1.0e-14, 1.0e-15)
        corner_points = points[:-1][lengths <= duplicate_tolerance].copy()
        keep = np.concatenate(([True], lengths > duplicate_tolerance))
        if not np.all(keep):
            points = points[keep]
            warnings.warn(
                "Consecutive duplicate airfoil points removed.",
                RuntimeWarning,
                stacklevel=2,
            )
            if len(points) < 5:
                raise ValueError(
                    "airfoil has fewer than five distinct consecutive points"
                )
        trailing_edge = 0.5 * (points[0] + points[-1])
        leading_index = int(np.argmax(np.linalg.norm(points - trailing_edge, axis=1)))
        if leading_index in {0, len(points) - 1}:
            raise ValueError("airfoil must run from upper TE around the LE to lower TE")
        closed = (
            np.vstack((points, points[0]))
            if has_open_trailing_edge(points)
            else points.copy()
        )
        closed[-1] = closed[0]
        edges = np.diff(closed, axis=0)
        following = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        edge_scale = np.linalg.norm(edges, axis=1) * np.linalg.norm(following, axis=1)
        if np.any(
            (np.abs(cross) <= 32.0 * np.finfo(float).eps * edge_scale)
            & (np.sum(edges * following, axis=1) < 0.0)
        ):
            warnings.warn(
                "Adjacent airfoil segments overlap.", RuntimeWarning, stacklevel=2
            )
        intersects = False
        for first in range(len(closed) - 1):
            for second in range(first + 2, len(closed) - 1):
                if first == 0 and second == len(closed) - 2:
                    continue
                if _segments_intersect(
                    closed[first], closed[first + 1], closed[second], closed[second + 1]
                ):
                    intersects = True
                    break
            if intersects:
                warnings.warn(
                    "Airfoil surface intersects itself.", RuntimeWarning, stacklevel=2
                )
                break
        object.__setattr__(self, "points", points.copy())
        corner_points.setflags(write=False)
        object.__setattr__(self, "corner_points", corner_points)

    @property
    def has_trailing_edge_gap(self) -> bool:
        return has_open_trailing_edge(self.points)

    @classmethod
    def read(cls, path: str | Path) -> AirfoilProfile:
        source = Path(path)
        values: list[tuple[float, float]] = []
        started = False
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            fields = line.replace(",", " ").split()
            if not fields:
                continue
            try:
                point = (float(fields[0]), float(fields[1]))
            except (ValueError, IndexError):
                if not started:
                    continue
                raise ValueError(
                    f"invalid coordinate on line {line_number} of {source}"
                ) from None
            started = True
            values.append(point)
        if len(values) < 5:
            raise ValueError(f"{source} does not contain at least five coordinates")
        points = np.asarray(values, dtype=float)
        if points[1, 1] < points[-2, 1]:
            points = points[::-1]
        return cls(points, source.stem)


def transform_airfoil(points: np.ndarray) -> np.ndarray:
    """Align the chord with x, place the LE at zero, and set the TE at one."""
    raw = np.asarray(points, dtype=float)
    trailing_edge = 0.5 * (raw[0] + raw[-1])
    leading_edge = raw[np.argmax(np.linalg.norm(raw - trailing_edge, axis=1))]
    chord_vector = trailing_edge - leading_edge
    scale = float(np.linalg.norm(chord_vector))
    if scale <= 1.0e-14:
        raise ValueError("cannot normalize an airfoil with zero trailing-edge chord")
    chord_direction = chord_vector / scale
    normal_direction = np.array((-chord_direction[1], chord_direction[0]))
    relative = raw - leading_edge
    return (
        np.column_stack((relative @ chord_direction, relative @ normal_direction))
        / scale
    )


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )


def interpolate_polyline(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    arc_lengths = cumulative_distance(points)
    targets = np.clip(np.asarray(distances, dtype=float), 0.0, arc_lengths[-1])
    return np.column_stack(
        (
            np.interp(targets, arc_lengths, points[:, 0]),
            np.interp(targets, arc_lengths, points[:, 1]),
        )
    )


def wall_distance(
    target_wall_y_plus: float, reynolds_number: float, reference_length: float
) -> float:
    """Return full first-layer height in chord units for a cell-centred solver.

    Re is based on the same length L/c supplied as reference_length. With
    Cf = (2 log10(Re) - 0.65)^-2.3, the estimated wall-to-cell-centre distance
    is y/c = y_plus * (L/c) / (Re * sqrt(Cf/2)). The mesher needs h/c = 2*y/c,
    not y/c. This assumes thin, nearly orthogonal cells and a smooth-wall,
    zero-pressure-gradient turbulent skin-friction correlation. Re alone
    cannot determine the actual local wall shear or the solved y-plus.
    """
    values = (target_wall_y_plus, reynolds_number, reference_length)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("wall-spacing inputs must be finite and positive")
    if reynolds_number <= 10.0**0.325:
        raise ValueError(
            "flow_reynolds_number must exceed 10**0.325 for the wall-spacing formula"
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        friction = (2.0 * np.log10(reynolds_number) - 0.65) ** -2.3
        distance = (
            2.0
            * target_wall_y_plus
            * reference_length
            / (reynolds_number * np.sqrt(0.5 * friction))
        )
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("wall-spacing inputs must produce a finite, positive distance")
    return float(distance)


def _optimized_widths(
    points: int, length: float, first: float, last: float
) -> np.ndarray:
    """Compatibility wrapper for the unified Bernstein-3 SLSQP solve."""
    return _unified_widths(points, length, first, last, "BERNSTEIN3")


def _unified_widths(
    points: int,
    length: float,
    first: float,
    last: float,
    method: str = "BERNSTEIN3",
) -> np.ndarray:
    """Solve one cubic-plus-sigmoid basis family by minimax-lambda SLSQP.

    ``SIGMOID5`` activates the quadratic background and two endpoint sigmoid
    terms (the original five-coefficient law). ``BERNSTEIN3`` activates only
    the cubic background, and ``HYBRID`` permits both sets of shape terms.
    Every mode uses the same basis-matrix elimination, feasibility phase,
    explicit adjacent-ratio lambda, SLSQP optimizer, and smoothness tie-break.
    """
    mode = method.upper()
    if mode not in {"BERNSTEIN3", "SIGMOID5", "HYBRID"}:
        raise ValueError("spacing_method must be BERNSTEIN3, SIGMOID5, or HYBRID")
    if points < 4 or min(length, first, last) <= 0.0 or length <= first + last:
        raise ValueError("surface spacing inputs do not define a positive distribution")
    n = points - 1
    mean = length / n
    peak_factor = 1.65
    if max(first, last) > peak_factor * mean:
        raise ValueError("endpoint spacing exceeds the maximum peak factor")
    x = np.linspace(0.0, 1.0, n)
    # This is tanh(2x) in sigmoid form.
    left = 2.0 / (1.0 + np.exp(-4.0 * x)) - 1.0
    right = 2.0 / (1.0 + np.exp(-4.0 * (1.0 - x))) - 1.0
    dependent_basis = np.column_stack((np.ones(n), x, x**2))
    free_basis = {
        "BERNSTEIN3": np.column_stack((x**3,)),
        "SIGMOID5": np.column_stack((left, right)),
        "HYBRID": np.column_stack((x**3, left, right)),
    }[mode]

    def constraint_columns(basis: np.ndarray) -> np.ndarray:
        return np.vstack((basis[0], basis[-1], np.sum(basis, axis=0)))

    target = np.array((first / mean, last / mean, n), dtype=float)
    dependent_constraints = constraint_columns(dependent_basis)
    free_constraints = constraint_columns(free_basis)
    dependent_offset = np.linalg.solve(dependent_constraints, target)
    dependent_from_free = np.linalg.solve(dependent_constraints, free_constraints)
    width_offset = dependent_basis @ dependent_offset
    width_from_free = free_basis - dependent_basis @ dependent_from_free

    def normalized_widths(free_coefficients: np.ndarray) -> np.ndarray:
        return width_offset + width_from_free @ free_coefficients

    epsilon = 1.0e-12
    coefficient_bounds = [(-50.0, 50.0)] * free_basis.shape[1]

    def feasibility_objective(free_coefficients: np.ndarray) -> float:
        values = normalized_widths(free_coefficients)
        below = np.minimum(values - epsilon, 0.0)
        above = np.maximum(values - peak_factor, 0.0)
        return float(below @ below + above @ above)

    seed = minimize(
        feasibility_objective,
        np.zeros(free_basis.shape[1]),
        method="SLSQP",
        bounds=coefficient_bounds,
        options={"ftol": 1.0e-14, "maxiter": 300},
    )
    seed_coefficients = np.asarray(seed.x, dtype=float)
    seed_widths = normalized_widths(seed_coefficients)
    if (
        np.min(seed_widths) < epsilon * 0.99
        or np.max(seed_widths) > peak_factor + 1.0e-10
    ):
        raise ValueError(f"{mode.lower()} spacing constraints are infeasible")
    seed_lambda = max(
        float(np.max(seed_widths[1:] / seed_widths[:-1] - 1.0)),
        float(np.max(seed_widths[:-1] / seed_widths[1:] - 1.0)),
        0.0,
    )
    initial = np.concatenate(
        (seed_coefficients, [seed_lambda * (1.0 + 1.0e-8) + 1.0e-10])
    )

    def minimax_constraints(variables: np.ndarray) -> np.ndarray:
        values = normalized_widths(variables[:-1])
        ratio = 1.0 + variables[-1]
        return np.concatenate(
            (
                values - epsilon,
                peak_factor - values,
                ratio * values[:-1] - values[1:],
                ratio * values[1:] - values[:-1],
            )
        )

    bounds = [*coefficient_bounds, (0.0, None)]
    primary = minimize(
        lambda variables: float(variables[-1]),
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "ineq", "fun": minimax_constraints},
        options={"ftol": 1.0e-12, "maxiter": 500},
    )
    if not primary.success or np.min(minimax_constraints(primary.x)) < -1.0e-8:
        raise ValueError(
            f"{mode.lower()} minimax spacing optimization failed: {primary.message}"
        )

    lambda_limit = float(primary.x[-1]) * (1.0 + 1.0e-8) + 1.0e-10

    def curvature(variables: np.ndarray) -> float:
        values = normalized_widths(variables[:-1])
        return float(np.sum(np.diff(np.log(values), 2) ** 2))

    secondary = minimize(
        curvature,
        primary.x,
        method="SLSQP",
        bounds=bounds,
        constraints=(
            {"type": "ineq", "fun": minimax_constraints},
            {"type": "ineq", "fun": lambda variables: lambda_limit - variables[-1]},
        ),
        options={"ftol": 1.0e-13, "maxiter": 300},
    )
    solution = secondary.x if secondary.success else primary.x
    values = normalized_widths(solution[:-1]) * mean
    values[0] = first
    values[-1] = last
    if (
        np.any(values <= 0.0)
        or np.max(values) > peak_factor * mean * (1.0 + 1.0e-10)
        or not np.isclose(np.sum(values), length, rtol=1.0e-10, atol=1.0e-13)
    ):
        raise ValueError(f"{mode.lower()} spacing optimization produced invalid widths")
    return values


def _leading_edge_arc(points: np.ndarray) -> float:
    arc_lengths = cumulative_distance(points)
    spline_x = CubicSpline(arc_lengths, points[:, 0], bc_type="natural")
    roots = spline_x.derivative().roots(extrapolate=False)
    interior = roots[(roots > 0.0) & (roots < arc_lengths[-1])]
    candidates = np.concatenate(([0.0], interior, [arc_lengths[-1]]))
    return float(candidates[np.argmin(spline_x(candidates))])


def _point_segment_distance(
    point: np.ndarray, first: np.ndarray, last: np.ndarray
) -> float:
    edge = last - first
    denominator = float(edge @ edge)
    if denominator <= np.finfo(float).tiny:
        return float(np.linalg.norm(point - first))
    fraction = float(np.clip((point - first) @ edge / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (first + fraction * edge)))


class GeometryPreparationError(ValueError):
    """Geometry failure with a stable machine-readable classification."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _condition_surface_segment(
    points: np.ndarray,
    mode: str = "AUTO",
    deviation_tolerance: float = 2.0e-4,
    maximum_depth: int = 10,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build and validate one local reference curve."""
    source = np.asarray(points, dtype=float)
    if len(source) < 3:
        return source.copy(), {"mode": "POLYLINE", "reason": "fewer than three points"}
    parameter = cumulative_distance(source)
    if np.any(np.diff(parameter) <= 1.0e-14):
        return source.copy(), {"mode": "POLYLINE", "reason": "coincident points"}

    requested = mode.upper()
    if requested == "POLYLINE":
        return source.copy(), {"mode": "POLYLINE", "reason": "explicit selection"}
    if requested not in {"AUTO", "SEGMENTED_HERMITE", "NATURAL_CUBIC_LEGACY"}:
        raise ValueError("unsupported geometry conditioning mode")

    if requested == "NATURAL_CUBIC_LEGACY":
        spline_x = CubicSpline(parameter, source[:, 0], bc_type="natural")
        spline_y = CubicSpline(parameter, source[:, 1], bc_type="natural")
        accepted_mode = "NATURAL_CUBIC_LEGACY"
    else:
        spline_x = PchipInterpolator(parameter, source[:, 0])
        spline_y = PchipInterpolator(parameter, source[:, 1])
        accepted_mode = "SEGMENTED_HERMITE"
    derivative_x = spline_x.derivative()
    derivative_y = spline_y.derivative()
    absolute_tolerance = deviation_tolerance
    maximum_turn = np.deg2rad(2.0)
    samples = [source[0].copy()]
    maximum_deviation = 0.0
    maximum_tangent_turn = 0.0
    maximum_deviation_ratio = 0.0

    def evaluate(value: float) -> np.ndarray:
        return np.array((spline_x(value), spline_y(value)), dtype=float)

    def tangent(value: float) -> np.ndarray:
        derivative = np.array((derivative_x(value), derivative_y(value)))
        magnitude = float(np.linalg.norm(derivative))
        return (
            derivative / magnitude if magnitude > np.finfo(float).tiny else derivative
        )

    def refine(
        start_s: float,
        end_s: float,
        start: np.ndarray,
        end: np.ndarray,
        segment_length: float,
        depth: int,
    ) -> None:
        nonlocal maximum_deviation, maximum_tangent_turn, maximum_deviation_ratio
        middle_s = 0.5 * (start_s + end_s)
        middle = evaluate(middle_s)
        deviation = _point_segment_distance(middle, start, end)
        local_tolerance = max(absolute_tolerance, 0.05 * segment_length)
        start_tangent = tangent(start_s)
        end_tangent = tangent(end_s)
        tangent_turn = float(np.arccos(np.clip(start_tangent @ end_tangent, -1.0, 1.0)))
        needs_refinement = (
            deviation > 0.25 * local_tolerance or tangent_turn > maximum_turn
        )
        if depth < maximum_depth and needs_refinement:
            refine(start_s, middle_s, start, middle, segment_length * 0.5, depth + 1)
            refine(middle_s, end_s, middle, end, segment_length * 0.5, depth + 1)
        else:
            maximum_deviation = max(maximum_deviation, deviation)
            maximum_deviation_ratio = max(
                maximum_deviation_ratio, deviation / local_tolerance
            )
            maximum_tangent_turn = max(maximum_tangent_turn, tangent_turn)
            samples.append(end.copy())

    for index in range(len(source) - 1):
        refine(
            float(parameter[index]),
            float(parameter[index + 1]),
            source[index],
            source[index + 1],
            float(parameter[index + 1] - parameter[index]),
            0,
        )
    conditioned = np.asarray(samples)
    rejected = np.any(~np.isfinite(conditioned)) or maximum_deviation_ratio > 1.0
    if rejected:
        if requested != "AUTO":
            raise GeometryPreparationError(
                "INTERPOLATION_REJECTED",
                f"local deviation ratio {maximum_deviation_ratio:.6g} exceeds one",
            )
        return source.copy(), {
            "mode": "POLYLINE",
            "reason": "deviation limit",
            "maximum_deviation": maximum_deviation,
            "maximum_deviation_ratio": maximum_deviation_ratio,
        }
    conditioned[0] = source[0]
    conditioned[-1] = source[-1]
    return conditioned, {
        "mode": accepted_mode,
        "reason": "accepted",
        "maximum_deviation": maximum_deviation,
        "maximum_deviation_ratio": maximum_deviation_ratio,
        "maximum_tangent_turn_degrees": float(np.degrees(maximum_tangent_turn)),
        "support_point_count": len(conditioned),
    }


def _condition_surface_with_breaks(
    points: np.ndarray,
    break_indices: tuple[int, ...],
    mode: str,
    deviation_tolerance: float,
    maximum_depth: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    interior = sorted(index for index in break_indices if 0 < index < len(points) - 1)
    boundaries = (0, *interior, len(points) - 1)
    pieces: list[np.ndarray] = []
    diagnostics: list[dict[str, object]] = []
    for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
        piece, diagnostic = _condition_surface_segment(
            points[first : last + 1], mode, deviation_tolerance, maximum_depth
        )
        pieces.append(piece if not pieces else piece[1:])
        diagnostics.append(diagnostic)
    return np.vstack(pieces), diagnostics


def _mean_camber_diagnostics(
    upper_te_to_le: np.ndarray, lower_le_to_te: np.ndarray
) -> dict[str, object]:
    upper = upper_te_to_le[::-1]
    lower = lower_le_to_te

    def unique_surface(surface: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(surface[:, 0], kind="stable")
        x = surface[order, 0]
        y = surface[order, 1]
        unique_x, inverse = np.unique(x, return_inverse=True)
        sums = np.bincount(inverse, weights=y)
        counts = np.bincount(inverse)
        return unique_x, sums / counts

    upper_x, upper_y = unique_surface(upper)
    lower_x, lower_y = unique_surface(lower)
    start = max(float(upper_x[0]), float(lower_x[0]))
    end = min(float(upper_x[-1]), float(lower_x[-1]))
    if end <= start:
        raise GeometryPreparationError(
            "INPUT_GEOMETRY_INVALID", "surfaces have no x overlap"
        )
    angle = np.linspace(np.pi, 0.0, 257)
    stations = start + 0.5 * (end - start) * (1.0 + np.cos(angle))
    upper_values = np.interp(stations, upper_x, upper_y)
    lower_values = np.interp(stations, lower_x, lower_y)
    camber = 0.5 * (upper_values + lower_values)
    thickness = upper_values - lower_values
    negative = thickness < -1.0e-10
    return {
        "station_count": len(stations),
        "minimum_thickness": float(np.min(thickness)),
        "maximum_thickness": float(np.max(thickness)),
        "maximum_absolute_camber": float(np.max(np.abs(camber))),
        "negative_thickness_station_count": int(np.count_nonzero(negative)),
    }


def _split_about_mean_camber(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, object]]:
    """Split the ordered contour at the LE and identify sides about mean camber.

    The two TE endpoints remain independent. Contour order supplies the two
    branches; the mean-camber line, rather than the sign of y, verifies which
    branch is upper and which is lower for cambered and rotated airfoils.
    """
    leading_index = int(np.argmin(points[:, 0]))
    if leading_index in {0, len(points) - 1}:
        raise GeometryPreparationError(
            "INPUT_GEOMETRY_INVALID",
            "airfoil reference curve has no interior leading edge",
        )
    first = points[: leading_index + 1]
    second = points[leading_index:]
    diagnostics = _mean_camber_diagnostics(first, second)
    if diagnostics["negative_thickness_station_count"]:
        swapped = _mean_camber_diagnostics(second[::-1], first[::-1])
        if swapped["negative_thickness_station_count"]:
            raise GeometryPreparationError(
                "INPUT_GEOMETRY_INVALID", "surface branches cross the mean-camber line"
            )
        first, second, diagnostics = second[::-1], first[::-1], swapped
    diagnostics["split_method"] = "ORDERED_CONTOUR_VALIDATED_BY_MEAN_CAMBER"
    diagnostics["upper_trailing_edge"] = first[0].tolist()
    diagnostics["lower_trailing_edge"] = second[-1].tolist()
    diagnostics["trailing_edge_gap"] = float(np.linalg.norm(first[0] - second[-1]))
    return first, second, leading_index, diagnostics


def _curvature_repanel(points: np.ndarray, final_cells: int) -> np.ndarray:
    """Create dense curvature-aware support points without moving the curve."""
    arc = cumulative_distance(points)
    support_count = max(5 * final_cells + 1, len(points))
    uniform_arc = np.linspace(0.0, arc[-1], support_count)
    dense = _evaluate_reference_curve(points, uniform_arc)
    segments = np.diff(dense, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= np.finfo(float).tiny):
        return points.copy()
    unit = segments / lengths[:, None]
    turns = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0))
    curvature = np.zeros(support_count)
    curvature[1:-1] = turns / np.maximum(
        0.5 * (lengths[:-1] + lengths[1:]), np.finfo(float).tiny
    )
    window = max(3, 2 * (support_count // 100) + 1)
    kernel = np.ones(window) / window
    smooth = np.convolve(curvature, kernel, mode="same")
    scale = float(np.max(smooth))
    attraction = np.ones(support_count)
    if scale > np.finfo(float).tiny:
        attraction += 2.0 * smooth / scale
    weighted = np.concatenate(
        ([0.0], np.cumsum(lengths * 0.5 * (attraction[:-1] + attraction[1:])))
    )
    weighted_targets = np.linspace(0.0, weighted[-1], support_count)
    selected_arc = np.interp(weighted_targets, weighted, uniform_arc)
    selected_arc = np.unique(np.concatenate((selected_arc, arc)))
    result = _evaluate_reference_curve(dense, selected_arc)
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def _evaluate_reference_curve(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Evaluate arc-distance targets on an adaptively sampled reference curve."""
    arc = cumulative_distance(points)
    targets = np.clip(np.asarray(distances, dtype=float), 0.0, arc[-1])
    result = np.column_stack(
        (np.interp(targets, arc, points[:, 0]), np.interp(targets, arc, points[:, 1]))
    )
    result[0] = points[0]
    result[-1] = points[-1]
    return result


def _curve_fidelity_metrics(points: np.ndarray, source: np.ndarray) -> dict[str, float]:
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    unit = segments / np.maximum(lengths[:, None], np.finfo(float).tiny)
    turns = np.degrees(
        np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1.0, 1.0))
    )
    curvature = turns / np.maximum(
        0.5 * (lengths[:-1] + lengths[1:]), np.finfo(float).tiny
    )
    starts = source[:-1]
    edges = np.diff(source, axis=0)
    edge_squared = np.sum(edges * edges, axis=1)
    offsets = points[:, None, :] - starts[None, :, :]
    fractions = np.sum(offsets * edges[None, :, :], axis=2) / np.maximum(
        edge_squared[None, :], np.finfo(float).tiny
    )
    projections = starts[None, :, :] + np.clip(fractions, 0.0, 1.0)[:, :, None] * edges
    deviations = np.min(
        np.linalg.norm(points[:, None, :] - projections, axis=2), axis=1
    )
    return {
        "maximum_polyline_deviation_chords": float(np.max(deviations)),
        "rms_polyline_deviation_chords": float(np.sqrt(np.mean(deviations**2))),
        "maximum_segment_turn_degrees": float(np.max(turns, initial=0.0)),
        "maximum_discrete_curvature_degrees_per_chord": float(
            np.max(curvature, initial=0.0)
        ),
    }


def redistribute_airfoil(
    points: np.ndarray,
    count: int,
    leading_spacing: float,
    trailing_spacing: float,
    spacing_method: str = "BERNSTEIN3",
    geometry_conditioning_mode: str = "AUTO",
    geometry_deviation_tolerance_chords: float = 2.0e-4,
    geometry_max_refinement_depth: int = 10,
    corner_indices: tuple[int, ...] = (),
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Condition geometry, then distribute nodes with fixed endpoint sizes."""
    raw = np.asarray(points, dtype=float)
    input_arc = cumulative_distance(raw)
    if np.any(np.diff(input_arc) <= 1.0e-14):
        keep = np.concatenate(([True], np.diff(input_arc) > 1.0e-14))
        raw = raw[keep]
    top_source, bottom_source, leading_index, camber_diagnostics = (
        _split_about_mean_camber(raw)
    )
    top_breaks = tuple(index for index in corner_indices if index <= leading_index)
    bottom_breaks = tuple(
        index - leading_index for index in corner_indices if index >= leading_index
    )
    top_reference, top_diagnostics = _condition_surface_with_breaks(
        top_source,
        top_breaks,
        geometry_conditioning_mode,
        geometry_deviation_tolerance_chords,
        geometry_max_refinement_depth,
    )
    bottom_reference, bottom_diagnostics = _condition_surface_with_breaks(
        bottom_source,
        bottom_breaks,
        geometry_conditioning_mode,
        geometry_deviation_tolerance_chords,
        geometry_max_refinement_depth,
    )
    top_length = cumulative_distance(top_reference)[-1]
    bottom_length = cumulative_distance(bottom_reference)[-1]
    total_cells = count - 1
    total_length = top_length + bottom_length
    top_cells = int(
        np.clip(round(total_cells * top_length / total_length), 2, total_cells - 2)
    )
    bottom_cells = total_cells - top_cells
    top_reference = _curvature_repanel(top_reference, top_cells)
    bottom_reference = _curvature_repanel(bottom_reference, bottom_cells)
    try:
        top_widths = _unified_widths(
            top_cells + 1, top_length, trailing_spacing, leading_spacing, spacing_method
        )
        bottom_widths = _unified_widths(
            bottom_cells + 1,
            bottom_length,
            leading_spacing,
            trailing_spacing,
            spacing_method,
        )
    except ValueError as error:
        raise GeometryPreparationError(
            "SURFACE_SPACING_INFEASIBLE", str(error)
        ) from error
    top_targets = np.concatenate(([0.0], np.cumsum(top_widths)))
    bottom_targets = np.concatenate(([0.0], np.cumsum(bottom_widths)))
    top = _evaluate_reference_curve(top_reference, top_targets)
    bottom = _evaluate_reference_curve(bottom_reference, bottom_targets)
    result = np.vstack((top, bottom[1:]))
    result[0] = raw[0]
    result[-1] = raw[-1]
    if np.any(~np.isfinite(result)) or np.any(
        np.linalg.norm(np.diff(result, axis=0), axis=1) <= 0
    ):
        raise GeometryPreparationError(
            "WALL_BOUNDARY_INVALID",
            "redistributed wall contains invalid or repeated points",
        )
    diagnostics = {
        "conditioning_mode": geometry_conditioning_mode.upper(),
        "input_point_count": len(raw),
        "leading_edge_input_index": leading_index,
        "upper": top_diagnostics,
        "lower": bottom_diagnostics,
        "mean_camber": camber_diagnostics,
        "upper_reference_point_count": len(top_reference),
        "lower_reference_point_count": len(bottom_reference),
        "upper_wall_fidelity": _curve_fidelity_metrics(top, top_source),
        "lower_wall_fidelity": _curve_fidelity_metrics(bottom, bottom_source),
        "corner_indices": list(corner_indices),
    }
    return (result, diagnostics) if return_diagnostics else result


def close_blunt_trailing_edge(
    points: np.ndarray, face_cell_count: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Close a blunt trailing edge without changing its supplied geometry.

    The airfoil runs from upper TE to lower TE.  The only added geometry is a
    uniformly subdivided straight segment from the lower TE back to the upper
    TE.  Both supplied endpoints remain bit-for-bit unchanged.
    """
    foil = np.asarray(points, dtype=float)
    upper = foil[0].copy()
    lower = foil[-1].copy()
    if not has_open_trailing_edge(foil):
        closed = foil.copy()
        closed[[0, -1]] = 0.5 * (upper + lower)
        return closed, (0, len(foil) - 1)
    te_face = np.linspace(lower, upper, face_cell_count + 1)
    closed = np.vstack((foil, te_face[1:]))
    return closed, (0, len(foil) - 1)


def _wall_has_self_intersection(points: np.ndarray) -> bool:
    for first in range(len(points) - 1):
        for second in range(first + 2, len(points) - 1):
            if first == 0 and second == len(points) - 2:
                continue
            if _segments_intersect(
                points[first], points[first + 1], points[second], points[second + 1]
            ):
                return True
    return False


def prepare_surface_with_diagnostics(
    airfoil: AirfoilProfile, options: MeshSettings
) -> tuple[np.ndarray, tuple[int, int], dict[str, object]]:
    """Prepare a faithful closed wall and return geometry provenance."""
    normalized = transform_airfoil(airfoil.points)
    has_gap = airfoil.has_trailing_edge_gap
    original_upper_te = normalized[0].copy()
    original_lower_te = normalized[-1].copy()
    if not has_gap:
        normalized[[0, -1]] = 0.5 * (normalized[0] + normalized[-1])
    corner_indices = tuple(
        index
        for index, point in enumerate(airfoil.points)
        if len(airfoil.corner_points)
        and np.any(
            np.all(np.isclose(point, airfoil.corner_points, atol=1.0e-14), axis=1)
        )
    )
    if options.surface_point_mode == "PRESERVE":
        surface = normalized.copy()
        diagnostics: dict[str, object] = {
            "conditioning_mode": "PRESERVE",
            "input_point_count": len(normalized),
            "corner_indices": list(corner_indices),
        }
    else:
        base_count = options.circumferential_node_count - (
            options.trailing_edge_face_cell_count if has_gap else 0
        )
        uniform = 2.0 / base_count
        le = (
            options.leading_edge_cell_length
            if options.leading_edge_cell_length is not None
            else uniform / 2.0
        )
        te = options.trailing_edge_cell_length
        if te is None:
            if has_gap:
                # This only sizes the panels beside the fixed TE face.
                te = max(
                    np.linalg.norm(normalized[0] - normalized[-1]) / 10.0, uniform / 1.5
                )
            else:
                te = uniform / 1.5
        try:
            surface, diagnostics = redistribute_airfoil(
                normalized,
                base_count,
                le,
                te,
                spacing_method=options.spacing_method,
                geometry_conditioning_mode=options.geometry_conditioning_mode,
                geometry_deviation_tolerance_chords=(
                    options.geometry_deviation_tolerance_chords
                ),
                geometry_max_refinement_depth=options.geometry_max_refinement_depth,
                corner_indices=corner_indices,
                return_diagnostics=True,
            )
        except GeometryPreparationError as error:
            if options.geometry_conditioning_mode != "AUTO" or error.code not in {
                "INTERPOLATION_REJECTED",
                "WALL_BOUNDARY_INVALID",
            }:
                raise
            surface, diagnostics = redistribute_airfoil(
                normalized,
                base_count,
                le,
                te,
                spacing_method=options.spacing_method,
                geometry_conditioning_mode="POLYLINE",
                geometry_deviation_tolerance_chords=(
                    options.geometry_deviation_tolerance_chords
                ),
                geometry_max_refinement_depth=options.geometry_max_refinement_depth,
                corner_indices=corner_indices,
                return_diagnostics=True,
            )
            diagnostics["auto_fallback_reason"] = str(error)
    if has_gap:
        surface, corners = close_blunt_trailing_edge(
            surface, options.trailing_edge_face_cell_count
        )
        if not np.array_equal(surface[corners[0]], original_upper_te):
            raise GeometryPreparationError(
                "TRAILING_EDGE_FIDELITY_FAILED", "upper blunt-TE endpoint moved"
            )
        if not np.array_equal(surface[corners[1]], original_lower_te):
            raise GeometryPreparationError(
                "TRAILING_EDGE_FIDELITY_FAILED", "lower blunt-TE endpoint moved"
            )
        expected_gap = float(np.linalg.norm(original_upper_te - original_lower_te))
        actual_gap = float(np.linalg.norm(surface[corners[0]] - surface[corners[1]]))
        if actual_gap != expected_gap:
            raise GeometryPreparationError(
                "TRAILING_EDGE_FIDELITY_FAILED", "blunt-TE gap changed"
            )
    else:
        if np.linalg.norm(surface[0] - surface[-1]) > 1.0e-12:
            surface = np.vstack((surface, surface[0]))
        surface[-1] = surface[0]
        corners = (1, len(surface) - 2)
        if not np.array_equal(surface[0], normalized[0]) or not np.array_equal(
            surface[-1], normalized[0]
        ):
            raise GeometryPreparationError(
                "TRAILING_EDGE_FIDELITY_FAILED",
                "sharp trailing edge is not exactly closed",
            )
    if (
        _wall_has_self_intersection(surface)
        and options.surface_point_mode == "REDISTRIBUTE"
        and options.geometry_conditioning_mode == "AUTO"
    ):
        surface, fallback_diagnostics = redistribute_airfoil(
            normalized,
            base_count,
            le,
            te,
            spacing_method=options.spacing_method,
            geometry_conditioning_mode="POLYLINE",
            geometry_deviation_tolerance_chords=(
                options.geometry_deviation_tolerance_chords
            ),
            geometry_max_refinement_depth=options.geometry_max_refinement_depth,
            corner_indices=corner_indices,
            return_diagnostics=True,
        )
        fallback_diagnostics["auto_fallback_reason"] = "conditioned wall intersection"
        diagnostics = fallback_diagnostics
        if has_gap:
            surface, corners = close_blunt_trailing_edge(
                surface, options.trailing_edge_face_cell_count
            )
        else:
            surface[-1] = surface[0]
            corners = (1, len(surface) - 2)
    if _wall_has_self_intersection(surface):
        raise GeometryPreparationError(
            "WALL_BOUNDARY_INVALID", "prepared wall intersects itself"
        )
    if has_gap and (
        not np.array_equal(surface[corners[0]], original_upper_te)
        or not np.array_equal(surface[corners[1]], original_lower_te)
    ):
        raise GeometryPreparationError(
            "TRAILING_EDGE_FIDELITY_FAILED", "blunt trailing-edge endpoints moved"
        )
    if not has_gap and not np.array_equal(surface[0], surface[-1]):
        raise GeometryPreparationError(
            "TRAILING_EDGE_FIDELITY_FAILED", "sharp trailing edge is not exactly closed"
        )
    diagnostics.update(
        {
            "surface_point_mode": options.surface_point_mode,
            "spacing_method": options.spacing_method,
            "trailing_edge_kind": "BLUNT" if has_gap else "SHARP",
            "trailing_edge_gap_chords": float(
                np.linalg.norm(original_upper_te - original_lower_te)
            ),
            "output_point_count": len(surface),
            "failure_classification": None,
        }
    )
    return surface, corners, diagnostics


def prepare_surface(
    airfoil: AirfoilProfile, options: MeshSettings
) -> tuple[np.ndarray, tuple[int, int]]:
    """Backward-compatible wall preparation API."""
    surface, corners, _ = prepare_surface_with_diagnostics(airfoil, options)
    return surface, corners


def surface_normals(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    tangent = periodic_first_derivative(points)
    magnitude = np.linalg.norm(tangent, axis=1)
    if np.any(magnitude <= 1.0e-14):
        raise ValueError("undefined surface normal caused by coincident points")
    return np.column_stack((tangent[:, 1], -tangent[:, 0])) / magnitude[:, None]


def wall_corner_indices(
    surface: np.ndarray, threshold_degrees: float = 45.0
) -> tuple[int, ...]:
    """Return wall nodes whose tangent turns more sharply than the threshold.

    The periodic surface normal is undefined at a tangent discontinuity such as
    a trailing edge, so these nodes must be constrained to their edge bisector
    during the march instead of using a centered normal.
    """
    unique = np.asarray(surface, dtype=float)[:-1]
    if len(unique) < 3:
        return ()
    incoming = unique - np.roll(unique, 1, axis=0)
    outgoing = np.roll(unique, -1, axis=0) - unique
    incoming_length = np.linalg.norm(incoming, axis=1)
    outgoing_length = np.linalg.norm(outgoing, axis=1)
    valid = (incoming_length > 1.0e-14) & (outgoing_length > 1.0e-14)
    cosine = np.ones(len(unique))
    np.divide(
        np.sum(incoming * outgoing, axis=1),
        incoming_length * outgoing_length,
        out=cosine,
        where=valid,
    )
    turn = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return tuple(
        int(index) for index in np.flatnonzero(valid & (turn > threshold_degrees))
    )


def create_farfield(inner: np.ndarray, options: MeshSettings) -> np.ndarray:
    """Project a marched boundary onto a circle without index-based rotation."""
    center = np.array((0.5, 0.0))
    relative = inner - center
    angles = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    bias = options.farfield_angular_bias
    # Unit bias leaves the marched rays alone.
    angles = angles - (bias - 1.0) / (2.0 * (bias + 1.0)) * np.sin(2.0 * angles)
    radius = options.farfield_radius_chords
    outer = np.column_stack((0.5 + radius * np.cos(angles), radius * np.sin(angles)))
    outer[-1] = outer[0]
    return outer


def fit_farfield(grid: np.ndarray, options: MeshSettings) -> np.ndarray:
    """Extend the outer region smoothly while retaining the marched ray angles."""
    center = np.array((0.5, 0.0))
    relative = grid - center
    radii = np.linalg.norm(relative, axis=-1)
    angles = np.arctan2(relative[..., 1], relative[..., 0])
    distances = np.vstack(
        (
            np.zeros(grid.shape[1]),
            np.cumsum(np.linalg.norm(np.diff(grid, axis=0), axis=-1), axis=0),
        )
    )
    fraction = np.divide(
        distances,
        distances[-1],
        out=np.zeros_like(distances),
        where=distances[-1] > 0.0,
    )
    # Leave the inner 10% alone.
    fraction = np.clip((fraction - 0.1) / 0.9, 0.0, 1.0)
    blend = fraction**2 * (3.0 - 2.0 * fraction)
    outer = create_farfield(grid[-1], options)
    outer_angle = np.arctan2(outer[:, 1], outer[:, 0] - 0.5)
    rotation = np.arctan2(
        np.sin(outer_angle - angles[-1]), np.cos(outer_angle - angles[-1])
    )
    angles = angles + blend * rotation
    radii = radii + blend * (options.farfield_radius_chords - radii[-1])
    result = center + radii[..., None] * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1
    )
    result[blend == 0.0] = grid[blend == 0.0]
    result[-1] = outer
    result[:, -1] = result[:, 0]
    return result
