"""Shared numerical operations, spacing, atomic writes, and report conversion."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import blake2b
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

import numpy as np


def periodic_first_derivative(values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return a centered derivative for an array with a repeated seam node."""
    values = np.asarray(values, dtype=float)
    unique = np.take(values, np.arange(values.shape[axis] - 1), axis=axis)
    derivative = 0.5 * (np.roll(unique, -1, axis=axis) - np.roll(unique, 1, axis=axis))
    return np.concatenate((derivative, np.take(derivative, [0], axis=axis)), axis=axis)


def _growth(forward: np.ndarray, backward: np.ndarray) -> np.ndarray:
    fp = np.linalg.norm(forward, axis=-1)
    bp = np.linalg.norm(backward, axis=-1)
    return np.divide(fp - bp, bp, out=np.zeros_like(fp), where=bp > 1.0e-30)


def signed_cell_areas(nodes: np.ndarray) -> np.ndarray:
    """Return the signed area of every quadrilateral cell."""
    lower_left = nodes[:-1, :-1]
    lower_right = nodes[:-1, 1:]
    upper_right = nodes[1:, 1:]
    upper_left = nodes[1:, :-1]
    first_diagonal = upper_right - lower_left
    second_diagonal = upper_left - lower_right
    return 0.5 * (
        first_diagonal[..., 0] * second_diagonal[..., 1]
        - first_diagonal[..., 1] * second_diagonal[..., 0]
    )


def dominant_orientation(nodes: np.ndarray) -> int:
    """Return the dominant cell orientation, or zero for a degenerate grid."""
    areas = signed_cell_areas(nodes)
    areas = areas[np.isfinite(areas)]
    floor = float(np.max(np.abs(areas), initial=0.0)) * 1.0e-13
    significant = areas[np.abs(areas) > floor]
    if significant.size == 0:
        return 0
    positive = np.count_nonzero(significant > 0.0)
    return 1 if positive >= significant.size - positive else -1


def node_signature(nodes: np.ndarray) -> bytes:
    """Identify coordinates and topology without keeping another mesh copy."""
    nodes = np.ascontiguousarray(nodes, dtype=float)
    digest = blake2b(digest_size=32)
    digest.update(np.asarray(nodes.shape, dtype=np.int64).tobytes())
    digest.update(memoryview(nodes).cast("B"))
    return digest.digest()


def has_consistent_orientation(nodes: np.ndarray, expected: int | None = None) -> bool:
    """Report whether every finite cell has one nonzero orientation."""
    corners = scaled_corner_jacobians(nodes)
    orientation = dominant_orientation(nodes) if expected is None else expected
    return bool(
        corners.size > 0
        and orientation != 0
        and np.all(np.isfinite(corners))
        and np.all(orientation * corners > 32.0 * np.finfo(float).eps)
    )


def scaled_corner_jacobians(nodes: np.ndarray) -> np.ndarray:
    """Return signed, normalized corner determinants for each quadrilateral."""
    vertices = np.stack(
        (nodes[:-1, :-1], nodes[:-1, 1:], nodes[1:, 1:], nodes[1:, :-1]), axis=-2
    )
    edges = np.roll(vertices, -1, axis=-2) - vertices
    following = np.roll(edges, -1, axis=-2)
    cross = edges[..., 0] * following[..., 1] - edges[..., 1] * following[..., 0]
    scale = np.linalg.norm(edges, axis=-1) * np.linalg.norm(following, axis=-1)
    # Edge k and edge k+1 meet at vertex k+1; align results with vertex IDs.
    return np.roll(
        np.divide(cross, scale, out=np.zeros_like(cross), where=scale > 0.0), 1, axis=-1
    )


def geometric_length(first: float, ratio: float, cells: int) -> float:
    if cells < 1:
        return 0.0
    if abs(ratio - 1.0) < 1.0e-12:
        return first * cells
    log_ratio = np.log(ratio)
    return float(first * np.expm1(cells * log_ratio) / np.expm1(log_ratio))


def growth_ratio(length: float, first: float, cells: int) -> float:
    """Find a constant cell-growth ratio for the specified total length."""
    return float(growth_ratios(np.asarray([length]), first, cells)[0])


def growth_ratios(lengths: np.ndarray, first: float, cells: int) -> np.ndarray:
    """Solve many geometric growth ratios with safeguarded Newton steps."""
    targets = np.asarray(lengths, dtype=float)
    if targets.ndim != 1:
        raise ValueError("lengths must be a one-dimensional array")
    if (
        not np.isfinite(first)
        or not np.all(np.isfinite(targets))
        or first <= 0.0
        or cells < 1
        or np.any(targets <= 0.0)
    ):
        raise ValueError("positive length/spacing and at least one cell are required")
    if cells == 1:
        if not np.allclose(targets, first, rtol=1.0e-12, atol=0.0):
            raise ValueError("one cell cannot satisfy the specified length and spacing")
        return np.ones_like(targets)

    uniform = first * cells
    uniform_mask = np.abs(targets - uniform) <= 1.0e-13 * np.maximum(targets, uniform)
    if np.any((targets <= first) & ~uniform_mask):
        raise ValueError("the total length must exceed the first spacing")

    lower = np.empty_like(targets)
    upper = np.empty_like(targets)
    expanding = targets > uniform
    lower[expanding] = 0.0
    upper[expanding] = np.log(targets[expanding] / first) / (cells - 1)
    shrinking = ~expanding
    tail_fraction = (targets[shrinking] - first) / (first * (cells - 1))
    lower[shrinking] = np.log(np.clip(0.5 * tail_fraction, np.finfo(float).tiny, 0.5))
    upper[shrinking] = 0.0

    relative_error = targets / uniform - 1.0
    log_ratio = np.clip(2.0 * relative_error / (cells - 1), lower, upper)
    powers = np.arange(cells, dtype=float)
    tolerance = 5.0e-14 * np.maximum(targets, uniform)
    active = ~uniform_mask
    for _ in range(40):
        exponentials = np.exp(log_ratio[:, None] * powers)
        totals = first * np.sum(exponentials, axis=1)
        errors = totals - targets
        active &= np.abs(errors) > tolerance
        if not np.any(active):
            break
        lower = np.where(active & (errors < 0.0), log_ratio, lower)
        upper = np.where(active & (errors >= 0.0), log_ratio, upper)
        derivatives = first * np.sum(powers * exponentials, axis=1)
        candidate = log_ratio - np.divide(
            errors,
            derivatives,
            out=np.zeros_like(errors),
            where=derivatives > 0.0,
        )
        midpoint = 0.5 * (lower + upper)
        unsafe = ~np.isfinite(candidate) | (candidate <= lower) | (candidate >= upper)
        log_ratio = np.where(active, np.where(unsafe, midpoint, candidate), log_ratio)
    if np.any(active):
        final_totals = first * np.sum(
            np.exp(log_ratio[:, None] * powers),
            axis=1,
        )
        active &= np.abs(final_totals - targets) > tolerance
    if np.any(active):
        raise RuntimeError("the batched growth-ratio solve did not converge")
    log_ratio[uniform_mask] = 0.0
    return np.exp(log_ratio)


def geometric_distances(length: float, first: float, cells: int) -> np.ndarray:
    ratio = growth_ratio(length, first, cells)
    widths = first * ratio ** np.arange(cells, dtype=float)
    distances = np.concatenate(([0.0], np.cumsum(widths)))
    distances[-1] = length
    return distances


_REPLACE_LOCK = Lock()


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    """Give each writer a unique temporary file beside its destination."""
    destination = Path(path)
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        yield temporary
        with _REPLACE_LOCK:
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def finite_report(value):
    """Replace undefined numeric values with None in a report, not in the mesh."""
    if isinstance(value, np.ndarray | np.generic):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: finite_report(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [finite_report(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
