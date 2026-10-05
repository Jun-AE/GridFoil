"""Budgeted full-resolution optimization around a user-requested mesh."""

from __future__ import annotations

import inspect
import math
import time
import warnings
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from .generator import GridGenerator
from .generator import generate_mesh as source_generate_mesh
from .geometry import AirfoilProfile
from .models import AirfoilMesh, MeshSettings

SETTING_NAMES = tuple(item.name for item in fields(MeshSettings))
OUTPUT_NAMES = (
    "output_directory",
    "project_name",
    "write_su2_output",
    "write_cgns_output",
    "write_vtu_output",
    "write_msh_output",
)
CALL_DEFAULTS = {
    name: inspect.signature(source_generate_mesh).parameters[name].default
    for name in (*OUTPUT_NAMES, *SETTING_NAMES)
}
LOGARITHMIC = {
    "wall_y_plus_target",
    "flow_reynolds_number",
    "wall_reference_length_chords",
}
SPACING = {"leading_edge_cell_length", "trailing_edge_cell_length"}
TOPOLOGY_COUNTS = {"circumferential_node_count", "wall_normal_node_count"}
NON_OPTIMIZABLE = {
    "spacing_method",
    "geometry_conditioning_mode",
    "geometry_deviation_tolerance_chords",
    "geometry_max_refinement_depth",
}
INVALID_COUNTS = (
    "inverted_cell_count",
    "degenerate_cell_count",
    "invalid_corner_cell_count",
    "nonfinite_cell_count",
    "overlapping_cell_count",
    "overlap_unchecked_cell_count",
    "poor_quality_cell_count",
    "undefined_dual_node_count",
)

# Merit assigned to an invalid or fallback candidate so the surrogate avoids it.
INVALID_PENALTY = 10.0
# Fixed sensitivity-screen size so the design is nested as the budget grows.
SCREEN_POINTS = 6
# Screen step in encoded units; large enough to expose the useful directions.
SCREEN_TRUST = 0.18
# Number of most-influential controls given axis-aligned probes.
PROBE_LIMIT = 3
# Largest number of controls handed to the model search.
FOCUS_LIMIT = 6
# Model-search trust box around the requested mesh in encoded units.
TRUST_RADIUS = 0.35
# Deterministic response-surface seed.
OPTIMIZER_SEED = 0
# Previously learned control values, used only as an extra warm-start candidate.
PRIOR_CONTROLS = {
    "leading_edge_cell_length": 0.0020841394958560056,
    "trailing_edge_cell_length": 0.0002490216626885873,
    "trailing_edge_face_cell_count": 16,
    "farfield_angular_bias": 1.1390696813749193,
    "hyperbolic_normal_coupling": 5.2028,
    "hyperbolic_implicit_smoothing": 28.314320000000002,
    "hyperbolic_explicit_smoothing": 0.4333199999999999,
    "farfield_uniformity_weight": 0.20264,
    "hyperbolic_area_smoothing_passes": 58,
    "hyperbolic_max_pseudo_aspect_ratio": 7.1,
}


@dataclass(frozen=True)
class MeshOptimizationTrial:
    """One evaluated candidate in an optimized mesh-generation request."""

    iteration: int
    kind: str
    settings: dict[str, object]
    valid: bool
    pareto_safe: bool
    material: bool
    merit: float
    metrics: dict[str, float | int | None]
    seconds: float
    error: str | None = None


@dataclass(frozen=True)
class OptimizedMeshResult:
    """Selected mesh, provenance, quality comparison, and optimization history."""

    mesh: AirfoilMesh
    files: dict[str, Path]
    settings: MeshSettings
    baseline_mesh: AirfoilMesh
    baseline_settings: MeshSettings
    baseline_metrics: dict[str, float | int | None]
    selected_metrics: dict[str, float | int | None]
    baseline_valid: bool
    selected_valid: bool
    explicit_controls: tuple[str, ...]
    free_controls: tuple[str, ...]
    inactive_controls: tuple[str, ...]
    improved: bool
    trials: tuple[MeshOptimizationTrial, ...]
    initial_mesh_seconds: float
    optimization_seconds: float


def _resolve_call(kwargs: dict[str, object]):
    """Resolve source defaults while retaining the exact explicitly supplied keys."""
    unknown = set(kwargs) - set(CALL_DEFAULTS)
    if unknown:
        raise TypeError(f"unexpected mesh keyword(s): {', '.join(sorted(unknown))}")
    explicit = tuple(name for name in SETTING_NAMES if name in kwargs)
    values = {name: kwargs.get(name, CALL_DEFAULTS[name]) for name in SETTING_NAMES}
    output = {name: kwargs.get(name, CALL_DEFAULTS[name]) for name in OUTPUT_NAMES}
    return MeshSettings(**values), explicit, output


def _inactive(
    profile: AirfoilProfile, settings: MeshSettings, explicit: set[str]
) -> set[str]:
    inactive = set(NON_OPTIMIZABLE)
    if not profile.has_trailing_edge_gap:
        inactive.add("trailing_edge_face_cell_count")
    if settings.surface_point_mode == "PRESERVE" and "surface_point_mode" in explicit:
        inactive.update(
            (
                "circumferential_node_count",
                "leading_edge_cell_length",
                "trailing_edge_cell_length",
            )
        )
    if (
        settings.surface_point_mode == "REDISTRIBUTE"
        and "surface_point_mode" not in explicit
        and explicit
        & {
            "circumferential_node_count",
            "leading_edge_cell_length",
            "trailing_edge_cell_length",
        }
    ):
        # PRESERVE would ignore the requested surface count or spacing.
        inactive.add("surface_point_mode")
    return inactive


def _bounds(
    settings: MeshSettings, profile: AirfoilProfile | None = None
) -> dict[str, tuple[float, float, str]]:
    """Derive per-instance search bounds from the requested settings.

    Every range is anchored to the resolved caller settings, so the search is
    tailored to the requested airfoil, resolution, and flow instead of a fixed
    cross-airfoil box.
    """
    values = asdict(settings)
    count = settings.circumferential_node_count
    panel = 2.0 / max(count, 1)
    blunt = profile is not None and profile.has_trailing_edge_gap
    bounds: dict[str, tuple[float, float, str]] = {}
    for name in SETTING_NAMES:
        value = values[name]
        if name in NON_OPTIMIZABLE:
            bounds[name] = (0.0, 1.0, "fixed")
        elif name == "surface_point_mode":
            bounds[name] = (0.0, 1.0, "mode")
        elif name == "circumferential_node_count":
            bounds[name] = (
                max(20, round(value * 0.8)),
                max(20, round(value * 1.2)),
                "int",
            )
        elif name == "wall_normal_node_count":
            bounds[name] = (
                max(3, round(value * 0.8)),
                max(3, round(value * 1.2)),
                "int",
            )
        elif name == "farfield_radius_chords":
            bounds[name] = (max(20.0, value * 0.75), max(20.0, value * 1.25), "linear")
        elif name in SPACING:
            bounds[name] = (0.02 * panel, 0.8 * panel, "spacing")
        elif name == "trailing_edge_face_cell_count":
            low = max(2, round(0.03 * count))
            high = max(low + 2, round(0.2 * count))
            if blunt:
                low = max(low, 16)
                high = max(high, low)
            bounds[name] = (low, high, "int")
        elif name == "farfield_angular_bias":
            bounds[name] = (0.5, 1.5, "log")
        elif name == "hyperbolic_normal_coupling":
            bounds[name] = (0.5 * value, 2.0 * value, "linear")
        elif name == "hyperbolic_implicit_smoothing":
            bounds[name] = (0.5 * value, 1.8 * value, "linear")
        elif name == "hyperbolic_explicit_smoothing":
            bounds[name] = (0.0, max(2.0 * value, 1.0e-6), "linear")
        elif name == "farfield_uniformity_weight":
            bounds[name] = (max(0.0, 0.1 * value), min(1.0, 2.5 * value), "linear")
        elif name == "hyperbolic_area_smoothing_passes":
            bounds[name] = (
                max(0, round(0.4 * value)),
                max(1, round(2.0 * value)),
                "int",
            )
        elif name == "hyperbolic_max_pseudo_aspect_ratio":
            bounds[name] = (5.0, 10.0, "linear")
        elif name in LOGARITHMIC:
            bounds[name] = (value * 0.5, value * 2.0, "log")
        else:  # pragma: no cover - forces updates when MeshSettings grows
            raise AssertionError(f"missing bounds for {name}")
    return bounds


def _encode(settings: MeshSettings, bounds) -> np.ndarray:
    encoded = []
    for name in SETTING_NAMES:
        lo, hi, scale = bounds[name]
        value = getattr(settings, name)
        if scale == "fixed":
            gene = 0.0
        elif scale == "mode":
            gene = float(value == "PRESERVE")
        elif scale == "spacing" and value is None:
            gene = 0.0
        elif scale in {"log", "spacing"}:
            raw = math.log(float(value) / lo) / math.log(hi / lo)
            gene = 0.15 + 0.85 * raw if scale == "spacing" else raw
        else:
            gene = (float(value) - lo) / max(hi - lo, 1.0e-12)
        encoded.append(float(np.clip(gene, 0.0, 1.0)))
    return np.asarray(encoded)


def _decode(
    genes: np.ndarray, baseline: MeshSettings, bounds, explicit: set[str]
) -> MeshSettings:
    result = {}
    for gene, name in zip(genes, SETTING_NAMES, strict=True):
        if name in explicit:
            result[name] = getattr(baseline, name)
            continue
        lo, hi, scale = bounds[name]
        gene = float(np.clip(gene, 0.0, 1.0))
        if scale == "fixed":
            value = getattr(baseline, name)
        elif scale == "mode":
            value = "PRESERVE" if gene >= 0.5 else "REDISTRIBUTE"
        elif scale == "spacing" and gene < 0.15:
            value = None
        elif scale in {"log", "spacing"}:
            fraction = (gene - 0.15) / 0.85 if scale == "spacing" else gene
            value = float(math.exp(math.log(lo) + fraction * math.log(hi / lo)))
        else:
            value = float(lo + gene * (hi - lo))
        result[name] = int(round(value)) if scale == "int" else value

    # Keep the TE count compatible with the circumferential count.
    required = result["trailing_edge_face_cell_count"] + 6
    if result["circumferential_node_count"] < required:
        if "circumferential_node_count" not in explicit:
            result["circumferential_node_count"] = required
        elif "trailing_edge_face_cell_count" not in explicit:
            result["trailing_edge_face_cell_count"] = (
                result["circumferential_node_count"] - 6
            )
    return MeshSettings(**result).validated()


def _objective_components(report, baseline) -> dict[str, float]:
    """Return every normalized equal-weight constituent and the scalar objective."""
    skew = float(report["maximum_skew_degrees"]) / max(
        float(baseline["maximum_skew_degrees"]), 1.0e-12
    )
    dual_orthogonality = (
        90.0 - float(report["minimum_dual_orthogonality_degrees"])
    ) / max(90.0 - float(baseline["minimum_dual_orthogonality_degrees"]), 1.0e-12)
    cell_orthogonality = (1.0 - float(report["minimum_cell_orthogonal_quality"])) / max(
        1.0 - float(baseline["minimum_cell_orthogonal_quality"]), 1.0e-12
    )
    orth = 0.5 * (dual_orthogonality + cell_orthogonality)
    area = math.log(max(float(report["maximum_neighbor_area_ratio"]), 1.0)) / max(
        math.log(max(float(baseline["maximum_neighbor_area_ratio"]), 1.0)), 1.0e-12
    )
    weight = 1.0 / 3.0
    return {
        "objective_skew_loss": skew,
        "objective_orthogonality_loss": orth,
        "objective_area_smoothness_loss": area,
        "objective_skew_contribution": weight * skew,
        "objective_orthogonality_contribution": weight * orth,
        "objective_area_smoothness_contribution": weight * area,
        "objective_total": weight * (skew + orth + area),
    }


def _record_objective(report, baseline) -> float:
    components = _objective_components(report, baseline)
    report.update(components)
    return components["objective_total"]


def _material(
    report, baseline, skew_degrees, orthogonality_degrees, area_fraction
) -> bool:
    return (
        float(baseline["maximum_skew_degrees"]) - float(report["maximum_skew_degrees"])
        >= skew_degrees
        or float(report["minimum_dual_orthogonality_degrees"])
        - float(baseline["minimum_dual_orthogonality_degrees"])
        >= orthogonality_degrees
        or (
            float(baseline["maximum_neighbor_area_ratio"])
            - float(report["maximum_neighbor_area_ratio"])
        )
        / max(float(baseline["maximum_neighbor_area_ratio"]), 1.0e-12)
        >= area_fraction
    )


def _quality_metrics(mesh: AirfoilMesh) -> dict[str, float | int | None]:
    report = mesh.quality.summary()
    ratios = mesh.cells.neighbor_area_ratios
    finite = ratios[np.isfinite(ratios)]
    report["maximum_neighbor_area_ratio"] = (
        float(np.max(finite)) if finite.size else math.inf
    )
    return report


def _valid(report, caught: list[warnings.WarningMessage]) -> bool:
    if any((report.get(name) or 0) != 0 for name in INVALID_COUNTS):
        return False
    return not any("fallback" in str(item.message).lower() for item in caught)


def _pareto_safe(report, baseline) -> bool:
    tolerance = 1.0e-9
    return (
        float(report["maximum_skew_degrees"])
        <= float(baseline["maximum_skew_degrees"]) + tolerance
        and float(report["minimum_dual_orthogonality_degrees"])
        >= float(baseline["minimum_dual_orthogonality_degrees"]) - tolerance
        and float(report["maximum_neighbor_area_ratio"])
        <= float(baseline["maximum_neighbor_area_ratio"]) + tolerance
        and float(report["minimum_cell_orthogonal_quality"])
        >= float(baseline["minimum_cell_orthogonal_quality"]) - tolerance
        and (
            float(report["maximum_skew_degrees"])
            < float(baseline["maximum_skew_degrees"]) - tolerance
            or float(report["minimum_dual_orthogonality_degrees"])
            > float(baseline["minimum_dual_orthogonality_degrees"]) + tolerance
            or float(report["maximum_neighbor_area_ratio"])
            < float(baseline["maximum_neighbor_area_ratio"]) - tolerance
        )
    )


def _generate(profile, settings):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh = GridGenerator(profile, settings).run()
    report = _quality_metrics(mesh)
    return mesh, report, caught, time.perf_counter() - start


def _hadamard(size: int) -> np.ndarray:
    matrix = np.ones((1, 1))
    while len(matrix) < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def _optimize_existing_mesh(
    profile: AirfoilProfile,
    baseline_mesh: AirfoilMesh,
    settings: MeshSettings,
    explicit_controls: tuple[str, ...],
    *,
    budget: int = 16,
    minimum_skew_improvement_degrees: float = 0.05,
    minimum_orthogonality_improvement_degrees: float = 0.05,
    minimum_area_ratio_improvement_fraction: float = 0.005,
):
    """Optimize omitted controls around the exact full-resolution baseline mesh."""
    if budget < 0:
        raise ValueError("budget cannot be negative")
    explicit = set(explicit_controls)
    inactive = _inactive(profile, settings, explicit)
    free = tuple(name for name in SETTING_NAMES if name not in explicit | inactive)
    baseline_report = _quality_metrics(baseline_mesh)
    _record_objective(baseline_report, baseline_report)
    if not free or budget == 0:
        return (
            baseline_mesh,
            settings,
            baseline_report,
            (),
            free,
            tuple(sorted(inactive)),
            0.0,
        )

    bounds = _bounds(settings, profile)
    x0 = _encode(settings, bounds)
    free_indices = np.asarray([SETTING_NAMES.index(name) for name in free], dtype=int)
    x0_free = x0[free_indices].copy()
    n_free = len(free)

    trials: list[MeshOptimizationTrial] = []
    cache: dict[tuple[object, ...], float] = {}
    feasible_candidates: list[tuple[float, AirfoilMesh, MeshSettings, dict]] = []
    start = time.perf_counter()

    def evaluate(genes, kind) -> float:
        if len(trials) >= budget:
            return INVALID_PENALTY
        key: tuple[object, ...] | None = None
        try:
            candidate = _decode(genes, settings, bounds, explicit)
            if any(
                getattr(candidate, name) != getattr(settings, name) for name in explicit
            ):
                raise AssertionError("candidate changed an explicit control")
            key = tuple(asdict(candidate).values())
            if key in cache:
                return cache[key]
            mesh, report, caught, seconds = _generate(profile, candidate)
            valid = _valid(report, caught)
            if TOPOLOGY_COUNTS <= explicit and not np.array_equal(
                mesh.cells.node_ids, baseline_mesh.cells.node_ids
            ):
                valid = False
            safe = valid and _pareto_safe(report, baseline_report)
            material = safe and _material(
                report,
                baseline_report,
                minimum_skew_improvement_degrees,
                minimum_orthogonality_improvement_degrees,
                minimum_area_ratio_improvement_fraction,
            )
            if valid:
                merit = _record_objective(report, baseline_report)
            else:
                report["objective_total"] = INVALID_PENALTY
                merit = INVALID_PENALTY
            trials.append(
                MeshOptimizationTrial(
                    len(trials) + 1,
                    kind,
                    asdict(candidate),
                    valid,
                    safe,
                    material,
                    merit,
                    report,
                    seconds,
                )
            )
            cache[key] = merit
            if material:
                feasible_candidates.append((merit, mesh, candidate, report))
            return merit
        except (
            ValueError,
            TypeError,
            FloatingPointError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as error:
            trials.append(
                MeshOptimizationTrial(
                    len(trials) + 1,
                    kind,
                    {},
                    False,
                    False,
                    False,
                    INVALID_PENALTY,
                    {},
                    0.0,
                    f"{type(error).__name__}: {error}",
                )
            )
            if key is not None:
                cache[key] = INVALID_PENALTY
            return INVALID_PENALTY

    pool_genes: list[np.ndarray] = []
    pool_merit: list[float] = []

    # Warm-start from the previously learned control values when they apply.
    prior = asdict(settings)
    prior.update(
        {name: value for name, value in PRIOR_CONTROLS.items() if name in free}
    )
    prior_genes = _encode(MeshSettings(**prior).validated(profile), bounds)
    if not np.allclose(prior_genes[free_indices], x0_free, rtol=0.0, atol=1.0e-12):
        pool_genes.append(prior_genes[free_indices])
        pool_merit.append(evaluate(prior_genes, "prior"))
    prior_count = len(pool_genes)

    # Sensitivity screen: a fixed orthogonal design around the requested mesh.
    screen_count = max(1, min(n_free, budget, SCREEN_POINTS))
    order = 1
    while order < max(screen_count + 1, n_free + 1):
        order *= 2
    design = _hadamard(order)[1 : screen_count + 1, 1 : n_free + 1]
    for row in design:
        free_genes = np.clip(x0_free + SCREEN_TRUST * row, 0.0, 1.0)
        for position, name in enumerate(free):
            if name == "surface_point_mode":
                free_genes[position] = 1.0 if row[position] > 0 else 0.0
            elif name in SPACING and x0_free[position] == 0.0 and row[position] > 0:
                free_genes[position] = 0.35
        genes = x0.copy()
        genes[free_indices] = free_genes
        pool_genes.append(free_genes)
        pool_merit.append(evaluate(genes, "screen"))
    screen_genes = pool_genes[prior_count:]
    screen_merit = pool_merit[prior_count:]

    # Rank controls by first-order impact on the penalty-augmented merit.
    usable = np.asarray(
        [np.isfinite(merit) and merit < INVALID_PENALTY for merit in screen_merit],
        dtype=bool,
    )
    if np.count_nonzero(usable) >= 2:
        delta = np.asarray(screen_genes)[usable] - x0_free
        response = np.asarray(screen_merit)[usable]
        model = np.column_stack((np.ones(len(delta)), delta))
        coefficients = np.linalg.lstsq(model, response, rcond=None)[0][1:]
        sensitivity = np.abs(coefficients)
    else:
        sensitivity = np.ones(n_free)
    ranking = np.argsort(-sensitivity, kind="stable")

    # Axis-aligned probes keep single-control improvements available to the model.
    for position in ranking[: min(n_free, PROBE_LIMIT)]:
        for sign in (-1.0, 1.0):
            if len(trials) >= budget:
                break
            free_genes = x0_free.copy()
            free_genes[position] = np.clip(
                free_genes[position] + sign * SCREEN_TRUST, 0.0, 1.0
            )
            genes = x0.copy()
            genes[free_indices] = free_genes
            pool_genes.append(free_genes)
            pool_merit.append(evaluate(genes, "probe"))

    # Model search over a fixed control set, so more budget only adds samples.
    remaining = budget - len(trials)
    if remaining >= 1:
        focus = ranking[: min(n_free, FOCUS_LIMIT)]
        low = np.clip(x0_free[focus] - TRUST_RADIUS, 0.0, 1.0)
        high = np.clip(x0_free[focus] + TRUST_RADIUS, 0.0, 1.0)
        dimensions = [
            Real(float(lo), float(hi)) for lo, hi in zip(low, high, strict=True)
        ]
        warm_x: list[list[float]] = []
        warm_y: list[float] = []
        seen: set[tuple[float, ...]] = set()
        for free_genes, merit in zip(pool_genes, pool_merit, strict=True):
            if not (np.isfinite(merit) and merit < INVALID_PENALTY):
                continue
            point = tuple(
                float(value) for value in np.clip(free_genes[focus], low, high)
            )
            if point in seen:
                continue
            seen.add(point)
            warm_x.append(list(point))
            warm_y.append(float(merit))

        def objective(point) -> float:
            genes = x0.copy()
            genes[free_indices[focus]] = np.clip(
                np.asarray(point, dtype=float), 0.0, 1.0
            )
            return evaluate(genes, "bo")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp_minimize(
                objective,
                dimensions,
                n_calls=max(1, remaining),
                x0=warm_x or None,
                y0=warm_y or None,
                n_initial_points=0 if warm_x else min(3, max(1, remaining)),
                random_state=OPTIMIZER_SEED,
                acq_func="EI",
            )

    seconds = time.perf_counter() - start
    if feasible_candidates:
        _, mesh, selected_settings, report = min(
            feasible_candidates,
            key=lambda item: (
                item[0],
                -float(item[3]["minimum_cell_orthogonal_quality"]),
            ),
        )
        return (
            mesh,
            selected_settings,
            report,
            tuple(trials),
            free,
            tuple(sorted(inactive)),
            seconds,
        )
    return (
        baseline_mesh,
        settings,
        baseline_report,
        tuple(trials),
        free,
        tuple(sorted(inactive)),
        seconds,
    )


def generate_optimized_mesh(
    airfoil: AirfoilProfile | str | Path,
    *,
    optimize: bool = True,
    optimization_budget: int = 16,
    minimum_skew_improvement_degrees: float = 0.05,
    minimum_orthogonality_improvement_degrees: float = 0.05,
    minimum_area_ratio_improvement_fraction: float = 0.005,
    **kwargs,
) -> OptimizedMeshResult:
    """Generate a mesh and optimize only ``MeshSettings`` controls the caller omitted.

    Explicit mesh keywords, including values equal to defaults and explicit
    ``None``, remain immutable. The initial full-resolution mesh is always the
    fallback. A candidate is selected only when it is valid, improves at least
    one primary quality metric by the configured material threshold, and does
    not worsen maximum skew, minimum dual orthogonality, or maximum neighboring
    cell-area ratio.

    Physical inputs such as Reynolds number and target y-plus are optimization
    variables when omitted. Supply them explicitly whenever they are physical
    requirements rather than design variables.
    """
    if isinstance(optimization_budget, bool) or not isinstance(
        optimization_budget, int
    ):
        raise TypeError("optimization_budget must be an integer")
    if optimization_budget < 0:
        raise ValueError("optimization_budget cannot be negative")
    thresholds = (
        minimum_skew_improvement_degrees,
        minimum_orthogonality_improvement_degrees,
        minimum_area_ratio_improvement_fraction,
    )
    if not np.isfinite(thresholds).all() or min(thresholds) < 0.0:
        raise ValueError(
            "optimization improvement thresholds must be finite and nonnegative"
        )
    settings, explicit, output = _resolve_call(kwargs)
    profile = (
        airfoil if isinstance(airfoil, AirfoilProfile) else AirfoilProfile.read(airfoil)
    )
    settings = settings.validated(profile)
    baseline_mesh, baseline_report, caught, initial_seconds = _generate(
        profile, settings
    )
    _record_objective(baseline_report, baseline_report)
    baseline_valid = _valid(baseline_report, caught)
    primary = (
        baseline_report.get("maximum_skew_degrees"),
        baseline_report.get("minimum_dual_orthogonality_degrees"),
        baseline_report.get("maximum_neighbor_area_ratio"),
    )
    comparable = all(value is not None and np.isfinite(value) for value in primary)

    if optimize and comparable:
        (
            mesh,
            selected,
            selected_report,
            trials,
            free,
            inactive,
            optimization_seconds,
        ) = _optimize_existing_mesh(
            profile,
            baseline_mesh,
            settings,
            explicit,
            budget=optimization_budget,
            minimum_skew_improvement_degrees=minimum_skew_improvement_degrees,
            minimum_orthogonality_improvement_degrees=(
                minimum_orthogonality_improvement_degrees
            ),
            minimum_area_ratio_improvement_fraction=(
                minimum_area_ratio_improvement_fraction
            ),
        )
    else:
        inactive = tuple(sorted(_inactive(profile, settings, set(explicit))))
        free = tuple(
            name for name in SETTING_NAMES if name not in set(explicit) | set(inactive)
        )
        mesh, selected, selected_report, trials, optimization_seconds = (
            baseline_mesh,
            settings,
            baseline_report,
            (),
            0.0,
        )
    selected_valid = True if mesh is not baseline_mesh else baseline_valid

    files: dict[str, Path] = {}
    if output["output_directory"] is not None:
        generator = GridGenerator(profile, selected)
        generator.mesh = mesh
        files = generator.write(
            output["output_directory"],
            output["project_name"],
            su2=output["write_su2_output"],
            cgns=output["write_cgns_output"],
            vtu=output["write_vtu_output"],
            msh=output["write_msh_output"],
        )
    return OptimizedMeshResult(
        mesh,
        files,
        selected,
        baseline_mesh,
        settings,
        baseline_report,
        selected_report,
        baseline_valid,
        selected_valid,
        explicit,
        free,
        inactive,
        mesh is not baseline_mesh,
        trials,
        initial_seconds,
        optimization_seconds,
    )
