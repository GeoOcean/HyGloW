"""Compare wave datasets: metrics and plots.

Holdout validation (:func:`validate`, :func:`sweep`) — metamodel vs SnapWave on MDA
test cases (requires :func:`utils.model.metamodel.fit`).

Generic comparison (:func:`compare`, :func:`resolve_comparison_variables`) — any
aligned estimate/reference pair on ``(time, sites)`` bulk variables.

Buoy validation (:func:`load_offshore_hindcast_at_buoys`, :func:`compare_all`) —
bulk / spectral reconstruction, dynamic SnapWave, and closest-goal offshore
hindcast vs user-processed buoy datasets (``10_Validate_Buoy_Observations.ipynb``).
Buoy site coordinates and labels come from the onshore-waves GeoJSON
(:mod:`utils.observation.onshore_sites`).

Site grouping for plots and metrics tables: :mod:`utils.geometry.sites`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.metrics import bias, r2, rmse
from bluemath_tk.core.plotting.old_scatter import density_scatter
from matplotlib.figure import Figure

from utils.config import (
    METAMODEL,
    OFFSHORE_WAVES,
    get_offshore_wave,
    load_goals,
    offshore_catalog_key,
    open_offshore_waves,
    resolve,
    validate_metamodel_config,
)
from utils.geometry.sites import (
    SITE_META,
    attach_site_meta,
    find_group,
    metric_groups,
    per_site_coord,
    resolve_sites,
)
from utils.model import metamodel as mm
from utils.observation import onshore_sites as obs

# Config mirrors metamodel module (case config → metamodel)
METRICS_VARIABLES = METAMODEL["validation"]["metrics_variables"]
MDA_CANDIDATES = mm.MDA_CANDIDATES

LOGGER_NAME = "hyglow.validation"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("logs/validation.log")),
    level="INFO",
    console=True,
    console_level="WARNING",
)

PARAMETERS: dict[str, dict[str, Any]] = {
    "hs": {"label": "Hs", "units": "m", "circular": False},
    "tp": {"label": "Tp", "units": "s", "circular": False},
    "dir": {"label": "Dir", "units": "deg", "circular": True},
    # Directional spread (degrees); linear metrics — not periodic like dir.
    "spr": {"label": "Spr", "units": "deg", "circular": False},
}


def _metrics_variables(variables: Iterable[str] | None) -> list[str]:
    """
    Return the list of variables to score.

    Parameters
    ----------
    variables
        Explicit variable names; default from the case config JSON validation section.

    Returns
    -------
    list[str]
        Variable keys (e.g. ``hs``, ``tp``, ``dir``, ``spr``).
    """
    return list(variables) if variables is not None else list(METRICS_VARIABLES)


def _available_metrics_variables(
    estimate: xr.Dataset,
    reference: xr.Dataset | None = None,
    variables: Iterable[str] | None = None,
) -> list[str]:
    """Return configured metrics variables present in estimate/reference datasets."""
    keys = _metrics_variables(variables)
    keys = [k for k in keys if k in PARAMETERS and k in estimate.data_vars]
    if reference is not None:
        keys = [k for k in keys if k in reference.data_vars]
    return keys


def resolve_comparison_variables(
    *datasets: xr.Dataset,
    variables: Iterable[str] | None = None,
) -> list[str]:
    """Return metric variables present in every dataset (preserving request order).

    Use this before :func:`compare` or :func:`compare_all` when the requested
    list (``hs``, ``tp``, ``dir``, ``spr``) may not exist in all inputs — e.g.
    processed buoy NetCDF without ``spr``, or models missing a variable.

    Parameters
    ----------
    *datasets
        Two or more aligned datasets (reference, estimate, …).
    variables
        Requested variables; default ``metamodel.validation.metrics_variables``.

    Returns
    -------
    list[str]
        Subset of *variables* found in every dataset.

    Raises
    ------
    ValueError
        If no datasets are passed or no requested variable is shared.
    """
    if not datasets:
        raise ValueError("resolve_comparison_variables requires at least one dataset")
    requested = _metrics_variables(variables)
    available = requested
    for ds in datasets:
        available = [v for v in available if v in PARAMETERS and v in ds.data_vars]
    skipped = [v for v in requested if v not in available]
    if skipped:
        logger.info(
            "Comparison skipping variables not in all datasets: %s",
            skipped,
        )
    if not available:
        raise ValueError(
            "No comparison variables shared by all datasets; "
            f"requested={requested}, datasets={[list(d.data_vars) for d in datasets]}"
        )
    return available


def _scatter_grid_shape(n: int) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` for up to four scatter panels in a 2×2 grid."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    ncols = min(4, int(np.ceil(np.sqrt(n))))
    return int(np.ceil(n / ncols)), ncols


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _theta_to_360(values: np.ndarray) -> np.ndarray:
    """Wrap direction angles to ``[0, 360)`` degrees."""
    return np.mod(values, 360.0)


def _circular_difference(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """Signed shortest angular difference ``estimate - reference`` in degrees."""
    return (est - ref + 180.0) % 360.0 - 180.0


def _coincident_values(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return paired finite values from *x* and *y*."""
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _circular_r2(reference: np.ndarray, estimate: np.ndarray) -> float:
    """R² for circular data using shortest angular deviations."""
    ref, est = _coincident_values(_theta_to_360(reference), _theta_to_360(estimate))
    if ref.size == 0:
        return np.nan
    diff = _circular_difference(ref, est)
    ss_res = float(np.sum(diff**2))
    mean_ref = float(
        np.arctan2(np.sin(np.deg2rad(ref)).mean(), np.cos(np.deg2rad(ref)).mean())
    )
    mean_ref = np.rad2deg(mean_ref)
    dev = _circular_difference(ref, np.full(ref.shape, mean_ref))
    ss_tot = float(np.sum(dev**2))
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def compute_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    circular: bool = False,
) -> dict[str, float]:
    """
    Bias, RMSE, R² and sample count for one variable pair.

    Parameters
    ----------
    estimate, reference
        Flat arrays of paired values.
    circular
        If ``True``, treat values as directions in degrees.

    Returns
    -------
    dict[str, float]
        Keys ``n``, ``bias``, ``rmse``, ``r2``.
    """
    if circular:
        ref, est = _coincident_values(_theta_to_360(reference), _theta_to_360(estimate))
        if ref.size == 0:
            return {"n": 0, "bias": np.nan, "rmse": np.nan, "r2": np.nan}
        diff = _circular_difference(ref, est)
        return {
            "n": int(ref.size),
            "bias": float(np.mean(diff)),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "r2": _circular_r2(ref, est),
        }

    x, y = _coincident_values(estimate, reference)
    if x.size == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "r2": np.nan}
    return {
        "n": int(x.size),
        "bias": float(bias(x, y)),
        "rmse": float(rmse(x, y)),
        "r2": float(r2(x, y)),
    }


# ---------------------------------------------------------------------------
# Site groups (re-exported from utils.geometry.sites)
# ---------------------------------------------------------------------------


def metrics_table(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Build a metrics table with one row per validation group and variable.

    Parameters
    ----------
    estimate, reference
        Aligned datasets with ``sites`` and wave variables.
    variables
        Subset of metrics variables; default from config.

    Returns
    -------
    pandas.DataFrame
        Columns include ``group``, ``site_source``, ``variable``, ``bias``,
        ``rmse``, ``r2``, ``n``.
    """
    params = _available_metrics_variables(estimate, reference, variables)
    rows: list[dict[str, Any]] = []

    for group_name, site_source, site_ids in metric_groups(estimate):
        est_g = estimate.sel(sites=site_ids)
        ref_g = reference.sel(sites=site_ids)
        for param in params:
            spec = PARAMETERS[param]
            metrics = compute_metrics(
                est_g[param].values.ravel(),
                ref_g[param].values.ravel(),
                circular=bool(spec["circular"]),
            )
            rows.append(
                {
                    "group": group_name,
                    "site_source": site_source,
                    "site_label": group_name
                    if site_source in ("isobath", "mesh")
                    else per_site_coord(estimate, "site_label").get(
                        site_ids[0], group_name
                    ),
                    "variable": param,
                    "label": spec["label"],
                    **metrics,
                }
            )

    out = pd.DataFrame(rows)
    logger.info("Computed %d validation metric row(s)", len(out))
    return out


def validation_metrics_path(goal: str | int) -> Path:
    """CSV path for :func:`sweep` output on one goal."""
    return mm.goal_output_dir(goal) / "validation_metrics.csv"


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two lon/lat points."""
    radius = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlon / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def closest_goal_for_point(
    lon: float,
    lat: float,
    goals: Mapping[str, dict[str, Any]],
    catalog_key: str,
) -> tuple[str, float]:
    """Nearest goal hindcast node in ``catalog_key`` (e.g. ``whacs``)."""
    best_goal: str | None = None
    best_dist = float("inf")
    for gid, goal in goals.items():
        snap = (goal.get("hindcast") or {}).get(catalog_key)
        if not snap:
            continue
        dist = _haversine_m(lon, lat, snap["longitude"], snap["latitude"])
        if dist < best_dist:
            best_dist = dist
            best_goal = str(gid)
    if best_goal is None:
        raise ValueError(f"No hindcast snaps for catalog {catalog_key!r} in goals")
    return best_goal, best_dist


def closest_goals_table(
    sites: Iterable[str],
    *,
    offshore_names: Iterable[str] | None = None,
    goals: Mapping[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Map each buoy site to the closest goal hindcast node per offshore source."""
    goals = dict(goals or load_goals())
    buoy_coords = obs.get_coordinates()
    names = (
        list(offshore_names)
        if offshore_names is not None
        else [e["name"] for e in OFFSHORE_WAVES]
    )

    rows: list[dict[str, Any]] = []
    for site in sites:
        site = obs.snapwave_site_id(site)
        iid = obs.instrument_id(site)
        if iid not in buoy_coords:
            raise KeyError(
                f"Buoy {site!r} not in onshore-waves GeoJSON (instrument {iid!r})"
            )
        lon = buoy_coords[iid]["lon"]
        lat = buoy_coords[iid]["lat"]
        for offshore_name in names:
            catalog_key = offshore_catalog_key(get_offshore_wave(offshore_name))
            goal, dist_m = closest_goal_for_point(lon, lat, goals, catalog_key)
            snap = goals[goal]["hindcast"][catalog_key]
            rows.append(
                {
                    "site": site,
                    "offshore": offshore_name,
                    "goal": int(goal),
                    "distance_m": round(dist_m, 1),
                    "hindcast_lon": snap["longitude"],
                    "hindcast_lat": snap["latitude"],
                }
            )
    return pd.DataFrame(rows)


def load_offshore_hindcast_at_buoys(
    time: str | slice,
    sites: Iterable[str],
    *,
    offshore_names: Iterable[str] | None = None,
    goals: Mapping[str, dict[str, Any]] | None = None,
    variables: Iterable[str] | None = None,
    load: bool = True,
) -> tuple[dict[str, xr.Dataset], pd.DataFrame]:
    """Bulk offshore hindcast at each buoy's closest goal node.

    One entry per configured offshore source (e.g. WHACS). Each dataset has
    ``(time, sites)`` with the requested bulk variables present in the zarr.
    """
    site_ids = [obs.snapwave_site_id(s) for s in sites]
    mapping = closest_goals_table(site_ids, offshore_names=offshore_names, goals=goals)
    vars_ = [v for v in _metrics_variables(variables) if v in PARAMETERS]

    estimates: dict[str, xr.Dataset] = {}
    for offshore_name in mapping["offshore"].unique():
        entry = get_offshore_wave(str(offshore_name))
        offshore = open_offshore_waves(entry=entry).sel(time=time)
        present = [v for v in vars_ if v in offshore]
        if not present:
            raise ValueError(f"Offshore {offshore_name!r} has none of {vars_}")

        sub = mapping[mapping["offshore"] == offshore_name]
        chunks: list[xr.Dataset] = []
        for row in sub.itertuples(index=False):
            goal_ds = offshore.sel(goal=int(row.goal))[present]
            chunks.append(goal_ds.expand_dims(sites=[row.site]))

        label = f"{offshore_name} (closest goal)"
        out = xr.concat(chunks, dim="sites").transpose("time", "sites")
        out = attach_site_meta(out, mm.load_cases_goal(1))
        estimates[label] = out.load() if load else out
        logger.info(
            "Offshore %s: %d site(s) mapped to goals %s",
            offshore_name,
            len(sub),
            sorted(sub["goal"].unique().tolist()),
        )

    return estimates, mapping


def align_pair(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    sites: str | Iterable[str] | None = None,
    time: str | slice | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Inner join on shared ``sites`` and ``time``."""
    if sites is not None:
        site_ids = resolve_sites(estimate, sites)
    else:
        site_ids = sorted(
            set(str(s) for s in estimate.sites.values)
            & set(str(s) for s in reference.sites.values),
            key=str,
        )
    if not site_ids:
        raise ValueError("No shared sites between estimate and reference")

    estimate = estimate.sel(sites=site_ids)
    reference = reference.sel(sites=site_ids)
    if time is not None:
        estimate = estimate.sel(time=time)
        reference = reference.sel(time=time)

    estimate, reference = xr.align(estimate, reference, join="inner")
    if estimate.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping time steps after alignment")
    return estimate, reference


def compare_all(
    reference: xr.Dataset,
    estimates: Mapping[str, xr.Dataset],
    *,
    sites: str | Iterable[str] | None = None,
    time: str | slice | None = None,
    variables: Iterable[str] | None = None,
    reference_label: str = "Buoy",
    pca_variance: mm.PcaVariance = None,
) -> dict[str, ComparisonResult]:
    """Run :func:`compare` for several models against the same buoy reference."""
    results: dict[str, ComparisonResult] = {}
    for name, est in estimates.items():
        aligned_est, aligned_ref = align_pair(est, reference, sites=sites, time=time)
        results[name] = compare(
            aligned_est,
            aligned_ref,
            variables=variables,
            estimate_label=name,
            reference_label=reference_label,
            pca_variance=pca_variance,
        )
    return results


def metrics_summary(results: Mapping[str, ComparisonResult]) -> pd.DataFrame:
    """Concatenate :attr:`ComparisonResult.metrics` tables with a ``model`` column."""
    return pd.concat(
        [r.metrics.assign(model=name) for name, r in results.items()],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Validation entry points
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Aligned estimate vs reference with site-group metrics and plot helpers."""

    estimate: xr.Dataset
    reference: xr.Dataset
    metrics: pd.DataFrame
    _groups: list[tuple[str, str, list[str]]] = field(repr=False)

    def site_ids(self, group: str) -> list[str]:
        return find_group(self.estimate, group)[2]

    def plot(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        estimate_label: str = "Estimate",
        reference_label: str = "Reference",
        **kwargs: Any,
    ) -> Figure:
        name, _, site_ids = find_group(self.estimate, group)
        return plot_group(
            self.estimate,
            self.reference,
            name,
            site_ids,
            variables=variables,
            estimate_label=estimate_label,
            reference_label=reference_label,
            **kwargs,
        )

    def plot_map(self, group: str, **kwargs: Any) -> tuple[Figure, plt.Axes]:
        name, _, site_ids = find_group(self.estimate, group)
        return plot_map(self.estimate, site_ids, group_name=name, **kwargs)

    def plot_scatter(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        estimate_label: str = "Estimate",
        reference_label: str = "Reference",
        **kwargs: Any,
    ) -> Figure:
        name, _, site_ids = find_group(self.estimate, group)
        return plot_scatter(
            self.estimate,
            self.reference,
            name,
            site_ids,
            variables=variables,
            estimate_label=estimate_label,
            reference_label=reference_label,
            **kwargs,
        )

    def plot_timeseries(
        self,
        site: str,
        *,
        variables: Iterable[str] | None = None,
        estimate_label: str = "Estimate",
        reference_label: str = "Reference",
        **kwargs: Any,
    ) -> Figure:
        return plot_timeseries(
            self.estimate,
            self.reference,
            site,
            variables=variables,
            estimate_label=estimate_label,
            reference_label=reference_label,
            **kwargs,
        )

    def plot_all(
        self,
        *,
        variables: Iterable[str] | None = None,
        groups: Iterable[str] | None = None,
        estimate_label: str = "Estimate",
        reference_label: str = "Reference",
        **kwargs: Any,
    ) -> list[Figure]:
        active = self._groups
        if groups is not None:
            wanted = set(groups)
            active = [g for g in active if g[0] in wanted]
        return [
            plot_group(
                self.estimate,
                self.reference,
                name,
                site_ids,
                variables=variables,
                estimate_label=estimate_label,
                reference_label=reference_label,
                **kwargs,
            )
            for name, _, site_ids in active
        ]


HOLDOUT_ESTIMATE_LABEL = "Metamodel"
HOLDOUT_REFERENCE_LABEL = "SnapWave"


@dataclass
class HoldoutValidationResult(ComparisonResult):
    """Holdout metamodel validation for one goal and MDA train size."""

    goal: str = ""
    mda_candidate: int = 0
    n_test_cases: int = 0
    _default_estimate_label: str = field(default=HOLDOUT_ESTIMATE_LABEL, repr=False)
    _default_reference_label: str = field(default=HOLDOUT_REFERENCE_LABEL, repr=False)

    @property
    def n(self) -> int:
        """Alias for :attr:`mda_candidate` (MDA train size)."""
        return self.mda_candidate

    def plot(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        estimate_label: str | None = None,
        reference_label: str | None = None,
        **kwargs: Any,
    ) -> Figure:
        return super().plot(
            group,
            variables=variables,
            estimate_label=estimate_label or self._default_estimate_label,
            reference_label=reference_label or self._default_reference_label,
            **kwargs,
        )

    def plot_scatter(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        estimate_label: str | None = None,
        reference_label: str | None = None,
        **kwargs: Any,
    ) -> Figure:
        return super().plot_scatter(
            group,
            variables=variables,
            estimate_label=estimate_label or self._default_estimate_label,
            reference_label=reference_label or self._default_reference_label,
            **kwargs,
        )

    def plot_all(
        self,
        *,
        variables: Iterable[str] | None = None,
        groups: Iterable[str] | None = None,
        estimate_label: str | None = None,
        reference_label: str | None = None,
        **kwargs: Any,
    ) -> list[Figure]:
        return super().plot_all(
            variables=variables,
            groups=groups,
            estimate_label=estimate_label or self._default_estimate_label,
            reference_label=reference_label or self._default_reference_label,
            **kwargs,
        )


def _holdout_pair(
    goal: str,
    mda_candidate: int,
    *,
    context: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    pca_variance: mm.PcaVariance = None,
) -> tuple[xr.Dataset, xr.Dataset, int]:
    """Metamodel estimate and SnapWave reference on MDA test cases.

    Returns estimate, reference (both ``case_num`` × ``sites``), and test count.
    """
    if context is None:
        context = mm.load_context(goal)
    test_idx = context["test_idx"]
    if models is None:
        models = mm.load_models(goal, mda_candidate, pca_variance=pca_variance)

    cases = mm.load_cases_goal(goal)
    idx = np.asarray(test_idx, dtype=int)
    estimate = mm.predict(
        goal,
        mda_candidate,
        mm.forcing_dataframe(cases).iloc[idx],
        models=models,
        pca_variance=pca_variance,
    )
    configured = _metrics_variables(None)
    vars_ = _available_metrics_variables(estimate, cases, variables=None)
    missing = [v for v in configured if v in PARAMETERS and v not in vars_]
    if missing:
        logger.warning(
            "Goal %s / n=%d: configured metrics variables missing from "
            "estimate or reference: %s",
            goal,
            mda_candidate,
            missing,
        )
    reference = cases.isel(case_num=idx)[vars_].assign(
        **{name: cases[name] for name in SITE_META if name in cases}
    )
    return estimate, reference, len(test_idx)


def compare(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    estimate_label: str = "Estimate",
    reference_label: str = "Reference",
    pca_variance: mm.PcaVariance = None,
) -> ComparisonResult:
    """
    Compare aligned estimate and reference datasets.

    Parameters
    ----------
    estimate, reference
        Datasets sharing ``sites`` (and ``time`` or ``case_num``).
    variables
        Metrics variables to score.
    estimate_label, reference_label
        Labels stored on the metrics table and used in plots. Always present
        as ``estimate_label``/``reference_label`` columns, matching the
        schema :func:`validate`/:func:`sweep` produce via ``_tag_metrics``.
    pca_variance
        Fully resolved PCA fractions (config defaults filled in for any
        variable omitted from a partial override dict), always stored in a
        ``pca_variances`` column — ``None`` resolves to the config defaults.

    Returns
    -------
    ComparisonResult
        Metrics plus plot helpers.
    """
    table = metrics_table(estimate, reference, variables=variables)
    table.insert(0, "pca_variances", str(mm.resolved_pca_variances(pca_variance)))
    table["estimate_label"] = estimate_label
    table["reference_label"] = reference_label
    return ComparisonResult(
        estimate=estimate,
        reference=reference,
        metrics=table,
        _groups=metric_groups(estimate),
    )


def _tag_metrics(
    table: pd.DataFrame,
    *,
    goal: str,
    n: int,
    n_test: int,
    pca_variance: mm.PcaVariance,
    estimate_label: str = HOLDOUT_ESTIMATE_LABEL,
    reference_label: str = HOLDOUT_REFERENCE_LABEL,
) -> pd.DataFrame:
    table.insert(0, "goal", goal)
    table.insert(1, "mda_candidate", n)
    table.insert(2, "n_test_cases", n_test)
    table.insert(3, "pca_variances", str(mm.resolved_pca_variances(pca_variance)))
    table["estimate_label"] = estimate_label
    table["reference_label"] = reference_label
    return table


def validate(
    goal: str | int,
    n: int,
    *,
    variables: Iterable[str] | None = None,
    pca_variance: mm.PcaVariance = None,
) -> HoldoutValidationResult:
    """
    Holdout validation for one goal and MDA train size.

    Parameters
    ----------
    goal, n
        Goal id and MDA train size (saved models must exist).
    variables
        Metrics variables to score.
    pca_variance
        PCA fractions used at fit time.

    Returns
    -------
    HoldoutValidationResult
        Estimate, reference, tagged metrics, and plot helpers.
    """
    goal = str(goal)
    n = int(n)
    validate_metamodel_config()
    estimate, reference, n_test = _holdout_pair(goal, n, pca_variance=pca_variance)
    table = _tag_metrics(
        metrics_table(estimate, reference, variables=variables),
        goal=goal,
        n=n,
        n_test=n_test,
        pca_variance=pca_variance,
    )
    return HoldoutValidationResult(
        goal=goal,
        mda_candidate=n,
        estimate=estimate,
        reference=reference,
        metrics=table,
        n_test_cases=n_test,
        _groups=metric_groups(estimate),
    )


def sweep(
    goal: str | int,
    *,
    candidates: Iterable[int] | None = None,
    variables: Iterable[str] | None = None,
    pca_variance: mm.PcaVariance = None,
    save: bool = True,
) -> pd.DataFrame:
    """Holdout metrics for every MDA candidate size on one goal."""
    goal = str(goal)
    cand = [int(c) for c in (candidates or MDA_CANDIDATES)]
    context = mm.load_context(goal)

    frames: list[pd.DataFrame] = []
    for n in cand:
        logger.info("Goal %s: validating n=%d", goal, n)
        estimate, reference, n_test = _holdout_pair(
            goal, n, context=context, pca_variance=pca_variance
        )
        frames.append(
            _tag_metrics(
                metrics_table(estimate, reference, variables=variables),
                goal=goal,
                n=n,
                n_test=n_test,
                pca_variance=pca_variance,
            )
        )

    out = pd.concat(frames, ignore_index=True)
    if save:
        path = validation_metrics_path(goal)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        logger.info("Wrote %s", path)
    return out


def sweep_all(
    goals: Iterable[str | int] | None = None,
    *,
    candidates: Iterable[int] | None = None,
    variables: Iterable[str] | None = None,
    pca_variance: mm.PcaVariance = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run :func:`sweep` for every goal and concatenate."""
    active = [str(g) for g in (goals or load_goals().keys())]
    frames = [
        sweep(
            goal,
            candidates=candidates,
            variables=variables,
            pca_variance=pca_variance,
            save=save,
        )
        for goal in active
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _paired_values(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    variable: str,
    site_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    est = estimate.sel(sites=site_ids)[variable].values.ravel()
    ref = reference.sel(sites=site_ids)[variable].values.ravel()
    return _coincident_values(est, ref)


MAX_DENSITY_POINTS = 5000
"""Cap on points passed to ``gaussian_kde`` (O(n^2)) in :func:`_scatter_panel`."""


def _subsample_for_density(
    x: np.ndarray, y: np.ndarray, max_points: int = MAX_DENSITY_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly subsample paired arrays for ``gaussian_kde``.

    Dense groups (e.g. a mesh polygon with thousands of sites) times many
    holdout cases can reach 10^5-10^6 paired points; ``gaussian_kde``
    evaluation is O(n^2) and becomes impractically slow well before that.
    Subsampling only affects the density-color rendering, not the metrics
    annotation (computed upstream on the full pair).
    """
    if x.size <= max_points:
        return x, y
    idx = np.random.default_rng(0).choice(x.size, size=max_points, replace=False)
    return x[idx], y[idx]


def _scatter_panel(
    ax: plt.Axes,
    estimate: np.ndarray,
    reference: np.ndarray,
    variable: str,
    *,
    estimate_label: str = "Estimate",
    reference_label: str = "Reference",
    cmap: str = "viridis",
) -> None:
    spec = PARAMETERS[variable]
    est, ref = _coincident_values(estimate, reference)
    if spec["circular"]:
        est = _theta_to_360(est)
        ref = _theta_to_360(ref)
        est, ref = _coincident_values(est, ref)

    if est.size == 0:
        ax.set_title(f"{spec['label']} (no data)")
        return

    est_plot, ref_plot = _subsample_for_density(est, ref)
    try:
        xs, ys, density = density_scatter(est_plot, ref_plot)
        ax.scatter(xs, ys, c=density, s=10, cmap=cmap, edgecolors="none", alpha=0.9)
    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.debug("density_scatter failed (%s); using plain scatter", exc)
        ax.scatter(
            est_plot, ref_plot, s=10, color="steelblue", edgecolors="none", alpha=0.5
        )
        xs, ys = est_plot, ref_plot

    if spec["circular"]:
        lo, hi = 0.0, 360.0
    else:
        lo = 0.0
        hi = float(np.ceil(max(est.max(), ref.max()) * 1.05))

    ax.plot([lo, hi], [lo, hi], "-r", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{estimate_label} {spec['label']} [{spec['units']}]")
    ax.set_ylabel(f"{reference_label} {spec['label']} [{spec['units']}]")
    ax.set_title(spec["label"])

    m = compute_metrics(est, ref, circular=bool(spec["circular"]))
    props = dict(
        boxstyle="round", facecolor="w", edgecolor="grey", linewidth=0.8, alpha=0.6
    )
    label = "\n".join(
        (
            rf"BIAS = {m['bias']:.3f}",
            rf"RMSE = {m['rmse']:.3f}",
            rf"R² = {m['r2']:.3f}",
            rf"n = {m['n']}",
        )
    )
    ax.text(
        0.04,
        0.96,
        label,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )


def plot_scatter(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    group_name: str,
    site_ids: list[str],
    *,
    variables: Iterable[str] | None = None,
    estimate_label: str = "Estimate",
    reference_label: str = "Reference",
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Density scatter panels for one validation group."""
    keys = _available_metrics_variables(estimate, reference, variables)
    nrows, ncols = _scatter_grid_shape(len(keys))
    if figsize is None:
        figsize = (4.2 * ncols, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes_flat, keys):
        est, ref = _paired_values(estimate, reference, key, site_ids)
        _scatter_panel(
            ax,
            est,
            ref,
            key,
            estimate_label=estimate_label,
            reference_label=reference_label,
            cmap=cmap,
        )

    for ax in axes_flat[len(keys) :]:
        ax.set_visible(False)

    fig.suptitle(group_name, fontsize=11)
    fig.tight_layout()
    return fig


def plot_map(
    estimate: xr.Dataset,
    site_ids: list[str],
    *,
    group_name: str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[Figure, plt.Axes]:
    """Map of validation sites coloured by source / depth label."""
    from utils.viz.case_plots import plot_base_map, plot_validation_sites

    subset = estimate.sel(sites=site_ids)
    if ax is None:
        fig, ax = plot_base_map(title=title or group_name or "Validation sites")
    else:
        fig = ax.figure

    plot_validation_sites(
        subset["site_longitude"].values,
        subset["site_latitude"].values,
        subset["site_source"].values,
        subset["site_label"].values,
        ax,
        site_ids=subset["sites"].values,
        annotate_buoys=True,
    )
    return fig, ax


def plot_group(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    group_name: str,
    site_ids: list[str],
    *,
    variables: Iterable[str] | None = None,
    estimate_label: str = "Estimate",
    reference_label: str = "Reference",
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Map + density scatters for one validation group."""
    from utils.viz.case_plots import plot_base_map, plot_validation_sites

    keys = _available_metrics_variables(estimate, reference, variables)
    scatter_rows, scatter_cols = _scatter_grid_shape(len(keys))
    if figsize is None:
        figsize = (4.5 * scatter_cols, 3.0 + 3.5 * scatter_rows)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1 + scatter_rows,
        scatter_cols,
        height_ratios=[1.25] + [1.0] * scatter_rows,
        hspace=0.3,
        wspace=0.3,
    )
    ax_map = fig.add_subplot(gs[0, :])

    plot_base_map(ax=ax_map, title=None)
    subset = estimate.sel(sites=site_ids)
    plot_validation_sites(
        subset["site_longitude"].values,
        subset["site_latitude"].values,
        subset["site_source"].values,
        subset["site_label"].values,
        ax_map,
        site_ids=subset["sites"].values,
    )
    ax_map.set_title(group_name, fontsize=11)

    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[1 + i // scatter_cols, i % scatter_cols])
        est, ref = _paired_values(estimate, reference, key, site_ids)
        _scatter_panel(
            ax,
            est,
            ref,
            key,
            estimate_label=estimate_label,
            reference_label=reference_label,
            cmap=cmap,
        )

    fig.tight_layout()
    return fig


_MODEL_COLORS: dict[str, str] = {
    "Buoy": "black",
    "Bulk metamodel": "C0",
    "Spectral metamodel": "C1",
    "Dynamic SnapWave": "C2",
    "WHACS (closest goal)": "C3",
}


def plot_timeseries(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    site: str,
    *,
    variables: Iterable[str] | None = None,
    estimate_label: str = "Estimate",
    reference_label: str = "Reference",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Line plot of estimate vs reference at one site."""
    site = str(site)
    keys = _available_metrics_variables(estimate, reference, variables)
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    if len(keys) == 1:
        axes = [axes]

    est_site = estimate.sel(sites=site)
    ref_site = reference.sel(sites=site)

    for ax, key in zip(axes, keys, strict=True):
        spec = PARAMETERS[key]
        est, ref = xr.align(est_site[key], ref_site[key], join="inner")
        est.plot(
            ax=ax,
            label=estimate_label,
            color=_MODEL_COLORS.get(estimate_label, "C0"),
            lw=1.0,
        )
        ref.plot(
            ax=ax,
            label=reference_label,
            color=_MODEL_COLORS.get(reference_label, "C1"),
            lw=1.2,
            alpha=0.85,
        )
        ax.set_ylabel(f"{spec['label']} [{spec['units']}]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time")
    fig.suptitle(site, fontsize=11)
    fig.tight_layout()
    return fig


def plot_buoy_timeseries(
    reference: xr.Dataset,
    estimates: Mapping[str, xr.Dataset],
    site: str,
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Buoy",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Time series at one buoy: observations vs all model estimates."""
    site = str(site)
    keys = _metrics_variables(variables)
    keys = [k for k in keys if k in PARAMETERS and k in reference.data_vars]
    keys = [k for k in keys if all(k in est.data_vars for est in estimates.values())]
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    if len(keys) == 1:
        axes = [axes]

    ref_site = reference.sel(sites=site)

    for ax, key in zip(axes, keys, strict=True):
        spec = PARAMETERS[key]
        ref = ref_site[key]
        if spec["circular"]:
            ax.scatter(
                ref.time.values,
                ref.values,
                label=reference_label,
                color=_MODEL_COLORS.get(reference_label, "black"),
                s=8,
                alpha=0.75,
                zorder=3,
            )
            for i, (name, est) in enumerate(estimates.items()):
                est_aligned, _ = xr.align(est.sel(sites=site)[key], ref, join="inner")
                ax.scatter(
                    est_aligned.time.values,
                    est_aligned.values,
                    label=name,
                    color=_MODEL_COLORS.get(name, f"C{i}"),
                    s=6,
                    alpha=0.65,
                    zorder=2,
                )
            ax.set_ylim(0.0, 360.0)
        else:
            ref.plot(
                ax=ax,
                label=reference_label,
                color=_MODEL_COLORS.get(reference_label, "black"),
                lw=1.2,
            )
            for i, (name, est) in enumerate(estimates.items()):
                est_aligned, _ = xr.align(est.sel(sites=site)[key], ref, join="inner")
                est_aligned.plot(
                    ax=ax,
                    label=name,
                    color=_MODEL_COLORS.get(name, f"C{i}"),
                    lw=1.0,
                    alpha=0.9,
                )
        ax.set_ylabel(f"{spec['label']} [{spec['units']}]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(site, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# MDA sweep summary plots
# ---------------------------------------------------------------------------

SWEEP_STATS = ("bias", "rmse", "r2")
SWEEP_BY = ("bulk", "source", "group", "label", "buoy")


def load_sweep_metrics(goal: str | int) -> pd.DataFrame:
    """Load ``validation_metrics.csv`` written by :func:`sweep`."""
    path = validation_metrics_path(goal)
    if not path.exists():
        raise FileNotFoundError(f"No sweep metrics at {path}. Run sweep() first.")
    df = pd.read_csv(path)
    df["goal"] = df["goal"].astype(str)
    return df


def _filter_sweep(
    metrics: pd.DataFrame,
    *,
    goals: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = metrics.copy()
    df["goal"] = df["goal"].astype(str)
    if goals is not None:
        wanted = {str(g) for g in goals}
        df = df[df["goal"].isin(wanted)]
    vars_ = _metrics_variables(variables)
    return df[df["variable"].isin(vars_)].copy()


def _weighted_stats(group: pd.DataFrame) -> pd.Series:
    w = group["n"].to_numpy(dtype=float)
    if w.sum() <= 0:
        return pd.Series({s: np.nan for s in SWEEP_STATS} | {"n": 0})
    return pd.Series(
        {
            "bias": float(np.average(group["bias"], weights=w)),
            "rmse": float(np.average(group["rmse"], weights=w)),
            "r2": float(np.average(group["r2"], weights=w)),
            "n": int(w.sum()),
        }
    )


def aggregate_sweep(
    metrics: pd.DataFrame,
    *,
    by: str = "bulk",
    goals: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Collapse sweep rows for line plots.

    ``by`` options:

    - ``bulk`` — n-weighted mean over all groups
    - ``source`` — per ``site_source`` (isobath / mesh / buoy)
    - ``group`` — per validation group (``5m``, ``10m``, ``Apia``, buoy id, …)
    - ``label`` — per ``site_label``
    - ``buoy`` — buoy groups only (same granularity as ``group``)
    """
    if by not in SWEEP_BY:
        raise ValueError(f"by must be one of {SWEEP_BY}, got {by!r}")

    df = _filter_sweep(metrics, goals=goals, variables=variables)
    if df.empty:
        return df

    if by in ("group", "buoy"):
        out = df.copy()
        if by == "buoy":
            out = out[out["site_source"] == "buoy"]
        return out.sort_values(["goal", "group", "variable", "mda_candidate"])

    key_cols = {
        "bulk": ["goal", "mda_candidate", "variable"],
        "source": ["goal", "mda_candidate", "site_source", "variable"],
        "label": ["goal", "mda_candidate", "site_label", "variable"],
    }[by]

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(key_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(key_cols, keys, strict=True))
        row.update(_weighted_stats(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key_cols)


def _sweep_line_color(name: str, source: str = "") -> str:
    from utils.viz.case_plots import DEFAULT_POINT_COLOR, ISOBATH_COLORS, MESH_COLORS

    if name in ISOBATH_COLORS:
        return ISOBATH_COLORS[name]
    if name in MESH_COLORS:
        return MESH_COLORS[name]
    if source == "buoy" or str(name).startswith("buoy_"):
        return plt.cm.tab10(abs(hash(name)) % 10)
    return DEFAULT_POINT_COLOR


def _sweep_series_key(row: pd.Series, by: str) -> str:
    if by == "bulk":
        return "all sites"
    if by == "source":
        return str(row["site_source"])
    if by == "label":
        return str(row["site_label"])
    return str(row["group"])


def _sweep_line_label(row: pd.Series, by: str, *, multi_goal: bool) -> str:
    if by == "bulk":
        return f"goal {row['goal']}" if multi_goal else "all sites"
    name = _sweep_series_key(row, by)
    return f"{name} (goal {row['goal']})" if multi_goal else name


def plot_sweep(
    metrics: pd.DataFrame,
    *,
    by: str = "bulk",
    goals: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> Figure:
    """Line plots of bias / RMSE / R² vs MDA size.

    Rows = metrics, columns = variables (hs, tp, dir, spr). One line per series
    defined by ``by`` (bulk, source, group, label, or buoy).
    """
    vars_ = _metrics_variables(variables)
    agg = aggregate_sweep(metrics, by=by, goals=goals, variables=vars_)
    if agg.empty:
        raise ValueError("No sweep data to plot after filtering.")

    multi_goal = agg["goal"].nunique() > 1
    nrows, ncols = len(SWEEP_STATS), len(vars_)
    if figsize is None:
        figsize = (4.2 * ncols, 3.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    series_cols = {
        "bulk": ["goal"],
        "source": ["goal", "site_source"],
        "label": ["goal", "site_label"],
        "group": ["goal", "group", "site_source"],
        "buoy": ["goal", "group"],
    }[by]

    for i, stat in enumerate(SWEEP_STATS):
        for j, var in enumerate(vars_):
            ax = axes[i, j]
            sub = agg[agg["variable"] == var]
            for key, grp in sub.groupby(series_cols, sort=False):
                grp = grp.sort_values("mda_candidate")
                row0 = grp.iloc[0]
                label = _sweep_line_label(row0, by, multi_goal=multi_goal)
                source = str(row0.get("site_source", ""))
                color = _sweep_line_color(_sweep_series_key(row0, by), source)
                ax.plot(
                    grp["mda_candidate"],
                    grp[stat],
                    "o-",
                    lw=1.5,
                    ms=4,
                    label=label,
                    color=color,
                )

            spec = PARAMETERS[var]
            if i == 0:
                ax.set_title(spec["label"])
            ylab = "R²" if stat == "r2" else f"{stat.upper()} [{spec['units']}]"
            ax.set_ylabel(ylab)
            if i == nrows - 1:
                ax.set_xlabel("MDA train cases")

            if stat == "r2":
                ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, -1].get_legend_handles_labels()
    if not handles:
        for ax in axes[0]:
            h, lab = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, lab
                break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(4, len(labels)),
            bbox_to_anchor=(0.5, 1.04),
            fontsize=8,
        )

    by_title = {
        "bulk": "all sites",
        "source": "by site source",
        "group": "by group",
        "label": "by site label",
        "buoy": "by buoy",
    }[by]
    goal_txt = ", ".join(sorted(agg["goal"].unique()))
    fig.suptitle(
        title or f"MDA sweep — {by_title} — goal(s) {goal_txt}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_sweep_suite(
    metrics: pd.DataFrame,
    *,
    goals: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
    by: Iterable[str] | None = None,
) -> dict[str, Figure]:
    """One :func:`plot_sweep` figure per grouping mode."""
    modes = list(by or SWEEP_BY)
    return {
        mode: plot_sweep(metrics, by=mode, goals=goals, variables=variables)
        for mode in modes
    }


__all__ = [
    "LOGGER_NAME",
    "MAX_DENSITY_POINTS",
    "ComparisonResult",
    "HoldoutValidationResult",
    "align_pair",
    "closest_goals_table",
    "compare",
    "compare_all",
    "compute_metrics",
    "load_offshore_hindcast_at_buoys",
    "logger",
    "metric_groups",
    "metrics_summary",
    "metrics_table",
    "resolve_comparison_variables",
    "aggregate_sweep",
    "load_sweep_metrics",
    "plot_buoy_timeseries",
    "plot_group",
    "plot_map",
    "plot_scatter",
    "plot_timeseries",
    "plot_sweep",
    "plot_sweep_suite",
    "sweep",
    "sweep_all",
    "validate",
    "validation_metrics_path",
    "SWEEP_BY",
    "SWEEP_STATS",
]
