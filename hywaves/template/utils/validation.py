"""Compare wave datasets: metrics and plots.

Holdout validation (:func:`validate`, :func:`sweep`) — metamodel vs SnapWave on MDA
test cases, per metamodel set (requires :func:`utils.metamodel.fit`).

Generic comparison (:func:`compare`, :func:`resolve_comparison_variables`) — any
aligned estimate/reference pair on ``(time, sites)`` bulk variables.

Buoy validation (:func:`load_offshore_hindcast_at_buoys`, :func:`compare_all`) —
bulk / spectral reconstruction, dynamic SnapWave, and closest-goal offshore
hindcast vs user-processed buoy datasets (the ``10*`` buoy-validation notebooks).
Buoy site coordinates and labels come from the onshore-waves GeoJSON
(:mod:`utils.onshore_sites`).

Site grouping for plots and metrics tables: :mod:`utils.geometry`.

Convention (shared with :mod:`bluemath_tk.core.metrics`): every function takes
the **reference** first and the **estimate** second. Metrics are
``metric(reference, estimate)`` (positive bias = the estimate overestimates),
and scatters put the reference on the x-axis, the estimate on the y-axis.
Which dataset is the reference depends on the comparison: SnapWave for the
holdout (06), dynamic SnapWave for linear summation (09), the buoy for buoy
validation (10).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core import metrics as bm_metrics
from bluemath_tk.core.geo import geodesic_distance
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.plotting.scatter import scatter_grid_shape, validation_scatter
from bluemath_tk.core.plotting.timeseries import plot_timeseries_comparison
from matplotlib.figure import Figure

from utils.config import (
    METAMODEL,
    OFFSHORE_WAVES,
    get_offshore_wave,
    load_goals,
    model_label,
    offshore_catalog_key,
    open_offshore_waves,
    resolve,
    validate_metamodel_config,
)
from utils.geometry import (
    SITE_META,
    attach_site_meta,
    find_group,
    metric_groups,
    per_site_coord,
    resolve_sites,
)
from utils import metamodel as mm
from utils import onshore_sites as obs

# Config mirrors metamodel module (case config → metamodel)
METRICS_VARIABLES = METAMODEL["validation"]["metrics_variables"]
MDA_CANDIDATES = mm.MDA_CANDIDATES

LOGGER_NAME = "hywaves.validation"
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

# Generous physical ceilings, not tight QC bounds: they exist only to catch
# corrupted dynamic-model output that np.isfinite doesn't — e.g. a
# wetting/drying instability at a handful of SnapWave isobath sites/hours
# producing finite-but-nonphysical hs (seen up to ~8.8e13 m at one point).
# A single such value squared and averaged over millions of otherwise-good
# points is enough to send a whole group's rmse/r2 to nonsense (observed:
# rmse=3.1e10, r2=-1.8e21 for the isobath "5m" group). Six years of this
# domain's hs (buoys and isobaths alike) tops out under 5 m, so 15 m leaves
# comfortable headroom for any real storm while excluding every corrupted
# value found so far (all >= 3.7e6 m). The lower bound only catches corrupt
# inputs: metamodel hs is floored at 0 in ``predict()`` (see
# ``utils.metamodel.MIN_HS``), so its calm-site estimates are scored,
# not dropped. Variables not listed here keep the isfinite-only filter
# unchanged.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "hs": (0.0, 15.0),
}

# Columns written to every metrics table / CSV (plus ``n``). The contract only
# requires bias/rmse/r2; the rest are extra columns. Circular variables (dir)
# get NaN for the metrics without a circular definition (rrmse, si, r).
METRICS: tuple[str, ...] = ("bias", "rmse", "rrmse", "si", "r", "r2")

# Metrics annotated on scatter panels.
SCATTER_METRICS: tuple[str, ...] = ("bias", "rmse", "rrmse", "r2")


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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _coincident_values(
    reference: np.ndarray, estimate: np.ndarray, *, variable: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired finite, physically-plausible ``(reference, estimate)`` values.

    Parameters
    ----------
    reference, estimate
        Paired arrays (flattened here).
    variable
        Key into :data:`PHYSICAL_BOUNDS`. When given and bounded, pairs where
        either side falls outside the bound are dropped in addition to the
        usual ``isfinite`` filter — see :data:`PHYSICAL_BOUNDS` for why.
    """
    ref, est = bm_metrics.paired_finite(reference, estimate)
    bounds = PHYSICAL_BOUNDS.get(variable) if variable else None
    if bounds is not None:
        lo, hi = bounds
        physical = (ref >= lo) & (ref <= hi) & (est >= lo) & (est <= hi)
        n_dropped = int(ref.size - physical.sum())
        if n_dropped:
            logger.warning(
                "%s: dropped %d pair(s) outside physical bound [%s, %s] "
                "(e.g. corrupted dynamic-model output)",
                variable,
                n_dropped,
                lo,
                hi,
            )
        ref, est = ref[physical], est[physical]
    return ref, est


def compute_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    circular: bool = False,
    variable: str | None = None,
    metrics: Iterable[str] = METRICS,
) -> dict[str, float]:
    """
    Validation metrics for one variable pair (reference first).

    Thin wrapper on :func:`bluemath_tk.core.metrics.compute_metrics` that
    also drops pairs outside :data:`PHYSICAL_BOUNDS`.

    Parameters
    ----------
    reference, estimate
        Paired arrays: what is trusted (SnapWave, buoy, ...) and what is
        evaluated (metamodel, hindcast, ...).
    circular
        If ``True``, treat values as directions in degrees.
    variable
        Key into :data:`PHYSICAL_BOUNDS`, e.g. ``"hs"``; passed through to
        :func:`_coincident_values` to drop corrupted-but-finite outliers.
    metrics
        Metric names; default :data:`METRICS`.

    Returns
    -------
    dict[str, float]
        ``n`` plus one key per metric (positive ``bias`` = overestimation).
    """
    ref, est = _coincident_values(reference, estimate, variable=variable)
    return bm_metrics.compute_metrics(ref, est, metrics, circular=circular)


# ---------------------------------------------------------------------------
# Site groups (re-exported from utils.geometry)
# ---------------------------------------------------------------------------


def metrics_table(
    reference: xr.Dataset,
    estimate: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Build a metrics table with one row per validation group and variable.

    Parameters
    ----------
    reference, estimate
        Aligned datasets with ``sites`` and wave variables.
    variables
        Subset of metrics variables; default from config.

    Returns
    -------
    pandas.DataFrame
        Columns include ``group``, ``site_source``, ``variable``, ``n`` and
        one column per :data:`METRICS` entry.
    """
    params = _available_metrics_variables(estimate, reference, variables)
    rows: list[dict[str, Any]] = []

    for group_name, site_source, site_ids in metric_groups(estimate):
        est_g = estimate.sel(sites=site_ids)
        ref_g = reference.sel(sites=site_ids)
        for param in params:
            spec = PARAMETERS[param]
            metrics = compute_metrics(
                ref_g[param].values,
                est_g[param].values,
                circular=bool(spec["circular"]),
                variable=param,
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


def validation_metrics_path(goal: str | int, metamodel: str) -> Path:
    """CSV path for :func:`sweep` output on one goal and metamodel set."""
    return mm.goal_output_dir(goal) / metamodel / "validation_metrics.csv"


def _slug(label: str) -> str:
    """Filesystem-safe stand-in for a period label used in output filenames."""
    return re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")


def metamodel_validation_path(period: str, metamodel: str) -> Path:
    """CSV path for the linear-summation-vs-dynamic comparison of one metamodel set
    (``09_Validate_Linear_Summation.ipynb``)."""
    return (
        resolve("outputs/metamodel_validation")
        / f"linear_summation_{metamodel}_{_slug(period)}.csv"
    )


def buoy_validation_path(name: str, period_label: str) -> Path:
    """CSV path for a buoy validation summary (``10a``/``10b`` notebooks).

    Parameters
    ----------
    name
        Buoy subset the summary covers, e.g. ``"offshore"`` or ``"onshore"``.
    period_label
        Human-readable period used in the notebook (e.g. ``"2001-2011"``).
    """
    return resolve("outputs/buoy_validation") / f"{name}_summary_{_slug(period_label)}.csv"


EARTH_RADIUS_M = 6_371_000.0


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two lon/lat points."""
    arc_deg = geodesic_distance(lat1, lon1, lat2, lon2)
    return float(np.radians(arc_deg) * EARTH_RADIUS_M)


def closest_goal_for_point(
    lon: float,
    lat: float,
    goals: Mapping[str, dict[str, Any]],
    catalog_key: str,
) -> tuple[str, float]:
    """Nearest goal hindcast node in ``catalog_key`` (e.g. ``whacs``).

    Parameters
    ----------
    lon, lat
        Query point coordinates.
    goals
        Goals mapping from :func:`utils.config.load_goals`.
    catalog_key
        Offshore catalog key (see :func:`utils.config.offshore_catalog_key`)
        used to look up each goal's ``hindcast`` snap point.

    Returns
    -------
    goal_id, distance_m
        Closest goal id and its great-circle distance to *(lon, lat)*.

    Raises
    ------
    ValueError
        If no goal has a hindcast snap for *catalog_key*.
    """
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
    """Map each buoy site to the closest goal hindcast node per offshore source.

    Parameters
    ----------
    sites
        SnapWave/buoy site labels (see :func:`utils.onshore_sites.snapwave_site_id`).
    offshore_names
        Offshore catalog names to map against; default every entry in
        ``offshore_waves`` from the case config JSON.
    goals
        Goals mapping; default :func:`utils.config.load_goals`.

    Returns
    -------
    pandas.DataFrame
        One row per ``(site, offshore)`` pair with columns ``site``,
        ``offshore``, ``goal``, ``distance_m``, ``hindcast_lon``, ``hindcast_lat``.

    Raises
    ------
    KeyError
        If a site's buoy instrument is not in the onshore-waves GeoJSON.
    ValueError
        If an offshore catalog has no hindcast snap for some goal (raised by
        :func:`closest_goal_for_point`).
    """
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

    Parameters
    ----------
    time
        Period passed to ``.sel(time=...)`` on the offshore zarr.
    sites
        Buoy/SnapWave site labels to map.
    offshore_names
        Offshore catalog names; default every entry in ``offshore_waves``
        from the case config JSON.
    goals
        Goals mapping; default :func:`utils.config.load_goals`.
    variables
        Bulk variables to keep; default ``metamodel.validation.metrics_variables``.
    load
        Eager-load each output dataset.

    Returns
    -------
    estimates
        ``{"{offshore_name} (closest goal)": dataset}`` on ``(time, sites)``.
    mapping
        Site → closest-goal table from :func:`closest_goals_table`.

    Raises
    ------
    ValueError
        If an offshore source has none of the requested variables.
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
    reference: xr.Dataset,
    estimate: xr.Dataset,
    *,
    sites: str | Iterable[str] | None = None,
    time: str | slice | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Inner join *reference* and *estimate* on shared ``sites`` and ``time``.

    Parameters
    ----------
    reference, estimate
        Datasets to align; must share a ``sites`` coordinate and a ``time`` dimension.
    sites
        Site selector resolved via :func:`utils.geometry.resolve_sites`
        (on *estimate*); default the intersection of both datasets' site labels.
    time
        Optional ``.sel(time=...)`` applied to both datasets before alignment.

    Returns
    -------
    reference, estimate
        Datasets restricted to shared sites and inner-joined on time.

    Raises
    ------
    ValueError
        If no sites are shared, or no time steps overlap after alignment.
    """
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

    reference, estimate = xr.align(reference, estimate, join="inner")
    if estimate.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping time steps after alignment")
    return reference, estimate


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
    """Run :func:`compare` for several models against the same buoy reference.

    Each estimate is independently aligned to *reference* via :func:`align_pair`
    before comparison, so models covering different sites/time spans are all handled.

    Parameters
    ----------
    reference
        Common reference dataset (e.g. buoy observations).
    estimates
        Model name → dataset to compare against *reference*.
    sites, time
        Passed to :func:`align_pair` for each estimate.
    variables
        Metrics variables to score.
    reference_label
        Reference label stored on each result's metrics table.
    pca_variance
        Passed through to :func:`compare` (labels metamodel PCA settings, if any).

    Returns
    -------
    dict[str, ComparisonResult]
        One result per estimate, keyed by model name.
    """
    results: dict[str, ComparisonResult] = {}
    for name, est in estimates.items():
        aligned_ref, aligned_est = align_pair(reference, est, sites=sites, time=time)
        results[name] = compare(
            aligned_ref,
            aligned_est,
            variables=variables,
            reference_label=reference_label,
            estimate_label=name,
            pca_variance=pca_variance,
        )
    return results


def metrics_summary(results: Mapping[str, ComparisonResult]) -> pd.DataFrame:
    """Concatenate :attr:`ComparisonResult.metrics` tables with a ``model`` column.

    Parameters
    ----------
    results
        Model name → result, e.g. from :func:`compare_all`.

    Returns
    -------
    pandas.DataFrame
        All metrics tables stacked, with an added ``model`` column.
    """
    return pd.concat(
        [r.metrics.assign(model=name) for name, r in results.items()],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Validation entry points
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Aligned reference vs estimate with site-group metrics and plot helpers."""

    reference: xr.Dataset
    estimate: xr.Dataset
    metrics: pd.DataFrame
    _groups: list[tuple[str, str, list[str]]] = field(repr=False)

    def site_ids(self, group: str) -> list[str]:
        """Site labels belonging to one validation group (see :func:`find_group`)."""
        return find_group(self.estimate, group)[2]

    def plot(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        reference_label: str = "Reference",
        estimate_label: str = "Estimate",
        **kwargs: Any,
    ) -> Figure:
        """Map + density scatters for one validation group (see :func:`plot_group`)."""
        name, _, site_ids = find_group(self.estimate, group)
        return plot_group(
            self.reference,
            self.estimate,
            name,
            site_ids,
            variables=variables,
            reference_label=reference_label,
            estimate_label=estimate_label,
            **kwargs,
        )

    def plot_map(self, group: str, **kwargs: Any) -> tuple[Figure, plt.Axes]:
        """Site map for one validation group (see :func:`plot_map`)."""
        name, _, site_ids = find_group(self.estimate, group)
        return plot_map(self.estimate, site_ids, group_name=name, **kwargs)

    def plot_scatter(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        reference_label: str = "Reference",
        estimate_label: str = "Estimate",
        **kwargs: Any,
    ) -> Figure:
        """Density scatter panels for one validation group (see :func:`plot_scatter`)."""
        name, _, site_ids = find_group(self.estimate, group)
        return plot_scatter(
            self.reference,
            self.estimate,
            name,
            site_ids,
            variables=variables,
            reference_label=reference_label,
            estimate_label=estimate_label,
            **kwargs,
        )

    def plot_timeseries(
        self,
        site: str,
        *,
        variables: Iterable[str] | None = None,
        reference_label: str = "Reference",
        estimate_label: str = "Estimate",
        **kwargs: Any,
    ) -> Figure:
        """Line plot of estimate vs reference at one site (see :func:`plot_timeseries`)."""
        return plot_timeseries(
            self.reference,
            self.estimate,
            site,
            variables=variables,
            reference_label=reference_label,
            estimate_label=estimate_label,
            **kwargs,
        )

    def plot_all(
        self,
        *,
        variables: Iterable[str] | None = None,
        groups: Iterable[str] | None = None,
        reference_label: str = "Reference",
        estimate_label: str = "Estimate",
        **kwargs: Any,
    ) -> list[Figure]:
        """Run :meth:`plot` for every validation group (or a *groups* subset)."""
        active = self._groups
        if groups is not None:
            wanted = set(groups)
            active = [g for g in active if g[0] in wanted]
        return [
            plot_group(
                self.reference,
                self.estimate,
                name,
                site_ids,
                variables=variables,
                reference_label=reference_label,
                estimate_label=estimate_label,
                **kwargs,
            )
            for name, _, site_ids in active
        ]


HOLDOUT_ESTIMATE_LABEL = "Metamodel"
#: The model that ran the cases (``simulations.model``): the holdout reference
#: and the dynamic runs of notebooks 09-13.
HOLDOUT_REFERENCE_LABEL = model_label()
DYNAMIC_REFERENCE_LABEL = f"Dynamic {model_label()}"


@dataclass
class HoldoutValidationResult(ComparisonResult):
    """Holdout metamodel validation for one goal, metamodel set and MDA train size."""

    goal: str = ""
    metamodel: str = ""
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
        reference_label: str | None = None,
        estimate_label: str | None = None,
        **kwargs: Any,
    ) -> Figure:
        """As :meth:`ComparisonResult.plot`, defaulting labels to Metamodel and the case model."""
        return super().plot(
            group,
            variables=variables,
            reference_label=reference_label or self._default_reference_label,
            estimate_label=estimate_label or self._default_estimate_label,
            **kwargs,
        )

    def plot_scatter(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        reference_label: str | None = None,
        estimate_label: str | None = None,
        **kwargs: Any,
    ) -> Figure:
        """As :meth:`ComparisonResult.plot_scatter`, defaulting labels to Metamodel and the case model."""
        return super().plot_scatter(
            group,
            variables=variables,
            reference_label=reference_label or self._default_reference_label,
            estimate_label=estimate_label or self._default_estimate_label,
            **kwargs,
        )

    def plot_all(
        self,
        *,
        variables: Iterable[str] | None = None,
        groups: Iterable[str] | None = None,
        reference_label: str | None = None,
        estimate_label: str | None = None,
        **kwargs: Any,
    ) -> list[Figure]:
        """As :meth:`ComparisonResult.plot_all`, defaulting labels to Metamodel and the case model."""
        return super().plot_all(
            variables=variables,
            groups=groups,
            reference_label=reference_label or self._default_reference_label,
            estimate_label=estimate_label or self._default_estimate_label,
            **kwargs,
        )


def _holdout_pair(
    goal: str,
    metamodel: str,
    mda_candidate: int,
    *,
    context: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
    pca_variance: mm.PcaVariance = None,
) -> tuple[xr.Dataset, xr.Dataset, int]:
    """Model reference and metamodel estimate on MDA test cases.

    Returns reference, estimate (both ``case_num`` × ``sites``), and test count.
    Dry reference cases stay NaN, so the metrics skip them.
    """
    if context is None:
        context = mm.load_context(goal)
    test_idx = context["test_idx"]
    if models is None:
        models = mm.load_models(
            goal, metamodel, mda_candidate, pca_variance=pca_variance
        )

    cases = mm.load_cases_goal(goal, fill_dry=False)
    cases = cases.sel(sites=mm.set_sites(cases, metamodel))
    idx = np.asarray(test_idx, dtype=int)
    estimate = mm.predict(
        goal,
        mda_candidate,
        mm.forcing_dataframe(cases).iloc[idx],
        metamodel=metamodel,
        models=models,
        pca_variance=pca_variance,
    )
    configured = _metrics_variables(None)
    vars_ = _available_metrics_variables(estimate, cases, variables=None)
    missing = [v for v in configured if v in PARAMETERS and v not in vars_]
    if missing:
        logger.warning(
            "Goal %s / %s / n=%d: configured metrics variables missing from "
            "estimate or reference: %s",
            goal,
            metamodel,
            mda_candidate,
            missing,
        )
    reference = (
        cases.isel(case_num=idx)[vars_]
        .assign(**{name: cases[name] for name in SITE_META if name in cases})
        .sel(sites=estimate.sites.values)
    )
    return reference, estimate, len(test_idx)


def compare(
    reference: xr.Dataset,
    estimate: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
    pca_variance: mm.PcaVariance = None,
) -> ComparisonResult:
    """
    Compare aligned reference and estimate datasets.

    Parameters
    ----------
    reference, estimate
        Datasets sharing ``sites`` (and ``time`` or ``case_num``). The
        reference is what is trusted (e.g. dynamic SnapWave, a buoy).
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
    table = metrics_table(reference, estimate, variables=variables)
    table.insert(0, "pca_variances", str(mm.resolved_pca_variances(pca_variance)))
    table["reference_label"] = reference_label
    table["estimate_label"] = estimate_label
    return ComparisonResult(
        reference=reference,
        estimate=estimate,
        metrics=table,
        _groups=metric_groups(estimate),
    )


def _tag_metrics(
    table: pd.DataFrame,
    *,
    goal: str,
    metamodel: str,
    n: int,
    n_test: int,
    pca_variance: mm.PcaVariance,
    estimate_label: str = HOLDOUT_ESTIMATE_LABEL,
    reference_label: str = HOLDOUT_REFERENCE_LABEL,
) -> pd.DataFrame:
    """Insert holdout identifying columns (goal, set, n, labels, …) into a metrics table in place."""
    table.insert(0, "goal", goal)
    table.insert(1, "metamodel", metamodel)
    table.insert(2, "mda_candidate", n)
    table.insert(3, "n_test_cases", n_test)
    table.insert(4, "pca_variances", str(mm.resolved_pca_variances(pca_variance)))
    table["reference_label"] = reference_label
    table["estimate_label"] = estimate_label
    return table


def validate(
    goal: str | int,
    n: int,
    *,
    metamodel: str,
    variables: Iterable[str] | None = None,
    pca_variance: mm.PcaVariance = None,
) -> HoldoutValidationResult:
    """
    Holdout validation for one goal, metamodel set and MDA train size.

    Parameters
    ----------
    goal, n
        Goal id and MDA train size (saved models must exist).
    metamodel
        Metamodel set.
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
    reference, estimate, n_test = _holdout_pair(
        goal, metamodel, n, pca_variance=pca_variance
    )
    table = _tag_metrics(
        metrics_table(reference, estimate, variables=variables),
        goal=goal,
        metamodel=metamodel,
        n=n,
        n_test=n_test,
        pca_variance=pca_variance,
    )
    return HoldoutValidationResult(
        goal=goal,
        metamodel=metamodel,
        mda_candidate=n,
        reference=reference,
        estimate=estimate,
        metrics=table,
        n_test_cases=n_test,
        _groups=metric_groups(estimate),
    )


def sweep(
    goal: str | int,
    *,
    metamodel: str,
    candidates: Iterable[int] | None = None,
    variables: Iterable[str] | None = None,
    pca_variance: mm.PcaVariance = None,
    save: bool = True,
) -> pd.DataFrame:
    """Holdout metrics for every MDA candidate size on one goal and metamodel set.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON (fitted models must exist).
    metamodel
        Metamodel set.
    candidates
        MDA train sizes to validate; default ``mda_candidates`` from config.
    variables
        Metrics variables to score.
    pca_variance
        PCA fractions used at fit time.
    save
        Write the concatenated table to :func:`validation_metrics_path`.

    Returns
    -------
    pandas.DataFrame
        One :func:`validate` metrics table per candidate, concatenated.
    """
    goal = str(goal)
    cand = [int(c) for c in (candidates or MDA_CANDIDATES)]
    context = mm.load_context(goal)

    frames: list[pd.DataFrame] = []
    for n in cand:
        logger.info("Goal %s / %s: validating n=%d", goal, metamodel, n)
        reference, estimate, n_test = _holdout_pair(
            goal, metamodel, n, context=context, pca_variance=pca_variance
        )
        frames.append(
            _tag_metrics(
                metrics_table(reference, estimate, variables=variables),
                goal=goal,
                metamodel=metamodel,
                n=n,
                n_test=n_test,
                pca_variance=pca_variance,
            )
        )

    out = pd.concat(frames, ignore_index=True)
    if save:
        path = validation_metrics_path(goal, metamodel)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        logger.info("Wrote %s", path)
    return out


def sweep_all(
    goals: Iterable[str | int] | None = None,
    *,
    metamodels: Iterable[str] | None = None,
    candidates: Iterable[int] | None = None,
    variables: Iterable[str] | None = None,
    pca_variance: mm.PcaVariance = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run :func:`sweep` for every goal and metamodel set and concatenate.

    Parameters
    ----------
    goals
        Goals to sweep; default all configured goals.
    metamodels
        Metamodel sets to sweep; default every set in ``metamodel.sets``.
    candidates
        MDA train sizes to validate; default ``mda_candidates`` from config.
    variables
        Metrics variables to score.
    pca_variance
        PCA fractions used at fit time.
    save
        Write each goal's table to :func:`validation_metrics_path`.

    Returns
    -------
    pandas.DataFrame
        All goals' :func:`sweep` tables concatenated.
    """
    active = [str(g) for g in (goals or load_goals().keys())]
    frames = [
        sweep(
            goal,
            metamodel=metamodel,
            candidates=candidates,
            variables=variables,
            pca_variance=pca_variance,
            save=save,
        )
        for goal in active
        for metamodel in metamodels or mm.SETS
    ]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _paired_values(
    reference: xr.Dataset,
    estimate: xr.Dataset,
    variable: str,
    site_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Flat, coincident-finite ``(reference, estimate)`` arrays for one variable and site subset."""
    ref = reference.sel(sites=site_ids)[variable].values
    est = estimate.sel(sites=site_ids)[variable].values
    return _coincident_values(ref, est, variable=variable)


MAX_DENSITY_POINTS = 5000
"""Points drawn per scatter panel (random subsample; metrics always use every pair)."""


def _scatter_panel(
    ax: plt.Axes,
    reference: np.ndarray,
    estimate: np.ndarray,
    variable: str,
    *,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
    cmap: str = "viridis",
) -> None:
    """Draw one density scatter panel: reference on x, estimate on y.

    Delegates to :func:`bluemath_tk.core.plotting.scatter.validation_scatter`
    after dropping pairs outside :data:`PHYSICAL_BOUNDS`, so the annotated
    metrics match :func:`compute_metrics`.

    Parameters
    ----------
    ax
        Target axes; drawn on in place.
    reference, estimate
        Paired values for one variable.
    variable
        Key into :data:`PARAMETERS` (selects label, units, circular handling).
    reference_label, estimate_label
        Axis label prefixes (x and y).
    cmap
        Colormap for the density-colored scatter.
    """
    spec = PARAMETERS[variable]
    ref, est = _coincident_values(reference, estimate, variable=variable)
    validation_scatter(
        ax,
        ref,
        est,
        f"{reference_label} {spec['label']}",
        f"{estimate_label} {spec['label']}",
        spec["label"],
        units=spec["units"],
        circular=bool(spec["circular"]),
        metrics=SCATTER_METRICS,
        qq=False,
        cmap=cmap,
        s=10,
        max_points=MAX_DENSITY_POINTS,
    )



def plot_scatter(
    reference: xr.Dataset,
    estimate: xr.Dataset,
    group_name: str,
    site_ids: list[str],
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Density scatter panels for one validation group (reference on x).

    Parameters
    ----------
    reference, estimate
        Aligned datasets with ``sites`` and wave variables.
    group_name
        Title for the figure.
    site_ids
        Sites in this group (see :func:`utils.geometry.metric_groups`).
    variables
        Metrics variables to plot; default from config.
    reference_label, estimate_label
        Axis labels (x and y) for each scatter panel.
    figsize
        Figure size; default scales with the panel grid.
    cmap
        Colormap for the density-colored scatter.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per variable, up to a 2x2 (or larger) grid.
    """
    keys = _available_metrics_variables(estimate, reference, variables)
    nrows, ncols = scatter_grid_shape(len(keys))
    if figsize is None:
        figsize = (4.2 * ncols, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes_flat, keys):
        ref, est = _paired_values(reference, estimate, key, site_ids)
        _scatter_panel(
            ax,
            ref,
            est,
            key,
            reference_label=reference_label,
            estimate_label=estimate_label,
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
    """Map of validation sites coloured by source / depth label.

    Parameters
    ----------
    estimate
        Dataset carrying ``SITE_META`` coordinates (``site_longitude``,
        ``site_latitude``, ``site_source``, ``site_label``).
    site_ids
        Sites to plot.
    group_name
        Used as the map title when *title* is not given.
    title
        Explicit map title; overrides *group_name*.
    ax
        Existing axes to draw on; a new base map is created if omitted.

    Returns
    -------
    fig, ax
        Figure and axes with the site map drawn.
    """
    from utils.viz import plot_base_map, plot_validation_sites

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
    reference: xr.Dataset,
    estimate: xr.Dataset,
    group_name: str,
    site_ids: list[str],
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
    figsize: tuple[float, float] | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Map + density scatters for one validation group.

    Combines :func:`plot_map` (top row) with the :func:`plot_scatter` panels
    (below) in one figure.

    Parameters
    ----------
    reference, estimate
        Aligned datasets with ``sites`` and wave variables.
    group_name
        Title for the site map.
    site_ids
        Sites in this group.
    variables
        Metrics variables to plot; default from config.
    reference_label, estimate_label
        Axis labels (x and y) for each scatter panel.
    figsize
        Figure size; default scales with the panel grid.
    cmap
        Colormap for the density-colored scatter.

    Returns
    -------
    matplotlib.figure.Figure
        Site map above a grid of per-variable scatter panels.
    """
    from utils.viz import plot_base_map, plot_validation_sites

    keys = _available_metrics_variables(estimate, reference, variables)
    scatter_rows, scatter_cols = scatter_grid_shape(len(keys))
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
        ref, est = _paired_values(reference, estimate, key, site_ids)
        _scatter_panel(
            ax,
            ref,
            est,
            key,
            reference_label=reference_label,
            estimate_label=estimate_label,
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
    reference: xr.Dataset,
    estimate: xr.Dataset,
    site: str,
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Line plot of reference vs estimate at one site.

    Parameters
    ----------
    reference, estimate
        Aligned datasets with ``sites`` and wave variables on a ``time`` axis.
    site
        Site label to plot.
    variables
        Metrics variables to plot; default from config.
    reference_label, estimate_label
        Line labels.
    figsize
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        One row per variable, sharing the time axis.
    """
    return plot_buoy_timeseries(
        reference,
        {estimate_label: estimate},
        site,
        variables=variables,
        reference_label=reference_label,
        figsize=figsize,
    )


def plot_buoy_timeseries(
    reference: xr.Dataset,
    estimates: Mapping[str, xr.Dataset],
    site: str,
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Buoy",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Time series at one site: reference vs one or more model estimates.

    Overlays several named model datasets on the same reference — e.g. bulk
    metamodel, spectral metamodel, dynamic SnapWave, closest-goal offshore
    hindcast against a buoy (see the ``10*`` buoy-validation notebooks). Each
    row is drawn by
    :func:`bluemath_tk.core.plotting.timeseries.plot_timeseries_comparison`,
    so estimates are restricted to the reference's time stamps and the x-axis
    is cropped to where the reference has data.

    Parameters
    ----------
    reference
        Reference dataset (e.g. buoy observations) with ``sites`` and wave
        variables.
    estimates
        Model name → dataset; only variables present in *every* estimate
        (and in *reference*) are plotted.
    site
        Site label to plot.
    variables
        Metrics variables to plot; default from config.
    reference_label
        Reference series label.
    figsize
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        One row per variable; circular variables (``dir``) use scatter markers.
    """
    site = str(site)
    keys = _metrics_variables(variables)
    keys = [k for k in keys if k in PARAMETERS and k in reference.data_vars]
    keys = [k for k in keys if all(k in est.data_vars for est in estimates.values())]
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    axes = np.atleast_1d(axes)

    taken = {_MODEL_COLORS[name] for name in estimates if name in _MODEL_COLORS}
    free = (f"C{i}" for i in range(20) if f"C{i}" not in taken)
    colors = {name: _MODEL_COLORS.get(name) or next(free) for name in estimates}
    ref_site = reference.sel(sites=site)
    for ax, key in zip(axes, keys, strict=True):
        spec = PARAMETERS[key]
        plot_timeseries_comparison(
            ax,
            ref_site[key],
            {name: est.sel(sites=site)[key] for name, est in estimates.items()},
            reference_label,
            colors=colors,
            reference_color=_MODEL_COLORS.get(reference_label, "black"),
            circular=bool(spec["circular"]),
            ylabel=f"{spec['label']} [{spec['units']}]",
        )

    axes[-1].set_xlabel("Time")
    fig.suptitle(site, fontsize=11)
    fig.tight_layout()
    return fig



# Kept separate from _MODEL_COLORS (used by plot_timeseries/plot_buoy_timeseries
# above) rather than repointing it — those keep matplotlib's default C0/C1/...
# cycle elsewhere; these hex values are the ones the GUI's own charts use
# (GUI/hywaves_gui/colors.py MODEL_COLORS), so a buoy's summary PNG reads as
# the same colors as the boxplots around it once embedded in the GUI.
_BUOY_SUMMARY_COLORS: dict[str, str] = {
    "Buoy": "#111111",
    "Bulk metamodel": "#22d3ee",
    "Spectral metamodel": "#a78bfa",
    "WHACS (closest goal)": "#fb923c",
}


def buoy_summary_figure_path(site: str) -> Path:
    """PNG path for :func:`plot_buoy_summary` on one buoy site."""
    return resolve("outputs/buoy_validation/figures") / f"{site}.png"


def plot_buoy_summary(
    reference: xr.Dataset,
    estimates: Mapping[str, xr.Dataset],
    site: str,
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Buoy",
    estimate_label: str = "Spectral metamodel",
    period_label: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """One combined figure per buoy: density scatters on top, one wide
    Hs/Tp time series below — both buoy vs. *estimate_label* only.

    Meant to be saved once per site (see ``10a``/``10b``) and shown as a
    static image in the GUI's Buoy Validation tab. Deliberately narrower
    than a full model comparison (see :func:`plot_buoy_timeseries` /
    :func:`plot_scatter` for that): this is a fast per-buoy skim of the
    production model (spectral metamodel) against the actual observation.

    Parameters
    ----------
    reference
        Buoy observations, ``(time, sites)``.
    estimates
        Model name -> aligned-shape dataset; only ``estimate_label`` is used.
    site
        SnapWave site id (e.g. ``"buoy_Sydney_Buoy"``).
    variables
        Subset of metrics variables (scatter row only); default from config.
    estimate_label
        Which model in *estimates* to compare against — the time series row
        is always just this one model vs. the buoy, not every model.
    period_label
        Human-readable period, drawn into the figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    site = str(site)
    if estimate_label not in estimates:
        raise KeyError(f"{estimate_label!r} not in estimates for site {site}: {sorted(estimates)}")
    estimate = estimates[estimate_label]

    keys = _metrics_variables(variables)
    keys = [k for k in keys if k in PARAMETERS and k in reference.data_vars and k in estimate.data_vars]
    if not keys:
        raise ValueError(f"{site}: no metrics variable present in both buoy and {estimate_label}")

    ncols = len(keys)
    if figsize is None:
        figsize = (4.2 * ncols, 7.4)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, ncols, height_ratios=[1.15, 0.85], hspace=0.35)

    ref_site = reference.sel(sites=site)
    est_site = estimate.sel(sites=site)
    colors = _BUOY_SUMMARY_COLORS

    # --- top row: one density scatter per variable, buoy vs estimate_label only ---
    for col, key in enumerate(keys):
        ax = fig.add_subplot(gs[0, col])
        ref_aligned, est_aligned = xr.align(ref_site[key], est_site[key], join="inner")
        _scatter_panel(
            ax,
            ref_aligned.values,
            est_aligned.values,
            key,
            reference_label=reference_label,
            estimate_label=estimate_label,
            cmap=cmap,
        )

    # --- bottom row: one wide Hs (left axis) / Tp (right axis) time series,
    # cropped to the span where the buoy actually has data (not the full
    # nominal period, which for most sites is mostly gaps) ---
    ax_hs = fig.add_subplot(gs[1, :])
    window: tuple[np.datetime64, np.datetime64] | None = None

    if "hs" in keys:
        hs_ref, hs_est = xr.align(ref_site["hs"], est_site["hs"], join="inner")
        finite = np.isfinite(hs_ref.values)
        if finite.any():
            times = hs_ref.time.values[finite]
            window = (times.min(), times.max())
            hs_ref = hs_ref.sel(time=slice(*window))
            hs_est = hs_est.sel(time=slice(*window))
        hs_ref.plot(ax=ax_hs, color=colors.get(reference_label, "black"), lw=1.1, label=f"{reference_label} Hs")
        hs_est.plot(ax=ax_hs, color=colors.get(estimate_label, "gray"), lw=1.0, alpha=0.9, label=f"{estimate_label} Hs")
        ax_hs.set_ylabel(f"Hs [{PARAMETERS['hs']['units']}]")
        ax_hs.set_title("")

    if "tp" in keys:
        ax_tp = ax_hs.twinx()
        tp_ref, tp_est = xr.align(ref_site["tp"], est_site["tp"], join="inner")
        if window is None:
            finite = np.isfinite(tp_ref.values)
            if finite.any():
                times = tp_ref.time.values[finite]
                window = (times.min(), times.max())
        if window is not None:
            tp_ref = tp_ref.sel(time=slice(*window))
            tp_est = tp_est.sel(time=slice(*window))
        tp_ref.plot(ax=ax_tp, color=colors.get(reference_label, "black"), lw=1.0, linestyle="--", alpha=0.8, label=f"{reference_label} Tp")
        tp_est.plot(ax=ax_tp, color=colors.get(estimate_label, "gray"), lw=0.9, linestyle="--", alpha=0.8, label=f"{estimate_label} Tp")
        ax_tp.set_ylabel(f"Tp [{PARAMETERS['tp']['units']}]")
        ax_tp.set_title("")

    if window is not None:
        ax_hs.set_xlim(*window)
    ax_hs.set_xlabel("Time")
    ax_hs.grid(True, alpha=0.3)
    handles, labels = ax_hs.get_legend_handles_labels()
    if "tp" in keys:
        h2, l2 = ax_tp.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax_hs.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)

    title = site if not period_label else f"{site}  —  {period_label}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# ---------------------------------------------------------------------------
# MDA sweep summary plots
# ---------------------------------------------------------------------------

SWEEP_STATS = ("bias", "rmse", "rrmse", "r2")
SWEEP_BY = ("bulk", "source", "group", "label", "buoy")


def load_sweep_metrics(goal: str | int) -> pd.DataFrame:
    """Load the ``validation_metrics.csv`` of every metamodel set of one goal.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Returns
    -------
    pandas.DataFrame
        Sweep metrics table with ``goal`` coerced to string.

    Raises
    ------
    FileNotFoundError
        If :func:`sweep` has not been run (and saved) for any set of this goal.
    """
    paths = [validation_metrics_path(goal, s) for s in mm.SETS]
    frames = [pd.read_csv(p) for p in paths if p.exists()]
    if not frames:
        raise FileNotFoundError(f"No sweep metrics for goal {goal}. Run sweep() first.")
    df = pd.concat(frames, ignore_index=True)
    df["goal"] = df["goal"].astype(str)
    return df


def _filter_sweep(
    metrics: pd.DataFrame,
    *,
    goals: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Subset a sweep metrics table to the requested goals and variables."""
    df = metrics.copy()
    df["goal"] = df["goal"].astype(str)
    if goals is not None:
        wanted = {str(g) for g in goals}
        df = df[df["goal"].isin(wanted)]
    vars_ = _metrics_variables(variables)
    return df[df["variable"].isin(vars_)].copy()


def _weighted_stats(group: pd.DataFrame) -> pd.Series:
    """Pool one sweep group's per-site-group metrics into one row (plus summed ``n``).

    ``bias`` (n-weighted mean) and ``rmse`` (root of the n-weighted mean
    square) are exact pooled values. Ratio/correlation metrics (``rrmse``,
    ``r2``) cannot be pooled without the raw pairs, so they are n-weighted
    means — an approximation. Metrics missing from older CSVs come out NaN.
    """
    w = group["n"].to_numpy(dtype=float)
    if w.sum() <= 0:
        return pd.Series({s: np.nan for s in SWEEP_STATS} | {"n": 0})
    out: dict[str, float] = {}
    for stat in SWEEP_STATS:
        if stat not in group:
            out[stat] = np.nan
        elif stat == "rmse":
            out[stat] = float(np.sqrt(np.average(group["rmse"] ** 2, weights=w)))
        else:
            out[stat] = float(np.average(group[stat], weights=w))
    out["n"] = int(w.sum())
    return pd.Series(out)


def aggregate_sweep(
    metrics: pd.DataFrame,
    *,
    by: str = "bulk",
    goals: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Collapse sweep rows for line plots.

    ``by`` options:

    - ``bulk`` — n-weighted mean over all groups of each metamodel set
    - ``source`` — per ``site_source`` (isobath / mesh / buoy)
    - ``group`` — per validation group (``5m``, ``10m``, ``Apia``, buoy id, …)
    - ``label`` — per ``site_label``
    - ``buoy`` — buoy groups only (same granularity as ``group``)

    Parameters
    ----------
    metrics
        Sweep metrics table (from :func:`sweep`, :func:`sweep_all`, or
        :func:`load_sweep_metrics`).
    by
        Grouping mode; one of :data:`SWEEP_BY`.
    goals
        Goals to keep; default all present in *metrics*.
    variables
        Metrics variables to keep; default from config.

    Returns
    -------
    pandas.DataFrame
        For ``group``/``buoy``, the filtered rows unchanged; otherwise one
        pooled row per ``(goal, metamodel, mda_candidate, ..., variable)`` group
        with one column per :data:`SWEEP_STATS` entry plus ``n`` (see
        :func:`_weighted_stats`).

    Raises
    ------
    ValueError
        If *by* is not one of :data:`SWEEP_BY`.
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
        return out.sort_values(
            ["goal", "metamodel", "group", "variable", "mda_candidate"]
        )

    key_cols = {
        "bulk": ["goal", "metamodel", "mda_candidate", "variable"],
        "source": ["goal", "metamodel", "mda_candidate", "site_source", "variable"],
        "label": ["goal", "metamodel", "mda_candidate", "site_label", "variable"],
    }[by]

    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(key_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(key_cols, keys, strict=True))
        row.update(_weighted_stats(group).to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key_cols)


def _sweep_line_color(name: str, source: str = "") -> str:
    """Line color for one sweep series: isobath/mesh palette, hashed for buoys, else default."""
    from utils.viz import DEFAULT_POINT_COLOR, ISOBATH_COLORS, MESH_COLORS

    if name in ISOBATH_COLORS:
        return ISOBATH_COLORS[name]
    if name in MESH_COLORS:
        return MESH_COLORS[name]
    if source == "buoy" or str(name).startswith("buoy_"):
        return plt.cm.tab10(abs(hash(name)) % 10)
    return DEFAULT_POINT_COLOR


def _sweep_series_key(row: pd.Series, by: str) -> str:
    """Series-grouping key for one sweep row under the given ``by`` mode."""
    if by == "bulk":
        return "all sites"
    if by == "source":
        return str(row["site_source"])
    if by == "label":
        return str(row["site_label"])
    return str(row["group"])


def _sweep_line_label(
    row: pd.Series, by: str, *, multi_goal: bool, multi_set: bool
) -> str:
    """Legend label for one sweep line, naming the goal / set when several are plotted."""
    name = str(row["metamodel"]) if by == "bulk" else _sweep_series_key(row, by)
    if by == "bulk" and not multi_set:
        name = "all sites"
    elif by != "bulk" and multi_set:
        name = f"{name} [{row['metamodel']}]"
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
    """Line plots of bias / RMSE / RRMSE / R² vs MDA size.

    Rows = metrics (those present in *metrics*; older CSVs may lack
    ``rrmse``), columns = variables (hs, tp, dir, spr). One line per series
    defined by ``by`` (bulk, source, group, label, or buoy).

    Parameters
    ----------
    metrics
        Sweep metrics table, aggregated internally via :func:`aggregate_sweep`.
    by
        Grouping mode; one of :data:`SWEEP_BY`.
    goals
        Goals to keep; default all present in *metrics*.
    variables
        Metrics variables (columns) to plot; default from config.
    figsize
        Figure size; default scales with the panel grid.
    title
        Figure title; default summarises *by* and the plotted goal(s).

    Returns
    -------
    matplotlib.figure.Figure
        One row per available :data:`SWEEP_STATS` metric by ``len(variables)``
        columns of line plots.

    Raises
    ------
    ValueError
        If no sweep rows remain after filtering by *goals*/*variables*.
    """
    vars_ = _metrics_variables(variables)
    agg = aggregate_sweep(metrics, by=by, goals=goals, variables=vars_)
    if agg.empty:
        raise ValueError("No sweep data to plot after filtering.")

    multi_goal = agg["goal"].nunique() > 1
    multi_set = agg["metamodel"].nunique() > 1
    stats = [s for s in SWEEP_STATS if s in agg and agg[s].notna().any()]
    nrows, ncols = len(stats), len(vars_)
    if figsize is None:
        figsize = (4.2 * ncols, 3.2 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    series_cols = ["goal", "metamodel"] + {
        "bulk": [],
        "source": ["site_source"],
        "label": ["site_label"],
        "group": ["group", "site_source"],
        "buoy": ["group"],
    }[by]

    for i, stat in enumerate(stats):
        for j, var in enumerate(vars_):
            ax = axes[i, j]
            sub = agg[agg["variable"] == var]
            for key, grp in sub.groupby(series_cols, sort=False):
                grp = grp.sort_values("mda_candidate")
                row0 = grp.iloc[0]
                label = _sweep_line_label(
                    row0, by, multi_goal=multi_goal, multi_set=multi_set
                )
                source = str(row0.get("site_source", ""))
                color = (
                    None  # default colour cycle: one colour per goal / set
                    if by == "bulk" and (multi_goal or multi_set)
                    else _sweep_line_color(_sweep_series_key(row0, by), source)
                )
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
            ylab = {
                "r2": "R²",
                "rrmse": "RRMSE [-]",
            }.get(stat, f"{stat.upper()} [{spec['units']}]")
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
    """One :func:`plot_sweep` figure per grouping mode.

    Parameters
    ----------
    metrics
        Sweep metrics table.
    goals
        Goals to keep; default all present in *metrics*.
    variables
        Metrics variables to plot; default from config.
    by
        Grouping modes to generate; default all of :data:`SWEEP_BY`.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        One figure per grouping mode, keyed by mode name.
    """
    modes = list(by or SWEEP_BY)
    return {
        mode: plot_sweep(metrics, by=mode, goals=goals, variables=variables)
        for mode in modes
    }


__all__ = [
    "ComparisonResult",
    "HoldoutValidationResult",
    "LOGGER_NAME",
    "MAX_DENSITY_POINTS",
    "METRICS",
    "SCATTER_METRICS",
    "SWEEP_BY",
    "SWEEP_STATS",
    "aggregate_sweep",
    "align_pair",
    "closest_goals_table",
    "compare",
    "compare_all",
    "compute_metrics",
    "load_offshore_hindcast_at_buoys",
    "load_sweep_metrics",
    "logger",
    "metric_groups",
    "metrics_summary",
    "metrics_table",
    "plot_buoy_timeseries",
    "plot_group",
    "plot_map",
    "plot_scatter",
    "plot_sweep",
    "plot_sweep_suite",
    "plot_timeseries",
    "resolve_comparison_variables",
    "sweep",
    "sweep_all",
    "validate",
    "validation_metrics_path",
]
