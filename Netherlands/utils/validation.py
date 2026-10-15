"""Compare wave datasets on the output sites: metrics tables and plots.

- Holdout validation of the metamodel (notebook 06): :func:`validate`,
  :func:`sweep`, :func:`sweep_all`, :func:`plot_sweep`.
- Any two datasets on ``(time | case_num, sites)``: :func:`compare`,
  :func:`compare_all` (e.g. linear summation vs the dynamic runs in 09, models
  vs buoys in 10), :func:`load_offshore_hindcast_at_buoys`.

Convention (:mod:`bluemath_tk.core.metrics`): the **reference** comes first,
the **estimate** second; bias = estimate − reference (positive =
overestimation); scatters put the reference on the x-axis. Metrics are per
site group (:func:`utils.geometry.metric_groups`): each isobath, mesh polygon
and buoy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.geo import geodesic_distance
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.metrics import DEFAULT_METRICS, compute_metrics as bm_compute_metrics
from bluemath_tk.core.metrics import pool_metrics
from bluemath_tk.core.plotting.scatter import scatter_grid_shape, validation_scatter
from bluemath_tk.core.plotting.timeseries import plot_timeseries_comparison
from bluemath_tk.waves.hywaves.plots import plot_mda_sweep
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
)
from utils.geometry import attach_site_meta, find_group, metric_groups, per_site_coord
from utils import metamodel as mm
from utils import onshore_sites as obs

LOGGER_NAME = "hywaves.validation"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("logs/validation.log")),
    level="INFO",
    console=True,
    console_level="WARNING",
)

METRICS_VARIABLES = METAMODEL["validation"]["metrics_variables"]

PARAMETERS: dict[str, dict[str, Any]] = {
    "hs": {"label": "Hs", "units": "m", "circular": False},
    "tp": {"label": "Tp", "units": "s", "circular": False},
    "dir": {"label": "Dir", "units": "deg", "circular": True},
    "spr": {"label": "Spr", "units": "deg", "circular": False},
}

# Generous physical ceilings that only catch corrupted-but-finite model
# output (a SnapWave wetting/drying instability once gave hs ~ 1e13 m at a few
# isobath sites and hours, enough to wreck a whole group's rmse/r2). Real
# storms here stay far below 15 m.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {"hs": (0.0, 15.0)}

#: Metric columns of every table (plus ``n``); NaN where a circular variable has no definition.
METRICS: tuple[str, ...] = DEFAULT_METRICS
#: Metrics annotated on scatter panels.
SCATTER_METRICS: tuple[str, ...] = ("bias", "rmse", "rrmse", "r2")
#: Points drawn per scatter panel (the metrics always use every pair).
MAX_DENSITY_POINTS = 5000

HOLDOUT_ESTIMATE_LABEL = "Metamodel"
#: The model that ran the cases: the holdout reference and the dynamic runs.
HOLDOUT_REFERENCE_LABEL = model_label()
DYNAMIC_REFERENCE_LABEL = f"Dynamic {model_label()}"


def _variables(variables: Iterable[str] | None) -> list[str]:
    """*variables*, default ``metamodel.validation.metrics_variables``."""
    return list(variables) if variables is not None else list(METRICS_VARIABLES)


def resolve_comparison_variables(
    *datasets: xr.Dataset, variables: Iterable[str] | None = None
) -> list[str]:
    """
    The requested variables present in every dataset (in request order).

    Raises
    ------
    ValueError
        If none is shared.
    """
    available = [
        v for v in _variables(variables)
        if v in PARAMETERS and all(v in ds.data_vars for ds in datasets)
    ]
    if not available:
        raise ValueError(f"No variable of {_variables(variables)} is in every dataset")
    return available


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(reference: np.ndarray, estimate: np.ndarray, variable: str) -> dict[str, float]:
    """
    Metrics of one variable (reference first), see :func:`bluemath_tk.core.metrics.compute_metrics`.

    Directions are circular, and pairs outside :data:`PHYSICAL_BOUNDS` are dropped.
    """
    return bm_compute_metrics(
        reference,
        estimate,
        METRICS,
        circular=PARAMETERS[variable]["circular"],
        bounds=PHYSICAL_BOUNDS.get(variable),
    )


def metrics_table(
    reference: xr.Dataset, estimate: xr.Dataset, variables: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Metrics per site group and variable.

    Parameters
    ----------
    reference, estimate
        Aligned datasets on ``sites`` with the site metadata coordinates.
    variables
        Variables to score; default the configured ones present in both.

    Returns
    -------
    pandas.DataFrame
        ``group``, ``site_source``, ``site_label``, ``variable``, ``label``,
        ``n`` and one column per :data:`METRICS` entry.
    """
    labels = per_site_coord(estimate, "site_label")
    rows = [
        {
            "group": group,
            "site_source": source,
            "site_label": group if source != "buoy" else labels.get(site_ids[0], group),
            "variable": var,
            "label": PARAMETERS[var]["label"],
            **compute_metrics(
                reference[var].sel(sites=site_ids).values,
                estimate[var].sel(sites=site_ids).values,
                var,
            ),
        }
        for group, source, site_ids in metric_groups(estimate)
        for var in resolve_comparison_variables(reference, estimate, variables=variables)
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    """Aligned reference and estimate, their metrics table and plots per site group."""

    reference: xr.Dataset
    estimate: xr.Dataset
    metrics: pd.DataFrame
    reference_label: str = "Reference"
    estimate_label: str = "Estimate"

    @property
    def groups(self) -> list[str]:
        """Site group names (see :func:`utils.geometry.metric_groups`)."""
        return [name for name, _, _ in metric_groups(self.estimate)]

    def site_ids(self, group: str) -> list[str]:
        """Sites of one group."""
        return find_group(self.estimate, group)[2]

    def plot(self, group: str, variables: Iterable[str] | None = None) -> Figure:
        """Site map and density scatters of one group."""
        return plot_group(
            self.reference, self.estimate, group, self.site_ids(group),
            variables=variables, reference_label=self.reference_label,
            estimate_label=self.estimate_label,
        )

    def plot_scatter(self, group: str, variables: Iterable[str] | None = None) -> Figure:
        """Density scatters of one group."""
        return plot_scatter(
            self.reference, self.estimate, group, self.site_ids(group),
            variables=variables, reference_label=self.reference_label,
            estimate_label=self.estimate_label,
        )

    def plot_all(self, variables: Iterable[str] | None = None) -> list[Figure]:
        """:meth:`plot` for every group."""
        return [self.plot(group, variables) for group in self.groups]

    def plot_timeseries(self, site: str, variables: Iterable[str] | None = None) -> Figure:
        """Reference and estimate time series at one site."""
        return plot_buoy_timeseries(
            self.reference, {self.estimate_label: self.estimate}, site,
            variables=variables, reference_label=self.reference_label,
        )


def align_pair(
    reference: xr.Dataset, estimate: xr.Dataset
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Inner join of *reference* and *estimate* on their shared sites and times.

    Raises
    ------
    ValueError
        If they share no site or no time step.
    """
    in_reference = set(reference.sites.values)
    shared = [s for s in estimate.sites.values if s in in_reference]
    if not shared:
        raise ValueError("No shared sites between estimate and reference")
    reference, estimate = xr.align(
        reference.sel(sites=shared), estimate.sel(sites=shared), join="inner"
    )
    if "time" in estimate.dims and estimate.sizes["time"] == 0:
        raise ValueError("No overlapping time steps")
    return reference, estimate


def compare(
    reference: xr.Dataset,
    estimate: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
) -> Comparison:
    """
    Metrics of *estimate* against *reference* on their shared sites and times.

    Parameters
    ----------
    reference, estimate
        Datasets on ``sites`` (and ``time`` or ``case_num``); the estimate
        needs the site metadata coordinates (groups).
    variables
        Variables to score.
    reference_label, estimate_label
        Names used in the plots.

    Returns
    -------
    Comparison
    """
    reference, estimate = align_pair(reference, estimate)
    return Comparison(
        reference, estimate, metrics_table(reference, estimate, variables),
        reference_label, estimate_label,
    )


def compare_all(
    reference: xr.Dataset,
    estimates: Mapping[str, xr.Dataset],
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Buoy",
) -> dict[str, Comparison]:
    """:func:`compare` each named estimate with the same reference."""
    return {
        name: compare(
            reference, est, variables=variables,
            reference_label=reference_label, estimate_label=name,
        )
        for name, est in estimates.items()
    }


def metrics_summary(results: Mapping[str, Comparison]) -> pd.DataFrame:
    """The metrics tables of :func:`compare_all`, stacked with a ``model`` column."""
    return pd.concat(
        [r.metrics.assign(model=name) for name, r in results.items()], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Holdout validation (notebook 06)
# ---------------------------------------------------------------------------


def validation_metrics_path(goal: str | int, metamodel: str) -> Path:
    """``outputs/goal_XX/{set}/validation_metrics.csv`` (contract §4)."""
    return mm.goal_output_dir(goal) / metamodel / "validation_metrics.csv"


def validate(
    goal: str | int, n: int, *, metamodel: str, variables: Iterable[str] | None = None
) -> Comparison:
    """
    Metamodel against the model on the holdout cases of one goal, set and MDA size.

    The reference keeps ``hs = 0`` where no energy reaches a site; its other
    fields are NaN there and skipped.

    Returns
    -------
    Comparison
        Metrics with ``goal``, ``metamodel``, ``mda_candidate`` and
        ``n_test_cases`` columns.
    """
    goal = str(goal)
    test_idx = mm.load_split(goal)[1]
    cases = mm.load_cases_goal(goal, fill_zero=False).isel(case_num=test_idx)
    estimate = mm.predict(goal, n, mm.forcing_dataframe(cases), metamodel=metamodel)
    reference = cases[resolve_comparison_variables(cases, estimate, variables=variables)]
    result = compare(
        reference.sel(sites=estimate.sites.values), estimate, variables=variables,
        reference_label=HOLDOUT_REFERENCE_LABEL, estimate_label=HOLDOUT_ESTIMATE_LABEL,
    )
    for i, (name, value) in enumerate(
        {"goal": goal, "metamodel": metamodel, "mda_candidate": int(n),
         "n_test_cases": len(test_idx)}.items()
    ):
        result.metrics.insert(i, name, value)
    return result


def sweep(
    goal: str | int,
    *,
    metamodel: str,
    candidates: Iterable[int] | None = None,
    variables: Iterable[str] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    :func:`validate` every MDA size of one goal and set.

    Parameters
    ----------
    goal, metamodel
        Goal id and metamodel set.
    candidates
        MDA sizes; default ``mda_candidates``.
    variables
        Variables to score.
    save
        Write :func:`validation_metrics_path`.

    Returns
    -------
    pandas.DataFrame
        The metrics tables, stacked.
    """
    out = pd.concat(
        [
            validate(goal, n, metamodel=metamodel, variables=variables).metrics
            for n in (candidates or mm.MDA_CANDIDATES)
        ],
        ignore_index=True,
    )
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
    **kwargs: Any,
) -> pd.DataFrame:
    """:func:`sweep` every goal and set (default all); *kwargs* as for :func:`sweep`."""
    return pd.concat(
        [
            sweep(goal, metamodel=metamodel, **kwargs)
            for goal in mm._goals(goals)
            for metamodel in metamodels or mm.SETS
        ],
        ignore_index=True,
    )


def load_sweep_metrics(goal: str | int) -> pd.DataFrame:
    """The saved :func:`sweep` tables of every set of one goal."""
    paths = [p for s in mm.SETS if (p := validation_metrics_path(goal, s)).exists()]
    if not paths:
        raise FileNotFoundError(f"No sweep metrics for goal {goal}; run sweep() first.")
    return pd.concat([pd.read_csv(p, dtype={"goal": str}) for p in paths], ignore_index=True)


#: Sweep grouping modes: the columns that make one line.
SWEEP_BY: dict[str, list[str]] = {
    "bulk": [],
    "source": ["site_source"],
    "label": ["site_label"],
    "group": ["group"],
    "buoy": ["group"],
}


def aggregate_sweep(
    metrics: pd.DataFrame,
    *,
    by: str = "bulk",
    goals: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Sweep metrics pooled per goal, set, MDA size, variable and *by* column.

    *by* (:data:`SWEEP_BY`): ``bulk`` (all sites of a set), ``source``,
    ``label``, ``group`` or ``buoy`` (buoy groups only); pooled with
    :func:`bluemath_tk.core.metrics.pool_metrics` (n-weighted).
    """
    df = metrics.astype({"goal": str})
    df = df[df["variable"].isin(_variables(variables))]
    if goals is not None:
        df = df[df["goal"].isin({str(g) for g in goals})]
    if by == "buoy":
        df = df[df["site_source"] == "buoy"]
    keys = ["goal", "metamodel", "mda_candidate", "variable", *SWEEP_BY[by]]
    return pool_metrics(df, keys)


def plot_sweep(
    metrics: pd.DataFrame,
    *,
    by: str = "bulk",
    goals: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Holdout metrics against the MDA size (:func:`bluemath_tk.waves.hywaves.plots.plot_mda_sweep`)."""
    from utils.viz import site_group_colors

    table = aggregate_sweep(metrics, by=by, goals=goals, variables=variables)
    series = [c for c in ("goal", "metamodel") if table[c].nunique() > 1] + SWEEP_BY[by]
    variables = [v for v in _variables(variables) if v in set(table["variable"])]
    return plot_mda_sweep(
        table,
        series or ["metamodel"],
        variables,
        units={v: PARAMETERS[v]["units"] for v in variables},
        colors=site_group_colors(),
        title=title or f"MDA sweep — {by} — goal(s) {', '.join(sorted(table['goal'].unique()))}",
    )


# ---------------------------------------------------------------------------
# Offshore hindcast at the buoys and output paths (notebooks 09, 10)
# ---------------------------------------------------------------------------


def _slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-")


def metamodel_validation_path(period: str, metamodel: str) -> Path:
    """``outputs/metamodel_validation/linear_summation_{set}_{period}.csv`` (notebook 09)."""
    return resolve("outputs/metamodel_validation") / f"linear_summation_{metamodel}_{_slug(period)}.csv"


def buoy_validation_path(name: str, period_label: str) -> Path:
    """``outputs/buoy_validation/{name}_summary_{period}.csv`` (notebook 10)."""
    return resolve("outputs/buoy_validation") / f"{name}_summary_{_slug(period_label)}.csv"


def buoy_summary_figure_path(site: str) -> Path:
    """``outputs/buoy_validation/figures/{site}.png`` (shown by the GUI)."""
    return resolve("outputs/buoy_validation/figures") / f"{site}.png"


def closest_goals_table(
    sites: Iterable[str], offshore_names: Iterable[str] | None = None
) -> pd.DataFrame:
    """
    Closest goal hindcast node of each buoy site, per offshore source.

    Returns
    -------
    pandas.DataFrame
        ``site``, ``offshore``, ``goal``, ``distance_m``, ``hindcast_lon``,
        ``hindcast_lat``.
    """
    goals = load_goals()
    coords = obs.get_coordinates()
    rows = []
    for name in offshore_names or [e["name"] for e in OFFSHORE_WAVES]:
        key = offshore_catalog_key(get_offshore_wave(name))
        nodes = {g: v["hindcast"][key] for g, v in goals.items() if key in v.get("hindcast", {})}
        for site in sites:
            buoy = coords[obs.instrument_id(site)]
            dist = {
                g: np.radians(geodesic_distance(buoy["lat"], buoy["lon"], n["latitude"], n["longitude"])) * 6_371_000.0
                for g, n in nodes.items()
            }
            goal = min(dist, key=dist.get)
            rows.append({
                "site": site, "offshore": name, "goal": int(goal),
                "distance_m": round(float(dist[goal]), 1),
                "hindcast_lon": nodes[goal]["longitude"], "hindcast_lat": nodes[goal]["latitude"],
            })
    return pd.DataFrame(rows)


def load_offshore_hindcast_at_buoys(
    time: str | slice,
    sites: Iterable[str],
    *,
    offshore_names: Iterable[str] | None = None,
    variables: Iterable[str] | None = None,
) -> tuple[dict[str, xr.Dataset], pd.DataFrame]:
    """
    Bulk offshore hindcast at each buoy's closest goal node.

    Returns
    -------
    estimates
        ``{"{offshore} (closest goal)": dataset on (time, sites)}``.
    mapping
        :func:`closest_goals_table`.
    """
    sites = list(sites)
    mapping = closest_goals_table(sites, offshore_names)
    estimates = {}
    for name, rows in mapping.groupby("offshore", sort=False):
        offshore = open_offshore_waves(name).sel(time=time)
        present = [v for v in _variables(variables) if v in offshore]
        ds = offshore[present].sel(goal=rows["goal"].to_numpy()).rename(goal="sites")
        ds = ds.assign_coords(sites=rows["site"].to_numpy()).transpose("time", "sites")
        estimates[f"{name} (closest goal)"] = attach_site_meta(ds.drop_vars("seapoint", errors="ignore")).load()
    return estimates, mapping


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _scatter_panel(
    ax: plt.Axes, reference: np.ndarray, estimate: np.ndarray, variable: str,
    reference_label: str, estimate_label: str,
) -> None:
    """One density scatter (reference on x), annotated like :func:`compute_metrics`."""
    spec = PARAMETERS[variable]
    validation_scatter(
        ax,
        reference,
        estimate,
        f"{reference_label} {spec['label']}",
        f"{estimate_label} {spec['label']}",
        spec["label"],
        units=spec["units"],
        circular=spec["circular"],
        metrics=SCATTER_METRICS,
        qq=False,
        cmap="viridis",
        s=10,
        max_points=MAX_DENSITY_POINTS,
        bounds=PHYSICAL_BOUNDS.get(variable),
    )


def plot_scatter(
    reference: xr.Dataset,
    estimate: xr.Dataset,
    title: str,
    site_ids: list[str],
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
) -> Figure:
    """Density scatters of the given sites, one panel per variable."""
    keys = resolve_comparison_variables(reference, estimate, variables=variables)
    nrows, ncols = scatter_grid_shape(len(keys))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for ax, key in zip(axes.ravel(), keys):
        _scatter_panel(
            ax, reference[key].sel(sites=site_ids).values, estimate[key].sel(sites=site_ids).values,
            key, reference_label, estimate_label,
        )
    for ax in axes.ravel()[len(keys):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def plot_group(
    reference: xr.Dataset,
    estimate: xr.Dataset,
    title: str,
    site_ids: list[str],
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Reference",
    estimate_label: str = "Estimate",
) -> Figure:
    """Map of the given sites above their density scatters."""
    from utils.viz import plot_base_map, plot_validation_sites

    keys = resolve_comparison_variables(reference, estimate, variables=variables)
    nrows, ncols = scatter_grid_shape(len(keys))
    fig = plt.figure(figsize=(4.5 * ncols, 3.0 + 3.5 * nrows))
    gs = fig.add_gridspec(1 + nrows, ncols, height_ratios=[1.25] + [1.0] * nrows)
    ax_map = fig.add_subplot(gs[0, :])
    plot_base_map(ax=ax_map, title=None)
    plot_validation_sites(estimate.sel(sites=site_ids), ax_map)
    ax_map.set_title(title, fontsize=11)
    for i, key in enumerate(keys):
        _scatter_panel(
            fig.add_subplot(gs[1 + i // ncols, i % ncols]),
            reference[key].sel(sites=site_ids).values, estimate[key].sel(sites=site_ids).values,
            key, reference_label, estimate_label,
        )
    fig.tight_layout()
    return fig


#: Line colours of the named estimates in the buoy figures (as the GUI's charts).
MODEL_COLORS: dict[str, str] = {
    "Buoy": "#111111",
    "Bulk metamodel": "#22d3ee",
    "Spectral metamodel": "#a78bfa",
    "Spectral metamodel (wind-corrected)": "#7c3aed",
    "Bulk metamodel (wind-corrected)": "#0891b2",
    DYNAMIC_REFERENCE_LABEL: "#16a34a",
}


def _model_colors(names: Iterable[str]) -> dict[str, str]:
    """:data:`MODEL_COLORS`, then the default cycle for other names."""
    free = iter(f"C{i}" for i in range(10))
    return {name: MODEL_COLORS.get(name) or next(free) for name in names}


def plot_buoy_timeseries(
    reference: xr.Dataset,
    estimates: Mapping[str, xr.Dataset],
    site: str,
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Buoy",
) -> Figure:
    """Time series at one site: the reference and every named estimate, one row per variable."""
    keys = resolve_comparison_variables(reference, *estimates.values(), variables=variables)
    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 2 + 2 * len(keys)), sharex=True, squeeze=False)
    colors = _model_colors(estimates)
    for ax, key in zip(axes[:, 0], keys):
        spec = PARAMETERS[key]
        plot_timeseries_comparison(
            ax,
            reference[key].sel(sites=site),
            {name: est[key].sel(sites=site) for name, est in estimates.items()},
            reference_label,
            colors=colors,
            reference_color=MODEL_COLORS.get(reference_label, "black"),
            circular=spec["circular"],
            ylabel=f"{spec['label']} [{spec['units']}]",
        )
    axes[-1, 0].set_xlabel("Time")
    fig.suptitle(site, fontsize=11)
    fig.tight_layout()
    return fig


def plot_buoy_summary(
    reference: xr.Dataset,
    estimate: xr.Dataset,
    site: str,
    *,
    variables: Iterable[str] | None = None,
    reference_label: str = "Buoy",
    estimate_label: str = "Spectral metamodel",
    period_label: str | None = None,
) -> Figure:
    """
    One figure per buoy for the GUI: density scatters, then Hs (and Tp) over time.

    The time series is cropped to the span where the buoy has data.
    """
    ref, est = xr.align(reference.sel(sites=site), estimate.sel(sites=site), join="inner")
    keys = resolve_comparison_variables(ref, est, variables=variables)
    fig = plt.figure(figsize=(4.2 * len(keys), 7.4))
    gs = fig.add_gridspec(2, len(keys), height_ratios=[1.15, 0.85], hspace=0.35)
    for col, key in enumerate(keys):
        _scatter_panel(fig.add_subplot(gs[0, col]), ref[key].values, est[key].values, key,
                       reference_label, estimate_label)

    ax = fig.add_subplot(gs[1, :])
    has_data = ref["hs"].notnull().values if "hs" in keys else np.ones(ref.sizes["time"], bool)
    if has_data.any():
        span = ref.time.values[has_data]
        ref, est = ref.sel(time=slice(span.min(), span.max())), est.sel(time=slice(span.min(), span.max()))
    colors = _model_colors([estimate_label])
    for name, axis, style in (("hs", ax, "-"), ("tp", ax.twinx() if "tp" in keys else None, "--")):
        if name not in keys:
            continue
        axis.plot(ref.time, ref[name], style, color="black", lw=1.0, label=f"{reference_label} {name}")
        axis.plot(est.time, est[name], style, color=colors[estimate_label], lw=0.9, label=f"{estimate_label} {name}")
        axis.set_ylabel(f"{PARAMETERS[name]['label']} [{PARAMETERS[name]['units']}]")
    handles = [h for a in fig.axes[len(keys):] for h in a.get_legend_handles_labels()[0]]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    fig.suptitle(site if not period_label else f"{site}  —  {period_label}", fontsize=13, fontweight="bold")
    return fig
