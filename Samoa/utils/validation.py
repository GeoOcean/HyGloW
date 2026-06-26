"""Propagation holdout validation for Samoa metamodels.

Validation **loads** saved models only — run :func:`utils.metamodel.fit` first.

Typical notebook flow::

    from utils import metamodel as mm
    from utils import validation as val

    mm.fit("1")                         # train once
    metrics = val.sweep("1")            # metrics for all MDA sizes → CSV
    result = val.validate("1", n=125)   # one size, with data for plotting
    result.plot("10m")                  # one isobath depth
    result.plot_all(variables=["hs"])   # every group

    metrics = val.sweep("1")
    val.plot_sweep(metrics, by="bulk")           # all sites, all metrics
    val.plot_sweep_suite(metrics)                # bulk + source + group + …
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.metrics import bias, r2, rmse
from bluemath_tk.core.plotting.old_scatter import density_scatter
from matplotlib.figure import Figure
from utils import metamodel as mm
from utils.config import (
    METAMODEL,
    load_goals,
    resolve,
)

_log_dir = resolve("outputs/logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_makedirs = os.makedirs
os.makedirs = lambda path, *args, **kwargs: (
    _makedirs(path, exist_ok=True) if path else None
)

LOGGER_NAME = "Samoa.validation"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("outputs/logs/validation.log")),
    level="INFO",
    console=True,
    console_level="INFO",
)

os.makedirs = _makedirs

PARAMETERS: dict[str, dict[str, Any]] = {
    "hs": {"label": "Hs", "units": "m", "circular": False},
    "tp": {"label": "Tp", "units": "s", "circular": False},
    "dir": {"label": "Dir", "units": "deg", "circular": True},
}


def _metrics_variables(variables: Iterable[str] | None) -> list[str]:
    if variables is not None:
        return list(variables)
    return list(METAMODEL["validation"]["metrics_variables"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _theta_to_360(values: np.ndarray) -> np.ndarray:
    return np.mod(values, 360.0)


def _circular_difference(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    return (est - ref + 180.0) % 360.0 - 180.0


def _coincident_values(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _circular_r2(reference: np.ndarray, estimate: np.ndarray) -> float:
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
    """Bias, RMSE, R² and sample count for one variable pair."""
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
# Site groups
# ---------------------------------------------------------------------------


def _per_site_coord(ds: xr.Dataset, name: str, default: str = "") -> dict[str, str]:
    if name not in ds.coords:
        return {str(s): default for s in ds.sites.values}
    values = ds[name].values
    labels = [str(s) for s in ds.sites.values]
    if values.ndim == 0:
        return {s: str(values) for s in labels}
    return {s: str(v) for s, v in zip(labels, values, strict=True)}


def _isobath_sort_key(label: str) -> float:
    try:
        return float(str(label).rstrip("m"))
    except ValueError:
        return float("inf")


def metric_groups(ds: xr.Dataset) -> list[tuple[str, str, list[str]]]:
    """Validation groups as ``(name, site_source, site_ids)``.

    - isobaths: one group per depth label (``5m``, ``10m``, …)
    - mesh: one group per polygon label (e.g. ``Apia``)
    - buoys: one group per site id
    """
    labels = [str(s) for s in ds.sites.values]
    source_by_site = _per_site_coord(ds, "site_source", "")
    label_by_site = _per_site_coord(ds, "site_label", "")

    groups: list[tuple[str, str, list[str]]] = []

    isobath_sites = [s for s in labels if source_by_site.get(s) == "isobath"]
    for iso_label in sorted(
        {label_by_site[s] for s in isobath_sites},
        key=_isobath_sort_key,
    ):
        site_ids = [s for s in isobath_sites if label_by_site[s] == iso_label]
        if site_ids:
            groups.append((iso_label, "isobath", site_ids))

    mesh_sites = [s for s in labels if source_by_site.get(s) == "mesh"]
    for mesh_label in sorted({label_by_site[s] for s in mesh_sites}):
        site_ids = [s for s in mesh_sites if label_by_site[s] == mesh_label]
        if site_ids:
            groups.append((mesh_label, "mesh", site_ids))

    for site_id in [s for s in labels if source_by_site.get(s) == "buoy"]:
        groups.append((site_id, "buoy", [site_id]))

    return groups


def _find_group(
    ds: xr.Dataset,
    group: str,
) -> tuple[str, str, list[str]]:
    for name, source, site_ids in metric_groups(ds):
        if name == group:
            return name, source, site_ids
    available = [g[0] for g in metric_groups(ds)]
    raise KeyError(f"Unknown group {group!r}. Available: {available}")


def metrics_table(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Metrics table with one row per group × variable."""
    params = _metrics_variables(variables)
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
                    else _per_site_coord(estimate, "site_label").get(
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
    return mm.goal_output_dir(goal) / "validation_metrics.csv"


# ---------------------------------------------------------------------------
# Validation entry points
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Metamodel vs SnapWave on held-out test cases for one goal and MDA size."""

    goal: str
    n: int
    estimate: xr.Dataset
    reference: xr.Dataset
    metrics: pd.DataFrame
    n_test_cases: int
    _groups: list[tuple[str, str, list[str]]] = field(repr=False)

    def site_ids(self, group: str) -> list[str]:
        """Site ids for a validation group (e.g. ``'10m'``, ``'Apia'``, buoy id)."""
        return _find_group(self.estimate, group)[2]

    def plot(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Map + density scatters for one group."""
        name, _, site_ids = _find_group(self.estimate, group)
        return plot_group(
            self.estimate,
            self.reference,
            name,
            site_ids,
            variables=variables,
            **kwargs,
        )

    def plot_map(self, group: str, **kwargs: Any) -> tuple[Figure, plt.Axes]:
        """Site map for one group."""
        name, _, site_ids = _find_group(self.estimate, group)
        return plot_map(self.estimate, site_ids, group_name=name, **kwargs)

    def plot_scatter(
        self,
        group: str,
        *,
        variables: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Density scatters only (no map) for one group."""
        name, _, site_ids = _find_group(self.estimate, group)
        return plot_scatter(
            self.estimate,
            self.reference,
            name,
            site_ids,
            variables=variables,
            **kwargs,
        )

    def plot_all(
        self,
        *,
        variables: Iterable[str] | None = None,
        groups: Iterable[str] | None = None,
        **kwargs: Any,
    ) -> list[Figure]:
        """One figure per group (optionally filtered by name)."""
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
                **kwargs,
            )
            for name, _, site_ids in active
        ]


def _pair(
    goal: str,
    n: int,
    *,
    context: dict[str, Any] | None = None,
    models: dict[str, Any] | None = None,
) -> tuple[xr.Dataset, xr.Dataset, int]:
    """Load models and return estimate, reference, n_test on holdout cases."""
    if context is None:
        context = mm.load_context(goal)
    test_idx = context["test_idx"]
    if models is None:
        models = mm.load_models(goal, n)
    estimate = mm.predict_cases(goal, n, test_idx, models=models)
    reference = mm.truth_cases(goal, test_idx)
    return estimate, reference, len(test_idx)


def validate(
    goal: str | int,
    n: int,
    *,
    variables: Iterable[str] | None = None,
) -> ValidationResult:
    """Validate one MDA size on holdout cases (loads saved models only)."""
    goal = str(goal)
    n = int(n)
    estimate, reference, n_test = _pair(goal, n)
    table = metrics_table(estimate, reference, variables=variables)
    table.insert(0, "goal", goal)
    table.insert(1, "mda_candidate", n)
    table.insert(2, "n_test_cases", n_test)
    return ValidationResult(
        goal=goal,
        n=n,
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
    save: bool = True,
) -> pd.DataFrame:
    """Metrics for every MDA candidate size on one goal."""
    goal = str(goal)
    cand = [int(c) for c in (candidates or METAMODEL["mda_candidates"])]
    context = mm.load_context(goal)

    frames: list[pd.DataFrame] = []
    for n in cand:
        logger.info("Goal %s: validating n=%d", goal, n)
        estimate, reference, n_test = _pair(goal, n, context=context)
        table = metrics_table(estimate, reference, variables=variables)
        table.insert(0, "goal", goal)
        table.insert(1, "mda_candidate", n)
        table.insert(2, "n_test_cases", n_test)
        frames.append(table)

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
    save: bool = True,
) -> pd.DataFrame:
    """Run :func:`sweep` for every goal and concatenate."""
    active = [str(g) for g in (goals or load_goals().keys())]
    frames = [
        sweep(
            goal,
            candidates=candidates,
            variables=variables,
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


def _scatter_panel(
    ax: plt.Axes,
    estimate: np.ndarray,
    reference: np.ndarray,
    variable: str,
    *,
    estimate_label: str = "Metamodel",
    reference_label: str = "SnapWave",
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

    xs, ys, density = density_scatter(est, ref)
    ax.scatter(xs, ys, c=density, s=10, cmap=cmap, edgecolors="none", alpha=0.9)

    if spec["circular"]:
        lo, hi = 0.0, 360.0
    else:
        lo = 0.0
        hi = float(np.ceil(max(xs.max(), ys.max()) * 1.05))

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
    estimate_label: str = "Metamodel",
    reference_label: str = "SnapWave",
    figsize: tuple[float, float] = (12, 4),
    cmap: str = "viridis",
) -> Figure:
    """Density scatter panels for one validation group."""
    keys = _metrics_variables(variables)
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=True):
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
    from utils.plotting import plot_base_map, plot_validation_sites

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
    estimate_label: str = "Metamodel",
    reference_label: str = "SnapWave",
    figsize: tuple[float, float] = (14, 9),
    cmap: str = "viridis",
) -> Figure:
    """Map + density scatters for one validation group."""
    from utils.plotting import plot_base_map, plot_validation_sites

    keys = _metrics_variables(variables)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        2, len(keys), height_ratios=[1.25, 1.0], hspace=0.3, wspace=0.3
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

    for col, key in enumerate(keys):
        ax = fig.add_subplot(gs[1, col])
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


def plot_all_groups(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    groups: list[tuple[str, str, list[str]]] | None = None,
    **kwargs: Any,
) -> list[Figure]:
    """One map + scatter figure per validation group."""
    active = groups or metric_groups(estimate)
    return [
        plot_group(
            estimate,
            reference,
            group_name,
            site_ids,
            variables=variables,
            **kwargs,
        )
        for group_name, _source, site_ids in active
    ]


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
    from utils.plotting import DEFAULT_POINT_COLOR, ISOBATH_COLORS, MESH_COLORS

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

    Rows = metrics, columns = variables (hs, tp, dir). One line per series
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
    "ValidationResult",
    "compute_metrics",
    "logger",
    "metric_groups",
    "metrics_table",
    "aggregate_sweep",
    "load_sweep_metrics",
    "plot_all_groups",
    "plot_group",
    "plot_map",
    "plot_scatter",
    "plot_sweep",
    "plot_sweep_suite",
    "sweep",
    "sweep_all",
    "validate",
    "validation_metrics_path",
    "SWEEP_BY",
    "SWEEP_STATS",
]
