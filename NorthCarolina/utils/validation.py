"""Validate spectral metamodel output against SnapWave dynamic runs.

Compares ``hs``, ``tp``, and ``dm`` from the metamodel spectral product
(``outputs/spectral_metamodel.nc``) against ``point_hm0``, ``point_tp``, and
``point_wavdir`` in the dynamic reference file
(``SnapWave_CASES_Dynamic/Dynamic_Cases_198601.nc``).

The metamodel NetCDF contains **QC-passed sites only** (see
:func:`utils.metamodel.reconstruct_spectral`). Validation aligns on those
sites and the matching dynamic records — no re-QC by default.

Typical notebook flow::

    from utils import validation as val

    result = val.validate_against_dynamic(time="1986-01")
    val.plot_scatter(result.metamodel, result.dynamic)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.metrics import bias, r2, rmse, si
from bluemath_tk.core.plotting.old_scatter import validation_scatter

from . import metamodel as mm
from .config import METAMODEL, PATHS, resolve
from .snapwave_wrapper import normalize_site_dims

# bluemath_tk get_file_logger calls makedirs without exist_ok=True
_log_dir = resolve("outputs/logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_makedirs = os.makedirs
os.makedirs = lambda path, *args, **kwargs: (
    _makedirs(path, exist_ok=True) if path else None
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

LOGGER_NAME = "NorthCarolina.validation"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("outputs/logs/validation.log")),
    level="INFO",
    console=True,
    console_level="INFO",
)

# ---------------------------------------------------------------------------
# Variable mapping
# ---------------------------------------------------------------------------

VARIABLES: dict[str, dict[str, Any]] = {
    "hs": {
        "meta": "hs",
        "dynamic": "point_hm0",
        "label": "Hs",
        "units": "m",
        "circular": False,
    },
    "tp": {
        "meta": "tp",
        "dynamic": "point_tp",
        "label": "Tp",
        "units": "s",
        "circular": False,
    },
    "dm": {
        "meta": "dm",
        "dynamic": "point_wavdir",
        "label": "Dm",
        "units": "deg",
        "circular": True,
    },
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def dynamic_output_path(period: str = "198601") -> Path:
    """Default dynamic reference NetCDF for one calendar month (``YYYYMM``)."""
    return resolve(Path(PATHS["cases_dynamic_dir"]) / f"Dynamic_Cases_{period}.nc")


def load_metamodel(path: str | Path | None = None) -> xr.Dataset:
    """Open the spectral metamodel product."""
    p = (
        resolve(path)
        if path is not None
        else resolve(METAMODEL["paths"]["spectral_output"])
    )
    logger.info("Loading metamodel from %s", p)
    return xr.open_dataset(p)


def load_dynamic(path: str | Path | None = None) -> xr.Dataset:
    """Open the SnapWave dynamic reference."""
    p = resolve(path) if path is not None else dynamic_output_path()
    logger.info("Loading dynamic reference from %s", p)
    return xr.open_dataset(p)


def load_pair(
    *,
    metamodel_path: str | Path | None = None,
    dynamic_path: str | Path | None = None,
    time: str | slice | None = None,
    sites: Iterable[str] | str | None = None,
    site_source: str | None = "isobath",
    apply_qc: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load and align metamodel and dynamic datasets for comparison.

    By default the metamodel file is trusted to contain QC-passed sites only.
    Set ``apply_qc=True`` to re-run spectral QC (e.g. on an older product).
    """
    meta = normalize_metamodel(load_metamodel(metamodel_path))
    if apply_qc:
        meta, _ = mm.filter_spectral_qc(meta)
    dynamic = normalize_dynamic(load_dynamic(dynamic_path))
    return align_datasets(
        meta,
        dynamic,
        time=time,
        sites=sites,
        site_source=site_source,
    )


# ---------------------------------------------------------------------------
# Dataset normalization / alignment
# ---------------------------------------------------------------------------


def normalize_metamodel(ds: xr.Dataset) -> xr.Dataset:
    """Ensure a ``sites`` dimension labelled with site ids (e.g. ``5m_10``)."""

    ds = normalize_site_dims(ds)
    if "sites" in ds.dims:
        return ds
    raise ValueError("Metamodel dataset has no sites dimension")


def normalize_dynamic(ds: xr.Dataset) -> xr.Dataset:
    """Index the dynamic dataset on the ``sites`` dimension."""

    ds = normalize_site_dims(ds)
    if "sites" in ds.dims:
        return ds
    raise ValueError("Dynamic dataset has no sites dimension")


def _site_source_values(ds: xr.Dataset) -> np.ndarray:
    n = ds.sizes["sites"]
    if "site_source" not in ds.coords:
        return np.array(["isobath"] * n, dtype=object)
    src = ds["site_source"].values
    if src.ndim == 0:
        return np.array([str(src)] * n, dtype=object)
    return src.astype(str)


def filter_sites(
    ds: xr.Dataset,
    *,
    sites: Iterable[str] | str | None = None,
    site_source: str | None = "isobath",
) -> xr.Dataset:
    """Subset by explicit site ids and/or ``site_source`` (default: isobaths)."""

    labels = [str(s) for s in ds.coords["sites"].values]

    if site_source is not None:
        src = _site_source_values(ds)
        allowed = {
            lab for lab, role in zip(labels, src, strict=True) if role == site_source
        }
        labels = [lab for lab in labels if lab in allowed]

    if sites is not None:
        if isinstance(sites, str):
            wanted = {sites}
        else:
            wanted = {str(s) for s in sites}
        labels = [lab for lab in labels if lab in wanted]

    if not labels:
        raise ValueError("Site filter left no sites to compare")
    return ds.sel(sites=labels)


def common_sites(
    meta: xr.Dataset,
    dynamic: xr.Dataset,
    *,
    site_source: str | None = "isobath",
) -> list[str]:
    """Site ids present in both datasets after optional source filtering."""

    meta = filter_sites(meta, site_source=site_source)
    dynamic = filter_sites(dynamic, site_source=site_source)
    shared = sorted(set(meta.sites.values) & set(dynamic.sites.values))
    if not shared:
        raise ValueError("No common sites between metamodel and dynamic data")
    return [str(s) for s in shared]


def align_datasets(
    meta: xr.Dataset,
    dynamic: xr.Dataset,
    *,
    time: str | slice | None = None,
    sites: Iterable[str] | str | None = None,
    site_source: str | None = "isobath",
) -> tuple[xr.Dataset, xr.Dataset]:
    """Return metamodel and dynamic subsets on shared ``time`` × ``sites``."""

    meta = normalize_metamodel(meta)
    dynamic = normalize_dynamic(dynamic)

    labels = common_sites(meta, dynamic, site_source=site_source)
    if sites is not None:
        if isinstance(sites, str):
            wanted = {sites}
        else:
            wanted = {str(s) for s in sites}
        labels = [s for s in labels if s in wanted]
        if not labels:
            raise ValueError("Requested sites are not in the common set")

    meta = meta.sel(sites=labels)
    dynamic = dynamic.sel(sites=labels)

    if time is not None:
        meta = meta.sel(time=time)
        dynamic = dynamic.sel(time=time)

    meta, dynamic = xr.align(meta, dynamic, join="inner")
    if meta.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping time steps between metamodel and dynamic data")

    logger.info(
        "Aligned %d site(s), %d time step(s) [%s … %s]",
        meta.sizes["sites"],
        meta.sizes["time"],
        str(meta.time.values[0])[:10],
        str(meta.time.values[-1])[:10],
    )
    return meta, dynamic


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _paired_arrays(
    meta: xr.Dataset,
    dynamic: xr.Dataset,
    meta_var: str,
    dyn_var: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(metamodel, dynamic)`` arrays on ``(time, sites)``."""

    m = meta[meta_var].transpose("time", "sites")
    d = dynamic[dyn_var].transpose("time", "sites")
    return m.values, d.values


def coincident_values(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop NaN pairs from two equally-shaped arrays."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _theta_to_360(values: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(values, dtype=float), 360.0)


def _circular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (
        np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0
    ) % 360.0 - 180.0


def compute_metrics(
    meta_vals: np.ndarray,
    dyn_vals: np.ndarray,
    *,
    circular: bool = False,
) -> dict[str, float]:
    """Validation metrics for paired (metamodel, dynamic) samples."""

    if circular:
        x = _theta_to_360(meta_vals)
        y = _theta_to_360(dyn_vals)
        x, y = coincident_values(x, y)
        if x.size == 0:
            return {"n": 0, "bias": np.nan, "rmse": np.nan, "si": np.nan, "r2": np.nan}
        diff = _circular_difference(y, x)
        return {
            "n": int(x.size),
            "bias": float(np.mean(diff)),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "si": np.nan,
            "r2": np.nan,
        }

    x, y = coincident_values(meta_vals, dyn_vals)
    if x.size == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "si": np.nan, "r2": np.nan}
    return {
        "n": int(x.size),
        "bias": float(bias(x, y)),
        "rmse": float(rmse(x, y)),
        "si": float(si(x, y)),
        "r2": float(r2(x, y)),
    }


def _coord_per_site(ds: xr.Dataset, key: str) -> list[str] | None:
    if key not in ds.coords:
        return None
    values = ds[key].values
    if values.ndim == 0:
        return [str(values)] * ds.sizes["sites"]
    return [str(v) for v in values]


def _site_metadata(ds: xr.Dataset) -> pd.DataFrame:
    rows: dict[str, list[Any]] = {"site": [str(s) for s in ds.sites.values]}
    for key in ("site_label", "site_depth", "site_source"):
        per_site = _coord_per_site(ds, key)
        if per_site is not None:
            rows[key] = per_site
    return pd.DataFrame(rows).set_index("site")


def metrics_table(
    meta: xr.Dataset,
    dynamic: xr.Dataset,
    *,
    groupby: str | None = None,
) -> pd.DataFrame:
    """Compute validation metrics for each variable (and optional grouping).

    Parameters
    ----------
    groupby
        ``None`` — one pooled row per variable.
        ``"site"`` — one row per site and variable.
        ``"site_label"`` — aggregate by isobath depth label (``5m``, ``10m``, …).
    """

    rows: list[dict[str, Any]] = []
    meta_info = _site_metadata(meta)

    if groupby is None:
        groups: list[tuple[str, list[str]]] = [("all", list(meta.sites.values))]
    elif groupby == "site":
        groups = [(str(s), [str(s)]) for s in meta.sites.values]
    elif groupby == "site_label":
        if "site_label" not in meta.coords:
            raise ValueError("site_label coordinate required for this groupby")
        labels = sorted({str(v) for v in meta["site_label"].values})
        groups = [
            (
                label,
                [
                    str(s)
                    for s, lab in zip(meta.sites.values, meta["site_label"].values)
                    if str(lab) == label
                ],
            )
            for label in labels
        ]
    else:
        raise ValueError(f"Unsupported groupby={groupby!r}")

    for group_name, site_ids in groups:
        meta_g = meta.sel(sites=site_ids)
        dyn_g = dynamic.sel(sites=site_ids)
        for key, spec in VARIABLES.items():
            meta_vals, dyn_vals = _paired_arrays(
                meta_g, dyn_g, spec["meta"], spec["dynamic"]
            )
            m = compute_metrics(
                meta_vals,
                dyn_vals,
                circular=spec["circular"],
            )
            row = {
                "group": group_name,
                "variable": key,
                "label": spec["label"],
                **m,
            }
            if groupby == "site" and group_name in meta_info.index:
                row["site_label"] = meta_info.loc[group_name].get("site_label")
                row["site_depth"] = meta_info.loc[group_name].get("site_depth")
            rows.append(row)

    out = pd.DataFrame(rows)
    logger.info(
        "Computed metrics (%s grouping): %d row(s)", groupby or "pooled", len(out)
    )
    return out


@dataclass
class ValidationResult:
    metamodel: xr.Dataset
    dynamic: xr.Dataset
    metrics: pd.DataFrame
    sites: list[str]
    excluded_sites: list[str]
    qc_report: pd.DataFrame | None = None


def validate_against_dynamic(
    *,
    metamodel_path: str | Path | None = None,
    dynamic_path: str | Path | None = None,
    time: str | slice | None = None,
    sites: Iterable[str] | str | None = None,
    site_source: str | None = "isobath",
    apply_qc: bool = False,
) -> ValidationResult:
    """End-to-end validation on QC-passed metamodel sites vs dynamic SnapWave.

    Sites in the saved metamodel product define the validation set; dynamic
    data is subset to the same labels. Set ``apply_qc=True`` only if you need
    to re-filter an unclean metamodel file.
    """

    meta_raw = normalize_metamodel(load_metamodel(metamodel_path))
    qc_report: pd.DataFrame | None = None
    excluded = list(mm.get_excluded_sites())
    if apply_qc:
        meta_raw, qc_report = mm.filter_spectral_qc(meta_raw)
        excluded = list(mm.get_excluded_sites())

    dynamic = normalize_dynamic(load_dynamic(dynamic_path))
    meta, dynamic = align_datasets(
        meta_raw,
        dynamic,
        time=time,
        sites=sites,
        site_source=site_source,
    )
    table = metrics_table(meta, dynamic)
    sites_out = [str(s) for s in meta.sites.values]
    for _, row in table.iterrows():
        logger.info(
            "%s: n=%d  RMSE=%.3f  BIAS=%.3f  R²=%.3f",
            row["label"],
            int(row["n"]),
            row["rmse"],
            row["bias"],
            row["r2"],
        )
    return ValidationResult(
        metamodel=meta,
        dynamic=dynamic,
        metrics=table,
        sites=sites_out,
        excluded_sites=excluded,
        qc_report=qc_report,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _pooled_series(
    meta: xr.Dataset, dynamic: xr.Dataset, meta_var: str, dyn_var: str
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (time, sites) arrays into 1-D vectors."""

    m, d = _paired_arrays(meta, dynamic, meta_var, dyn_var)
    return coincident_values(m.ravel(), d.ravel())


def plot_scatter(
    meta: xr.Dataset,
    dynamic: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    figsize: tuple[float, float] = (12, 4),
    cmap: str = "rainbow",
) -> plt.Figure:
    """Density scatter panels (metamodel vs dynamic) for hs / tp / dm."""

    keys = list(variables or VARIABLES)
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=True):
        spec = VARIABLES[key]
        x, y = _pooled_series(meta, dynamic, spec["meta"], spec["dynamic"])
        if x.size == 0:
            ax.set_title(f"{spec['label']} (no data)")
            continue
        validation_scatter(
            axs=ax,
            x=x,
            y=y,
            xlabel=f"Metamodel {spec['label']} [{spec['units']}]",
            ylabel=f"Dynamic {spec['label']} [{spec['units']}]",
            title=spec["label"],
            cmap=cmap,
        )
    fig.tight_layout()
    return fig


def plot_timeseries(
    meta: xr.Dataset,
    dynamic: xr.Dataset,
    site: str,
    *,
    variables: Iterable[str] | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Time-series comparison at one site."""

    site = str(site)
    keys = list(variables or VARIABLES)
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=True):
        spec = VARIABLES[key]
        meta.sel(sites=site)[spec["meta"]].plot(
            ax=ax, label="Metamodel", color="darkred", lw=1.0
        )
        dynamic.sel(sites=site)[spec["dynamic"]].plot(
            ax=ax, label="Dynamic", color="darkgreen", lw=1.0, alpha=0.85
        )
        ax.set_ylabel(f"{spec['label']} [{spec['units']}]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Site {site}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_metrics_by_depth(
    meta: xr.Dataset,
    dynamic: xr.Dataset,
    *,
    metric: str = "rmse",
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Bar chart of a metric grouped by isobath depth (``site_label``)."""

    table = metrics_table(meta, dynamic, groupby="site_label")
    keys = list(VARIABLES)
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=True):
        spec = VARIABLES[key]
        sub = table[table["variable"] == key].sort_values("group")
        ax.bar(sub["group"], sub[metric], color="steelblue", alpha=0.85)
        ax.set_title(spec["label"])
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{metric.upper()} by isobath depth")
    fig.tight_layout()
    return fig


__all__ = [
    "LOGGER_NAME",
    "VARIABLES",
    "ValidationResult",
    "align_datasets",
    "coincident_values",
    "common_sites",
    "compute_metrics",
    "dynamic_output_path",
    "filter_sites",
    "load_dynamic",
    "load_metamodel",
    "load_pair",
    "logger",
    "metrics_table",
    "normalize_dynamic",
    "normalize_metamodel",
    "plot_metrics_by_depth",
    "plot_scatter",
    "plot_timeseries",
    "validate_against_dynamic",
]

os.makedirs = _makedirs
