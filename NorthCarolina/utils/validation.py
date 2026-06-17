"""Validation utilities for the North Carolina HyGloW pipeline.

Data flow (production)
----------------------
1. **SnapWave raw** — static cases + dynamic runs, post-processed with
   :mod:`utils.snapwave_wrapper` (``sites``, ``site_source``, ``site_label``).
2. **Metamodel product** — ``outputs/spectral_metamodel.nc`` (``hs``, ``tp``, ``tm01``, ``dm``;
   built by :mod:`utils.metamodel`).
3. **Dynamic reference** — ``SnapWave_CASES_Dynamic/Dynamic_Cases_*.nc``.
4. **Buoy observations** — ``inputs/JennBuoys/processed/buoy_*_bulk_parameters.pkl``.

Validation modes
----------------
* **Isobaths** — metamodel vs dynamic (same site grid, ``site_source='isobath'``).
  Cluster by ``site_label`` (``5m`` / ``10m`` / ``15m``).
* **Buoys** — buoy observations as reference; compare **metamodel** and **dynamic**
  at matching buoy sites for a user-specified time window (notebook:
  ``06_Validate_Against_Buoys.ipynb``).

Typical isobath notebook flow::

    from utils import validation as val

    result = val.validate_isobaths(time="1986-01-15", site_label="10m")
    val.metrics_table(result.estimate, result.reference, groupby="site_label")
    val.plot_scatter(result.estimate, result.reference)

Typical buoy notebook flow (when ready)::

    result = val.validate_buoys(time="1986-01")
    val.plot_buoy_skill(result.metrics, metric="r2")
    val.plot_buoy_timeseries(result, site="buoy_41013")
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.metrics import bias, r2, rmse, si
from bluemath_tk.core.plotting.old_scatter import validation_scatter

from . import metamodel as mm
from .config import (
    METAMODEL,
    PATHS,
    get_buoy_coordinates,
    resolve,
    validation_period_var,
)
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
# Canonical parameters & source registry
# ---------------------------------------------------------------------------

PARAMETERS: dict[str, dict[str, Any]] = {
    "hs": {"label": "Hs", "units": "m", "circular": False},
    "tp": {"label": "Tp", "units": "s", "circular": False},
    "tm01": {"label": "Tm01", "units": "s", "circular": False},
    "dm": {"label": "Dm", "units": "deg", "circular": True},
}

# Per-source NetCDF / table variable names for each canonical parameter.
SOURCE_VARS: dict[str, dict[str, str]] = {
    "metamodel": {"hs": "hs", "tp": "tp", "tm01": "tm01", "dm": "dm"},
    "dynamic": {
        "hs": "point_hm0",
        "tp": "point_tp",
        "tm01": "point_tp",
        "dm": "point_wavdir",
    },
    "downscaled": {"hs": "hs", "tp": "tp", "tm01": "tp", "dm": "wd"},
    "buoy": {"hs": "hs", "tp": "tp", "tm01": "tp", "dm": "dm"},
}

# JennBuoys processed pickle column names → canonical parameters.
BUOY_OBS_COLUMNS: dict[str, str] = {
    "hs": "Hs_Buoy",
    "tp": "Tp_Buoy",
    "dm": "Dir_Buoy",
}

BUOY_OBS_DIR = Path("inputs/JennBuoys/processed")
BUOY_OBS_PATTERN = "buoy_{buoy_id}_bulk_parameters.pkl"

# Models compared against buoys (reference = buoy observations).
BUOY_MODEL_SOURCES: tuple[str, ...] = ("metamodel", "dynamic")


def validation_parameters() -> list[str]:
    """Canonical parameters used in isobath / buoy skill metrics."""
    period = validation_period_var()
    return ["hs", period, "dm"]


# Backward-compatible alias (legacy notebooks; pooled isobath validation params).
VARIABLES: dict[str, dict[str, Any]] = {
    key: {
        **PARAMETERS[key],
        "meta": SOURCE_VARS["metamodel"][key],
        "dynamic": SOURCE_VARS["dynamic"][key],
    }
    for key in ("hs", "tp", "tm01", "dm")
}

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def dynamic_output_path(period: str = "198601") -> Path:
    """Default dynamic reference NetCDF for one calendar month (``YYYYMM``)."""
    return resolve(Path(PATHS["cases_dynamic_dir"]) / f"Dynamic_Cases_{period}.nc")


def buoy_obs_path(buoy_id: str) -> Path:
    """Processed buoy bulk-parameters pickle for one instrument id."""
    return resolve(BUOY_OBS_DIR / BUOY_OBS_PATTERN.format(buoy_id=_buoy_id(buoy_id)))


def whacs_path() -> Path:
    """WHACS reanalysis zarr (offshore forcing; same path as metamodel)."""
    return mm.whacs_path()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


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


def load_whacs(**kwargs: Any) -> xr.Dataset:
    """Open WHACS offshore reanalysis (zarr). Extra kwargs passed to ``open_zarr``."""
    logger.info("Loading WHACS from %s", whacs_path())
    return xr.open_zarr(whacs_path(), **kwargs)


def load_metamodel_spectral(
    time: str | slice,
    *,
    sites: int | slice | Iterable[int] | str | None = "buoy",
    metamodel_path: str | Path | None = None,
    recompute: bool = False,
) -> xr.Dataset:
    """Spectral metamodel (``hs``, ``tp``, ``dm``) for selected sites.

    Reads ``spectral_metamodel.nc`` when all requested sites are present and
    ``recompute=False``. Otherwise runs bulk + spectral reconstruction via
    :mod:`utils.metamodel` (e.g. buoy sites omitted from the saved product).
    """
    if not recompute:
        try:
            on_disk = normalize_source(load_metamodel(metamodel_path), "metamodel")
            wanted = mm.resolve_sites(on_disk, sites)
            if set(wanted).issubset(set(on_disk.sites.values.astype(str))):
                out = on_disk.sel(sites=wanted)
                if time is not None:
                    out = out.sel(time=time)
                return out
        except (FileNotFoundError, OSError, ValueError):
            pass

    logger.info("Recomputing spectral metamodel for sites=%r time=%r", sites, time)
    bulk = mm.apply_missing_forcing_zeros(mm.reconstruct_bulk(time=time))
    return mm.reconstruct_spectral(bulk, sites=sites, time=time)


def load_downscaled_bulk(
    time: str | slice,
    *,
    sites: int | slice | Iterable[int] | str | None = None,
    partition: str = "sea",
) -> xr.Dataset:
    """GPR-downscaled bulk (sea partition) from WHACS forcing.

    Collapses the ``goal`` dimension by averaging finite values per site.
    Direction is exposed as ``wd`` then renamed to ``dm`` during normalization.
    """
    bulk = mm.apply_missing_forcing_zeros(mm.reconstruct_bulk(time=time))
    bulk = normalize_site_dims(bulk)
    if partition not in bulk.partition.values:
        raise ValueError(f"Partition {partition!r} not in bulk reconstruction")
    part = bulk.sel(partition=partition)
    hs = part.hs.mean(dim="goal", skipna=True)
    tp = part.tp.mean(dim="goal", skipna=True)
    wd = part.wd.mean(dim="goal", skipna=True)
    ds = xr.Dataset(
        {"hs": hs, "tp": tp, "wd": wd},
        coords={"time": hs.time, "sites": hs.sites},
    )
    if sites is not None:
        site_ids = mm.resolve_sites(ds, sites)
        ds = ds.sel(sites=site_ids)
    return normalize_source(ds, "downscaled")


def _buoy_id(site_or_id: str) -> str:
    text = str(site_or_id)
    return text.removeprefix("buoy_")


def _resolve_buoy_ids(
    buoys: str | Iterable[str] | None = None,
    *,
    role: str | None = "validation",
) -> list[str]:
    """Return canonical site ids ``buoy_<id>`` for validation."""

    if buoys is None:
        coords = get_buoy_coordinates(role=role)
        return [f"buoy_{iid}" for iid in coords]
    if isinstance(buoys, str):
        buoys = [buoys]
    out: list[str] = []
    for item in buoys:
        text = str(item)
        out.append(text if text.startswith("buoy_") else f"buoy_{text}")
    return out


def load_buoy_observations(
    buoys: str | Iterable[str] | None = None,
    *,
    time: str | slice | None = None,
    role: str | None = "validation",
) -> xr.Dataset:
    """Load buoy bulk-parameter time series as a single ``(time, sites)`` dataset.

    Each site is one buoy; variables are canonical ``hs``, ``tp``, ``dm``.
    Missing pickle files are skipped with a warning.
    """
    site_ids = _resolve_buoy_ids(buoys, role=role)
    chunks: list[xr.Dataset] = []

    for site in site_ids:
        path = buoy_obs_path(_buoy_id(site))
        if not path.is_file():
            logger.warning("Buoy observations not found: %s", path)
            continue
        frame: pd.DataFrame = pickle.loads(path.read_bytes())
        rename = {col: param for param, col in BUOY_OBS_COLUMNS.items() if col in frame}
        data = frame.rename(columns=rename)[list(rename.values())]
        ds = (
            data.to_xarray()
            .rename({"datetime": "time"})
            .expand_dims(sites=[site])
            .transpose("time", "sites")
        )
        chunks.append(ds)

    if not chunks:
        raise FileNotFoundError(
            "No buoy observation files found — check inputs/JennBuoys/processed/"
        )

    out = xr.concat(chunks, dim="sites")
    if time is not None:
        out = out.sel(time=time)
    return normalize_source(out, "buoy")


def load_source(
    name: str,
    *,
    time: str | slice | None = None,
    sites: int | slice | Iterable[int] | str | None = None,
    path: str | Path | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Dispatch loader for a registered source name."""

    if name == "metamodel":
        if time is None:
            ds = load_metamodel(path)
            if sites is not None:
                ds = ds.sel(sites=mm.resolve_sites(normalize_source(ds, name), sites))
            return ds
        return load_metamodel_spectral(time, sites=sites, metamodel_path=path, **kwargs)
    if name == "dynamic":
        ds = normalize_source(load_dynamic(path), name)
        if sites is not None:
            ds = ds.sel(sites=mm.resolve_sites(ds, sites))
        if time is not None:
            ds = ds.sel(time=time)
        return ds
    if name == "downscaled":
        if time is None:
            raise ValueError("downscaled source requires time=...")
        return load_downscaled_bulk(time, sites=sites, **kwargs)
    if name == "buoy":
        return load_buoy_observations(buoys=sites, time=time, **kwargs)
    if name == "whacs":
        return load_whacs()
    raise ValueError(
        f"Unknown source {name!r}; expected one of {list(SOURCE_VARS) + ['whacs']}"
    )


# Sources whose NetCDFs use canonical names (metamodel product, downscaled bulk).
_CANONICAL_SOURCES = frozenset({"metamodel", "downscaled"})

# SnapWave dynamic keeps native ``point_*`` names; lookup via SOURCE_VARS only.
_SNAPWAVE_SOURCES = frozenset({"dynamic"})


def _require_sites_dim(ds: xr.Dataset, kind: str) -> xr.Dataset:
    ds = normalize_site_dims(ds)
    if "sites" not in ds.dims:
        raise ValueError(f"{kind} dataset has no sites dimension")
    return ds


def normalize_source(ds: xr.Dataset, source: str) -> xr.Dataset:
    """Ensure ``sites`` dimension; rename to canonical names where appropriate.

    SnapWave **dynamic** data keep native ``point_hm0`` / ``point_tp`` /
    ``point_wavdir`` — variable mapping happens in :func:`_var_name`.
    """
    ds = _require_sites_dim(ds, source)
    if source in _SNAPWAVE_SOURCES:
        return ds
    if source not in _CANONICAL_SOURCES:
        return ds
    var_map = SOURCE_VARS.get(source, {})
    seen_src: set[str] = set()
    rename: dict[str, str] = {}
    for param, src in var_map.items():
        if src in seen_src or src not in ds.data_vars or param in ds.data_vars:
            continue
        if src != param:
            rename[src] = param
            seen_src.add(src)
    if rename:
        ds = ds.rename(rename)
    return ds


def normalize_metamodel(ds: xr.Dataset) -> xr.Dataset:
    """Backward-compatible alias for :func:`normalize_source` (metamodel)."""
    return normalize_source(ds, "metamodel")


def normalize_dynamic(ds: xr.Dataset) -> xr.Dataset:
    """Backward-compatible alias for :func:`normalize_source` (dynamic)."""
    return normalize_source(ds, "dynamic")


def _as_site_set(sites: Iterable[str] | str | None) -> set[str] | None:
    if sites is None:
        return None
    if isinstance(sites, str):
        return {sites}
    return {str(s) for s in sites}


def _per_site_coord(ds: xr.Dataset, name: str, default: str) -> dict[str, str]:
    site_ids = [str(s) for s in ds.coords["sites"].values]
    if name not in ds.coords:
        return dict.fromkeys(site_ids, default)
    values = ds[name].values
    if values.ndim == 0:
        return dict.fromkeys(site_ids, str(values))
    return dict(zip(site_ids, values.astype(str), strict=True))


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
    site_label: str | Iterable[str] | None = None,
) -> xr.Dataset:
    """Subset by site id, ``site_source``, and/or ``site_label`` cluster."""

    labels = [str(s) for s in ds.coords["sites"].values]

    if site_source is not None:
        src = _site_source_values(ds)
        allowed = {
            lab for lab, role in zip(labels, src, strict=True) if role == site_source
        }
        labels = [lab for lab in labels if lab in allowed]

    if site_label is not None:
        wanted = _as_site_set(site_label)
        label_by_site = _per_site_coord(ds, "site_label", "")
        labels = [lab for lab in labels if label_by_site.get(lab, "") in wanted]

    if sites is not None:
        selected = mm.resolve_sites(ds, sites)
        wanted = set(selected)
        labels = [lab for lab in labels if lab in wanted]

    if not labels:
        raise ValueError("Site filter left no sites to compare")
    return ds.sel(sites=labels)


def common_sites(
    *datasets: xr.Dataset,
    site_source: str | None = "isobath",
    site_label: str | Iterable[str] | None = None,
) -> list[str]:
    """Site ids present in all datasets after optional filters."""

    if not datasets:
        raise ValueError("At least one dataset required")

    filtered = [
        filter_sites(ds, site_source=site_source, site_label=site_label)
        for ds in datasets
    ]
    shared = set(str(s) for s in filtered[0].sites.values)
    for ds in filtered[1:]:
        shared &= {str(s) for s in ds.sites.values}
    if not shared:
        raise ValueError("No common sites across datasets")
    return sorted(shared)


def align_pair(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    time: str | slice | None = None,
    sites: Iterable[str] | str | None = None,
    site_source: str | None = "isobath",
    site_label: str | Iterable[str] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Align two site-labelled datasets on shared ``time`` × ``sites``."""

    estimate = normalize_site_dims(estimate)
    reference = normalize_site_dims(reference)

    labels = common_sites(
        estimate, reference, site_source=site_source, site_label=site_label
    )
    wanted = _as_site_set(sites)
    if wanted is not None:
        labels = [s for s in labels if s in wanted]
        if not labels:
            raise ValueError("Requested sites are not in the common set")

    estimate = estimate.sel(sites=labels)
    reference = reference.sel(sites=labels)

    if time is not None:
        estimate = estimate.sel(time=time)
        reference = reference.sel(time=time)

    estimate, reference = xr.align(estimate, reference, join="inner")
    if estimate.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping time steps between datasets")

    logger.info(
        "Aligned %d site(s), %d time step(s) [%s … %s]",
        estimate.sizes["sites"],
        estimate.sizes["time"],
        str(estimate.time.values[0])[:10],
        str(estimate.time.values[-1])[:10],
    )
    return estimate, reference


def align_datasets(
    meta: xr.Dataset,
    dynamic: xr.Dataset,
    *,
    time: str | slice | None = None,
    sites: Iterable[str] | str | None = None,
    site_source: str | None = "isobath",
    site_label: str | Iterable[str] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Backward-compatible alias: metamodel (estimate) vs dynamic (reference)."""
    return align_pair(
        meta,
        dynamic,
        time=time,
        sites=sites,
        site_source=site_source,
        site_label=site_label,
    )


def align_models_to_reference(
    reference: xr.Dataset,
    models: Mapping[str, xr.Dataset],
    *,
    time: str | slice | None = None,
    sites: Iterable[str] | str | None = None,
) -> tuple[xr.Dataset, dict[str, xr.Dataset]]:
    """Align every model dataset to the reference ``time`` × ``sites`` grid."""

    reference = normalize_site_dims(reference)
    aligned_ref = reference
    aligned_models: dict[str, xr.Dataset] = {}

    site_list = [str(s) for s in reference.sites.values]
    if sites is not None:
        wanted = _as_site_set(sites)
        site_list = [s for s in site_list if s in wanted]

    if time is not None:
        aligned_ref = aligned_ref.sel(time=time, sites=site_list)
    else:
        aligned_ref = aligned_ref.sel(sites=site_list)

    for name, ds in models.items():
        model = normalize_site_dims(ds).sel(sites=site_list)
        if time is not None:
            model = model.sel(time=time)
        model, ref = xr.align(model, aligned_ref, join="inner")
        aligned_models[name] = model
        aligned_ref = ref

    if aligned_ref.sizes.get("time", 0) == 0:
        raise ValueError("No overlapping time steps after alignment to reference")

    logger.info(
        "Aligned %d buoy site(s), %d time step(s) for models %s",
        aligned_ref.sizes["sites"],
        aligned_ref.sizes["time"],
        list(models),
    )
    return aligned_ref, aligned_models


def load_pair(
    *,
    metamodel_path: str | Path | None = None,
    dynamic_path: str | Path | None = None,
    time: str | slice | None = None,
    sites: Iterable[str] | str | None = None,
    site_source: str | None = "isobath",
    site_label: str | Iterable[str] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Load and align metamodel and dynamic datasets for comparison."""
    meta = normalize_metamodel(load_metamodel(metamodel_path))
    dynamic = normalize_dynamic(load_dynamic(dynamic_path))
    return align_datasets(
        meta,
        dynamic,
        time=time,
        sites=sites,
        site_source=site_source,
        site_label=site_label,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _var_name(ds: xr.Dataset, source: str, param: str) -> str:
    """Resolve a canonical parameter to a variable name in ``ds``."""
    mapped = SOURCE_VARS.get(source, {}).get(param)
    candidates: list[str] = []
    if mapped:
        candidates.append(mapped)
    candidates.append(param)
    # Spectral product: tm01 may be absent on older files; fall back to tp.
    if param in ("tm01", "tp") and source in ("metamodel", "downscaled"):
        for alt in ("tm01", "tp"):
            if alt not in candidates:
                candidates.append(alt)
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if candidate in ds.data_vars:
            return candidate
    raise KeyError(f"Parameter {param!r} not found in {source} dataset")


def _paired_arrays(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    param: str,
    *,
    estimate_source: str = "metamodel",
    reference_source: str = "dynamic",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(estimate, reference)`` arrays on ``(time, sites)``."""

    ev = _var_name(estimate, estimate_source, param)
    rv = _var_name(reference, reference_source, param)
    est = estimate[ev].transpose("time", "sites")
    ref = reference[rv].transpose("time", "sites")
    return est.values, ref.values


def coincident_values(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _theta_to_360(values: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(values, dtype=float), 360.0)


def _circular_difference(reference: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return (
        np.asarray(reference, dtype=float) - np.asarray(estimate, dtype=float) + 180.0
    ) % 360.0 - 180.0


def _circular_mean_deg(values: np.ndarray) -> float:
    rad = np.deg2rad(_theta_to_360(values))
    return float(
        np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 360.0
    )


def circular_r2(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Nash–Sutcliffe efficiency using circular residuals."""

    ref = _theta_to_360(reference)
    est = _theta_to_360(estimate)
    ref, est = coincident_values(ref, est)
    if ref.size == 0:
        return np.nan
    resid = _circular_difference(ref, est)
    ss_res = float(np.sum(resid**2))
    mean_ref = _circular_mean_deg(ref)
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
    """Validation metrics for paired (estimate, reference) samples."""

    if circular:
        ref = _theta_to_360(reference)
        est = _theta_to_360(estimate)
        ref, est = coincident_values(ref, est)
        if ref.size == 0:
            return {"n": 0, "bias": np.nan, "rmse": np.nan, "si": np.nan, "r2": np.nan}
        diff = _circular_difference(ref, est)
        return {
            "n": int(ref.size),
            "bias": float(np.mean(diff)),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "si": np.nan,
            "r2": circular_r2(ref, est),
        }

    x, y = coincident_values(estimate, reference)
    if x.size == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "si": np.nan, "r2": np.nan}
    return {
        "n": int(x.size),
        "bias": float(bias(x, y)),
        "rmse": float(rmse(x, y)),
        "si": float(si(x, y)),
        "r2": float(r2(x, y)),
    }


def _site_metadata(ds: xr.Dataset) -> pd.DataFrame:
    rows: dict[str, list[Any]] = {"site": [str(s) for s in ds.sites.values]}
    for key in ("site_label", "site_depth", "site_source"):
        if key not in ds.coords:
            continue
        values = ds[key].values
        if values.ndim == 0:
            rows[key] = [str(values)] * ds.sizes["sites"]
        else:
            rows[key] = [str(v) for v in values]
    return pd.DataFrame(rows).set_index("site")


def _metric_groups(
    ds: xr.Dataset,
    groupby: str | None,
) -> list[tuple[str, list[str]]]:
    if groupby is None:
        return [("all", [str(s) for s in ds.sites.values])]
    if groupby == "site":
        return [(str(s), [str(s)]) for s in ds.sites.values]
    if groupby == "site_label":
        if "site_label" not in ds.coords:
            raise ValueError("site_label coordinate required for this groupby")
        labels = sorted({str(v) for v in ds["site_label"].values})
        label_by_site = _per_site_coord(ds, "site_label", "")
        return [
            (
                label,
                [
                    s
                    for s in ds.sites.values.astype(str)
                    if label_by_site.get(s) == label
                ],
            )
            for label in labels
        ]
    raise ValueError(f"Unsupported groupby={groupby!r}")


def metrics_table(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    groupby: str | None = None,
    variables: Iterable[str] | None = None,
    estimate_source: str = "metamodel",
    reference_source: str = "dynamic",
) -> pd.DataFrame:
    """Compute validation metrics for each parameter (and optional grouping)."""

    params = list(variables or validation_parameters())
    rows: list[dict[str, Any]] = []
    meta_info = _site_metadata(estimate)

    for group_name, site_ids in _metric_groups(estimate, groupby):
        est_g = estimate.sel(sites=site_ids)
        ref_g = reference.sel(sites=site_ids)
        for param in params:
            spec = PARAMETERS[param]
            est_vals, ref_vals = _paired_arrays(
                est_g,
                ref_g,
                param,
                estimate_source=estimate_source,
                reference_source=reference_source,
            )
            m = compute_metrics(est_vals, ref_vals, circular=spec["circular"])
            row = {
                "group": group_name,
                "variable": param,
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


def compare_models_table(
    reference: xr.Dataset,
    models: Mapping[str, xr.Dataset],
    *,
    groupby: str | None = "site",
    reference_source: str = "buoy",
) -> pd.DataFrame:
    """Metrics for several models against one reference (buoy validation)."""

    rows: list[dict[str, Any]] = []
    for model_name, estimate in models.items():
        sub = metrics_table(
            estimate,
            reference,
            groupby=groupby,
            estimate_source=model_name,
            reference_source=reference_source,
        )
        sub.insert(0, "model", model_name)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _log_metrics_table(table: pd.DataFrame, *, prefix: str = "") -> None:
    for _, row in table.iterrows():
        if "model" in row:
            label = f"{row['model']} {row['label']}"
            if row.get("group") not in (None, "all"):
                label += f" ({row['group']})"
        else:
            label = str(row["label"])
        r2_val = row["r2"]
        r2_txt = f"{r2_val:.3f}" if np.isfinite(r2_val) else "n/a"
        logger.info(
            "%s%s: n=%d  RMSE=%.3f  BIAS=%.3f  R²=%s",
            prefix,
            label,
            int(row["n"]),
            row["rmse"],
            row["bias"],
            r2_txt,
        )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Aligned estimate vs reference with pooled metrics."""

    estimate: xr.Dataset
    reference: xr.Dataset
    metrics: pd.DataFrame
    sites: list[str]
    estimate_name: str = "metamodel"
    reference_name: str = "dynamic"

    @property
    def metamodel(self) -> xr.Dataset:
        return self.estimate

    @property
    def dynamic(self) -> xr.Dataset:
        return self.reference


@dataclass
class BuoyValidationResult:
    """Buoy reference with aligned model outputs and comparison metrics."""

    reference: xr.Dataset
    models: dict[str, xr.Dataset]
    metrics: pd.DataFrame
    sites: list[str]


# Backward-compatible alias
ValidationResult = ComparisonResult


# ---------------------------------------------------------------------------
# High-level validation entry points
# ---------------------------------------------------------------------------


def validate_isobaths(
    *,
    metamodel_path: str | Path | None = None,
    dynamic_path: str | Path | None = None,
    time: str | slice | None = None,
    sites: Iterable[str] | str | None = None,
    site_source: str | None = "isobath",
    site_label: str | Iterable[str] | None = None,
) -> ComparisonResult:
    """Validate spectral metamodel against SnapWave dynamic on isobath sites."""

    meta_raw = normalize_metamodel(load_metamodel(metamodel_path))
    dynamic = normalize_dynamic(load_dynamic(dynamic_path))
    estimate, reference = align_pair(
        meta_raw,
        dynamic,
        time=time,
        sites=sites,
        site_source=site_source,
        site_label=site_label,
    )
    table = metrics_table(
        estimate,
        reference,
        estimate_source="metamodel",
        reference_source="dynamic",
    )
    sites_out = [str(s) for s in estimate.sites.values]
    _log_metrics_table(table)
    return ComparisonResult(
        estimate=estimate,
        reference=reference,
        metrics=table,
        sites=sites_out,
        estimate_name="metamodel",
        reference_name="dynamic",
    )


def validate_against_dynamic(**kwargs: Any) -> ComparisonResult:
    """Backward-compatible alias for :func:`validate_isobaths`."""
    return validate_isobaths(**kwargs)


def validate_buoys(
    *,
    time: str | slice,
    buoys: str | Iterable[str] | None = None,
    models: Sequence[str] = BUOY_MODEL_SOURCES,
    buoy_role: str | None = "validation",
    dynamic_path: str | Path | None = None,
    metamodel_path: str | Path | None = None,
    recompute_metamodel: bool = False,
) -> BuoyValidationResult:
    """Compare metamodel and dynamic SnapWave against buoy observations.

    Parameters
    ----------
    time
        Time selection applied to all sources (e.g. ``"2010-01"`` or
        ``"1986-01-20"``). Must overlap buoy records and model outputs.
    buoys
        Buoy ids or ``buoy_<id>`` site labels; defaults to validation-role buoys.
    models
        Subset of ``"metamodel"`` and ``"dynamic"`` (default: both).
    recompute_metamodel
        If ``True``, run spectral reconstruction for buoy sites instead of reading
        ``spectral_metamodel.nc``.
    """
    reference = load_buoy_observations(buoys, time=time, role=buoy_role)
    site_ids = [str(s) for s in reference.sites.values]

    raw_models: dict[str, xr.Dataset] = {}
    for name in models:
        if name == "metamodel":
            if recompute_metamodel:
                raw_models[name] = load_metamodel_spectral(
                    time,
                    sites=site_ids,
                    metamodel_path=metamodel_path,
                    recompute=True,
                )
            else:
                meta = normalize_metamodel(load_metamodel(metamodel_path))
                raw_models[name] = meta.sel(sites=site_ids, time=time)
        elif name == "dynamic":
            raw_models[name] = load_source(
                "dynamic", time=time, sites=site_ids, path=dynamic_path
            )
        else:
            raise ValueError(
                f"Unknown buoy model source {name!r}; use 'metamodel' and/or 'dynamic'"
            )

    reference, aligned = align_models_to_reference(reference, raw_models, time=time)
    table = compare_models_table(reference, aligned, groupby="site")
    _log_metrics_table(table, prefix="[buoy] ")
    return BuoyValidationResult(
        reference=reference,
        models=aligned,
        metrics=table,
        sites=[str(s) for s in reference.sites.values],
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_MODEL_COLORS = {
    "metamodel": "darkred",
    "dynamic": "darkgreen",
    "downscaled": "darkorange",
    "buoy": "black",
}


def _pooled_series(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    param: str,
    *,
    estimate_source: str = "metamodel",
    reference_source: str = "dynamic",
) -> tuple[np.ndarray, np.ndarray]:
    est, ref = _paired_arrays(
        estimate,
        reference,
        param,
        estimate_source=estimate_source,
        reference_source=reference_source,
    )
    return coincident_values(est.ravel(), ref.ravel())


def plot_scatter(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    variables: Iterable[str] | None = None,
    estimate_source: str = "metamodel",
    reference_source: str = "dynamic",
    estimate_label: str = "Metamodel",
    reference_label: str = "Dynamic",
    figsize: tuple[float, float] = (12, 4),
    cmap: str = "rainbow",
) -> plt.Figure:
    """Density scatter panels (estimate vs reference) for hs / tp / dm."""

    keys = list(variables or validation_parameters())
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=True):
        spec = PARAMETERS[key]
        x, y = _pooled_series(
            estimate,
            reference,
            key,
            estimate_source=estimate_source,
            reference_source=reference_source,
        )
        if x.size == 0:
            ax.set_title(f"{spec['label']} (no data)")
            continue
        if spec["circular"]:
            x = _theta_to_360(x)
            y = _theta_to_360(y)
        validation_scatter(
            axs=ax,
            x=x,
            y=y,
            xlabel=f"{estimate_label} {spec['label']} [{spec['units']}]",
            ylabel=f"{reference_label} {spec['label']} [{spec['units']}]",
            title=spec["label"],
            cmap=cmap,
        )
    fig.tight_layout()
    return fig


def plot_timeseries(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    site: str,
    *,
    variables: Iterable[str] | None = None,
    estimate_source: str = "metamodel",
    reference_source: str = "dynamic",
    estimate_label: str = "Metamodel",
    reference_label: str = "Dynamic",
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Time-series comparison at one site."""

    site = str(site)
    keys = list(variables or validation_parameters())
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=True):
        spec = PARAMETERS[key]
        ev = _var_name(estimate, estimate_source, key)
        rv = _var_name(reference, reference_source, key)
        estimate.sel(sites=site)[ev].plot(
            ax=ax,
            label=estimate_label,
            color=_MODEL_COLORS.get(estimate_source, "C0"),
            lw=1.0,
        )
        reference.sel(sites=site)[rv].plot(
            ax=ax,
            label=reference_label,
            color=_MODEL_COLORS.get(reference_source, "C1"),
            lw=1.0,
            alpha=0.85,
        )
        ax.set_ylabel(f"{spec['label']} [{spec['units']}]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Site {site}", fontsize=11)
    fig.tight_layout()
    return fig


def plot_timeseries_sites(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    sites: Iterable[str],
    *,
    variable: str = "hs",
    ncol: int = 2,
    figsize_per_panel: tuple[float, float] = (5, 3),
) -> plt.Figure:
    """Small multiples of one variable for several sites."""

    site_list = [str(s) for s in sites]
    n = len(site_list)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(figsize_per_panel[0] * ncol, figsize_per_panel[1] * nrow),
        sharex=True,
        squeeze=False,
    )
    spec = PARAMETERS[variable]
    ev = _var_name(estimate, "metamodel", variable)
    rv = _var_name(reference, "dynamic", variable)

    for ax, site in zip(axes.ravel(), site_list, strict=False):
        estimate.sel(sites=site)[ev].plot(
            ax=ax, label="Metamodel", color="darkred", lw=0.9
        )
        reference.sel(sites=site)[rv].plot(
            ax=ax, label="Dynamic", color="darkgreen", lw=0.9, alpha=0.85
        )
        ax.set_title(site, fontsize=9)
        ax.set_ylabel(spec["label"])
        ax.grid(True, alpha=0.3)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"{spec['label']} time series", fontsize=11)
    fig.tight_layout()
    return fig


def plot_metrics_by_label(
    estimate: xr.Dataset,
    reference: xr.Dataset,
    *,
    metric: str = "rmse",
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Bar chart of a metric grouped by ``site_label`` (``5m``, ``10m``, …)."""

    table = metrics_table(estimate, reference, groupby="site_label")
    keys = list(validation_parameters())
    fig, axes = plt.subplots(1, len(keys), figsize=figsize)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=True):
        spec = PARAMETERS[key]
        sub = table[table["variable"] == key].sort_values("group")
        ax.bar(sub["group"], sub[metric], color="steelblue", alpha=0.85)
        ax.set_title(spec["label"])
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"{metric.upper()} by site_label")
    fig.tight_layout()
    return fig


plot_metrics_by_depth = plot_metrics_by_label


def plot_buoy_timeseries(
    result: BuoyValidationResult,
    site: str,
    *,
    variables: Iterable[str] | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Time series at one buoy: observations vs all model sources."""

    site = str(site)
    keys = list(variables or validation_parameters())
    fig, axes = plt.subplots(len(keys), 1, figsize=figsize, sharex=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys, strict=True):
        spec = PARAMETERS[key]
        rv = _var_name(result.reference, "buoy", key)
        result.reference.sel(sites=site)[rv].plot(
            ax=ax, label="Buoy", color=_MODEL_COLORS["buoy"], lw=1.2
        )
        for model_name, ds in result.models.items():
            ev = _var_name(ds, model_name, key)
            ds.sel(sites=site)[ev].plot(
                ax=ax,
                label=model_name,
                color=_MODEL_COLORS.get(model_name, None),
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


def plot_buoy_skill(
    metrics: pd.DataFrame,
    *,
    metric: str = "r2",
    variable: str | None = None,
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """Grouped bar chart of model skill per buoy and variable."""

    table = metrics if variable is None else metrics[metrics["variable"] == variable]
    params = list(validation_parameters()) if variable is None else [variable]
    fig, axes = plt.subplots(1, len(params), figsize=figsize, sharey=True)
    if len(params) == 1:
        axes = [axes]

    models = list(table["model"].unique())
    width = 0.8 / max(len(models), 1)
    for ax, param in zip(axes, params, strict=True):
        spec = PARAMETERS[param]
        sub = table[table["variable"] == param]
        groups = sorted(sub["group"].unique())
        x = np.arange(len(groups))
        for i, model_name in enumerate(models):
            vals = [
                float(
                    sub[(sub["group"] == g) & (sub["model"] == model_name)][
                        metric
                    ].iloc[0]
                )
                if len(sub[(sub["group"] == g) & (sub["model"] == model_name)])
                else np.nan
                for g in groups
            ]
            ax.bar(
                x + i * width,
                vals,
                width=width,
                label=model_name,
                color=_MODEL_COLORS.get(model_name, f"C{i}"),
                alpha=0.85,
            )
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_title(spec["label"])
        ax.set_ylabel(metric.upper())
        ax.grid(True, axis="y", alpha=0.3)

    fig.legend(loc="upper center", ncol=len(models), fontsize=9)
    fig.suptitle(f"Buoy validation — {metric.upper()}")
    fig.tight_layout()
    return fig


__all__ = [
    "LOGGER_NAME",
    "PARAMETERS",
    "SOURCE_VARS",
    "VARIABLES",
    "BUOY_MODEL_SOURCES",
    "BUOY_OBS_COLUMNS",
    "BUOY_OBS_DIR",
    "BuoyValidationResult",
    "ComparisonResult",
    "ValidationResult",
    "align_datasets",
    "align_models_to_reference",
    "align_pair",
    "buoy_obs_path",
    "circular_r2",
    "coincident_values",
    "common_sites",
    "compare_models_table",
    "compute_metrics",
    "dynamic_output_path",
    "filter_sites",
    "load_buoy_observations",
    "load_downscaled_bulk",
    "load_dynamic",
    "load_metamodel",
    "load_metamodel_spectral",
    "load_pair",
    "load_source",
    "load_whacs",
    "logger",
    "metrics_table",
    "normalize_dynamic",
    "normalize_metamodel",
    "normalize_source",
    "plot_buoy_skill",
    "plot_buoy_timeseries",
    "plot_metrics_by_depth",
    "plot_metrics_by_label",
    "plot_scatter",
    "plot_timeseries",
    "plot_timeseries_sites",
    "validate_against_dynamic",
    "validate_buoys",
    "validate_isobaths",
    "validation_parameters",
    "whacs_path",
]

os.makedirs = _makedirs
