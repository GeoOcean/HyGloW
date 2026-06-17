"""Metamodel training and reconstruction for the North Carolina case.

Typical notebook flow::

    from utils.config import METAMODEL, GOALS
    from utils import metamodel as mm

    models = mm.fit_models()                          # train on all SnapWave cases
    bulk   = mm.reconstruct_bulk("1986-01")           # all sites, no QC
    bulk   = mm.apply_missing_forcing_zeros(bulk)     # NaN → 0 where forcing was missing
    spec   = mm.reconstruct_spectral(                # per-site spectral QC
        bulk,
        sites="isobath",
        time="1986-01-15",
    )
    mm.save_spectral(spec)                            # QC-passed sites only

Site and time selectors are plain xarray-style arguments so you can
experiment quickly without touching the training code.
"""

from __future__ import annotations

# bluemath_tk get_file_logger calls makedirs without exist_ok=True
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.logging import get_file_logger

from .config import GOALS, METAMODEL, OFFSHORE_WAVES, PATHS, resolve
from .snapwave_wrapper import normalize_site_dims

_log_dir = resolve("outputs/logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_makedirs = os.makedirs
os.makedirs = lambda path, *args, **kwargs: (
    _makedirs(path, exist_ok=True) if path else None
)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

LOGGER_NAME = "NorthCarolina.metamodel"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("outputs/logs/metamodel.log")),
    level="INFO",
    console=True,
    console_level="INFO",
)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _goal_ids(goals: Iterable[int] | None = None) -> list[int]:
    return [int(g) for g in (goals or GOALS)]


def _num_mda_points() -> int:
    return int(METAMODEL["num_mda_points"])


def _meta_dir(key: str) -> Path:
    template = METAMODEL["paths"][key]
    return resolve(template.format(num_mda_points=_num_mda_points()))


def cases_path(goal: int) -> Path:
    """SnapWave metamodel cases NetCDF for one goal."""
    stem = Path(PATHS["cases_static_dir"]) / "output_sites_{goal}.nc"
    return resolve(stem.as_posix().format(goal=int(goal)))


def model_path(kind: str, goal: int) -> Path:
    """``kind`` is one of ``mda``, ``pca``, ``gp``."""
    return _meta_dir(f"{kind}_dir") / f"{kind}_model_{int(goal)}.pkl"


def whacs_path() -> Path:
    return resolve(OFFSHORE_WAVES[0]["data_path"])


def spectral_output_path() -> Path:
    """Default NetCDF path for the spectral metamodel product."""
    return resolve(METAMODEL["paths"]["spectral_output"])


def bulk_output_path() -> Path:
    """Alias for :func:`spectral_output_path` (legacy name)."""
    return spectral_output_path()


# ---------------------------------------------------------------------------
# Partition / forcing helpers
# ---------------------------------------------------------------------------


def partition_variable_map() -> dict[str, dict[str, str]]:
    """``partition -> {model_var: whacs_var}`` from ``NorthCarolina.json``."""
    return dict(OFFSHORE_WAVES[0]["partition_map"])


def spectral_config() -> dict[str, Any]:
    return dict(METAMODEL["spectral_reconstruction"])


def spectral_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(freq, dir)`` arrays used by :func:`construct_partition`."""
    cfg = spectral_config()
    freq_cfg = cfg["freq"]
    dir_cfg = cfg["dir"]
    freq = np.logspace(
        np.log10(float(freq_cfg["start"])),
        np.log10(float(freq_cfg["stop"])),
        int(freq_cfg["n"]),
    )
    step = int(dir_cfg["step"])
    direction = np.arange(
        float(dir_cfg["start"]),
        float(dir_cfg["stop"]),
        step,
    )
    return freq, direction


def load_cases_goal(goal: int) -> xr.Dataset:
    """Load SnapWave cases for ``goal`` and add wave-direction UV components."""
    from bluemath_tk.core.operations import get_uv_components

    ds = normalize_site_dims(xr.open_dataset(cases_path(goal)))
    u, v = get_uv_components(ds.point_wavdir)
    ds = ds.assign(point_wavdir_u=u, point_wavdir_v=v)
    return ds


# ---------------------------------------------------------------------------
# Site QC  (spectral reconstruction output only)
# ---------------------------------------------------------------------------

CASES_QC_VARS: dict[str, str] = {
    "hs": "point_hm0",
    "tp": "point_tp",
    "dm": "point_wavdir",
}
SPECTRAL_QC_VARS: dict[str, str] = {
    "hs": "hs",
    "tp": "tp",
    "dm": "dm",
}

_EXCLUDED_SITES: set[str] | None = None


_SITE_QC_DEFAULTS: dict[str, dict[str, float | None]] = {
    "hs": {"max": 10.0, "min_mean": 0.005},
    "tp": {"max": 20.0, "min_mean": 0.25},
    "dm": {"max": 360.0, "min_mean": None},
}


def site_qc_config() -> dict[str, dict[str, float]]:
    """Anomaly thresholds for spectral-reconstruction QC (``metamodel.site_qc``).

    Each variable may define ``max`` (spike rejection) and ``min_mean``
    (dead-site rejection, computed over values above ``active_min``).
    Legacy ``spectral_qc`` entries ``[lo, hi]`` are mapped to ``max=hi``.
    """

    raw = (
        METAMODEL.get("site_qc")
        or METAMODEL.get("station_qc")
        or METAMODEL.get("spectral_qc", {})
    )
    active_min = float(raw.get("active_min", 1e-4))
    out: dict[str, dict[str, float]] = {}
    for key, default in _SITE_QC_DEFAULTS.items():
        spec = raw.get(key, default)
        if isinstance(spec, (list, tuple)) and len(spec) == 2:
            merged: dict[str, float | None] = {
                "max": float(spec[1]),
                "min_mean": default.get("min_mean"),
            }
        elif isinstance(spec, dict):
            merged = {**default, **spec}
        else:
            merged = dict(default)
        entry: dict[str, float] = {"max": float(merged["max"])}
        min_mean = merged.get("min_mean")
        if min_mean is not None:
            entry["min_mean"] = float(min_mean)
        entry["active_min"] = active_min
        out[key] = entry
    return out


def site_qc_bounds() -> dict[str, tuple[float, float]]:
    """Legacy ``(0, max)`` view of :func:`site_qc_config`."""

    return {key: (0.0, cfg["max"]) for key, cfg in site_qc_config().items()}


def spectral_qc_bounds() -> dict[str, tuple[float, float]]:
    """Alias for :func:`site_qc_bounds`."""
    return site_qc_bounds()


def excluded_sites_path() -> Path:
    return resolve(
        METAMODEL["paths"].get(
            "excluded_sites",
            METAMODEL["paths"].get("excluded_stations", "outputs/excluded_sites.json"),
        )
    )


def get_excluded_sites() -> list[str]:
    """Site ids excluded by spectral-reconstruction QC."""

    global _EXCLUDED_SITES
    if _EXCLUDED_SITES is None:
        import json

        for path in (
            excluded_sites_path(),
            resolve("outputs/excluded_stations.json"),
        ):
            if path.is_file():
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                names = data if isinstance(data, list) else data.get("excluded", [])
                _EXCLUDED_SITES = {str(s) for s in names}
                break
        else:
            _EXCLUDED_SITES = set()
    return sorted(_EXCLUDED_SITES)


def _save_excluded_sites() -> None:
    import json

    path = excluded_sites_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted(_EXCLUDED_SITES or []), f, indent=2)


def reset_excluded_sites() -> None:
    """Clear the in-memory and on-disk exclusion list."""

    global _EXCLUDED_SITES
    _EXCLUDED_SITES = set()
    path = excluded_sites_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]\n", encoding="utf-8")
    logger.info("Reset excluded-sites list")


def register_excluded_sites(
    sites: Iterable[str],
    *,
    source: str = "qc",
) -> list[str]:
    """Add site labels to the persistent exclusion list."""

    global _EXCLUDED_SITES
    if _EXCLUDED_SITES is None:
        get_excluded_sites()
    assert _EXCLUDED_SITES is not None

    new = {str(s) for s in sites} - _EXCLUDED_SITES
    if new:
        _EXCLUDED_SITES.update(new)
        _save_excluded_sites()
        preview = ", ".join(sorted(new)[:8])
        if len(new) > 8:
            preview += f", … (+{len(new) - 8} more)"
        logger.warning(
            "QC (%s) excluded %d site(s): %s",
            source,
            len(new),
            preview,
        )
    return get_excluded_sites()


def refresh_excluded_from_cases(
    goals: Iterable[int] | None = None,
    *,
    site_source: str | None = "isobath",
    reset: bool = True,
) -> list[str]:
    """Diagnostic only — cases have many zeros per goal so QC is not applied there.

    Use :func:`qc_cases_report` to inspect individual goals manually.
    Returns the current exclusion list unchanged.
    """

    logger.info(
        "refresh_excluded_from_cases is a no-op; site QC runs in "
        "reconstruct_spectral only"
    )
    return get_excluded_sites()


def _site_labels(ds: xr.Dataset, site_dim: str) -> list[str]:
    if site_dim in ds.coords:
        return [str(v) for v in ds[site_dim].values]
    return [str(v) for v in ds[site_dim].values]


def _site_coord_value(
    ds: xr.Dataset,
    name: str,
    site_dim: str,
    index: int,
) -> str | None:
    """Read a per-site metadata coordinate (scalar or aligned on ``site_dim``)."""

    if name not in ds.coords:
        return None
    coord = ds[name]
    if site_dim in coord.dims:
        return str(coord.isel({site_dim: index}).values)
    return str(coord.values)


def _site_finite_values(
    ds: xr.Dataset,
    var: str,
    site_dim: str,
    index: int,
    *,
    reduce_dims: Iterable[str] | None = None,
) -> np.ndarray:
    sub = ds[var]
    if site_dim in sub.dims:
        sub = sub.isel({site_dim: index})
    if reduce_dims:
        flat_dims = [d for d in reduce_dims if d in sub.dims]
        if flat_dims:
            sub = sub.stack(_qc_flat=flat_dims)
    vals = np.asarray(sub.values, dtype=float).ravel()
    return vals[np.isfinite(vals)]


def _site_anomaly_report(
    ds: xr.Dataset,
    var_map: Mapping[str, str],
    site_dim: str,
    *,
    goal: int | None = None,
    config: Mapping[str, Mapping[str, float]] | None = None,
    reduce_dims: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Per-site QC: reject spikes (``max``) and dead sites (``min_mean``)."""

    cfg = dict(config or site_qc_config())
    if reduce_dims is None:
        reduce_dims = tuple(d for d in ds.dims if d != site_dim)

    labels = _site_labels(ds, site_dim)
    rows: list[dict[str, Any]] = []
    for i, site in enumerate(labels):
        passed = True
        reasons: list[str] = []
        row: dict[str, Any] = {"site": site, "passed": True, "reason": ""}
        if goal is not None:
            row["goal"] = int(goal)
        for meta_key in ("site_label", "site_source", "site_depth"):
            meta_val = _site_coord_value(ds, meta_key, site_dim, i)
            if meta_val is not None:
                row[meta_key] = meta_val

        for key, var in var_map.items():
            limits = cfg[key]
            finite = _site_finite_values(ds, var, site_dim, i, reduce_dims=reduce_dims)
            row[f"{key}_min"] = float(np.min(finite)) if finite.size else np.nan
            row[f"{key}_max"] = float(np.max(finite)) if finite.size else np.nan
            row[f"{key}_mean"] = float(np.mean(finite)) if finite.size else np.nan

            if finite.size == 0:
                passed = False
                reasons.append(f"{key}:no_data")
                continue

            vmax = float(np.max(finite))
            if vmax > limits["max"]:
                passed = False
                reasons.append(f"{key}:max")

            min_mean = limits.get("min_mean")
            if min_mean is not None:
                active_min = float(limits.get("active_min", 1e-4))
                active = finite[finite > active_min]
                mean_active = float(np.mean(active)) if active.size else 0.0
                row[f"{key}_mean_active"] = mean_active
                if active.size == 0 or mean_active < min_mean:
                    passed = False
                    reasons.append(f"{key}:dead")

        row["passed"] = passed
        row["reason"] = ",".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def qc_cases_report(cases: xr.Dataset, goal: int | None = None) -> pd.DataFrame:
    """Per-site anomaly report on SnapWave cases (diagnostic; not used in pipeline)."""
    return _site_anomaly_report(
        cases, CASES_QC_VARS, "sites", goal=goal, reduce_dims=("case_num",)
    )


def _filter_cases_sites(cases: xr.Dataset) -> xr.Dataset:
    excluded = set(get_excluded_sites())
    if not excluded:
        return cases
    names = [str(x) for x in cases["sites"].values]
    keep = [i for i, n in enumerate(names) if n not in excluded]
    if not keep:
        raise ValueError("SnapWave cases QC removed all sites")
    return cases.isel(sites=keep)


def _filter_labeled_dataset(ds: xr.Dataset, dim: str) -> xr.Dataset:
    excluded = set(get_excluded_sites())
    if not excluded:
        return ds
    labels = [str(v) for v in ds[dim].values]
    keep = [lb for lb in labels if lb not in excluded]
    if not keep:
        raise ValueError(f"QC removed all sites along {dim!r}")
    if len(keep) == len(labels):
        return ds
    logger.info(
        "Dropping %d excluded site(s) along %r (%d remaining)",
        len(labels) - len(keep),
        dim,
        len(keep),
    )
    return ds.sel({dim: keep})


def _site_indices_for_prediction(goal: int, n_sites: int) -> np.ndarray:
    """Align PCA site count with labelled SnapWave case metadata."""

    meta = normalize_site_dims(xr.open_dataset(cases_path(goal)))
    all_names = meta["sites"].values.astype(str)
    excluded = set(get_excluded_sites())
    keep_idx = np.array(
        [i for i, name in enumerate(all_names) if name not in excluded],
        dtype=int,
    )

    if n_sites == len(all_names):
        return np.arange(len(all_names), dtype=int)
    if n_sites == len(keep_idx):
        return keep_idx
    raise ValueError(
        f"Goal {goal}: model predicts {n_sites} sites but cases file has "
        f"{len(all_names)} ({len(keep_idx)} after QC exclusions). "
        "Retrain with fit_models(force_retrain=True) after refreshing exclusions."
    )


BULK_QC_VARS: dict[str, str] = {
    "hs": "hs",
    "tp": "tp",
    "dm": "wd",
}


def qc_bulk_report(bulk: xr.Dataset) -> pd.DataFrame:
    """Per-site anomaly report on bulk output (diagnostic; not used in pipeline)."""

    reduce_dims = tuple(d for d in bulk.dims if d != "sites")
    return _site_anomaly_report(bulk, BULK_QC_VARS, "sites", reduce_dims=reduce_dims)


def filter_bulk_sites(bulk: xr.Dataset) -> xr.Dataset:
    """Drop sites excluded by a previous spectral QC run (no bulk anomaly test)."""

    return _filter_labeled_dataset(bulk, "sites")


def _forcing_dataframe(
    whacs: xr.Dataset,
    goal: int,
    partition: str,
    *,
    zs: float = 0.0,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Build the metamodel forcing DataFrame for one goal / partition."""

    var_map = partition_variable_map()[partition]
    src_vars = list(var_map.values())
    renamed = {src: dst for dst, src in var_map.items()}

    part_ds = whacs.sel(goal=goal)[src_vars].rename(renamed)
    df = part_ds.to_dataframe()

    if "tp_inv" in df.columns:
        df["tp"] = 1.0 / df.pop("tp_inv")
    df = df[["hs", "tp", "wd", "ds"]]
    df["zs"] = zs
    return df, part_ds["hs"]


def forcing_valid_mask(forcing_df: pd.DataFrame) -> pd.Series:
    """Boolean mask of timesteps with finite WHACS forcing (``hs/tp/wd/ds``).

    Sea and swell partitions often have NaN ``hs`` when that component is
    absent.  Those timesteps are **skipped** during GP prediction; they are
    left as NaN in :func:`reconstruct_bulk` and should be converted to zero
    with :func:`apply_missing_forcing_zeros` before the spectral step.
    """
    return forcing_df[["hs", "tp", "wd", "ds"]].notna().all(axis=1)


def apply_missing_forcing_zeros(
    ds: xr.Dataset,
    *,
    vars: Iterable[str] = ("hs", "tp", "wd", "ds"),
) -> xr.Dataset:
    """Replace NaN with ``0.0`` where offshore forcing was missing.

    :func:`reconstruct_bulk` only runs the metamodel on timesteps with valid
    WHACS forcing and leaves NaN elsewhere (e.g. no sea/swell at that time).
    Call this function **before** :func:`reconstruct_spectral` so wavespectra
    sees zeros instead of NaN.  Setting ``hs`` (and the other bulk variables)
    to zero at those times is intentional: it means "no contribution from this
    partition".

    Parameters
    ----------
    ds
        Bulk or spectral dataset to clean.
    vars
        Data variables to fill.  Defaults to the four bulk parameters.

    Returns
    -------
    xarray.Dataset
        Copy with NaN replaced by ``0.0`` in the selected variables.
    """
    out = ds.copy()
    filled = 0
    for name in vars:
        if name in out:
            n = int(out[name].isnull().sum())
            if n:
                filled += n
                out[name] = out[name].fillna(0.0)
    if filled:
        logger.info(
            "Zero-filled %d NaN values in %s",
            filled,
            ", ".join(name for name in vars if name in ds),
        )
    else:
        logger.info("No NaN values to zero-fill")
    return out


def _goal_coords(goal: int) -> dict[str, float | int]:
    """Offshore goal location from ``NorthCarolina.json`` → ``goals``."""
    entry = GOALS[str(int(goal))]
    return {
        "latitude": float(entry["lat"]),
        "longitude": float(entry["lon"]),
        "seapoint": int(entry["seapoint"]),
    }


def _attach_site_coords(ds: xr.Dataset, goal: int) -> xr.Dataset:
    """Label the ``sites`` dimension and attach site + goal metadata.

    Site ids match the PCA output width (all SnapWave case sites).
    """
    meta = normalize_site_dims(xr.open_dataset(cases_path(goal)))
    idx = _site_indices_for_prediction(goal, ds.sizes["sites"])
    meta = meta.isel(sites=idx)
    names = meta["sites"].values.astype(str)
    goal_meta = _goal_coords(goal)

    coords: dict[str, Any] = {
        "sites": names,
        "latitude": (("goal",), [goal_meta["latitude"]]),
        "longitude": (("goal",), [goal_meta["longitude"]]),
        "seapoint": (("goal",), [goal_meta["seapoint"]]),
    }
    for name in ("site_source", "site_label", "site_depth"):
        if name in meta:
            coords[name] = (("sites",), meta[name].values)

    return ds.assign_coords(coords)


# ---------------------------------------------------------------------------
# Model training / loading
# ---------------------------------------------------------------------------


def fit_models(
    goals: Iterable[int] | None = None,
    *,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
) -> dict[str, dict[int, Any]]:
    """Train or load MDA, PCA and GP models for each goal.

    Returns ``{"mda": {goal: model, ...}, "pca": {...}, "gp": {...}}``.
    """

    from bluemath_tk.core.io import load_model
    from bluemath_tk.datamining.mda import MDA
    from bluemath_tk.datamining.pca import PCA
    from bluemath_tk.interpolation.gps import ExactGPInterpolation

    active = _goal_ids(goals)
    num_mda = _num_mda_points()
    pca_var = float(METAMODEL["pca_variance"])
    dir_vars = list(METAMODEL["directional_variables"])
    vars_stack = list(METAMODEL["vars_to_stack"])

    models: dict[str, dict[int, Any]] = {"mda": {}, "pca": {}, "gp": {}}

    logger.info(
        "Fitting models for %d goal(s)  (num_mda_points=%d, force_retrain=%s)",
        len(active),
        num_mda,
        force_retrain,
    )

    for goal in active:
        mda_p, pca_p, gp_p = (
            model_path("mda", goal),
            model_path("pca", goal),
            model_path("gp", goal),
        )

        if not force_retrain and mda_p.exists() and pca_p.exists() and gp_p.exists():
            models["mda"][goal] = load_model(mda_p)
            models["pca"][goal] = load_model(pca_p)
            models["gp"][goal] = load_model(gp_p)
            logger.info("Goal %2d: loaded existing models", goal)
            continue

        logger.info("Goal %2d: training MDA → PCA → GP", goal)
        cases = load_cases_goal(goal)
        cases_df = cases[["tp", "wd", "ds", "zs"]].to_dataframe()

        mda = MDA(num_centers=num_mda)
        mda.fit(cases_df, directional_variables=dir_vars)
        mda_p.parent.mkdir(parents=True, exist_ok=True)
        mda.save_model(mda_p)
        models["mda"][goal] = mda

        pca = PCA(n_components=pca_var)
        pca.fit_transform(
            data=cases.isel(case_num=mda.centroid_real_indices),
            vars_to_stack=vars_stack,
            coords_to_stack=["sites"],
            pca_dim_for_rows="case_num",
        )
        pca_p.parent.mkdir(parents=True, exist_ok=True)
        pca.save_model(pca_p)
        models["pca"][goal] = pca

        gp = ExactGPInterpolation(epochs=gp_epochs)
        gp.fit(
            subset_data=mda.centroids[["tp", "wd", "ds", "zs"]],
            subset_directional_variables=dir_vars,
            target_data=pca.pcs_df,
        )
        gp_p.parent.mkdir(parents=True, exist_ok=True)
        gp.save_model(gp_p)
        models["gp"][goal] = gp
        logger.info("Goal %2d: models saved to %s", goal, mda_p.parent)

    logger.info("Model fitting complete for %d goal(s)", len(active))
    return models


def load_models(goals: Iterable[int] | None = None) -> dict[str, dict[int, Any]]:
    """Load previously saved models from disk."""
    from bluemath_tk.core.io import load_model

    active = _goal_ids(goals)
    logger.info("Loading models for %d goal(s)", len(active))
    models = {
        kind: {goal: load_model(model_path(kind, goal)) for goal in active}
        for kind in ("mda", "pca", "gp")
    }
    logger.info("Models loaded")
    return models


# ---------------------------------------------------------------------------
# Bulk reconstruction  (hs / tp / wd / ds at every site)
# ---------------------------------------------------------------------------


def reconstruct_bulk(
    time: str | slice,
    *,
    goals: Iterable[int] | None = None,
    partitions: Iterable[str] | None = None,
    models: Mapping[str, Mapping[int, Any]] | None = None,
    zs: float = 0.0,
) -> xr.Dataset:
    """Reconstruct partition bulk parameters at all sites.

    Parameters
    ----------
    time
        Passed to ``whacs.sel(time=...)`` — e.g. ``"1986-01"`` or a slice.
    goals, partitions
        Subset to run; defaults to every goal / every WHACS partition.
    models
        Pre-loaded dict from :func:`fit_models` or :func:`load_models`.
        Loaded from disk when omitted.
    zs
        Constant water level added to the forcing (m).  North Carolina
        currently uses ``0``; hook up CODEC here later if needed.

    Returns
    -------
    xarray.Dataset
        Variables ``hs``, ``tp``, ``wd``, ``ds`` on ``(time, sites)`` plus
        ``goal`` and ``partition`` dimensions.  Timesteps with missing WHACS
        forcing are NaN — run :func:`apply_missing_forcing_zeros` before the
        spectral step.
    """

    from bluemath_tk.core.operations import get_degrees_from_uv

    active_goals = _goal_ids(goals)
    active_parts = list(partitions or partition_variable_map())
    if models is None:
        models = load_models(active_goals)

    whacs = xr.open_zarr(whacs_path()).sel(time=time).load()
    chunks: list[xr.Dataset] = []

    logger.info(
        "Bulk reconstruction: time=%r, goals=%s, partitions=%s",
        time,
        active_goals,
        active_parts,
    )

    for partition in active_parts:
        for goal in active_goals:
            forcing_df, hs_offshore = _forcing_dataframe(whacs, goal, partition, zs=zs)
            valid = forcing_valid_mask(forcing_df)
            forcing_valid = forcing_df.loc[valid]
            n_total = len(forcing_df)
            n_valid = len(forcing_valid)

            if forcing_valid.empty:
                logger.warning(
                    "Goal %2d / %-7s: no valid forcing (%d timesteps) — skipped",
                    goal,
                    partition,
                    n_total,
                )
                continue

            logger.info(
                "Goal %2d / %-7s: reconstructing %d / %d timesteps",
                goal,
                partition,
                n_valid,
                n_total,
            )

            pcs = models["gp"][goal].predict(dataset=forcing_valid).values
            points = models["pca"][goal].inverse_transform(
                xr.Dataset(
                    {"PCs": (("case_num", "n_component"), pcs)},
                    coords={"case_num": forcing_valid.index},
                )
            )
            points = normalize_site_dims(points)
            n_st = points.sizes["sites"]
            pred = xr.Dataset(
                {
                    "hs": (("time", "sites"), points.point_hm0.values),
                    "tp": (("time", "sites"), points.point_tp.values),
                    "wd": (
                        ("time", "sites"),
                        get_degrees_from_uv(
                            points.point_wavdir_u.values,
                            points.point_wavdir_v.values,
                        ),
                    ),
                    "ds": (
                        ("time", "sites"),
                        np.repeat(
                            forcing_valid["ds"].to_numpy()[:, None], n_st, axis=1
                        ),
                    ),
                },
                coords={"time": forcing_valid.index},
            )
            pred["hs"] = pred["hs"] * hs_offshore.sel(time=forcing_valid.index)
            pred = pred.reindex(time=forcing_df.index)
            pred = pred.expand_dims(goal=[goal], partition=[partition])
            pred = _attach_site_coords(pred, goal)
            chunks.append(pred.load())

    if not chunks:
        logger.warning("Bulk reconstruction produced no chunks")
        raise ValueError(
            "Bulk reconstruction produced no chunks (check forcing/time range)"
        )

    result = normalize_site_dims(xr.merge(chunks))
    logger.info(
        "Bulk reconstruction done: %d site(s), %d chunk(s), %.2f GB",
        result.sizes["sites"],
        len(chunks),
        result.nbytes * 1e-9,
    )
    return result


# ---------------------------------------------------------------------------
# Spectral reconstruction  (user-facing site / time selectors)
# ---------------------------------------------------------------------------


def resolve_sites(
    ds: xr.Dataset,
    sites: int | slice | Iterable[int | str] | str | None = None,
) -> list[str | int]:
    """Resolve a flexible site selector to coordinate labels for ``.sel()``.

    When ``sites`` is labelled (the default after :func:`reconstruct_bulk`),
    returned values are strings suitable for ``ds.sel(sites=...)``.

    Examples
    --------
    ``None``                  → all site labels
    ``0`` / ``slice(0, 3)``   → by position
    ``"buoy"`` / ``"isobath"``→ filter on ``site_source``
    ``"buoy_41013"``          → exact / substring match on the label
    ``["5m_0", "buoy_41013"]``→ explicit list
    """

    labels = [str(lb) for lb in ds.coords["sites"].values]
    excluded = set(get_excluded_sites())
    labels = [lb for lb in labels if lb not in excluded]

    if sites is None:
        return labels
    if isinstance(sites, int):
        return [labels[sites]]
    if isinstance(sites, slice):
        return labels[sites]
    if isinstance(sites, str):
        if sites in ("buoy", "isobath") and "site_source" in ds.coords:
            return [
                lb
                for lb, src in zip(
                    ds.coords["sites"].values.astype(str),
                    ds["site_source"].astype(str).values,
                    strict=True,
                )
                if src == sites and lb not in excluded
            ]
        hits = [lb for lb in labels if sites in str(lb)]
        if hits:
            return hits
        if sites in labels:
            return [sites]
        raise ValueError(f"Could not resolve site selector {sites!r}")
    selected = [labels[int(i)] if str(i).isdigit() else str(i) for i in sites]
    return [lb for lb in selected if lb not in excluded]


def resolve_site_indices(
    ds: xr.Dataset,
    sites: int | slice | Iterable[int | str] | str | None = None,
) -> list[int]:
    """Like :func:`resolve_sites` but return integer positions (legacy)."""
    labels = list(ds.coords["sites"].values)
    selected = resolve_sites(ds, sites)
    return [labels.index(s) for s in selected]


def reconstruct_spectral(
    bulk: xr.Dataset,
    *,
    sites: int | slice | Iterable[int] | str | None = None,
    time: str | slice | None = None,
    partitions_used: Iterable[str] | None = None,
    load: bool = True,
) -> xr.Dataset:
    """Reconstruct spectral bulk stats (``hs``, ``tp``, ``dm``) for selected sites.

    This is the expensive step.  Pass a **small** ``sites`` subset while
    experimenting (e.g. ``sites="buoy"`` or ``sites=slice(0, 3)``),
    then widen once happy with the settings.

    Parameters
    ----------
    bulk
        Output of :func:`reconstruct_bulk` (ideally after
        :func:`apply_missing_forcing_zeros`).
    sites
        Flexible site selector — see :func:`resolve_sites`.
    time
        Optional sub-selection applied before the spectral sum.
    partitions_used
        Partition names to stack (default from config:
        ``metamodel.spectral_reconstruction.partitions_used``).
    load
        If ``True``, call ``.load()`` on the result (eager array).
    """

    from wavespectra.construct import construct_partition

    bulk = normalize_site_dims(filter_bulk_sites(bulk))
    spec_cfg = spectral_config()
    freq, direction = spectral_grid()
    parts = list(partitions_used or spec_cfg["partitions_used"])

    ds = bulk.sel(partition=parts)
    if time is not None:
        ds = ds.sel(time=time)

    site_labels = resolve_sites(ds, sites)
    out_sites: list[xr.Dataset] = []
    qc_rows: list[dict[str, Any]] = []
    failed_sites: list[str] = []
    n_sites = len(site_labels)

    logger.info(
        "Spectral reconstruction: %d site(s), partitions=%s, time=%r",
        n_sites,
        parts,
        time,
    )

    for i, site in enumerate(site_labels, start=1):
        logger.info("Spectral site %3d / %d: %s", i, n_sites, site)
        partitions = (
            ds.sel(sites=site)
            .stack(part=("goal", "partition"))
            .rename({"sites": "site"})
        )
        partitions = partitions.assign_coords(site=site)

        full_spec = construct_partition(
            freq_name=spec_cfg["freq_name"],
            freq_kwargs={
                "freq": freq,
                "fp": 1.0 / partitions.tp,
                "hs": partitions.hs,
            },
            dir_name=spec_cfg["dir_name"],
            dir_kwargs={
                "dir": direction,
                "dm": partitions.wd,
                "dspr": partitions.ds,
            },
        )
        bulk_site = full_spec.sum(dim="part").spec.stats(["hs", "tp", "dm"])
        bulk_site = bulk_site.expand_dims(sites=[site])

        passed, row = qc_spectral_site(bulk_site, site)
        qc_rows.append(row)
        if passed:
            out_sites.append(bulk_site)
        else:
            failed_sites.append(site)
            logger.warning(
                "Spectral QC rejected %s (%s)",
                site,
                row.get("reason") or "unknown",
            )

    if failed_sites:
        register_excluded_sites(failed_sites, source="spectral_reconstruction")

    if not out_sites:
        raise ValueError(
            "Spectral QC removed all sites — check bulk reconstruction and thresholds"
        )

    result = xr.concat(out_sites, dim="sites")
    if load:
        result = result.load()
    logger.info(
        "Spectral reconstruction done: %d / %d site(s) passed QC, %.2f MB",
        len(out_sites),
        n_sites,
        result.nbytes * 1e-6,
    )
    return result


# ---------------------------------------------------------------------------
# Spectral QC  (applied automatically in :func:`reconstruct_spectral`)
# ---------------------------------------------------------------------------


def _spectral_site_dim(ds: xr.Dataset) -> str:
    if "sites" in ds.dims:
        return "sites"
    if "site" in ds.dims:
        return "site"
    raise ValueError("Spectral dataset has no sites dimension")


def qc_spectral_site(
    ds_site: xr.Dataset,
    site: str,
    config: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Return whether one reconstructed site passes spectral QC."""

    if "sites" not in ds_site.dims:
        ds_site = ds_site.expand_dims(sites=[site])
    report = _site_anomaly_report(
        ds_site,
        SPECTRAL_QC_VARS,
        "sites",
        config=config,
        reduce_dims=tuple(d for d in ds_site.dims if d != "sites"),
    )
    row = report.iloc[0].to_dict()
    return bool(row["passed"]), row


def qc_spectral_mask(
    ds: xr.Dataset,
    config: Mapping[str, Mapping[str, float]] | None = None,
) -> xr.DataArray:
    """Boolean mask along the sites dimension (``True`` = passed)."""

    dim = _spectral_site_dim(ds)
    report = qc_spectral_report(ds, config=config)
    passed = dict(zip(report["site"].astype(str), report["passed"], strict=True))
    return xr.DataArray(
        [passed.get(str(lb), False) for lb in ds[dim].values],
        dims=[dim],
        coords={dim: ds[dim].values},
    )


def qc_spectral_report(
    ds: xr.Dataset,
    config: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Per-site QC report for spectral ``hs`` / ``tp`` / ``dm``."""

    dim = _spectral_site_dim(ds)
    reduce_dims = tuple(d for d in ds.dims if d != dim)
    return _site_anomaly_report(
        ds,
        SPECTRAL_QC_VARS,
        dim,
        config=config,
        reduce_dims=reduce_dims,
    )


def filter_spectral_qc(
    ds: xr.Dataset,
    config: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Drop sites that fail spectral QC and persist the exclusion list."""

    report = qc_spectral_report(ds, config=config)
    dim = _spectral_site_dim(ds)
    passed = report.loc[report["passed"], "site"].tolist()
    failed = report.loc[~report["passed"], "site"].tolist()

    if failed:
        register_excluded_sites(failed, source="spectral_reconstruction")

    if not passed:
        raise ValueError("Spectral QC removed all sites — check bulk reconstruction")

    filtered = ds.sel({dim: passed})
    logger.info(
        "Spectral QC kept %d / %d site(s) (config: %s)",
        len(passed),
        len(report),
        config or site_qc_config(),
    )
    return filtered, report


def save_spectral(
    ds: xr.Dataset,
    path: str | Path | None = None,
) -> Path:
    """Write QC-passed spectral sites to NetCDF."""
    out = resolve(path) if path is not None else spectral_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    dim = _spectral_site_dim(ds)
    logger.info(
        "Wrote spectral metamodel to %s (%d site(s), %.2f MB)",
        out,
        ds.sizes[dim],
        ds.nbytes * 1e-6,
    )
    return out


__all__ = [
    "BULK_QC_VARS",
    "CASES_QC_VARS",
    "SPECTRAL_QC_VARS",
    "apply_missing_forcing_zeros",
    "bulk_output_path",
    "cases_path",
    "excluded_sites_path",
    "filter_bulk_sites",
    "filter_spectral_qc",
    "fit_models",
    "forcing_valid_mask",
    "get_excluded_sites",
    "load_cases_goal",
    "load_models",
    "logger",
    "LOGGER_NAME",
    "model_path",
    "partition_variable_map",
    "qc_bulk_report",
    "qc_cases_report",
    "qc_spectral_mask",
    "qc_spectral_report",
    "qc_spectral_site",
    "refresh_excluded_from_cases",
    "register_excluded_sites",
    "reset_excluded_sites",
    "reconstruct_bulk",
    "reconstruct_spectral",
    "resolve_sites",
    "resolve_site_indices",
    "save_spectral",
    "spectral_config",
    "spectral_grid",
    "spectral_output_path",
    "spectral_qc_bounds",
    "site_qc_bounds",
    "site_qc_config",
    "whacs_path",
]

os.makedirs = _makedirs
