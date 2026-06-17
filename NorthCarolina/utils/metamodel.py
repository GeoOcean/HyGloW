"""Metamodel training and reconstruction for the North Carolina case.

Typical notebook flow::

    from utils.config import METAMODEL, GOALS
    from utils import metamodel as mm

    models = mm.fit_models()                          # MDA + PCA + GP per goal
    bulk   = mm.reconstruct_bulk("1986-01")           # hs/tp/wd/ds at all sites
    bulk   = mm.apply_missing_forcing_zeros(bulk)     # NaN → 0 where forcing was missing
    spec   = mm.reconstruct_spectral(                # spectral bulk (bottleneck)
        bulk,
        stations="buoy",
        time="1986-01-15",
    )
    mm.save_spectral(spec)

Station and time selectors are plain xarray-style arguments so you can
experiment quickly without touching the training code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.logging import get_file_logger

from .config import GOALS, METAMODEL, OFFSHORE_WAVES, PATHS, resolve

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

    ds = xr.open_dataset(cases_path(goal))
    u, v = get_uv_components(ds.point_wavdir)
    ds = ds.assign(point_wavdir_u=u, point_wavdir_v=v)
    ds = ds.assign_coords(stations=np.arange(ds.sizes["stations"]))
    return ds


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


def _attach_station_coords(ds: xr.Dataset, goal: int) -> xr.Dataset:
    """Label the ``stations`` dimension and attach site + goal metadata.

    After this call you can select sites with xarray label indexing, e.g.
    ``ds.sel(stations="buoy_41013")`` or ``ds.sel(station_source="buoy")``,
    and inspect the offshore goal with ``latitude`` / ``longitude`` /
    ``seapoint`` along the ``goal`` dimension.
    """
    meta = xr.open_dataset(cases_path(goal))
    names = meta["station_name"].values.astype(str)
    goal_meta = _goal_coords(goal)

    coords: dict[str, Any] = {
        "stations": names,
        "latitude": (("goal",), [goal_meta["latitude"]]),
        "longitude": (("goal",), [goal_meta["longitude"]]),
        "seapoint": (("goal",), [goal_meta["seapoint"]]),
    }
    for name in ("station_source", "station_label", "station_depth"):
        if name in meta:
            coords[name] = (("stations",), meta[name].values)

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
            coords_to_stack=["stations"],
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
# Bulk reconstruction  (hs / tp / wd / ds at every station)
# ---------------------------------------------------------------------------


def reconstruct_bulk(
    time: str | slice,
    *,
    goals: Iterable[int] | None = None,
    partitions: Iterable[str] | None = None,
    models: Mapping[str, Mapping[int, Any]] | None = None,
    zs: float = 0.0,
) -> xr.Dataset:
    """Reconstruct partition bulk parameters at all stations.

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
        Variables ``hs``, ``tp``, ``wd``, ``ds`` on ``(time, stations)`` plus
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
            n_st = points.sizes["stations"]
            pred = xr.Dataset(
                {
                    "hs": (("time", "stations"), points.point_hm0.values),
                    "tp": (("time", "stations"), points.point_tp.values),
                    "wd": (
                        ("time", "stations"),
                        get_degrees_from_uv(
                            points.point_wavdir_u.values,
                            points.point_wavdir_v.values,
                        ),
                    ),
                    "ds": (
                        ("time", "stations"),
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
            pred = _attach_station_coords(pred, goal)
            chunks.append(pred.load())

    if not chunks:
        logger.warning("Bulk reconstruction produced no chunks")
        raise ValueError(
            "Bulk reconstruction produced no chunks (check forcing/time range)"
        )

    result = xr.merge(chunks)
    logger.info(
        "Bulk reconstruction done: %d chunk(s), %.2f GB",
        len(chunks),
        result.nbytes * 1e-9,
    )
    return result


# ---------------------------------------------------------------------------
# Spectral reconstruction  (user-facing station / time selectors)
# ---------------------------------------------------------------------------


def resolve_stations(
    ds: xr.Dataset,
    stations: int | slice | Iterable[int | str] | str | None = None,
) -> list[str | int]:
    """Resolve a flexible station selector to coordinate labels for ``.sel()``.

    When ``stations`` is labelled (the default after :func:`reconstruct_bulk`),
    returned values are strings suitable for ``ds.sel(stations=...)``.

    Examples
    --------
    ``None``                  → all station labels
    ``0`` / ``slice(0, 3)``   → by position
    ``"buoy"`` / ``"isobath"``→ filter on ``station_source``
    ``"buoy_41013"``          → exact / substring match on the label
    ``["5m_0", "buoy_41013"]``→ explicit list
    """

    labels = list(ds.coords["stations"].values)

    if stations is None:
        return labels
    if isinstance(stations, int):
        return [labels[stations]]
    if isinstance(stations, slice):
        return labels[stations]
    if isinstance(stations, str):
        if stations in ("buoy", "isobath") and "station_source" in ds.coords:
            src = ds["station_source"].values
            return [labels[i] for i, s in enumerate(src) if s == stations]
        hits = [lb for lb in labels if stations in str(lb)]
        if hits:
            return hits
        if stations in labels:
            return [stations]
        raise ValueError(f"Could not resolve station selector {stations!r}")
    return [labels[int(i)] if str(i).isdigit() else i for i in stations]


def resolve_station_indices(
    ds: xr.Dataset,
    stations: int | slice | Iterable[int | str] | str | None = None,
) -> list[int]:
    """Like :func:`resolve_stations` but return integer positions (legacy)."""
    labels = list(ds.coords["stations"].values)
    selected = resolve_stations(ds, stations)
    return [labels.index(s) for s in selected]


def reconstruct_spectral(
    bulk: xr.Dataset,
    *,
    stations: int | slice | Iterable[int] | str | None = None,
    time: str | slice | None = None,
    partitions_used: Iterable[str] | None = None,
    load: bool = True,
) -> xr.Dataset:
    """Reconstruct spectral bulk stats (``hs``, ``tp``, ``dm``) for selected sites.

    This is the expensive step.  Pass a **small** ``stations`` subset while
    experimenting (e.g. ``stations="buoy"`` or ``stations=slice(0, 3)``),
    then widen once happy with the settings.

    Parameters
    ----------
    bulk
        Output of :func:`reconstruct_bulk` (ideally after
        :func:`apply_missing_forcing_zeros`).
    stations
        Flexible site selector — see :func:`resolve_stations`.
    time
        Optional sub-selection applied before the spectral sum.
    partitions_used
        Partition names to stack (default from config:
        ``metamodel.spectral_reconstruction.partitions_used``).
    load
        If ``True``, call ``.load()`` on the result (eager array).
    """

    from wavespectra.construct import construct_partition

    spec_cfg = spectral_config()
    freq, direction = spectral_grid()
    parts = list(partitions_used or spec_cfg["partitions_used"])

    ds = bulk.sel(partition=parts)
    if time is not None:
        ds = ds.sel(time=time)

    site_labels = resolve_stations(ds, stations)
    out_sites: list[xr.Dataset] = []
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
            ds.sel(stations=site)
            .stack(part=("goal", "partition"))
            .rename({"stations": "site"})
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
        bulk_site = bulk_site.assign_coords(station=site)
        out_sites.append(bulk_site)

    result = xr.concat(out_sites, dim="site")
    if load:
        result = result.load()
    logger.info(
        "Spectral reconstruction done: %d site(s), %.2f MB",
        n_sites,
        result.nbytes * 1e-6,
    )
    return result


def save_spectral(ds: xr.Dataset, path: str | Path | None = None) -> Path:
    """Write the spectral reconstruction to NetCDF."""
    out = resolve(path) if path is not None else spectral_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    logger.info("Wrote spectral metamodel to %s (%.2f MB)", out, ds.nbytes * 1e-6)
    return out


__all__ = [
    "apply_missing_forcing_zeros",
    "bulk_output_path",
    "cases_path",
    "fit_models",
    "forcing_valid_mask",
    "load_cases_goal",
    "load_models",
    "logger",
    "LOGGER_NAME",
    "model_path",
    "partition_variable_map",
    "reconstruct_bulk",
    "reconstruct_spectral",
    "resolve_stations",
    "resolve_station_indices",
    "save_spectral",
    "spectral_config",
    "spectral_grid",
    "spectral_output_path",
    "whacs_path",
]
