"""Satellite validation and calibration of an offshore hindcast at the goals.

The offshore hindcast (any ``offshore_waves`` entry: WHACS, ERA5, ...) is
compared with satellite altimetry around each goal's hindcast node, then
calibrated per goal with the directional method of
:class:`bluemath_tk.waves.calibration.CalVal` (wind sea and swell, per
direction sector). This module is the case layer, driven by the
``satellite`` block of the case JSON:

- :func:`partition_energy_share` — how much of the bulk energy the hindcast
  partitions hold; :func:`closed_hindcast` scales them to the bulk
  (:func:`bluemath_tk.waves.hywaves.offshore.scale_partitions_to_bulk`) unless
  ``close_partitions`` is false; every step below reads it, so the calibrated
  hindcast is closed (or not) accordingly;
- :func:`satellite_pairs` — altimeter passes within ``radius_km`` of each
  goal node (deeper than ``min_depth``), one median Hs per pass, with the
  hindcast at the pass time (``outputs/calibration/{offshore}/pairs.csv``);
- :func:`fit_calibration` — one :class:`CalVal` per goal, fitted on
  ``train_fraction`` of the passes and validated on the rest
  (``calval_goal_XX.pkl``, ``coefficients.csv``, ``metrics.csv``);
- :func:`fit_quantile_mapping` — one tail map per goal on its own passes
  (:func:`quantile_mapping_pairs`; ``quantile_mapping_goal_XX.pkl``,
  ``quantile_mapping.csv``), on the directionally calibrated hindcast: the
  sectors fix the level, this the storm tail; the tail quantile of each goal
  keeps ``min_tail_samples`` passes above it (at most ``tail_quantile``);
- :func:`write_calibrated_hindcast` — the calibrated hindcast as a new zarr,
  to be added to ``offshore_waves`` (e.g. ``WHACS_cal``) and used by the
  dynamic runs and the reconstruction like any other offshore source;
- :func:`buoy_check_sites`, :func:`buoy_check`, :func:`buoy_check_metrics` —
  the independent check: raw and calibrated Hs at the closest goal node
  against deep buoys near the goals.

Notebook ``07b_Satellite_Calibration.ipynb``; method in
:mod:`bluemath_tk.waves.calibration` and :mod:`bluemath_tk.waves.altimetry`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.io import load_model
from bluemath_tk.core.metrics import compute_metrics
from bluemath_tk.waves.altimetry import altimeter_near, altimeter_passes, collocate
from bluemath_tk.waves.calibration import CalVal, QuantileMapping
from bluemath_tk.waves.hywaves.offshore import scale_partitions_to_bulk

from utils.config import (
    SATELLITE,
    get_offshore_wave,
    load_goals,
    offshore_catalog_key,
    open_offshore_waves,
    open_satellite,
    resolve,
)
from utils import onshore_sites as obs
from utils.geometry import site_meta
from utils import validation as val

logger = logging.getLogger("hywaves.calibration")

#: Defaults of the ``satellite`` block (the case JSON overrides them).
DEFAULTS: dict[str, Any] = {
    "name": "altimeter",
    "radius_km": 50.0,
    "min_depth": 15.0,
    "flags": [1, 2],
    "max_time_diff": "30min",
    "max_gap": "5min",
    "min_points": 3,
    "train_fraction": 0.75,
    "split": "random",
    "seed": 0,
    "sector_size": 22.5,
    "min_samples": 20,
    "tail_quantile": 0.999,
    "min_tail_samples": 5,
    "close_partitions": True,
}


def config() -> dict[str, Any]:
    """The ``satellite`` block over :data:`DEFAULTS`."""
    return {**DEFAULTS, **SATELLITE}


def calibration_dir(offshore_name: str) -> Path:
    """``outputs/calibration/{offshore_name}/`` (created)."""
    path = resolve(f"outputs/calibration/{offshore_name}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def goal_nodes(offshore_name: str) -> pd.DataFrame:
    """Hindcast node of each goal for *offshore_name* (goals JSON ``hindcast``).

    Returns
    -------
    pandas.DataFrame
        Indexed by goal id (str): ``lon``, ``lat`` of the node whose series
        the offshore zarr holds (the goal target when the goals JSON has no
        snap for this hindcast).
    """
    key = offshore_catalog_key(get_offshore_wave(offshore_name))
    rows = {}
    for gid, goal in load_goals().items():
        snap = (goal.get("hindcast") or {}).get(key) or goal["target"]
        rows[str(gid)] = {"lon": float(snap["longitude"]), "lat": float(snap["latitude"])}
    return pd.DataFrame.from_dict(rows, orient="index")


def partition_indices(ds: xr.Dataset) -> list[int]:
    """Partition numbers ``i`` of the ``phs{i}`` in *ds* (empty for a bulk-only hindcast)."""
    return sorted(int(n[3:]) for n in ds.data_vars if n.startswith("phs") and n[3:].isdigit())


def partition_energy_share(offshore_name: str) -> pd.DataFrame:
    """Share of the bulk energy (Hs²) the hindcast partitions hold, per goal.

    Parameters
    ----------
    offshore_name
        ``offshore_waves`` entry (as written by 07a, before any closure).

    Returns
    -------
    pd.DataFrame
        Indexed by goal: ``partitions`` (how many the hindcast has),
        ``energy_share`` (Σ phs² / Σ hs² over every hour), ``hours_last_full``
        (share of hours whose last partition holds energy) and
        ``energy_share_last_full`` / ``energy_share_others`` (the share in
        those hours and in the rest). A hindcast that writes a fixed number of
        partitions, in order, loses energy only when its last slot is used and
        more systems exist. Empty for a bulk-only hindcast.
    """
    ds = open_offshore_waves(offshore_name)
    parts = partition_indices(ds)
    if not parts:
        logger.info("%s has no partitions: nothing to close", offshore_name)
        return pd.DataFrame()
    rows = {}
    for gid in ds["goal"].values:
        node = ds.sel(goal=gid)[["hs", *(f"phs{i}" for i in parts)]].load()
        bulk = node["hs"].values ** 2
        phs = np.stack([np.nan_to_num(node[f"phs{i}"].values) for i in parts])
        energy = (phs**2).sum(axis=0)
        ok = np.isfinite(bulk) & (bulk > 0)
        full = ok & (phs[-1] > 0)
        share = lambda m: energy[m].sum() / bulk[m].sum() if m.any() else np.nan  # noqa: E731
        rows[str(int(gid))] = {
            "partitions": len(parts),
            "energy_share": share(ok),
            "hours_last_full": full.sum() / max(ok.sum(), 1),
            "energy_share_last_full": share(full),
            "energy_share_others": share(ok & ~full),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("goal")


def closed_hindcast(offshore_name: str) -> xr.Dataset:
    """The offshore hindcast with its partitions scaled to the bulk energy.

    Every step of the calibration reads it, so the calibrated hindcast is closed
    (a spectral sum of its partitions starts at the bulk energy). Unchanged when
    ``close_partitions`` is false, for a hindcast without partitions or whose
    partitions already add up.
    """
    hindcast = open_offshore_waves(offshore_name)
    return scale_partitions_to_bulk(hindcast) if config()["close_partitions"] else hindcast


def satellite_pairs(
    offshore_name: str,
    *,
    goals: Iterable[str | int] | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Altimeter passes around each goal node with the hindcast at the same time.

    Parameters
    ----------
    offshore_name
        ``offshore_waves`` entry (e.g. ``"WHACS"``).
    goals
        Goal ids (default all).
    refresh
        Recompute instead of reading ``pairs.csv`` (also recomputed when the
        offshore zarr is newer than it).

    Returns
    -------
    pandas.DataFrame
        One row per pass and goal: ``goal``, ``time``, the pass (``sat_hs``,
        ``sat_n``, ``sat_lon``, ``sat_lat``, ``sat_distance_km``,
        ``sat_depth``, ``sat_satellite``) and the hindcast variables
        (partitions closed, :func:`closed_hindcast`).
    """
    cfg = config()
    path = calibration_dir(offshore_name) / "pairs.csv"
    wanted = [str(g) for g in goals] if goals is not None else None
    source = resolve(get_offshore_wave(offshore_name)["data_path"])
    if path.exists() and not refresh and path.stat().st_mtime > source.stat().st_mtime:
        pairs = pd.read_csv(path, parse_dates=["time"], dtype={"goal": str})
        return pairs if wanted is None else pairs[pairs["goal"].isin(wanted)]

    nodes = goal_nodes(offshore_name)
    if wanted is not None:
        nodes = nodes.loc[wanted]
    altimeter = open_satellite()
    hindcast = closed_hindcast(offshore_name)
    chunks = []
    for gid, node in nodes.iterrows():
        obs = altimeter_near(
            altimeter,
            node["lon"],
            node["lat"],
            radius_km=float(cfg["radius_km"]),
            flags=tuple(cfg["flags"]),
            min_depth=cfg["min_depth"],
        )
        passes = altimeter_passes(obs, max_gap=cfg["max_gap"], min_points=int(cfg["min_points"]))
        series = (
            hindcast.sel(goal=int(gid))
            .drop_vars([c for c in hindcast.coords if c != "time"])
            .load()
        )
        pairs = collocate(series, passes, max_time_diff=cfg["max_time_diff"])
        logger.info(
            "Goal %s: %d obs, %d passes, %d pairs (%s)",
            gid, len(obs), len(passes), len(pairs), offshore_name,
        )
        chunks.append(pairs.reset_index().assign(goal=str(gid)))
    out = pd.concat(chunks, ignore_index=True)
    out = out[["goal", *[c for c in out.columns if c != "goal"]]]
    if wanted is None:
        out.to_csv(path, index=False)
    return out


def calval_path(offshore_name: str, goal: str | int) -> Path:
    """``outputs/calibration/{offshore}/calval_goal_XX.pkl``."""
    return calibration_dir(offshore_name) / f"calval_goal_{int(goal):02d}.pkl"


def fit_calibration(
    offshore_name: str,
    *,
    goals: Iterable[str | int] | None = None,
    train_fraction: float | None = None,
    split: str | None = None,
    seed: int | None = None,
) -> dict[str, CalVal]:
    """Fit one directional calibration per goal; save models and tables.

    Parameters
    ----------
    offshore_name
        ``offshore_waves`` entry to calibrate.
    goals
        Goal ids (default all).
    train_fraction, split, seed
        Override the ``satellite`` block (defaults 0.75, ``"random"``, 0): the
        share of passes used to fit; the rest only validates.

    Returns
    -------
    dict[str, CalVal]
        Fitted calibration per goal id. Also writes ``calval_goal_XX.pkl``,
        ``coefficients.csv`` and ``metrics.csv`` in
        ``outputs/calibration/{offshore_name}/``.
    """
    cfg = config()
    pairs = satellite_pairs(offshore_name, goals=goals)
    out: dict[str, CalVal] = {}
    coefficients, metrics = [], []
    for gid, group in pairs.groupby("goal", sort=False):
        data = group.drop(columns="goal").set_index("time")
        cv = CalVal(
            sector_size=float(cfg["sector_size"]), min_samples=int(cfg["min_samples"])
        ).fit(
            data,
            train_fraction=float(train_fraction if train_fraction is not None else cfg["train_fraction"]),
            split=split or cfg["split"],
            seed=int(seed if seed is not None else cfg["seed"]),
        )
        cv.save_model(str(calval_path(offshore_name, gid)))
        out[str(gid)] = cv
        coefficients.append(cv.coefficients.assign(goal=str(gid)))
        metrics.append(cv.metrics().assign(goal=str(gid)))

    folder = calibration_dir(offshore_name)
    table = pd.concat(coefficients, ignore_index=True)
    table[["goal", *[c for c in table if c != "goal"]]].to_csv(
        folder / "coefficients.csv", index=False
    )
    summary = pd.concat(metrics, ignore_index=True).assign(offshore=offshore_name)
    first = ["offshore", "goal", "set", "stage"]
    summary[first + [c for c in summary if c not in first]].to_csv(
        folder / "metrics.csv", index=False
    )
    return out


def quantile_mapping_path(offshore_name: str, goal: str | int) -> Path:
    """``outputs/calibration/{offshore_name}/quantile_mapping_goal_XX.pkl``."""
    return calibration_dir(offshore_name) / f"quantile_mapping_goal_{int(goal):02d}.pkl"


def quantile_mapping_pairs(offshore_name: str) -> pd.DataFrame:
    """Passes of every goal with the hindcast Hs before and after the directional step.

    Parameters
    ----------
    offshore_name
        ``offshore_waves`` entry, already through :func:`fit_calibration`.

    Returns
    -------
    pd.DataFrame
        ``goal``, ``time``, ``sat_hs``, ``hs_raw`` and ``hs`` (directionally
        calibrated), in the order :func:`fit_quantile_mapping` fits them.
    """
    calibrations = load_calibration(offshore_name)
    corrected = []
    for gid, group in satellite_pairs(offshore_name).groupby("goal", sort=False):
        data = group.set_index("time")
        corrected.append(
            pd.DataFrame(
                {
                    "goal": str(gid),
                    "sat_hs": data["sat_hs"],
                    "hs_raw": data["hs"],
                    "hs": calibrations[str(gid)].correct(data.drop(columns="goal"))["hs"],
                }
            )
        )
    return pd.concat(corrected).reset_index()


def fit_quantile_mapping(offshore_name: str) -> dict[str, QuantileMapping]:
    """Fit one tail map per goal on its own passes, after the directional step.

    Each goal's tail quantile is the highest that keeps ``min_tail_samples``
    passes above it, at most ``tail_quantile``
    (:meth:`QuantileMapping.tail_quantile_for`). Fitted on all the passes, so
    the test passes of :func:`fit_calibration` are not independent of this step.

    Parameters
    ----------
    offshore_name
        ``offshore_waves`` entry, already through :func:`fit_calibration`.

    Returns
    -------
    dict[str, QuantileMapping]
        Fitted map per goal id. Also writes ``quantile_mapping_goal_XX.pkl``
        and ``quantile_mapping.csv`` (``goal``, ``n``, ``tail_quantile`` and
        the quantile table) in ``outputs/calibration/{offshore_name}/``.
    """
    cfg = config()
    out: dict[str, QuantileMapping] = {}
    tables = []
    for gid, group in quantile_mapping_pairs(offshore_name).groupby("goal", sort=False):
        n = int(group[["hs", "sat_hs"]].notna().all(axis=1).sum())
        tail = QuantileMapping.tail_quantile_for(
            n, float(cfg["tail_quantile"]), int(cfg["min_tail_samples"])
        )
        mapping = QuantileMapping(
            tail_quantile=tail, min_tail_samples=int(cfg["min_tail_samples"])
        ).fit(group["hs"].to_numpy(), group["sat_hs"].to_numpy())
        mapping.save_model(str(quantile_mapping_path(offshore_name, gid)))
        out[str(gid)] = mapping
        tables.append(mapping.table.assign(goal=str(gid), n=n, tail_quantile=tail))
    table = pd.concat(tables, ignore_index=True)
    first = ["goal", "n", "tail_quantile"]
    table[first + [c for c in table if c not in first]].to_csv(
        calibration_dir(offshore_name) / "quantile_mapping.csv", index=False
    )
    return out


def load_quantile_mapping(offshore_name: str) -> dict[str, QuantileMapping]:
    """Fitted tail maps of *offshore_name* by goal id (from :func:`fit_quantile_mapping`)."""
    found = {}
    for path in sorted(calibration_dir(offshore_name).glob("quantile_mapping_goal_*.pkl")):
        found[str(int(path.stem.rsplit("_", 1)[-1]))] = load_model(str(path))
    if not found:
        raise FileNotFoundError(f"No quantile mapping fitted for {offshore_name!r}; run fit_quantile_mapping")
    return found


def mapped_pairs(offshore_name: str) -> pd.DataFrame:
    """:func:`quantile_mapping_pairs` plus ``hs_mapped``, each goal through its own map."""
    mappings = load_quantile_mapping(offshore_name)
    pairs = quantile_mapping_pairs(offshore_name)
    mapped = pd.Series(np.nan, index=pairs.index)
    for gid, group in pairs.groupby("goal"):
        mapped[group.index] = mappings[str(gid)].map(group["hs"].to_numpy())
    return pairs.assign(hs_mapped=mapped)


def load_calibration(offshore_name: str) -> dict[str, CalVal]:
    """Fitted calibrations of *offshore_name* by goal id (from :func:`fit_calibration`)."""
    found = {}
    for path in sorted(calibration_dir(offshore_name).glob("calval_goal_*.pkl")):
        found[str(int(path.stem.rsplit("_", 1)[-1]))] = load_model(str(path))
    if not found:
        raise FileNotFoundError(f"No calibration fitted for {offshore_name!r}; run fit_calibration")
    return found


def calibration_metrics(offshore_name: str) -> pd.DataFrame:
    """``metrics.csv`` of :func:`fit_calibration` (goal, set, stage, n, bias, rmse, ...)."""
    return pd.read_csv(calibration_dir(offshore_name) / "metrics.csv", dtype={"goal": str})


def write_calibrated_hindcast(
    offshore_name: str,
    *,
    name: str | None = None,
    path: str | Path | None = None,
) -> dict[str, str]:
    """Apply the per-goal calibration to the whole hindcast and write a zarr.

    Reads :func:`closed_hindcast`. Per goal, partition Hs are multiplied by
    their sector factor and the bulk Hs scaled by the change in partition
    energy; then the goal's tail map of
    :func:`fit_quantile_mapping` corrects the bulk Hs and the partitions by
    the same ratio. Periods, directions and spreads are unchanged. Every goal
    needs both fits.

    Parameters
    ----------
    offshore_name
        Source ``offshore_waves`` entry.
    name
        Name of the new entry (default ``f"{offshore_name}_cal"``).
    path
        Output zarr (default next to the source, ``*_cal.zarr``).

    Returns
    -------
    dict[str, str]
        The ``offshore_waves`` entry to add to the case JSON: ``name``,
        ``data_path`` and ``hindcast`` (the goals JSON key of the source, so
        its nodes are reused).
    """
    entry = get_offshore_wave(offshore_name)
    source = resolve(entry["data_path"])
    name = name or f"{offshore_name}_cal"
    target = Path(path) if path is not None else source.with_name(
        source.name.replace(".zarr", "") + "_cal.zarr"
    )
    calibrations = load_calibration(offshore_name)
    mappings = load_quantile_mapping(offshore_name)
    ds = closed_hindcast(offshore_name).load()
    goals = [str(int(g)) for g in ds["goal"].values]
    missing = sorted(set(goals) - (set(calibrations) & set(mappings)), key=int)
    if missing:
        raise ValueError(f"Goals {missing} of {offshore_name!r} have no calibration or tail map")
    out = xr.concat(
        [mappings[g].correct(calibrations[g].correct(ds.sel(goal=int(g)))) for g in goals],
        dim="goal",
    )
    cfg = config()
    out.attrs = {
        **ds.attrs,
        "hywaves_calibration": json.dumps(
            {
                "source": offshore_name,
                "reference": cfg["name"],
                "method": ("partitions scaled to the bulk energy, then " if cfg["close_partitions"] else "")
                + "per goal: directional CalVal (wind sea / swell per sector), QuantileMapping",
                "close_partitions": cfg["close_partitions"],
                "partition_energy_share": ds.attrs.get("partition_energy_share"),
                "tail_quantile": {g: m.tail_quantile for g, m in mappings.items()},
                "radius_km": cfg["radius_km"],
                "min_depth": cfg["min_depth"],
                "train_fraction": cfg["train_fraction"],
            }
        ),
    }
    for var in out.variables:
        out[var].encoding = {}
    out.chunk({"time": -1, "goal": 1}).to_zarr(target, mode="w")
    case_relative = target
    try:
        case_relative = target.relative_to(resolve("."))
    except ValueError:
        pass
    new_entry = {
        "name": name,
        "data_path": str(case_relative),
        "hindcast": entry.get("hindcast", entry["name"]),
    }
    logger.info("Calibrated %s written to %s; add to offshore_waves: %s", offshore_name, target, new_entry)
    return new_entry


def buoy_check_sites(
    offshore_name: str, *, max_distance_km: float = 50.0, min_depth: float = 24.0
) -> pd.DataFrame:
    """Buoys where the offshore hindcast itself can be checked: deep and near a goal node.

    Parameters
    ----------
    offshore_name
        ``offshore_waves`` entry whose goal nodes are used.
    max_distance_km, min_depth
        Largest distance to the closest goal node and smallest mesh depth.

    Returns
    -------
    pd.DataFrame
        ``site``, ``goal``, ``distance_km``, ``depth`` of the kept buoys.
    """
    table = val.closest_goals_table(obs.list_snapwave_buoy_sites(), [offshore_name])
    depth = site_meta(table["site"])["site_mesh_depth"].to_numpy()
    table = table.assign(distance_km=table["distance_m"] / 1000.0, depth=depth)
    keep = (table["distance_km"] <= max_distance_km) & (table["depth"] >= min_depth)
    return table.loc[keep, ["site", "goal", "distance_km", "depth"]].reset_index(drop=True)


def buoy_check(
    offshore_name: str,
    entry: dict[str, str],
    sites: pd.DataFrame,
    period: str | slice = slice(None),
) -> tuple[xr.DataArray, dict[str, xr.DataArray]]:
    """Buoy Hs and the raw and calibrated hindcast Hs at each buoy's closest goal node.

    Parameters
    ----------
    offshore_name
        Source ``offshore_waves`` entry.
    entry
        Calibrated entry (:func:`write_calibrated_hindcast`); need not be in
        the case JSON yet.
    sites
        :func:`buoy_check_sites`.
    period
        Time selection.

    Returns
    -------
    buoy, estimates
        Hs on ``(time, sites)``; ``{offshore_name: ..., entry["name"]: ...}``
        on the buoy hours.
    """
    buoy = val.load_buoy_observations(period, sites["site"], variables=["hs"])["hs"]
    rows = sites.set_index("site").loc[buoy["sites"].values]
    estimates = {}
    for name, ds in (
        (offshore_name, open_offshore_waves(offshore_name)),
        (entry["name"], open_offshore_waves(entry=entry)),
    ):
        hs = ds["hs"].sel(goal=rows["goal"].to_numpy()).rename(goal="sites")
        hs = hs.assign_coords(sites=rows.index.to_numpy()).drop_vars("seapoint", errors="ignore")
        estimates[name] = hs.reindex(time=buoy["time"]).transpose("time", "sites").load()
    return buoy, estimates


def buoy_check_metrics(
    buoy: xr.DataArray,
    estimates: dict[str, xr.DataArray],
    probabilities: tuple[float, ...] = (0.5, 0.99, 0.999),
) -> pd.DataFrame:
    """Per buoy and estimate: metrics and quantile errors on the shared hours.

    Returns
    -------
    pd.DataFrame
        ``site``, ``estimate``, ``n``, ``bias``, ``rmse``, ``r2`` (reference
        first: bias = hindcast − buoy) and ``q{p}``: hindcast − buoy Hs at
        that quantile (m).
    """
    rows = []
    for site in buoy["sites"].values:
        reference = buoy.sel(sites=site).values
        for name, hs in estimates.items():
            estimate = hs.sel(sites=site).values
            ok = np.isfinite(reference) & np.isfinite(estimate)
            row = {"site": site, "estimate": name, "n": int(ok.sum())}
            row.update(compute_metrics(reference=reference[ok], estimate=estimate[ok], metrics=("bias", "rmse", "r2")))
            for p in probabilities:
                row[f"q{100 * p:g}"] = np.quantile(estimate[ok], p) - np.quantile(reference[ok], p)
            rows.append(row)
    return pd.DataFrame(rows)
