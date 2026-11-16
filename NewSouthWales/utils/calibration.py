"""Satellite validation and calibration of an offshore hindcast at the goals.

The offshore hindcast (any ``offshore_waves`` entry: WHACS, ERA5, ...) is
compared with satellite altimetry around each goal's hindcast node, then
calibrated per goal with the directional method of
:class:`bluemath_tk.waves.calibration.CalVal` (wind sea and swell, per
direction sector). This module is the case layer, driven by the
``satellite`` block of the case JSON:

- :func:`satellite_pairs` — altimeter passes within ``radius_km`` of each
  goal node (deeper than ``min_depth``), one median Hs per pass, with the
  hindcast at the pass time (``outputs/calibration/{offshore}/pairs.csv``);
- :func:`fit_calibration` — one :class:`CalVal` per goal, fitted on
  ``train_fraction`` of the passes and validated on the rest
  (``calval_goal_XX.pkl``, ``coefficients.csv``, ``metrics.csv``);
- :func:`fit_quantile_mapping` — one tail map over the passes of every goal
  (:func:`quantile_mapping_pairs`; ``quantile_mapping.pkl``,
  ``quantile_mapping.csv``), on the directionally calibrated hindcast: the
  sectors fix the level, this the storm tail;
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
        Recompute instead of reading ``pairs.csv``.

    Returns
    -------
    pandas.DataFrame
        One row per pass and goal: ``goal``, ``time``, the pass (``sat_hs``,
        ``sat_n``, ``sat_lon``, ``sat_lat``, ``sat_distance_km``,
        ``sat_depth``, ``sat_satellite``) and the hindcast variables.
    """
    cfg = config()
    path = calibration_dir(offshore_name) / "pairs.csv"
    wanted = [str(g) for g in goals] if goals is not None else None
    if path.exists() and not refresh:
        pairs = pd.read_csv(path, parse_dates=["time"], dtype={"goal": str})
        return pairs if wanted is None else pairs[pairs["goal"].isin(wanted)]

    nodes = goal_nodes(offshore_name)
    if wanted is not None:
        nodes = nodes.loc[wanted]
    altimeter = open_satellite()
    hindcast = open_offshore_waves(offshore_name)
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


def quantile_mapping_path(offshore_name: str) -> Path:
    """``outputs/calibration/{offshore_name}/quantile_mapping.pkl``."""
    return calibration_dir(offshore_name) / "quantile_mapping.pkl"


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


def fit_quantile_mapping(offshore_name: str) -> QuantileMapping:
    """Fit the tail map on every pass of every goal, after the directional step.

    Pooled because one goal has too few storm passes; on all the passes, so the
    test passes of :func:`fit_calibration` are not independent of this step.

    Parameters
    ----------
    offshore_name
        ``offshore_waves`` entry, already through :func:`fit_calibration`.

    Returns
    -------
    QuantileMapping
        Fitted on :func:`quantile_mapping_pairs`. Also writes
        ``quantile_mapping.pkl`` and ``quantile_mapping.csv`` in
        ``outputs/calibration/{offshore_name}/``.
    """
    pairs = quantile_mapping_pairs(offshore_name)
    mapping = QuantileMapping(tail_quantile=float(config()["tail_quantile"])).fit(
        pairs["hs"].to_numpy(), pairs["sat_hs"].to_numpy()
    )
    mapping.save_model(str(quantile_mapping_path(offshore_name)))
    mapping.table.to_csv(calibration_dir(offshore_name) / "quantile_mapping.csv", index=False)
    return mapping


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

    Partition Hs are multiplied by their sector factor and the bulk Hs scaled
    by the change in partition energy; then the tail map of
    :func:`fit_quantile_mapping` corrects the bulk Hs and the partitions by
    the same ratio. Periods, directions and spreads are unchanged. Goals
    without a fitted calibration only get the tail map.

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
    ds = open_offshore_waves(offshore_name).load()
    parts = []
    for gid in ds["goal"].values:
        node = ds.sel(goal=gid)
        cv = calibrations.get(str(int(gid)))
        parts.append(cv.correct(node) if cv is not None else node)
    mapping = load_model(str(quantile_mapping_path(offshore_name)))
    out = mapping.correct(xr.concat(parts, dim="goal"))
    cfg = config()
    out.attrs = {
        **ds.attrs,
        "hywaves_calibration": json.dumps(
            {
                "source": offshore_name,
                "reference": cfg["name"],
                "method": "directional CalVal (wind sea / swell per sector), QuantileMapping",
                "tail_quantile": cfg["tail_quantile"],
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
