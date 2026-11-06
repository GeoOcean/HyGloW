"""
Build the dashboard store: every result of the Netherlands case for 2013, laid out for fast reading.

Run from this folder (``python build_store.py [--only part ...]``); it reads the
case through ``utils`` and ``analysis/results.py`` and writes ``store/`` (git-ignored):

- ``cube.zarr``: ``{estimate}/{variable}`` on ``(time, site)`` (8760 x 5156), the
  estimates of :data:`data.ESTIMATES`, plus ``wind/u10``, ``wind/u10dir`` (ERA5 at each site);
- ``wind_grid.zarr``: ERA5 ``u10``, ``v10`` on their grid (the shaded wind layer);
- ``sites.parquet``, ``goals.parquet``, ``offshore.zarr`` (hindcasts and water level at the goals);
- ``conditions.parquet`` (one row per hour and source), ``storms.csv``;
- ``metamodel/``: training cases per goal (inputs, train/test, Hs and Dir fields), holdout
  metrics, linear-sum metrics;
- ``calibration/``: satellite pairs, coefficients, metrics, quantile mapping, 1979-2023 quantiles;
- ``numerical_tests/``: the result tables and figures of ``numerical_tests/``;
- ``manifest.json``: build time and what each part read.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
sys.path[:0] = [str(CASE), str(CASE / "analysis")]

import results as rs  # noqa: E402
from utils import config as cfg  # noqa: E402
from utils import metamodel as mm  # noqa: E402

from data import COAST_REGIONS, ESTIMATES, OFFSHORE, STORE, VARIABLES  # noqa: E402

logger = logging.getLogger("hywaves.dashboard")
CHUNK_TIME = 744


def _sites() -> xr.Dataset:
    """Every site of the dynamic runs (isobaths, buoys, DeenHaag mesh nodes), with their metadata."""
    with xr.open_dataset(rs.sim.cases_dir("dynamic", offshore="WHACS", wind=False) / "Dynamic_Cases_2013.nc") as ds:
        return ds.drop_vars(list(ds.data_vars)).load()


def _write_array(group: zarr.Group, name: str, values: np.ndarray) -> None:
    group.create_array(name, data=values.astype("float32"), chunks=(CHUNK_TIME, values.shape[1]), overwrite=True)


def build_sites(sites: xr.Dataset) -> None:
    """``sites.parquet`` (with coast region and closest goal) and ``goals.parquet``."""
    goals = cfg.load_goals()
    node = {g: v["hindcast"]["whacs"] for g, v in goals.items()}
    lat, lon = sites["site_latitude"].values, sites["site_longitude"].values
    gid = np.array(list(node))
    glat = np.array([node[g]["latitude"] for g in gid])
    glon = np.array([node[g]["longitude"] for g in gid])
    d = (lat[:, None] - glat) ** 2 + ((lon[:, None] - glon) * np.cos(np.deg2rad(lat[:, None]))) ** 2
    region = np.array(list(COAST_REGIONS))[np.searchsorted(list(COAST_REGIONS.values()), lat)]
    table = pd.DataFrame({
        "site": sites["sites"].values.astype(str),
        "source": sites["site_source"].values.astype(str),
        "label": sites["site_label"].values.astype(str),
        "depth": sites["site_mesh_depth"].values,
        "nominal_depth": sites["site_nominal_depth"].values,
        "lon": lon, "lat": lat, "region": region,
        "goal": gid[d.argmin(1)].astype(int),
    })
    table.to_parquet(STORE / "sites.parquet")
    rows = [{"goal": int(g), "lon": v["target"]["longitude"], "lat": v["target"]["latitude"],
             "angle_left": v["angles"][0], "angle_right": v["angles"][1],
             "node_lon": node[g]["longitude"], "node_lat": node[g]["latitude"]} for g, v in goals.items()]
    pd.DataFrame(rows).to_parquet(STORE / "goals.parquet")


def build_offshore() -> xr.Dataset:
    """``offshore.zarr``: each source's hindcast at the goals in 2013, and the CODEC water level."""
    out = {}
    for name in OFFSHORE:
        ds = cfg.open_offshore_waves(name).sel(time=rs.PERIOD)[list(VARIABLES)].drop_vars("seapoint", errors="ignore")
        for var in VARIABLES:
            out[f"{name}_{var}"] = ds[var].astype("float32")
    wl = cfg.open_water_level().sel(time=rs.PERIOD)
    out["wl"] = wl.drop_vars([c for c in wl.coords if c not in ("time", "goal")]).astype("float32")
    offshore = xr.Dataset(out).transpose("time", "goal").load()
    offshore["goal"] = offshore["goal"].astype(int)
    offshore.to_zarr(STORE / "offshore.zarr", mode="w")
    return offshore


def build_cube(sites: xr.Dataset, offshore: xr.Dataset) -> None:
    """``cube.zarr``: every estimate on the dynamic-run sites, and the ERA5 wind there."""
    ids = sites["sites"].values
    root = zarr.open_group(STORE / "cube.zarr", mode="a")
    time_index = pd.DatetimeIndex(sites["time"].values)
    root.attrs.update({"time": [str(t) for t in time_index], "sites": [str(s) for s in ids]})

    def put(key: str, ds: xr.Dataset) -> None:
        ds = ds.reindex(time=time_index, sites=ids).transpose("time", "sites")
        group = root.require_group(key)
        for var in VARIABLES:
            _write_array(group, var, ds[var].values)
        logger.info("cube: %s", key)

    buoy = rs.buoys()
    put("buoy", buoy)
    site_goal = pd.read_parquet(STORE / "sites.parquet").set_index("site").loc[ids, "goal"].to_numpy()
    for off in OFFSHORE:
        near = xr.Dataset({v: offshore[f"{off}_{v}"].sel(goal=site_goal).rename(goal="sites")
                           .assign_coords(sites=ids) for v in VARIABLES})
        put(f"offshore_{off}", near)
        for spectral, key in ((False, "mm_bulk"), (True, "mm_spec")):
            parts = [rs.metamodel(off, s, spectral) for s in ("isobaths", "buoys", "deenhaag")]
            put(f"{key}_{off}", xr.concat([p[list(VARIABLES)] for p in parts], "sites"))
        for wind, key in ((False, "dyn_off"), (True, "dyn_on")):
            put(f"{key}_{off}", rs.dynamic(off, wind, ids))
    era5 = rs.wind_at_sites(sites)
    group = root.require_group("wind")
    for var in ("u10", "u10dir"):
        _write_array(group, var, era5[var].transpose("time", "sites").values)
    logger.info("cube: wind")


def build_wind_grid() -> None:
    """``wind_grid.zarr``: the ERA5 10 m wind on its own grid (the shaded layer of the maps)."""
    wind = cfg.open_wind(rs.PERIOD)[["u10", "v10"]].astype("float32").load()
    wind.to_zarr(STORE / "wind_grid.zarr", mode="w")


def build_conditions(offshore: xr.Dataset) -> None:
    """``conditions.parquet`` (per hour and source) and ``storms.csv``."""
    root = zarr.open_group(STORE / "cube.zarr", mode="r")
    time_index = pd.DatetimeIndex(root.attrs["time"])
    sites = pd.read_parquet(STORE / "sites.parquet")
    iso20 = (sites["label"] == "20m").to_numpy()
    u10, u10dir = root["wind/u10"][:], root["wind/u10dir"][:]
    rad = np.deg2rad(u10dir)
    frames = []
    for off in OFFSHORE:
        h0 = root[f"dyn_off_{off}/hs"][:]
        h1 = root[f"dyn_on_{off}/hs"][:]
        d0 = root[f"dyn_off_{off}/dir"][:]
        e20 = h0[:, iso20] ** 2
        w20 = np.deg2rad(d0[:, iso20])
        hs, tp, wd = (offshore[f"{off}_{v}"].values for v in ("hs", "tp", "dir"))
        energy = hs**2
        df = pd.DataFrame({
            "time": time_index, "offshore": off,
            "u10": np.nanmean(u10, 1),
            "u10_max": np.nanmax(u10, 1),
            "u10dir": np.rad2deg(np.arctan2(np.nanmean(u10 * np.sin(rad), 1), np.nanmean(u10 * np.cos(rad), 1))) % 360,
            "hs20": np.nanmean(h0[:, iso20], 1),
            "dir20": np.rad2deg(np.arctan2(np.nansum(e20 * np.sin(w20), 1), np.nansum(e20 * np.cos(w20), 1))) % 360,
            "hs_off": np.nanmean(hs, 1), "hs_off_max": np.nanmax(hs, 1),
            "tp_off": np.nansum(energy * tp, 1) / np.nansum(energy, 1),
            "dir_off": np.rad2deg(np.arctan2(np.nansum(energy * np.sin(np.deg2rad(wd)), 1),
                                             np.nansum(energy * np.cos(np.deg2rad(wd)), 1))) % 360,
            "wl": np.nanmean(offshore["wl"].values, 1),
            "wind_converged": np.isfinite(h1).any(1),
        })
        frames.append(df)
    conditions = pd.concat(frames, ignore_index=True)
    conditions.to_parquet(STORE / "conditions.parquet")
    _storms(conditions[conditions["offshore"] == "WHACS_cal"].set_index("time")).to_csv(STORE / "storms.csv", index=False)


NAMED_STORMS = {"2013-12-05": "Xaver", "2013-10-28": "St Jude (Christian)"}


def _storms(cond: pd.DataFrame, quantile: float = 0.99, gap_hours: int = 48, pad_hours: int = 36) -> pd.DataFrame:
    """Peaks of the mean offshore Hs above its *quantile*, declustered by *gap_hours*."""
    hs = cond["hs_off"]
    above = hs[hs > hs.quantile(quantile)]
    events, start, prev = [], None, None
    for t in above.index:
        if prev is None or (t - prev) > pd.Timedelta(hours=gap_hours):
            if start is not None:
                events.append((start, prev))
            start = t
        prev = t
    if start is not None:
        events.append((start, prev))
    rows = []
    for a, b in events:
        window = hs[a:b]
        peak = window.idxmax()
        name = NAMED_STORMS.get(str(peak.date()), "") or NAMED_STORMS.get(str((peak - pd.Timedelta(days=1)).date()), "")
        rows.append({"peak": peak, "start": a - pd.Timedelta(hours=pad_hours), "end": b + pd.Timedelta(hours=pad_hours),
                     "hs_off_peak": float(window.max()), "hs_off_max_peak": float(cond.loc[peak, "hs_off_max"]),
                     "u10_peak": float(cond.loc[peak, "u10"]), "u10dir_peak": float(cond.loc[peak, "u10dir"]),
                     "dir_off_peak": float(cond.loc[peak, "dir_off"]), "name": name})
    out = pd.DataFrame(rows).sort_values("hs_off_peak", ascending=False).reset_index(drop=True)
    out.insert(0, "event", [f"{r.peak:%d %b %H:%M}" + (f" · {r.name}" if r.name else "") for r in out.itertuples()])
    return out


def build_metamodel() -> None:
    """Training cases per goal, holdout and linear-sum metrics."""
    out = STORE / "metamodel"
    out.mkdir(exist_ok=True)
    fields = zarr.open_group(out / "case_fields.zarr", mode="w")
    tables = []
    for goal in cfg.load_goals():
        train, test = mm.load_split(goal)
        with xr.open_dataset(mm.cases_path(goal)) as ds:
            df = ds[[f"{v}_forcing" for v in mm.MDA_INPUTS]].to_dataframe()
            df.columns = mm.MDA_INPUTS
            group = fields.require_group(str(goal))
            for var in ("hs", "dir"):
                group.create_array(var, data=ds[var].values.astype("float32"), overwrite=True)
            group.attrs["sites"] = [str(s) for s in ds["sites"].values]
        df["role"] = "unused"
        df.loc[df.index[test], "role"] = "test"
        df.loc[df.index[train], "role"] = "train"
        df["mda_order"] = -1
        df.loc[df.index[train], "mda_order"] = np.arange(len(train))
        df["goal"] = int(goal)
        removed = mm.removed_sites(goal)
        fields[str(goal)].attrs["removed_sites"] = list(removed)
        tables.append(df.reset_index())
    pd.concat(tables, ignore_index=True).to_parquet(out / "cases.parquet")
    holdout = [pd.read_csv(p) for p in sorted(cfg.resolve("outputs").glob("goal_*/*/validation_metrics.csv"))]
    pd.concat(holdout, ignore_index=True).to_parquet(out / "holdout.parquet")
    for path in cfg.resolve("outputs/metamodel_validation").glob("*.csv"):
        shutil.copy(path, out / path.name)


QUANTILE_PROBS = np.concatenate([np.linspace(0.01, 0.99, 99), [0.995, 0.999, 0.9995, 0.9999]])


def build_calibration() -> None:
    """
    Satellite calibration per hindcast (``WHACS``, ``ERA5``): the 07b tables, the passes at each stage
    (raw, directional, mapped), the 1979-2023 quantiles per goal, and the deep-buoy check.
    """
    out = STORE / "calibration"
    out.mkdir(exist_ok=True)
    rows = []
    for raw, cal_name in (("WHACS", "WHACS_cal"), ("ERA5", "ERA5_cal")):
        src = cfg.resolve(f"outputs/calibration/{raw}")
        if not src.exists():
            continue
        for name in ("coefficients", "metrics", "quantile_mapping"):
            if (src / f"{name}.csv").exists():
                shutil.copy(src / f"{name}.csv", out / f"{raw}_{name}.csv")
        _calibration_passes(raw).to_csv(out / f"{raw}_passes.csv", index=False)
        for name in (raw, cal_name):
            hs = cfg.open_offshore_waves(name)["hs"].load()
            for goal in hs["goal"].values:
                q = np.nanquantile(hs.sel(goal=goal).values, QUANTILE_PROBS)
                rows += [{"source": name, "hindcast": raw, "goal": int(goal), "p": p, "hs": v}
                         for p, v in zip(QUANTILE_PROBS, q)]
        for name, table in _buoy_check(raw, cal_name).items():
            table.to_csv(out / f"{raw}_buoy_{name}.csv", index=False)
        logger.info("calibration: %s", raw)
    pd.DataFrame(rows).to_parquet(out / "quantiles.parquet")


def _calibration_passes(raw: str) -> pd.DataFrame:
    """Every satellite pass with the hindcast Hs raw, after the directional step and after the tail map."""
    from bluemath_tk.core.io import load_model
    from utils import calibration as cal

    passes = cal.quantile_mapping_pairs(raw).rename(columns={"hs": "hs_directional"})
    mapping = load_model(str(cal.quantile_mapping_path(raw)))
    return passes.assign(hs_mapped=mapping.map(passes["hs_directional"].to_numpy()))


def _buoy_check(raw: str, cal_name: str) -> dict[str, pd.DataFrame]:
    """07b section 5: deep buoys near a goal node, raw and calibrated hindcast, 1979-2023 (not used in the fit)."""
    from utils import calibration as cal

    sites = cal.buoy_check_sites(raw)
    buoy, estimates = cal.buoy_check(raw, cfg.get_offshore_wave(cal_name), sites)
    metrics = cal.buoy_check_metrics(buoy, estimates)
    rows = []
    for site in buoy["sites"].values:
        reference = buoy.sel(sites=site).values
        for name, hs in estimates.items():
            estimate = hs.sel(sites=site).values
            ok = np.isfinite(reference) & np.isfinite(estimate)
            if ok.sum() < 100:
                continue
            rows += [{"site": str(site), "estimate": name, "p": p, "buoy": b, "hindcast": h}
                     for p, b, h in zip(QUANTILE_PROBS, np.quantile(reference[ok], QUANTILE_PROBS),
                                        np.quantile(estimate[ok], QUANTILE_PROBS))]
    return {"sites": sites, "metrics": metrics, "quantiles": pd.DataFrame(rows)}


def build_numerical_tests() -> None:
    """Copy the result tables, figures and README of ``numerical_tests/``."""
    src = CASE / "numerical_tests"
    out = STORE / "numerical_tests"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src / "results", out, ignore=shutil.ignore_patterns("raw"))
    shutil.copy(src / "README.md", out / "README.md")


PARTS = {"sites": None, "offshore": None, "cube": None, "conditions": None, "wind_grid": build_wind_grid,
         "metamodel": build_metamodel, "calibration": build_calibration, "numerical_tests": build_numerical_tests}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--only", nargs="*", choices=list(PARTS), help="parts to (re)build; default all")
    args = parser.parse_args()
    parts = args.only or list(PARTS)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    STORE.mkdir(exist_ok=True)
    manifest_path = STORE / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    sites = _sites() if {"sites", "cube"} & set(parts) else None
    offshore = None
    for part in parts:
        t0 = time.time()
        if part == "sites":
            build_sites(sites)
        elif part == "offshore":
            offshore = build_offshore()
        elif part in ("cube", "conditions"):
            offshore = offshore if offshore is not None else xr.open_zarr(STORE / "offshore.zarr").load()
            (build_cube(sites, offshore) if part == "cube" else build_conditions(offshore))
        else:
            PARTS[part]()
        manifest[part] = {"built": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0)}
        logger.info("%s done in %.0f s", part, time.time() - t0)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
