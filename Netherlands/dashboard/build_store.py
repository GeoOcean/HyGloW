"""
Build the dashboard store: every result of the Netherlands case for 2013, laid out for fast reading.

Run from this folder (``python build_store.py [--only part ...]``); it reads the
case through ``utils`` (:mod:`sources`) and writes ``store/`` (git-ignored):

- ``cube.zarr``: ``{estimate}/{variable}`` on ``(time, site)`` (8760 x 5156), the
  estimates of :data:`data.ESTIMATES`, plus ``wind/u10``, ``wind/u10dir`` (ERA5 at each site),
  ``wind_only/`` (SnapWave without boundary waves, storm hours only: part ``wind_only``),
  ``dyn_*/dw, df, zs`` of the SnapWave runs (part ``breaking``) and ``goals_{source}/dw, qb``
  (each goal alone through SnapWave's breaking: part ``goal_breaking``);
- ``breaking/``: SnapWave's breaking parameters, ``dw`` and ``zs`` of the training cases;
- ``wind_grid.zarr``: ERA5 ``u10``, ``v10`` on their grid (the shaded wind layer);
- ``sites.parquet``, ``goals.parquet``, ``offshore.zarr`` (hindcasts and water level at the goals);
- ``conditions.parquet`` (one row per hour and source; SWAN's final accuracy), ``storms.csv``;
- ``maps.zarr``: grid maps of the dynamic runs (SnapWave, SWAN) at the hours where two or more
  have one, on the mesh nodes, with the image raster;
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
sys.path.insert(0, str(CASE))

import breaking as brk  # noqa: E402
import sources as src  # noqa: E402
from utils import config as cfg  # noqa: E402
from utils import metamodel as mm  # noqa: E402
from utils import simulation as sim  # noqa: E402

from data import COAST_REGIONS, DYNAMIC, DYNAMIC_KINDS, ESTIMATES, OFFSHORE, STORE, VARIABLES  # noqa: E402

logger = logging.getLogger("hywaves.dashboard")
CHUNK_TIME = 744


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
        ds = cfg.open_offshore_waves(name).sel(time=src.PERIOD)[list(VARIABLES)].drop_vars("seapoint", errors="ignore")
        for var in VARIABLES:
            out[f"{name}_{var}"] = ds[var].astype("float32")
    wl = cfg.open_water_level().sel(time=src.PERIOD)
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

    buoy = src.buoys()
    put("buoy", buoy)
    site_goal = pd.read_parquet(STORE / "sites.parquet").set_index("site").loc[ids, "goal"].to_numpy()
    for off in OFFSHORE:
        near = xr.Dataset({v: offshore[f"{off}_{v}"].sel(goal=site_goal).rename(goal="sites")
                           .assign_coords(sites=ids) for v in VARIABLES})
        put(f"offshore_{off}", near)
        for spectral, key in ((False, "mm_bulk"), (True, "mm_spec")):
            parts = [src.metamodel(off, s, spectral) for s in ("isobaths", "buoys", "deenhaag")]
            put(f"{key}_{off}", xr.concat([p[list(VARIABLES)] for p in parts], "sites"))
    for key in DYNAMIC:
        model, wind = DYNAMIC_KINDS[ESTIMATES[key]["kind"]]
        put(key, src.dynamic(ESTIMATES[key]["offshore"], wind, model, ids))
    era5 = src.wind_at_sites(sites)
    group = root.require_group("wind")
    for var in ("u10", "u10dir"):
        _write_array(group, var, era5[var].transpose("time", "sites").values)
    logger.info("cube: wind")


def build_wind_only(sites: xr.Dataset) -> None:
    """``cube.zarr`` ``wind_only/``: the SnapWave wind-only run on the storm hours (NaN elsewhere)."""
    ids = sites["sites"].values
    ds = src.wind_only(ids)
    if ds is None:
        logger.warning("wind_only: no Dynamic_Cases_storms.nc (run wind_only_storms.py)")
        return
    ds = ds.reindex(time=pd.DatetimeIndex(sites["time"].values), sites=ids).transpose("time", "sites")
    group = zarr.open_group(STORE / "cube.zarr", mode="a").require_group("wind_only")
    for var in VARIABLES:
        _write_array(group, var, ds[var].values)
    logger.info("cube: wind_only (%d hours)", int(ds["hs"].notnull().any("sites").sum()))


def build_breaking(sites: xr.Dataset) -> None:
    """
    SnapWave's breaking: the parameters of the templates (``breaking/params.json``), ``dw``, ``df``, ``zs`` of every
    SnapWave dynamic run in the cube (NaN at the hours the cube drops), and ``dw``, ``zs`` of the training cases
    (``breaking/cases.zarr``, one group per goal, cases and sites as the goal's case file).
    """
    out = STORE / "breaking"
    out.mkdir(exist_ok=True)
    params = {kind: brk.read_params(Path(sim.templates_dir(kind, "snapwave")) / "snapwave.inp")
              for kind in ("metamodel", "dynamic")}
    if params["metamodel"] != params["dynamic"]:
        raise ValueError(f"the SnapWave templates break differently: {params}")
    (out / "params.json").write_text(json.dumps(params["dynamic"], indent=2))

    root = zarr.open_group(STORE / "cube.zarr", mode="a")
    time_index = pd.DatetimeIndex(root.attrs["time"])
    n_sites = sites.sizes["sites"]
    for key in DYNAMIC:
        model, wind = DYNAMIC_KINDS[ESTIMATES[key]["kind"]]
        if model != "snapwave":
            continue
        values = src.dynamic_dissipation(ESTIMATES[key]["offshore"], wind, time_index, n_sites)
        valid = np.isfinite(root[f"{key}/hs"][:])
        for var, arr in values.items():
            _write_array(root[key], var, np.where(valid, arr, np.nan))
        logger.info("breaking: %s", key)

    cases = zarr.open_group(out / "cases.zarr", mode="w")
    for goal in cfg.load_goals():
        with xr.open_dataset(mm.cases_path(goal)) as ds:
            case_num, ids = ds["case_num"].values, ds["sites"].values
        values = src.case_dissipation(goal, case_num, len(ids))
        group = cases.require_group(str(goal))
        group.create_array("dw", data=values["dw"], overwrite=True)
        group.create_array("zs", data=np.nanmean(values["zs"], 1).astype("float32"), overwrite=True)
        group.attrs.update({"sites": [str(s) for s in ids], "case_num": [int(c) for c in case_num]})
        logger.info("breaking: training cases of goal %s", goal)


def build_goal_breaking(offshore: xr.Dataset) -> None:
    """
    Breaking of each goal alone in the 2013 reconstruction (bulk, MDA 600, CODEC): every goal's contribution
    (``goal_contributions``, before the sum) through SnapWave's formula at the depth of that goal's water level;
    cube ``goals_{source}/dw`` (sum over the goals, W/m²) and ``/qb`` (largest Baldock factor of a goal).

    Uses the printed ``reconstruction_resources`` proposal (~5 min per source on the 48-core node).
    """
    params = json.loads((STORE / "breaking" / "params.json").read_text())
    root = zarr.open_group(STORE / "cube.zarr", mode="a")
    time_index = pd.DatetimeIndex(root.attrs["time"])
    columns = pd.Series(np.arange(len(root.attrs["sites"])), index=root.attrs["sites"])
    depth = pd.read_parquet(STORE / "sites.parquet").set_index("site")["depth"]
    for off in OFFSHORE:
        resources = mm.reconstruction_resources(src.YEAR, src.MDA_N, metamodel="isobaths", spectral=False,
                                                offshore_name=off, wl="forcing", recompute=True)
        dw = np.full((len(time_index), len(columns)), np.nan, dtype="float32")
        qb = np.full_like(dw, np.nan)
        for month in pd.period_range(f"{src.YEAR}-01", f"{src.YEAR}-12", freq="M"):
            rows = np.flatnonzero(time_index.to_period("M") == month)
            for metamodel_set in ("isobaths", "buoys", "deenhaag"):
                parts = src.goal_contributions(off, metamodel_set, str(month), resources)
                parts = parts.reindex(time=time_index[rows])
                ids = parts["sites"].values.astype(str)
                wl = offshore["wl"].sel(goal=parts["goal"].values, time=time_index[rows]).transpose("goal", "time")
                h = brk.depth(depth.loc[ids].to_numpy()[None, None, :], wl.values[:, :, None], params["hmin"])
                hs, tp = parts["hs"].transpose("goal", "time", "sites").values, parts["tp"].transpose("goal", "time", "sites").values
                dw_goal = brk.dissipation(hs, tp, h, params["gamma"], params["alpha"])
                qb_goal = brk.qb(hs, h, params["gamma"])
                some = np.isfinite(hs).any(0)
                cols = columns.loc[ids].to_numpy()
                dw[np.ix_(rows, cols)] = np.where(some, np.nansum(dw_goal, 0), np.nan)
                qb[np.ix_(rows, cols)] = np.where(some, np.nanmax(np.where(np.isfinite(qb_goal), qb_goal, -1), 0), np.nan)
            logger.info("goal_breaking: %s %s", off, month)
        group = root.require_group(f"goals_{off}")
        _write_array(group, "dw", dw)
        _write_array(group, "qb", qb)


def build_wind_grid() -> None:
    """``wind_grid.zarr``: the ERA5 10 m wind on its own grid (the shaded layer of the maps)."""
    wind = cfg.open_wind(src.PERIOD)[["u10", "v10"]].astype("float32").load()
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
        for key in (k for k in DYNAMIC if ESTIMATES[k]["offshore"] == off and DYNAMIC_KINDS[ESTIMATES[k]["kind"]][0] == "swan"):
            wind = DYNAMIC_KINDS[ESTIMATES[key]["kind"]][1]
            df[f"{ESTIMATES[key]['kind']}_accuracy"] = src.swan_accuracy(off, wind, time_index).to_numpy()
        frames.append(df)
    conditions = pd.concat(frames, ignore_index=True)
    conditions.to_parquet(STORE / "conditions.parquet")
    _storms(conditions[conditions["offshore"] == "WHACS_cal"].set_index("time")).to_csv(STORE / "storms.csv", index=False)


#: Pixel size of the map images (deg lon, deg lat; ~400 m).
MAP_PIXEL = (0.006, 0.004)
#: Maps read in parallel before they are written (bounds the memory).
MAP_BATCH = 128


def build_maps(sites: xr.Dataset) -> None:
    """
    ``maps.zarr``: the grid maps of every dynamic run at the hours where at least two runs have one
    (float16 on the mesh nodes, one chunk per hour), the mesh, the image raster and each site's nearest node.

    Hours where a run is NaN in the cube (not converged) are left out of that run.
    """
    from concurrent.futures import ThreadPoolExecutor

    from scipy.spatial import cKDTree

    cube = zarr.open_group(STORE / "cube.zarr", mode="r")
    time_index = pd.DatetimeIndex(cube.attrs["time"])
    hours = {}
    for key in DYNAMIC:
        model, wind = DYNAMIC_KINDS[ESTIMATES[key]["kind"]]
        valid = time_index[np.isfinite(cube[f"{key}/hs"][:]).any(1)]
        hours[key] = src.map_hours(ESTIMATES[key]["offshore"], wind, model, valid)
        logger.info("maps: %s has %d maps", key, len(hours[key]))
    counts = pd.Series(np.concatenate([h.to_numpy() for h in hours.values()])).value_counts()
    shared = pd.DatetimeIndex(counts.index[counts >= 2])

    mesh = src.mesh()
    root = zarr.open_group(STORE / "maps.zarr", mode="w")
    for var in ("lon", "lat", "depth"):
        root.create_array(f"mesh/{var}", data=mesh[var].values.astype("float32"))
    _write_raster(root, mesh)
    lat0 = float(np.deg2rad(mesh["lat"].mean()))
    tree = cKDTree(np.c_[mesh["lon"].values * np.cos(lat0), mesh["lat"].values])
    _, node = tree.query(np.c_[sites["site_longitude"].values * np.cos(lat0), sites["site_latitude"].values])
    root.create_array("site_node", data=node.astype("int32"))

    n_node = mesh.sizes["node"]
    for key in DYNAMIC:
        model, wind = DYNAMIC_KINDS[ESTIMATES[key]["kind"]]
        off = ESTIMATES[key]["offshore"]
        times = hours[key].intersection(shared).sort_values()
        group = root.require_group(f"runs/{key}")
        group.attrs["time"] = [str(t) for t in times]
        arrays = {v: group.create_array(v, shape=(len(times), n_node), chunks=(1, n_node), dtype="float16")
                  for v in VARIABLES}
        read = lambda t: src.read_map(off, wind, model, t)
        # Threads read, this thread writes: parallel chunk creation on Lustre failed (EACCES) once
        with ThreadPoolExecutor(src.THREADS) as ex:
            for start in range(0, len(times), MAP_BATCH):
                for i, ds in enumerate(ex.map(read, times[start:start + MAP_BATCH]), start):
                    for v in VARIABLES:
                        arrays[v][i] = ds[v].values.astype("float16")
                logger.info("maps: %s, %d / %d hours", key, min(start + MAP_BATCH, len(times)), len(times))


def _write_raster(root: zarr.Group, mesh: xr.Dataset) -> None:
    """Pixels of the map image (north up) inside the mesh, with their triangle's nodes and barycentric weights."""
    from matplotlib.tri import Triangulation

    lon, lat, tri = mesh["lon"].values, mesh["lat"].values, mesh["triangles"].values
    x = np.arange(lon.min(), lon.max() + MAP_PIXEL[0], MAP_PIXEL[0])
    y = np.arange(lat.max(), lat.min() - MAP_PIXEL[1], -MAP_PIXEL[1])
    X, Y = np.meshgrid(x, y)
    found = Triangulation(lon, lat, tri).get_trifinder()(X.ravel(), Y.ravel())
    pixel = np.flatnonzero(found >= 0)
    nodes = tri[found[pixel]]
    px, py = X.ravel()[pixel], Y.ravel()[pixel]
    (x1, x2, x3), (y1, y2, y3) = lon[nodes].T, lat[nodes].T
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    w1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det
    w2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det
    weights = np.c_[w1, w2, 1 - w1 - w2]
    group = root.require_group("raster")
    group.create_array("pixel", data=pixel.astype("int32"))
    group.create_array("nodes", data=nodes.astype("int32"))
    group.create_array("weights", data=weights.astype("float32"))
    half = np.array(MAP_PIXEL) / 2
    group.attrs.update({"shape": [len(y), len(x)],
                        "extent": [float(x[0] - half[0]), float(x[-1] + half[0]),
                                   float(y[-1] - half[1]), float(y[0] + half[1])]})


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
        folder = cfg.resolve(f"outputs/calibration/{raw}")
        if not folder.exists():
            continue
        for name in ("coefficients", "metrics", "quantile_mapping"):
            if (folder / f"{name}.csv").exists():
                shutil.copy(folder / f"{name}.csv", out / f"{raw}_{name}.csv")
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
    folder = CASE / "numerical_tests"
    out = STORE / "numerical_tests"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(folder / "results", out, ignore=shutil.ignore_patterns("raw"))
    shutil.copy(folder / "README.md", out / "README.md")


PARTS = {"sites": None, "offshore": None, "cube": None, "wind_only": None, "conditions": None, "maps": None,
         "breaking": None, "goal_breaking": None, "wind_grid": build_wind_grid,
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
    sites = src.dynamic_sites() if {"sites", "cube", "wind_only", "maps", "breaking"} & set(parts) else None
    offshore = None
    for part in parts:
        t0 = time.time()
        if part == "sites":
            build_sites(sites)
        elif part == "maps":
            build_maps(sites)
        elif part == "wind_only":
            build_wind_only(sites)
        elif part == "breaking":
            build_breaking(sites)
        elif part == "offshore":
            offshore = build_offshore()
        elif part in ("cube", "conditions", "goal_breaking"):
            offshore = offshore if offshore is not None else xr.open_zarr(STORE / "offshore.zarr").load()
            {"cube": build_cube, "conditions": build_conditions,
             "goal_breaking": build_goal_breaking}[part](*((sites, offshore) if part == "cube" else (offshore,)))
        else:
            PARTS[part]()
        manifest[part] = {"built": pd.Timestamp.now().isoformat(timespec="seconds"), "seconds": round(time.time() - t0)}
        logger.info("%s done in %.0f s", part, time.time() - t0)
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
