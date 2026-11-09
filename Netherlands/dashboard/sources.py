"""
The case results the store is built from, for 2013, on ``(time, sites)``.

Read through the case ``utils`` (the case folder must be on ``sys.path``):
the dynamic runs of notebook 08 (SnapWave and SWAN, sites and grid maps), the
linear sum from the reconstruction cache (MDA :data:`MDA_N`, CODEC water level),
the buoys and the ERA5 wind. Every dynamic read names its model, whatever
``simulations.model`` the case JSON is on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils import metamodel as mm
from utils import onshore_sites as obs
from utils import simulation as sim
from utils import validation as val
from utils.geometry import attach_site_meta

PERIOD = slice("2013-01-01", "2013-12-31 23:00")
YEAR = "2013"
MDA_N = 600
VARIABLES = ("hs", "tp", "dir", "spr")
#: Grid-map file of each model in a dynamic case folder.
MAP_FILES = {"snapwave": "output_map.nc", "swan": "output_map.mat"}
#: SnapWave map variables -> short names (as the site output).
SNAPWAVE_MAP_VARS = {"hm0": "hs", "tp": "tp", "wd": "dir", "wdspr": "spr"}
THREADS = 32


def _valid(ds: xr.Dataset) -> xr.Dataset:
    """Hs outside (0, 15) m (not converged, dry or corrupted) -> NaN in every variable."""
    return ds.where((ds["hs"] > 0) & (ds["hs"] < 15))


def _case_dir(model: str, offshore: str, wind: bool, time: pd.Timestamp) -> Path:
    return sim.cases_dir("dynamic", model, offshore=offshore, wind=wind) / f"{time:%Y%m%d %H%M%S}"


def dynamic(offshore: str, wind: bool, model: str, sites: Iterable[str] | None = None) -> xr.Dataset | None:
    """
    Dynamic run of *model* forced by *offshore* (wind on or off), or None if not postprocessed.

    Hours with Hs outside (0, 15) m (not converged or corrupted) are NaN in every variable.
    """
    path = sim.cases_dir("dynamic", model, offshore=offshore, wind=wind) / f"Dynamic_Cases_{YEAR}.nc"
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        ds = ds[list(VARIABLES)]
        ds = (ds.sel(sites=list(sites)) if sites is not None else ds).load()
    return _valid(ds)


def map_hours(offshore: str, wind: bool, model: str, times: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The *times* whose case folder holds a grid map (``save_map`` cases of notebook 08)."""
    name = MAP_FILES[model]
    with ThreadPoolExecutor(THREADS) as ex:
        has = list(ex.map(lambda t: (_case_dir(model, offshore, wind, t) / name).exists(), times))
    return times[np.asarray(has, bool)]


def read_map(offshore: str, wind: bool, model: str, time: pd.Timestamp) -> xr.Dataset:
    """Grid map of one hour on ``node`` (``hs, tp, dir, spr``), Hs outside (0, 15) m as NaN."""
    path = _case_dir(model, offshore, wind, time) / MAP_FILES[model]
    if model == "swan":
        from bluemath_tk.wrappers.swan.swan_utils import read_swan_map
        ds = read_swan_map(str(path))[list(VARIABLES)]
    else:
        with xr.open_dataset(path) as nc:
            ds = xr.Dataset({short: ("node", nc[name].values.ravel()) for name, short in SNAPWAVE_MAP_VARS.items()})
    return _valid(ds)


def mesh() -> xr.Dataset:
    """The mesh of the maps (SWAN's ``.grd`` = SnapWave's ``.nc`` nodes): ``lon, lat, depth`` on ``node``, ``triangles``."""
    from bluemath_tk.wrappers.swan.swan_utils import read_adcirc_grid

    grid = read_adcirc_grid(str(sim.grid_file("swan")))
    lon, lat, depth = grid["vertices"].T
    return xr.Dataset({"lon": ("node", lon), "lat": ("node", lat), "depth": ("node", depth),
                       "triangles": (("triangle", "corner"), grid["triangles"].astype("int32"))})


def swan_accuracy(offshore: str, wind: bool, times: pd.DatetimeIndex) -> pd.Series:
    """
    SWAN's final accuracy per hour (% of wet nodes meeting ``STOPC``; 99.5 is the target), NaN without ``PRINT``.

    Cases that reach the 50-iteration cap stop below the target.
    """
    pattern = re.compile(r"accuracy OK in\s+([\d.]+)\s*%")

    def one(t: pd.Timestamp) -> float:
        path = _case_dir("swan", offshore, wind, t) / "PRINT"
        if not path.exists():
            return np.nan
        hits = pattern.findall(path.read_text(errors="ignore"))
        return float(hits[-1]) if hits else np.nan

    with ThreadPoolExecutor(THREADS) as ex:
        return pd.Series(list(ex.map(one, times)), index=times)


def dynamic_sites() -> xr.Dataset:
    """Every site of the dynamic runs (isobaths, buoys, DeenHaag mesh nodes) with its metadata and the time axis."""
    path = sim.cases_dir("dynamic", "snapwave", offshore="WHACS", wind=False) / f"Dynamic_Cases_{YEAR}.nc"
    with xr.open_dataset(path) as ds:
        return ds.drop_vars(list(ds.data_vars)).load()


def metamodel(offshore: str, metamodel_set: str, spectral: bool, sites: Iterable[str] | None = None) -> xr.Dataset:
    """
    Linear sum of *offshore* from the reconstruction cache (raises if a site-month is missing).

    Parameters
    ----------
    offshore
        ``offshore_waves`` entry.
    metamodel_set
        ``buoys``, ``isobaths`` or ``deenhaag``.
    spectral
        Partitions summed (True) or the bulk sea state.
    sites
        Default every site of the set.
    """
    store = mm.reconstruction_store(MDA_N, metamodel=metamodel_set, spectral=spectral,
                                    offshore_name=offshore, wl="forcing")
    site_ids = mm.reconstruction_sites(metamodel_set, None if sites is None else list(sites))
    return attach_site_meta(store.read(PERIOD, site_ids))


def buoys(variables: Iterable[str] = VARIABLES) -> xr.Dataset:
    """Buoy records of :data:`PERIOD` (the ``ok`` buoys with Hs)."""
    return val.load_buoy_observations(PERIOD, obs.list_snapwave_buoy_sites(), variables=list(variables))


def wind_at_sites(sites: xr.Dataset) -> xr.Dataset:
    """ERA5 wind (``forcing.wind``) at the sites of *sites* and on its time axis: ``u10`` speed, ``u10dir`` (from)."""
    from bluemath_tk.waves.wind import wind_speed_direction

    from utils.config import open_wind

    wind = open_wind(PERIOD).interp(
        time=sites["time"],
        latitude=xr.DataArray(sites["site_latitude"].values, dims="sites"),
        longitude=xr.DataArray(sites["site_longitude"].values, dims="sites"),
    ).load()
    speed, direction = wind_speed_direction(wind["u10"].values, wind["v10"].values)
    coords = {"time": sites["time"].values, "sites": sites["sites"].values}
    return xr.Dataset({"u10": (("time", "sites"), speed), "u10dir": (("time", "sites"), direction)}, coords=coords)
