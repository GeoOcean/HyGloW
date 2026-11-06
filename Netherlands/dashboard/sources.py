"""
The case results the store is built from, for 2013, on ``(time, sites)``.

Read through the case ``utils`` (the case folder must be on ``sys.path``):
the dynamic runs of notebook 08, the linear sum from the reconstruction cache
(MDA :data:`MDA_N`, CODEC water level), the buoys and the ERA5 wind.
"""

from __future__ import annotations

from collections.abc import Iterable

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


def dynamic(offshore: str, wind: bool, sites: Iterable[str] | None = None) -> xr.Dataset | None:
    """
    Dynamic run of *offshore* (wind on or off), or None if not postprocessed.

    Hours with Hs outside (0, 15) m (not converged or corrupted) are NaN in every variable.
    """
    path = sim.cases_dir("dynamic", offshore=offshore, wind=wind) / f"Dynamic_Cases_{YEAR}.nc"
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        ds = ds[list(VARIABLES)]
        ds = (ds.sel(sites=list(sites)) if sites is not None else ds).load()
    return ds.where((ds["hs"] > 0) & (ds["hs"] < 15))


def dynamic_sites() -> xr.Dataset:
    """Every site of the dynamic runs (isobaths, buoys, DeenHaag mesh nodes) with its metadata and the time axis."""
    path = sim.cases_dir("dynamic", offshore="WHACS", wind=False) / f"Dynamic_Cases_{YEAR}.nc"
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
