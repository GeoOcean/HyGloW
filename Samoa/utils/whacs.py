"""WHACS at the goal nodes, read from a local copy of the global archive.

The archive (``archive`` of the WHACS ``offshore_waves`` entry) holds one
folder per variable with monthly files on the global SMC mesh
(``<archive>/<variable>/*_YYYYMMDDhhmm-YYYYMMDDhhmm.nc``, ``(time, seapoint)``).
:func:`whacs_at_goals` reads the bulk (``hs``, ``fp``, ``dir``, ``spr``) and the
four partitions (``phs``, ``ptp``, ``pdir``, ``pspr``) at the WHACS node of
every goal (goals JSON ``hindcast.whacs``), reading only the file chunks that
hold those nodes. Notebook ``07a_Offshore_WHACS.ipynb``.

The archive is CSIRO's WHACS release (doi:10.25919/shdk-7p29), not shipped with
HyGloW: on the GeoOcean cluster it is ``/lustre/geocean/DATA/CSIRO/WHACS/global-smc``;
elsewhere download the :data:`VARIABLES` (~14 TB for 1979-2026) or ask GeoOcean
for the goal series.
"""

from __future__ import annotations

import logging
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from utils.config import get_offshore_wave, load_goals, offshore_catalog_key

logger = logging.getLogger("hywaves.whacs")

#: Archive variables read: bulk, then partition 0 (wind sea) and 1-3 (swells).
VARIABLES: tuple[str, ...] = ("hs", "fp", "dir", "spr") + tuple(
    f"p{v}{i}" for i in range(4) for v in ("hs", "tp", "dir", "spr")
)

#: Global attributes of a monthly file that describe only that file.
FILE_ATTRS: tuple[str, ...] = (
    "product_name", "start_date", "stop_date", "time_coverage_start",
    "time_coverage_end", "time_coverage_duration",
)

#: Largest distance (degrees) between a goals JSON node and its seapoint.
MAX_NODE_DISTANCE = 0.01

_MONTH = re.compile(r"_(\d{6})\d{6}-\d{12}\.nc$")


def archive_path(offshore_name: str = "WHACS") -> Path:
    """``archive`` of the *offshore_name* entry of ``offshore_waves``."""
    entry = get_offshore_wave(offshore_name)
    if not entry.get("archive"):
        raise KeyError(f"offshore_waves entry {offshore_name!r} has no 'archive'")
    return Path(entry["archive"])


def archive_files(archive: Path, variable: str, period: slice | None = None) -> list[Path]:
    """Monthly files of *variable*, sorted, within *period* (``slice("1979", "2026-01")``).

    The month comes from the file name; *period* bounds are inclusive
    (pandas partial dates).
    """
    files = sorted((archive / variable).glob("*.nc"))
    if not files:
        raise FileNotFoundError(
            f"No WHACS files in {archive / variable}: point offshore_waves[].archive at a local "
            "copy of the WHACS global archive (download it, doi:10.25919/shdk-7p29, or ask GeoOcean)"
        )
    months = pd.PeriodIndex([_MONTH.search(f.name).group(1) for f in files], freq="M")
    keep = np.ones(len(files), bool)
    if period is not None and period.start is not None:
        keep &= months >= pd.Period(pd.Timestamp(period.start), "M")
    if period is not None and period.stop is not None:
        keep &= months <= pd.Period(period.stop).end_time.to_period("M")
    return [f for f, k in zip(files, keep) if k]


def goal_seapoints(archive: Path, offshore_name: str = "WHACS") -> pd.DataFrame:
    """Seapoint of each goal's WHACS node (goals JSON ``hindcast.<key>``).

    Returns
    -------
    pd.DataFrame
        Indexed by goal id (int): ``seapoint``, ``longitude``, ``latitude``
        (of the seapoint) and ``distance`` (degrees to the JSON node).

    Raises
    ------
    ValueError
        If a node is farther than :data:`MAX_NODE_DISTANCE` from every seapoint.
    """
    key = offshore_catalog_key(get_offshore_wave(offshore_name))
    with netCDF4.Dataset(archive_files(archive, "hs")[0]) as nc:
        lon = np.ma.filled(nc["longitude"][:], np.nan).astype("float64")
        lat = np.ma.filled(nc["latitude"][:], np.nan).astype("float64")
    lon = np.where(lon > 180, lon - 360, lon)
    rows = {}
    for gid, goal in load_goals().items():
        node = goal["hindcast"][key]
        node_lon = (node["longitude"] + 180) % 360 - 180
        distance = np.hypot(lon - node_lon, lat - node["latitude"])
        i = int(np.nanargmin(distance))
        rows[int(gid)] = {"seapoint": i, "longitude": lon[i], "latitude": lat[i], "distance": distance[i]}
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("goal")
    far = table[table["distance"] > MAX_NODE_DISTANCE]
    if not far.empty:
        raise ValueError(f"Goal nodes not on the WHACS mesh (distance in degrees):\n{far}")
    return table


def read_file(path: str, variable: str, seapoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One monthly file at *seapoints*: ``(times, values (time, seapoint))``.

    Reads whole storage chunks along ``seapoint`` (only those holding a
    seapoint); hours are rounded to the minute.
    """
    with netCDF4.Dataset(path) as nc:
        var = nc[variable]
        size = var.chunking()[1] if isinstance(var.chunking(), list) else var.shape[1]
        out = np.empty((var.shape[0], seapoints.size))
        for block in np.unique(seapoints // size):
            inside = seapoints // size == block
            lo = int(block) * size
            values = np.ma.filled(var[:, lo : lo + size].astype("float64"), np.nan)
            out[:, inside] = values[:, seapoints[inside] - lo]
        t = nc["time"]
        times = netCDF4.num2date(t[:], t.units, only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    times = pd.DatetimeIndex(np.array(times, dtype="datetime64[ns]")).round("min").to_numpy()
    return times, out


def whacs_at_goals(
    offshore_name: str = "WHACS",
    *,
    period: slice | None = None,
    workers: int = 8,
) -> xr.Dataset:
    """WHACS bulk and partitions at every goal node, from the local archive.

    Parameters
    ----------
    offshore_name
        ``offshore_waves`` entry with the ``archive`` (and the goals JSON key).
    period
        Months to read (``slice("1979", "2026-01")``); default all.
    workers
        Processes reading files in parallel.

    Returns
    -------
    xr.Dataset
        On ``(time, goal)``: :data:`VARIABLES` plus ``tp = 1 / fp``; coord
        ``seapoint`` per goal; the global attributes of the archive files.

    Raises
    ------
    ValueError
        If the variables do not share the same hours.
    """
    archive = archive_path(offshore_name)
    nodes = goal_seapoints(archive, offshore_name)
    seapoints = nodes["seapoint"].to_numpy()
    data, times = {}, None
    # spawn, not fork: HDF5 is not fork-safe once the parent has opened a file
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for variable in VARIABLES:
            files = archive_files(archive, variable, period)
            parts = list(pool.map(read_file, map(str, files), [variable] * len(files), [seapoints] * len(files)))
            var_times = np.concatenate([p[0] for p in parts])
            if times is None:
                times = var_times
            elif not np.array_equal(times, var_times):
                raise ValueError(f"{variable}: hours differ from hs ({archive})")
            data[variable] = (("time", "goal"), np.concatenate([p[1] for p in parts]))
            logger.info("%s: %d months", variable, len(files))
    ds = xr.Dataset(
        data,
        coords={
            "time": times,
            "goal": nodes.index.to_numpy(),
            "seapoint": ("goal", seapoints.astype("int64")),
        },
    )
    ds["tp"] = 1.0 / ds["fp"]
    ds["tp"].attrs = {"long_name": "Wave peak period", "units": "s"}
    with netCDF4.Dataset(archive_files(archive, "hs")[0]) as nc:
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs() if k not in FILE_ATTRS}
    ds.attrs = {**attrs, "hywaves_source": f"WHACS global archive {archive}"}
    return ds
