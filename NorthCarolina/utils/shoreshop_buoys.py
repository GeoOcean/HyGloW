"""NDBC buoy records of the North Carolina case (case-only code).

The hourly bulk tables come from the ShoreShop pipeline
(``01_BinWaves/inputs/buoy_data/buoy_<id>_bulk_parameters.pkl``: pandas frames
with ``Hs_Buoy``, ``Tp_Buoy``, ``Dir_Buoy`` on an hourly index, downloaded from
NDBC and cleaned there). They hold no directional spreading.
"""

from __future__ import annotations

import importlib
import logging
import pickle
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger("hywaves.shoreshop_buoys")

SHORESHOP_BUOYS = Path(
    "/lustre/geocean/WORK/users/montanoj/projects/shoreshop3_2026_pipeline/01_BinWaves/inputs/buoy_data"
)
COLUMNS = {"Hs_Buoy": "hs", "Tp_Buoy": "tp", "Dir_Buoy": "dir", "Spr_Buoy": "spr"}


class _FrameUnpickler(pickle.Unpickler):
    """Load a pandas frame only (numpy/pandas classes), written with numpy 2 or 1."""

    def find_class(self, module: str, name: str):
        if module.split(".")[0] not in ("numpy", "pandas", "builtins"):
            raise pickle.UnpicklingError(f"unexpected class {module}.{name}")
        try:
            return getattr(importlib.import_module(module), name)
        except ModuleNotFoundError:
            return getattr(importlib.import_module(module.replace("numpy._core", "numpy.core")), name)


def read_table(station: str, folder: Path = SHORESHOP_BUOYS) -> pd.DataFrame:
    """Hourly ``hs``, ``tp``, ``dir``, ``spr`` of one NDBC station (rows without Hs dropped)."""
    with (folder / f"buoy_{station}_bulk_parameters.pkl").open("rb") as fh:
        frame = _FrameUnpickler(fh).load()
    frame = frame.rename(columns=COLUMNS)[list(COLUMNS.values())].astype("float32")
    frame.index = pd.DatetimeIndex(frame.index, name="time")
    return frame[frame["hs"].notna()].sort_index()


def write_processed(observations: dict[str, dict], out_dir: Path) -> pd.DataFrame:
    """
    Write ``<out_dir>/<instrument_id>.nc`` for every observation with a ``station``.

    Parameters
    ----------
    observations
        The notebook 03 dict; each entry names its NDBC ``station``.
    out_dir
        ``inputs/Observations/processed``.

    Returns
    -------
    pandas.DataFrame
        Per buoy: first and last hour, hours with Hs, with direction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for iid, site in observations.items():
        frame = read_table(site["station"])
        ds = xr.Dataset.from_dataframe(frame)
        ds.attrs = {
            "source": "NDBC, hourly bulk tables of the ShoreShop pipeline",
            "station": site["station"],
            "station_id": iid,
            "lon": site["lon"],
            "lat": site["lat"],
        }
        ds.to_netcdf(out_dir / f"{iid}.nc")
        rows.append(
            {
                "buoy": iid,
                "first": frame.index[0],
                "last": frame.index[-1],
                "hours_hs": len(frame),
                "hours_dir": int(frame["dir"].notna().sum()),
            }
        )
        logger.info("%s: %d hours", iid, len(frame))
    return pd.DataFrame(rows).set_index("buoy")
