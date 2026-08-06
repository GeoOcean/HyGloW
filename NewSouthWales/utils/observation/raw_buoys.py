"""Raw NSW buoy discovery and bulk-parameter standardization.

Case-specific — every case's raw buoy layout is different (file naming,
column sets, timezone), so this lives here rather than in the vendored
``utils.observation.onshore_sites`` module. See ``00_Buoys_Inspection.ipynb``
for the underlying data survey this module's mappings are based on.

Two raw networks under ``inputs/Observations/raw/``:

- **Offshore MHL Waverider buoys** (``buoys/*.nc``) — hourly bulk
  parameters only, already UTC, no directional spread.
- **Nearshore Spotter buoys** (``nearshore_buoys/wave_parameters_csv/*.csv``)
  — 30 min bulk parameters *and* directional spread (``Dm_sp``/``Dp_sp``)
  and QC flags, none of which are in the sibling ``wave_parameters_netcdf/``
  files for the same deployments — the CSV is the richer source. Timestamps
  are fixed Eastern Standard Time (UTC+10), **not** adjusted for daylight
  saving (verified against the NetCDF, which is UTC, across DST and
  non-DST months) — see :func:`read_nearshore_bulk`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from utils.config import resolve

RAW_DIR = "inputs/Observations/raw"
OFFSHORE_SUBDIR = "buoys"
NEARSHORE_CSV_SUBDIR = "nearshore_buoys/wave_parameters_csv"

NSW_LOCAL_UTC_OFFSET_HOURS = 10  # fixed AEST, never AEDT — see module docstring

# 3-letter MHL station code -> instrument_id in the observations catalogue
# (notebook 03 / inputs/Observations/NewSouthWales_observations.json)
OFFSHORE_CODES = {
    "SYD": "Sydney_Buoy",
    "KEM": "PortKembla_Buoy",
    "CRO": "CrowdyHead_Buoy",
    "COH": "CoffsHarbour_Buoy",
    "BYR": "ByronBay_Buoy",
    "EDE": "Eden_Buoy",
    "BAT": "BatemansBay_Buoy",
}

# Raw column/variable -> HyGloW bulk name.
#
# "hs" tracks the sibling NetCDF's "Hs", which is the statistical/zero-
# crossing estimate (matches "Hsig", not the spectral "Hm0") — kept for
# continuity with 00_Buoys_Inspection.ipynb's stats, computed against that
# Hs. Hm0 is kept too, as a distinct variable, not a replacement.
NEARSHORE_VAR_MAP = {
    "Hsig": "hs",
    "Hm0": "hm0",
    "Hmax": "hmax",
    "Tp": "tp",
    "Tm02": "tm02",
    "Tm01": "tm01",
    "Tz": "tz",
    "Dm": "dir",
    "Dm_sp": "spr",
    "Dp": "dp",
    "Dp_sp": "dp_spr",
    "Qflag": "qflag",
    "Qcode": "qcode",
    "Percent": "percent",
    "Dof": "dof",
}

OFFSHORE_VAR_MAP = {
    "Hs": "hs",
    "Hmax": "hmax",
    "Tp": "tp",
    "Tm02": "tm02",
    "Dir": "dir",
}

VAR_ATTRS: dict[str, dict[str, str]] = {
    "hs": {
        "standard_name": "sea_surface_wave_significant_height",
        "units": "m",
        "long_name": "Significant wave height (Hsig, zero-crossing estimate)",
    },
    "hm0": {
        "standard_name": "sea_surface_wave_significant_height",
        "units": "m",
        "long_name": "Spectral significant wave height (Hm0 = 4*sqrt(m0))",
    },
    "hmax": {"units": "m", "long_name": "Maximum wave height"},
    "tp": {
        "standard_name": "sea_surface_wave_period_at_variance_spectral_density_maximum",
        "units": "s",
        "long_name": "Peak wave period",
    },
    "tm02": {
        "standard_name": "sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment",
        "units": "s",
        "long_name": "Mean period (spectral, second moment)",
    },
    "tm01": {
        "standard_name": "sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
        "units": "s",
        "long_name": "Mean period (spectral, first moment)",
    },
    "tz": {"units": "s", "long_name": "Zero-crossing period"},
    "dir": {
        "standard_name": "sea_surface_wave_from_direction",
        "units": "degree",
        "long_name": "Mean wave-from direction",
    },
    "spr": {
        "standard_name": "sea_surface_wave_directional_spread",
        "units": "degree",
        "long_name": "Directional spread about the mean direction",
    },
    "dp": {"units": "degree", "long_name": "Peak wave-from direction"},
    "dp_spr": {"units": "degree", "long_name": "Directional spread about the peak direction"},
    "qflag": {"long_name": "MHL/NSW DCCEEW quality flag, as provided (not interpreted here)"},
    "qcode": {"long_name": "MHL/NSW DCCEEW quality reason code, as provided (not interpreted here)"},
    "percent": {"units": "%", "long_name": "Percent of expected raw samples used in this record"},
    "dof": {"long_name": "Spectral estimate degrees of freedom"},
}


def match_raw_buoy_files(meta: dict[str, dict]) -> dict[str, dict[str, Any]]:
    """Match every raw buoy file to its catalogue ``instrument_id``.

    Parameters
    ----------
    meta
        ``instrument_id`` -> metadata, from ``obs.get_coordinates()``.

    Returns
    -------
    dict[str, dict]
        ``instrument_id`` -> ``{"path": Path, "network": str, "format": str}``.
    """
    offshore_dir = resolve(f"{RAW_DIR}/{OFFSHORE_SUBDIR}")
    nearshore_dir = resolve(f"{RAW_DIR}/{NEARSHORE_CSV_SUBDIR}")
    meta_lower = {iid.lower(): iid for iid in meta}

    file_index: dict[str, dict[str, Any]] = {}

    for f in sorted(offshore_dir.glob("*.nc")):
        iid = OFFSHORE_CODES.get(f.stem.split("_")[0])
        if iid:
            file_index[iid] = {"path": f, "network": "offshore (MHL)", "format": "netcdf"}

    for f in sorted(nearshore_dir.glob("*.csv")):
        # e.g. NSWENV_20160302-20160517_12m_Narrabeen_WAVE_Parameters.csv
        parts = f.stem.split("_")
        candidate = f"{parts[3]}_buoy" if len(parts) > 3 else None
        iid = meta_lower.get(candidate.lower()) if candidate else None
        if iid:
            file_index[iid] = {"path": f, "network": "nearshore (Spotter)", "format": "csv"}

    return file_index


def _apply_attrs(ds: xr.Dataset) -> xr.Dataset:
    for name, attrs in VAR_ATTRS.items():
        if name in ds:
            ds[name].attrs.update(attrs)
    return ds


def read_offshore_bulk(path: Path) -> xr.Dataset:
    """Read one offshore MHL NetCDF and rename to HyGloW bulk names.

    Timestamps are used as provided (UTC per the source file's own CF time
    encoding) — there is no independent second source to cross-check this
    assumption against, unlike the nearshore buoys.
    """
    raw = xr.open_dataset(path)
    rename = {k: v for k, v in OFFSHORE_VAR_MAP.items() if k in raw.data_vars}
    ds = raw[list(rename)].rename(rename)
    ds = ds.sortby("time")
    ds = ds.drop_duplicates("time", keep="first")
    return _apply_attrs(ds)


def read_nearshore_bulk(path: Path) -> xr.Dataset:
    """Read one nearshore Spotter CSV, convert local time to UTC, and rename.

    The source ``Time`` column is fixed Eastern Standard Time (UTC+10),
    verified constant (never UTC+11 / AEDT) across both DST and non-DST
    months by cross-checking against the sibling NetCDF's UTC timestamps.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    time_utc = (
        pd.to_datetime(df["Time"].str.strip(), format="%d-%b-%Y %H:%M:%S")
        - pd.Timedelta(hours=NSW_LOCAL_UTC_OFFSET_HOURS)
    )

    data = {}
    for raw_col, name in NEARSHORE_VAR_MAP.items():
        if raw_col not in df.columns:
            continue
        values = pd.to_numeric(df[raw_col], errors="coerce")
        data[name] = ("time", values.to_numpy())

    ds = xr.Dataset(data, coords={"time": time_utc.to_numpy()})
    ds = ds.sortby("time")
    ds = ds.drop_duplicates("time", keep="first")
    return _apply_attrs(ds)


def read_bulk(info: dict[str, Any]) -> xr.Dataset:
    """Dispatch to :func:`read_offshore_bulk` / :func:`read_nearshore_bulk`."""
    if info["format"] == "netcdf":
        return read_offshore_bulk(info["path"])
    if info["format"] == "csv":
        return read_nearshore_bulk(info["path"])
    raise ValueError(f"Unknown raw buoy format: {info['format']!r}")


def standardize_and_write(
    instrument_id: str,
    info: dict[str, Any],
    meta: dict[str, Any],
    out_dir: Path,
) -> Path:
    """Read, standardize, attach provenance attrs, and write one buoy NetCDF.

    Parameters
    ----------
    instrument_id
        Catalogue instrument id (also the output file stem).
    info
        One entry from :func:`match_raw_buoy_files`.
    meta
        Full metadata dict for *this* instrument (from ``obs.get_coordinates()``).
    out_dir
        Destination directory (``inputs/Observations/processed/``).

    Returns
    -------
    Path
        Written file path.
    """
    ds = read_bulk(info)
    ds.attrs.update(
        {
            "instrument_id": instrument_id,
            "station_name": str(meta.get("name") or ""),
            "institution": str(meta.get("institution") or ""),
            "network": info["network"],
            "longitude": float(meta["lon"]),
            "latitude": float(meta["lat"]),
            "depth_m": float(meta["depth"]) if meta.get("depth") is not None else float("nan"),
            "source_file": info["path"].name,
            "time_convention": "UTC",
            "processing": (
                "Bulk wave parameters standardized to HyGloW/CF names from raw "
                f"{info['network']} data ({info['format']}). No gap-filling, "
                "interpolation, or QC-based filtering applied — NaNs and "
                "(where present) qflag/qcode/percent/dof are passed through "
                "as provided."
            ),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{instrument_id}.nc"
    ds.to_netcdf(out_path)
    return out_path


__all__ = [
    "NEARSHORE_VAR_MAP",
    "NSW_LOCAL_UTC_OFFSET_HOURS",
    "OFFSHORE_CODES",
    "OFFSHORE_VAR_MAP",
    "VAR_ATTRS",
    "match_raw_buoy_files",
    "read_bulk",
    "read_nearshore_bulk",
    "read_offshore_bulk",
    "standardize_and_write",
]
