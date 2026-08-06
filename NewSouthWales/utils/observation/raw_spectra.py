"""Raw NSW MHL directional spectral buoy processing.

Three offshore MHL stations (Sydney, Byron Bay, Batemans Bay) have raw
frequency-direction spectral data alongside their bulk files, under
``inputs/Observations/raw/buoys/WaveSpectra/MHL_Spectra/``. Two file types
per station, joined by timestamp:

- **Yearly files** (e.g. ``SYDNEY1999.TXT``) — one row per (hour, frequency
  bin), 64 frequency bins/hour: ``freq``, ``dirn`` (mean direction),
  ``energy`` (spectral density **normalized to peak=1 within that hour**),
  ``spread``, ``skewness``, ``kurtosis``.
- **SD file** (e.g. ``SYDNEYSD.TXT``) — one row per hour: ``spect_den``, the
  true spectral density scale (m^2/Hz) that hour's normalized ``energy``
  must be multiplied by to get an absolute energy-density spectrum.

Both files are timestamped "Eastern Standard" per the SD file's own header
— a fixed UTC+10 offset, never adjusted for daylight saving (same
convention verified and applied for the bulk nearshore buoys in
``raw_buoys.py``; not independently re-verified here since MHL's own header
states it explicitly).

**Sydney's SD file is corrupted** in the raw archive (entirely null bytes,
confirmed by :func:`_read_sd_file`) — only the peak-normalized ``energy``
is available for Sydney, not an absolute energy density. Byron Bay and
Batemans Bay both have valid SD files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils.config import resolve

RAW_SPECTRA_DIR = "inputs/Observations/raw/buoys/WaveSpectra/MHL_Spectra"
NSW_LOCAL_UTC_OFFSET_HOURS = 10  # fixed Eastern Standard, never AEDT

STATIONS = {
    "Sydney_Buoy": {"folder": "Sydney", "prefix": "SYDNEY", "sd_file": "SYDNEYSD.TXT"},
    "ByronBay_Buoy": {"folder": "Byron Bay", "prefix": "BYRON", "sd_file": "BYRONSD.TXT"},
    "BatemansBay_Buoy": {"folder": "Batemans Bay", "prefix": "BATEMANS", "sd_file": "BATEMANSSD.TXT"},
}

_YEARLY_COLS = ["date", "hour", "freq", "dirn", "energy", "spread", "skewness", "kurtosis"]
_VALUE_COLS = ["dirn", "energy", "spread", "skewness", "kurtosis"]

VAR_ATTRS: dict[str, dict[str, str]] = {
    "energy_norm": {
        "units": "1",
        "long_name": "Spectral energy density, normalized to peak=1 within each hour",
    },
    "energy_density": {
        "standard_name": "sea_surface_wave_variance_spectral_density",
        "units": "m2 s",
        "long_name": "Spectral energy density (energy_norm * spect_den)",
    },
    "dirn": {
        "standard_name": "sea_surface_wave_from_direction",
        "units": "degree",
        "long_name": "Mean wave-from direction per frequency bin",
    },
    "spread": {"units": "degree", "long_name": "Directional spread per frequency bin"},
    "skewness": {"units": "1", "long_name": "Directional skewness per frequency bin"},
    "kurtosis": {"units": "1", "long_name": "Directional kurtosis per frequency bin"},
    "spect_den": {
        "units": "m2 s",
        "long_name": "Total spectral density scale factor (multiplies energy_norm)",
    },
    "freq": {"standard_name": "sea_surface_wave_frequency", "units": "Hz"},
}


def _station_dir(info: dict[str, str]) -> Path:
    return resolve(f"{RAW_SPECTRA_DIR}/{info['folder']}")


def yearly_files(info: dict[str, str]) -> list[Path]:
    """All yearly spectral files for one station, sorted, excluding the SD file."""
    d = _station_dir(info)
    return sorted(
        f for f in d.glob(f"{info['prefix']}*.TXT") if f.name != info["sd_file"]
    )


def _read_yearly_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", skiprows=3, names=_YEARLY_COLS, engine="python")
    # A handful of files (e.g. BYRON2006/2007) carry one spurious trailing
    # freq=0 / all-zero row after each real 64-bin hour for a several-month
    # stretch — the 64 real bins that hour are complete and valid, this is
    # just an export artifact. Drop it rather than let it become a phantom
    # 65th, all-NaN frequency bin after pivoting.
    df = df[df["freq"] > 0]
    df["time_local"] = pd.to_datetime(
        df["date"] + " " + df["hour"].astype(str), format="%d-%b-%Y %H"
    )
    return df


def _read_sd_file(path: Path) -> pd.Series:
    """``spect_den`` indexed by local (Eastern Standard) timestamp.

    Returns an empty series if the file is unusable (e.g. Sydney's, which is
    entirely null bytes in the raw archive) rather than raising, so callers
    can fall back to normalized-only output.
    """
    if not path.is_file() or not any(path.read_bytes()):
        return pd.Series(dtype=float)
    df = pd.read_csv(path, sep=r"\s+", skiprows=8, names=["date", "time", "spect_den"], engine="python")
    dt = pd.to_datetime(df["date"] + " " + df["time"], format="%d/%m/%Y %H:%M:%S")
    s = pd.Series(df["spect_den"].to_numpy(float), index=dt)
    return s[~s.index.duplicated(keep="first")]


def read_station_spectra(instrument_id: str) -> xr.Dataset:
    """Read, join, and pivot one station's full raw spectral record.

    Returns
    -------
    xarray.Dataset
        Dims ``(time, freq)`` for ``energy_norm``/``dirn``/``spread``/
        ``skewness``/``kurtosis``, dim ``(time,)`` for ``spect_den``, and
        ``energy_density`` only when a usable SD file exists. Time is UTC.
    """
    info = STATIONS[instrument_id]
    years = yearly_files(info)
    if not years:
        raise FileNotFoundError(f"No yearly spectral files found for {instrument_id}")

    frames = [_read_yearly_file(p) for p in years]
    df = pd.concat(frames, ignore_index=True)

    freq_coord = np.sort(df["freq"].unique())

    indexed = df.set_index(["time_local", "freq"])[_VALUE_COLS]
    indexed = indexed[~indexed.index.duplicated(keep="first")]
    ds = indexed.to_xarray().reindex(freq=freq_coord).sortby("time_local")
    ds = ds.rename({"energy": "energy_norm", "time_local": "time"})

    spect_den = _read_sd_file(_station_dir(info) / info["sd_file"])
    if not spect_den.empty:
        spect_den = spect_den.reindex(ds["time"].to_index())
        ds = ds.assign(spect_den=("time", spect_den.to_numpy()))
        ds["energy_density"] = ds["energy_norm"] * ds["spect_den"]

    utc_time = ds["time"].to_index() - pd.Timedelta(hours=NSW_LOCAL_UTC_OFFSET_HOURS)
    ds = ds.assign_coords(time=utc_time.to_numpy())

    for name, attrs in VAR_ATTRS.items():
        if name in ds.variables:
            ds[name].attrs.update(attrs)

    return ds


def standardize_and_write(
    instrument_id: str,
    meta: dict,
    out_dir: Path,
) -> Path:
    """Read, standardize, attach provenance attrs, and write one station's spectra.

    Parameters
    ----------
    instrument_id
        Catalogue instrument id (also the output file stem), must be a key
        of :data:`STATIONS`.
    meta
        Metadata dict for this instrument (from ``obs.get_coordinates()``).
    out_dir
        Destination directory (``inputs/Observations/processed/spectra/``).

    Returns
    -------
    Path
        Written file path.
    """
    info = STATIONS[instrument_id]
    ds = read_station_spectra(instrument_id)
    has_density = "energy_density" in ds

    ds.attrs.update(
        {
            "instrument_id": instrument_id,
            "station_name": str(meta.get("name") or ""),
            "institution": str(meta.get("institution") or ""),
            "network": "offshore (MHL)",
            "longitude": float(meta["lon"]),
            "latitude": float(meta["lat"]),
            "depth_m": float(meta["depth"]) if meta.get("depth") is not None else float("nan"),
            "source_files": ", ".join(p.name for p in yearly_files(info)) + f", {info['sd_file']}",
            "time_convention": "UTC (converted from Eastern Standard, fixed UTC+10, no DST)",
            "processing": (
                "Directional frequency spectra (64 bins/hour) standardized from raw MHL "
                "yearly spectral text files. 'energy_norm' is peak-normalized to 1 per "
                "hour as provided; "
                + (
                    "'energy_density' (m^2/Hz) is energy_norm scaled by the paired SD "
                    "file's spect_den."
                    if has_density
                    else (
                        "no 'energy_density' — this station's SD (spect_den) file is "
                        "corrupted (all null bytes) in the raw archive, so only the "
                        "normalized shape is available, not an absolute spectrum."
                    )
                )
            ),
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{instrument_id}.nc"
    ds.to_netcdf(out_path)
    return out_path


__all__ = [
    "NSW_LOCAL_UTC_OFFSET_HOURS",
    "RAW_SPECTRA_DIR",
    "STATIONS",
    "VAR_ATTRS",
    "read_station_spectra",
    "standardize_and_write",
    "yearly_files",
]
