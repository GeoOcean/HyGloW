"""Netherlands wave-buoy archive (Waterinfo + CMEMS, 2026-10) -> HyGloW processed buoys.

Case-specific (not in the template). ``inputs/Observations/raw/paulina/``
holds the original delivery (``transfer_3757173_files_801acfe7.zip``) with the
station list ``paulina.csv``, and, unpacked from it, ``Waterinfo.zip``
(Rijkswaterstaat DDL NetCDFs, one folder per location, one file per quantity),
``CMEMS.zip`` (in-situ yearly CSVs per variable, all of Europe) and
``observations.py`` (the provider's reader, for reference). The earlier
Rijkswaterstaat exports in ``inputs/Observations/raw/`` (:data:`LEGACY_RWS`)
are a third source.

- :func:`catalog` — candidate stations (``paulina.csv`` + legacy exports) with
  an inside-the-AoI flag;
- :func:`group_stations` — stations within ``max_distance_m`` of each other are
  one buoy (co-located sensors, renamed stations, the same buoy in both
  sources);
- :func:`build_buoys` — one hourly NetCDF per buoy in
  ``inputs/Observations/processed/<instrument_id>.nc`` with ``hs`` (m), ``tp``
  (s), ``dir`` and ``spr`` (deg). Per variable, every matching source series is
  resampled to hourly means (directions as vector means) and they are merged
  longest-record first (within a tier, see :data:`WATERINFO_QUANTITIES`), the
  others only filling its gaps.

Waterinfo quantities: Hm0 30-500 mHz (cm), peak frequency (Hz, ``tp = 1/fp``),
mean spectral direction and directional spreading 30-500 mHz; only quality
code ``00`` is kept. CMEMS: ``VHM0``, ``VTPK``, ``VMDR``, ``VPSP`` with
``value_qc == 1``.
"""

from __future__ import annotations

import logging
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

from utils.config import PATHS, resolve

logger = logging.getLogger("hywaves.buoy_archive")

CATALOG_FILE = "inputs/Observations/raw/paulina/paulina.csv"
ARCHIVE_DIR = "inputs/Observations/raw/paulina"
PROCESSED_DIR = "inputs/Observations/processed"

#: Waterinfo quantities per HyGloW variable: ``(file-name fragment, conversion,
#: tier)``. Lower tiers are merged first; a higher tier only fills their gaps
#: (Hm0 over 30-1000 mHz, slightly above the 30-500 mHz one, is a fallback).
WATERINFO_QUANTITIES: dict[str, list[tuple[str, str, int]]] = {
    "hs": [
        ("Significante_golfhoogte_in_het_spectrale_domein_in_Oppervlaktewater_golffrequentie_tussen_30_en_500_mHz_in_cm", "cm", 0),
        ("Significante_golfhoogte_in_het_spectrale_domein_in_Oppervlaktewater_golffrequentie_tussen_30_en_1000_mHz_in_cm", "cm", 1),
    ],
    "tp": [("Golffrequentie_bij_maximum_v_h__variantiedichtheidspectrum_in_Oppervlaktewater_in_Hz", "inverse", 0)],
    "dir": [("Gemiddelde_golfrichting_in_het_spectrale_domein_in_Oppervlaktewater_golffrequentie_tussen_30_en_500_mHz_in_graad", "deg", 0)],
    "spr": [("Gemiddelde_golfrichtingspreiding_in_het_spectrale_domein_in_Oppervlaktewater_golffrequentie_tussen_30_en_500_mHz_in_graad", "deg", 0)],
}
#: CMEMS in-situ variable per HyGloW variable.
CMEMS_VARIABLES = {"hs": "VHM0", "tp": "VTPK", "dir": "VMDR", "spr": "VPSP"}
#: Physically plausible ranges (anything else is a sentinel or a bad sample).
VALID_RANGE = {"hs": (0.0, 20.0), "tp": (0.5, 30.0), "dir": (0.0, 360.0), "spr": (0.0, 90.0)}
DIRECTIONAL = ("dir",)
VARIABLES = ("hs", "tp", "dir", "spr")

#: Earlier Rijkswaterstaat exports (hs/dir), kept as a source: id -> (raw file, lon, lat, name).
LEGACY_RWS: dict[str, tuple[str, float, float, str]] = {
    "AmelanderZeegat_1_1": ("amelanderzeegat.1.1-Amelander_zeegat_1_1.nc", 5.4814, 53.5076, "Amelander Zeegat, 1.1"),
    "AmelanderZeegat_1_2": ("amelanderzeegat.1.2-Amelander_zeegat_1_2.nc", 5.48442, 53.5076, "Amelander Zeegat, 1.2"),
    "EierlandseGat": ("eierlandsegat-Eierlandse_Gat.nc", 4.66035, 53.2762, "Eierlandse Gat"),
    "EurogeulE13": ("eurogeul.e13-Eurogeul_E13.nc", 3.7418, 52.00918, "Eurogeul, E13"),
    "IJmuidenMunitiestort_1": ("ijmuiden.munitiestort.1-IJmuiden_munitiestort_1.nc", 4.05702, 52.54922, "IJmuiden, munitiestort, 1"),
    "PettenMeetraai01": ("petten.meetraai1-Petten_meetraai_01.nc", 4.55678, 52.80685, "Petten, meetraai 01"),
    "StortemelkWest": ("stortemelk.west-Stortemelk_west.nc", 4.996, 53.32072, "Stortemelk, west"),
    "WestereemsWest": ("westereems.west-Westereems_west.nc", 6.36667, 53.61806, "Westereems, west"),
    "ZandmoterBoeiNoord": ("zandmoter.boei.noord-Zandmoter_boei_noord.nc", 4.15827, 52.04809, "Zandmotor, boei noord"),
}

_M_PER_DEG = 111_195.0


def _distance_m(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Equirectangular distance (m), fine at the km scale."""
    dx = (np.asarray(lon2) - lon1) * np.cos(np.deg2rad(lat1)) * _M_PER_DEG
    dy = (np.asarray(lat2) - lat1) * _M_PER_DEG
    return np.hypot(dx, dy)


def archive_dir() -> Path:
    """``inputs/Observations/raw/paulina``."""
    return resolve(ARCHIVE_DIR)


# ---------------------------------------------------------------------------
# Stations
# ---------------------------------------------------------------------------


def catalog() -> pd.DataFrame:
    """
    Candidate stations: ``paulina.csv`` and the legacy exports, flagged inside the AoI.

    Returns
    -------
    pandas.DataFrame
        ``station, lon, lat, depth, start, end, origin`` (``paulina`` or
        ``legacy``), ``inside_aoi``.
    """
    paulina = pd.read_csv(resolve(CATALOG_FILE))
    rows = [
        {
            "station": str(r.Location),
            "lon": float(r.Lon_degE),
            "lat": float(r.Lat_degN),
            "depth": float(r.Depth_m),
            "start": str(r.Obs_start),
            "end": str(r.Obs_end),
            "origin": "paulina",
        }
        for r in paulina.itertuples()
    ]
    rows += [
        {"station": sid, "lon": lon, "lat": lat, "depth": np.nan, "start": "", "end": "", "origin": "legacy"}
        for sid, (_, lon, lat, _) in LEGACY_RWS.items()
    ]
    stations = pd.DataFrame(rows)
    aoi = gpd.read_file(resolve(PATHS["aoi_file"])).to_crs(4326).geometry.union_all()
    stations["inside_aoi"] = [aoi.contains(Point(x, y)) for x, y in zip(stations.lon, stations.lat)]
    return stations


def group_stations(stations: pd.DataFrame, max_distance_m: float = 1000.0) -> pd.DataFrame:
    """
    One buoy per cluster of stations within *max_distance_m* (single linkage).

    The buoy keeps a legacy id when the cluster has one (continuity with
    earlier results), else the ``paulina.csv`` name.

    Parameters
    ----------
    stations
        Output of :func:`catalog` (usually only ``inside_aoi`` rows).
    max_distance_m
        Stations closer than this are the same buoy.

    Returns
    -------
    pandas.DataFrame
        *stations* with ``instrument_id`` (the buoy each station belongs to).
    """
    stations = stations.reset_index(drop=True).copy()
    label = np.arange(len(stations))
    for i, r in stations.iterrows():
        near = np.flatnonzero(_distance_m(r.lon, r.lat, stations.lon, stations.lat) <= max_distance_m)
        old = np.unique(label[near])
        label[np.isin(label, old)] = old.min()
    stations["cluster"] = label
    ids = {}
    for cluster, members in stations.groupby("cluster"):
        legacy = sorted(members.loc[members.origin == "legacy", "station"])
        ids[cluster] = legacy[0] if legacy else members.sort_values("station")["station"].iloc[0]
    stations["instrument_id"] = stations["cluster"].map(ids)
    return stations.drop(columns="cluster")


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def waterinfo_locations() -> pd.DataFrame:
    """Waterinfo folders and their coordinates (from each folder's smallest file, cached)."""
    cache = archive_dir() / "waterinfo_locations.csv"
    if cache.is_file():
        return pd.read_csv(cache)
    archive = zipfile.ZipFile(archive_dir() / "Waterinfo.zip")
    smallest: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.filename.endswith(".nc"):
            folder = info.filename.split("/")[1]
            if folder not in smallest or info.file_size < smallest[folder].file_size:
                smallest[folder] = info
    rows = []
    for folder, info in smallest.items():
        path = _extract(archive, info.filename)
        with xr.open_dataset(path) as ds:
            rows.append({"folder": folder, "lon": float(ds.attrs.get("Lon", np.nan)), "lat": float(ds.attrs.get("Lat", np.nan))})
    table = pd.DataFrame(rows).sort_values("folder")
    table.to_csv(cache, index=False)
    return table


def _extract(archive: zipfile.ZipFile, member: str) -> Path:
    """Extract one Waterinfo member under ``raw/paulina/`` once."""
    target = archive_dir() / member
    if not target.is_file():
        archive.extract(member, archive_dir())
    return target


def _hourly(series: pd.Series, var: str) -> pd.Series:
    """Hourly means within the valid range (vector mean for directions)."""
    low, high = VALID_RANGE[var]
    series = series[(series >= low) & (series < high)].dropna()
    if series.empty:
        return series
    if var in DIRECTIONAL:
        rad = np.deg2rad(series)
        s = np.sin(rad).resample("1h").mean()
        c = np.cos(rad).resample("1h").mean()
        return np.mod(np.rad2deg(np.arctan2(s, c)), 360.0).dropna()
    return series.resample("1h").mean().dropna()


def waterinfo_series(folder: str, var: str, tier: int = 0) -> pd.Series:
    """
    Hourly series of one variable and tier at one Waterinfo folder (all its matching files).

    Parameters
    ----------
    folder
        Waterinfo location folder.
    var
        ``hs``, ``tp``, ``dir`` or ``spr``.
    tier
        Quantity tier (see :data:`WATERINFO_QUANTITIES`).

    Returns
    -------
    pandas.Series
        Hourly values (empty if the folder has no such quantity).
    """
    matches = [(f, c) for f, c, t in WATERINFO_QUANTITIES[var] if t == tier]
    if not matches:
        return pd.Series(dtype=float)
    fragment, conversion = matches[0]
    archive = zipfile.ZipFile(archive_dir() / "Waterinfo.zip")
    members = [n for n in archive.namelist() if n.startswith(f"Waterinfo/{folder}/") and fragment in n]
    parts = []
    for member in members:
        with xr.open_dataset(_extract(archive, member)) as ds:
            values = ds["Meetwaarde.Waarde_Numeriek"]
            code = ds.get("WaarnemingMetadata.Kwaliteitswaardecode")
            if code is not None:
                values = values.where(code.astype(str) == "00")
            parts.append(values.to_series().dropna())
    if not parts:
        return pd.Series(dtype=float)
    series = pd.concat(parts).sort_index()
    series.index = pd.DatetimeIndex(series.index)
    if conversion == "cm":
        series = series / 100.0
    elif conversion == "inverse":
        series = 1.0 / series[series > 0]
    return _hourly(series, var)


def extract_cmems(variables: tuple[str, ...] = tuple(CMEMS_VARIABLES.values()), margin_deg: float = 0.2) -> None:
    """
    Stream the CMEMS yearly CSVs out of ``CMEMS.zip``; keep QC-1 rows near the AoI.

    Writes ``raw/paulina/cmems_aoi/cmems_<VAR>.parquet`` (``platform_id, time,
    longitude, latitude, value``). Slow (tens of GB of text); run once.
    """
    aoi = gpd.read_file(resolve(PATHS["aoi_file"])).to_crs(4326).total_bounds
    lon0, lat0, lon1, lat1 = aoi[0] - margin_deg, aoi[1] - margin_deg, aoi[2] + margin_deg, aoi[3] + margin_deg
    zip_path = archive_dir() / "CMEMS.zip"
    out_dir = archive_dir() / "cmems_aoi"
    out_dir.mkdir(exist_ok=True)
    names = zipfile.ZipFile(zip_path).namelist()
    columns = ["platform_id", "time", "longitude", "latitude", "value", "value_qc"]
    for var in variables:
        parts = []
        for name in (n for n in names if n.startswith(f"CMEMS/{var}/") and n.endswith(".csv")):
            proc = subprocess.Popen(["unzip", "-p", str(zip_path), name], stdout=subprocess.PIPE)
            for chunk in pd.read_csv(proc.stdout, usecols=columns, dtype={"platform_id": str}, chunksize=2_000_000):
                keep = (
                    chunk.longitude.between(lon0, lon1)
                    & chunk.latitude.between(lat0, lat1)
                    & (chunk.value_qc == 1)
                )
                parts.append(chunk.loc[keep, columns[:-1]])
            proc.wait()
        table = pd.concat(parts, ignore_index=True)
        table["time"] = pd.to_datetime(table["time"], utc=True).dt.tz_localize(None)
        table.to_parquet(out_dir / f"cmems_{var}.parquet")
        logger.info("CMEMS %s: %d rows near the AoI", var, len(table))


def cmems_platforms() -> pd.DataFrame:
    """CMEMS platforms near the AoI with their median position (from ``cmems_VHM0``)."""
    table = pd.read_parquet(archive_dir() / "cmems_aoi" / "cmems_VHM0.parquet", columns=["platform_id", "longitude", "latitude"])
    return table.groupby("platform_id")[["longitude", "latitude"]].median().rename(columns={"longitude": "lon", "latitude": "lat"}).reset_index()


def cmems_series(platform: str, var: str) -> pd.Series:
    """Hourly series of one variable at one CMEMS platform (empty if none)."""
    path = archive_dir() / "cmems_aoi" / f"cmems_{CMEMS_VARIABLES[var]}.parquet"
    if not path.is_file():
        return pd.Series(dtype=float)
    table = pd.read_parquet(path, filters=[("platform_id", "==", platform)], columns=["time", "value"])
    series = table.groupby("time")["value"].mean().sort_index()
    series.index = pd.DatetimeIndex(series.index)
    return _hourly(series, var)


def legacy_series(instrument: str, var: str) -> pd.Series:
    """Hourly ``hs``/``dir`` of a legacy Rijkswaterstaat export (empty otherwise)."""
    if instrument not in LEGACY_RWS or var not in ("hs", "dir"):
        return pd.Series(dtype=float)
    from utils.rijkswaterstaat import process_raw_buoy

    ds = process_raw_buoy(Path("inputs/Observations/raw") / LEGACY_RWS[instrument][0])
    series = ds[var].to_series().dropna()
    series.index = pd.DatetimeIndex(series.index)
    return _hourly(series, var)


# ---------------------------------------------------------------------------
# Processed buoys
# ---------------------------------------------------------------------------


def _sources(members: pd.DataFrame, locations: pd.DataFrame, platforms: pd.DataFrame, max_distance_m: float) -> dict[str, list]:
    """Waterinfo folders, CMEMS platforms and legacy ids matching any cluster member."""
    folders, plats = set(), set()
    for r in members.itertuples():
        d = _distance_m(r.lon, r.lat, locations.lon, locations.lat)
        folders |= set(locations.folder[d <= max_distance_m])
        d = _distance_m(r.lon, r.lat, platforms.lon, platforms.lat)
        plats |= set(platforms.platform_id[d <= max_distance_m])
        plats |= set(platforms.platform_id[platforms.platform_id.str.lower() == r.station.lower()])
    legacy = sorted(set(members.station) & set(LEGACY_RWS))
    return {"waterinfo": sorted(folders), "cmems": sorted(plats), "legacy": legacy}


def build_buoys(
    stations: pd.DataFrame,
    *,
    max_distance_m: float = 1000.0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Write one processed NetCDF per buoy and return the buoy table.

    Parameters
    ----------
    stations
        Output of :func:`group_stations`.
    max_distance_m
        Matching radius for Waterinfo folders and CMEMS platforms.
    overwrite
        Rebuild files that exist.

    Returns
    -------
    pandas.DataFrame
        One row per buoy: ``instrument_id, lon, lat, depth, name, stations,
        sources, start, end`` and the number of valid hours per variable
        (``n_hs`` ...). Buoys without any ``hs`` are kept with ``n_hs = 0``
        and no file.
    """
    out_dir = resolve(PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    locations = waterinfo_locations().dropna(subset=["lon", "lat"])
    platforms = cmems_platforms()
    rows = []
    for instrument, members in stations.groupby("instrument_id"):
        src = _sources(members, locations, platforms, max_distance_m)
        path = out_dir / f"{instrument}.nc"
        if path.is_file() and not overwrite:
            with xr.open_dataset(path) as ds:
                data = {v: ds[v].to_series().dropna() for v in VARIABLES if v in ds}
        else:
            data = {}
            for var in VARIABLES:
                tiers = sorted({t for _, _, t in WATERINFO_QUANTITIES[var]})
                candidates = (
                    [(t, f"waterinfo:{f}" + (f"[tier {t}]" if t else ""), waterinfo_series(f, var, t))
                     for t in tiers for f in src["waterinfo"]]
                    + [(0, f"legacy:{i}", legacy_series(i, var)) for i in src["legacy"]]
                    + [(0, f"cmems:{p}", cmems_series(p, var)) for p in src["cmems"]]
                )
                candidates = sorted((c for c in candidates if len(c[2])), key=lambda c: (c[0], -len(c[2])))
                if not candidates:
                    continue
                merged = candidates[0][2]
                for _, _, series in candidates[1:]:
                    merged = merged.combine_first(series)
                data[var] = merged.sort_index()
                data[f"_{var}_sources"] = [name for _, name, _ in candidates]
        main = members.sort_values("origin").iloc[0]  # legacy coordinates (exact) first
        info = {
            "instrument_id": instrument,
            "lon": float(main.lon),
            "lat": float(main.lat),
            "depth": float(members.depth.dropna().iloc[0]) if members.depth.notna().any() else np.nan,
            "name": LEGACY_RWS[instrument][3] if instrument in LEGACY_RWS else instrument,
            "stations": ", ".join(sorted(members.station)),
        }
        if "hs" in data and (not path.is_file() or overwrite):
            index = pd.DatetimeIndex(sorted(set().union(*(data[v].index for v in VARIABLES if v in data))))
            ds = xr.Dataset({v: ("time", data[v].reindex(index).values) for v in VARIABLES if v in data}, coords={"time": index})
            for var, (units, standard) in {
                "hs": ("m", "sea_surface_wave_significant_height"),
                "tp": ("s", "sea_surface_wave_period_at_variance_spectral_density_maximum"),
                "dir": ("degree", "sea_surface_wave_from_direction"),
                "spr": ("degree", "sea_surface_wave_directional_spread"),
            }.items():
                if var in ds:
                    ds[var].attrs = {"units": units, "standard_name": standard, "sources": ", ".join(data[f"_{var}_sources"])}
            ds.attrs.update(
                source="Rijkswaterstaat Waterinfo/DDL + CMEMS in-situ (hourly means)",
                station_id=instrument,
                stations=info["stations"],
                lon=info["lon"],
                lat=info["lat"],
            )
            ds.to_netcdf(path)
        hs = data.get("hs", pd.Series(dtype=float))
        info.update(
            sources=" | ".join(f"{k}: {', '.join(v)}" for k, v in src.items() if v),
            start=str(hs.index.min().date()) if len(hs) else "",
            end=str(hs.index.max().date()) if len(hs) else "",
            **{f"n_{v}": int(len(data.get(v, []))) for v in VARIABLES},
        )
        rows.append(info)
        logger.info("%s: %s", instrument, {f"n_{v}": info[f"n_{v}"] for v in VARIABLES})
    return pd.DataFrame(rows)


def observations_dict(buoys: pd.DataFrame, min_hours: int = 24 * 30) -> dict[str, dict[str, Any]]:
    """
    Notebook-03 ``observations`` entries for buoys with at least *min_hours* of ``hs``.

    Parameters
    ----------
    buoys
        Output of :func:`build_buoys`.
    min_hours
        Minimum number of valid hourly ``hs`` values.

    Returns
    -------
    dict
        ``{instrument_id: {lon, lat, type, institution, recording, measured_depth, name}}``
        (``measured_depth`` from the station metadata; the model uses the mesh
        depth, see :func:`utils.geometry.output_sites`).
    """
    out = {}
    for r in buoys[buoys.n_hs >= min_hours].sort_values("instrument_id").itertuples():
        out[r.instrument_id] = {
            "lon": round(r.lon, 5),
            "lat": round(r.lat, 5),
            "type": "waves",
            "institution": "Rijkswaterstaat / CMEMS",
            "recording": f"{r.start} to {r.end}",
            "measured_depth": None if np.isnan(r.depth) else float(r.depth),
            "name": r.name,
        }
    return out


__all__ = [
    "LEGACY_RWS",
    "build_buoys",
    "catalog",
    "cmems_platforms",
    "extract_cmems",
    "group_stations",
    "observations_dict",
    "waterinfo_locations",
]
