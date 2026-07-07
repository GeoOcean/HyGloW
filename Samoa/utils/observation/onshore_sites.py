"""Onshore observation sites — GeoJSON I/O and lookups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from utils.config import ONSHORE_WAVES, resolve

REQUIRED_KEYS = ("lon", "lat", "type", "institution", "recording")
COORD_KEYS = ("lon", "lat")


def geojson_path() -> Path:
    """
    Resolved path to the onshore-waves GeoJSON.

    Returns
    -------
    Path
        ``onshore_waves`` path from ``Samoa.json``.
    """
    return resolve(ONSHORE_WAVES)


def _load_geojson(path: Path | str | None = None) -> dict:
    """Load a GeoJSON feature collection from disk."""
    with resolve(path or geojson_path()).open(encoding="utf-8") as f:
        return json.load(f)


def get_coordinates(
    role: str | None = None,
    institution: str | None = None,
    *,
    path: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load site metadata keyed by ``instrument_id``.

    Parameters
    ----------
    role, institution
        Optional filters (case-insensitive).
    path
        Override GeoJSON path.

    Returns
    -------
    dict[str, dict]
        ``instrument_id`` → lon/lat and metadata fields.
    """
    features = _load_geojson(path).get("features", [])
    requested_role = role.strip().lower() if role else None
    requested_inst = institution.strip().lower() if institution else None

    sites: dict[str, dict[str, Any]] = {}
    for feat in features:
        props = feat.get("properties") or {}
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        if len(coords) < 2:
            continue

        iid = props.get("instrument_id")
        if iid is None:
            continue
        iid = str(iid)

        site_role = (props.get("role") or props.get("type") or "").strip().lower()
        site_inst = (props.get("institution") or "").strip().lower()

        if requested_role and site_role != requested_role:
            continue
        if requested_inst and site_inst != requested_inst:
            continue

        sites[iid] = {
            "lon": float(coords[0]),
            "lat": float(coords[1]),
            "role": site_role,
            "institution": props.get("institution"),
            "type": props.get("type"),
            "recording": props.get("recording"),
            "depth": props.get("depth"),
            "name": props.get("name"),
        }

    return sites


def dict_to_geojson(observations: dict[str, dict]) -> dict:
    """
    Convert an observations dict to a GeoJSON FeatureCollection.

    Parameters
    ----------
    observations
        ``instrument_id`` → property dict with required keys.

    Returns
    -------
    dict
        GeoJSON FeatureCollection.

    Raises
    ------
    KeyError
        If required property keys are missing.
    """
    features = []
    for instrument_id, props in observations.items():
        missing = [k for k in REQUIRED_KEYS if k not in props]
        if missing:
            raise KeyError(f"{instrument_id}: missing {missing}")

        lon, lat = float(props["lon"]), float(props["lat"])
        properties = {k: v for k, v in props.items() if k not in COORD_KEYS}
        properties["instrument_id"] = str(instrument_id)
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def geojson_to_dataframe(feature_collection: dict) -> pd.DataFrame:
    """
    Flatten a GeoJSON FeatureCollection to a DataFrame.

    Parameters
    ----------
    feature_collection
        GeoJSON with point features.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``instrument_id`` with lon/lat columns.
    """
    rows = []
    for feat in feature_collection["features"]:
        props = dict(feat["properties"])
        props["lon"], props["lat"] = feat["geometry"]["coordinates"][:2]
        rows.append(props)
    return pd.DataFrame(rows).set_index("instrument_id")


def dict_to_gdf(observations: dict[str, dict]) -> gpd.GeoDataFrame:
    """
    Build a GeoDataFrame from an observations dict.

    Parameters
    ----------
    observations
        ``instrument_id`` → property dict with ``lon`` and ``lat``.

    Returns
    -------
    geopandas.GeoDataFrame
        Points in ``EPSG:4326``.
    """
    rows = [{**props, "instrument_id": iid} for iid, props in observations.items()]
    return gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy([r["lon"] for r in rows], [r["lat"] for r in rows]),
        crs="EPSG:4326",
    )


def save_geojson(
    feature_collection: dict, output_path: Path | str | None = None
) -> Path:
    """
    Write a GeoJSON FeatureCollection to disk.

    Parameters
    ----------
    feature_collection
        GeoJSON dict to serialize.
    output_path
        Destination path; default ``onshore_waves`` from config.

    Returns
    -------
    Path
        Written file path.
    """
    path = resolve(output_path or geojson_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(feature_collection, f, indent=2)
        f.write("\n")
    return path


__all__ = [
    "COORD_KEYS",
    "REQUIRED_KEYS",
    "dict_to_gdf",
    "dict_to_geojson",
    "geojson_path",
    "geojson_to_dataframe",
    "get_coordinates",
    "save_geojson",
]
