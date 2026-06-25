"""Isobath extraction and SnapWave output-site construction."""

from __future__ import annotations

from typing import Any, Mapping

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import shapely
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon
from utils import observations as obs
from utils.config import PATHS, resolve

_X_NAMES = ("x", "lon", "longitude")
_Y_NAMES = ("y", "lat", "latitude")

DEFAULT_OUTPUT_SITES: dict[str, Any] = {
    "isobaths": [
        {"name": "10m", "level": -10.0, "spacing": 0.003, "longest_only": True},
        {"name": "20m", "level": -20.0, "spacing": 0.003, "longest_only": True},
    ],
    "include_buoys": True,
    "clip_to_aoi": True,
}


def _xy_names(da: xr.DataArray) -> tuple[str, str]:
    x = next((n for n in _X_NAMES if n in da.coords), None)
    y = next((n for n in _Y_NAMES if n in da.coords), None)
    if x is None or y is None:
        raise KeyError(
            f"Could not find x/y coords in {tuple(da.coords)}; "
            f"expected one of {_X_NAMES} and {_Y_NAMES}."
        )
    return x, y


def get_isobath_points(
    da: xr.DataArray,
    level: float,
    *,
    spacing: float | None = None,
    longest_only: bool = False,
) -> np.ndarray:
    """Return ``(N, 2)`` array of ``(x, y)`` points along an isobath."""

    xname, yname = _xy_names(da)
    x = da[xname].values
    y = da[yname].values
    z = np.asarray(da.values, dtype=float)
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"Expected da.shape == (len({yname}), len({xname})); "
            f"got {z.shape} vs ({y.size}, {x.size}). "
            f"Try da.transpose({yname!r}, {xname!r})."
        )

    fig, ax = plt.subplots()
    cs = ax.contour(x, y, z, levels=[float(level)])
    segments = [np.asarray(p) for p in cs.allsegs[0] if len(p) >= 2]
    plt.close(fig)

    if not segments:
        return np.empty((0, 2), dtype=float)

    if longest_only:
        segments = [max(segments, key=_polyline_length)]

    if spacing is not None and spacing > 0:
        segments = [_resample_polyline(s, spacing) for s in segments]
        segments = [s for s in segments if s.shape[0] > 0]

    return np.vstack(segments) if segments else np.empty((0, 2), dtype=float)


def compute_output_sites(
    *,
    config: Mapping[str, Any] | None = None,
    topobathy: xr.DataArray | None = None,
    aoi_polygon: Polygon | MultiPolygon | None = None,
) -> pd.DataFrame:
    """Build output sites from isobaths (topobathy) and onshore observation buoys."""

    cfg = {**DEFAULT_OUTPUT_SITES, **dict(config or {})}
    rows: list[dict[str, Any]] = []

    iso_specs = cfg.get("isobaths") or []
    if iso_specs:
        if topobathy is None:
            topobathy = rioxarray.open_rasterio(
                resolve(PATHS["topobathy_file"])
            ).squeeze("band", drop=True)
        for spec in iso_specs:
            name = spec.get("name") or f"{abs(spec['level']):g}m"
            pts = get_isobath_points(
                topobathy,
                level=float(spec["level"]),
                spacing=spec.get("spacing"),
                longest_only=bool(spec.get("longest_only", False)),
            )
            for i, (lon, lat) in enumerate(pts):
                rows.append(
                    {
                        "lon": float(lon),
                        "lat": float(lat),
                        "source": "isobath",
                        "source_id": f"{name}_{i}",
                        "source_label": name,
                        "depth": float(spec["level"]),
                        "name": None,
                    }
                )

    if cfg.get("include_buoys", True):
        for iid, site in obs.get_coordinates(role=cfg.get("buoy_role")).items():
            rows.append(
                {
                    "lon": float(site["lon"]),
                    "lat": float(site["lat"]),
                    "source": "buoy",
                    "source_id": f"buoy_{iid}",
                    "source_label": site.get("type") or "buoy",
                    "depth": site.get("depth"),
                    "name": site.get("name") or site.get("institution"),
                }
            )

    if not rows:
        raise ValueError("No output-site points produced.")

    df = pd.DataFrame(rows).reset_index(drop=True)

    if bool(cfg.get("clip_to_aoi", True)):
        if aoi_polygon is None:
            aoi_polygon = gpd.read_file(resolve(PATHS["aoi_file"])).geometry.iloc[0]
        mask = shapely.contains_xy(
            aoi_polygon, df["lon"].to_numpy(float), df["lat"].to_numpy(float)
        )
        df = df.loc[mask].reset_index(drop=True)
        if df.empty:
            raise ValueError("All output-site points were clipped by the AoI.")

    return df


def write_output_sites(
    df: pd.DataFrame,
    out_dir: str,
    filename: str = "output_sites.txt",
) -> None:
    """Write ``lon lat`` pairs to a SnapWave observation file."""

    path = resolve(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, df[["lon", "lat"]].to_numpy(float), fmt="%.8f")


def _polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _resample_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 0.0:
        return np.empty((0, 2), dtype=float)

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    ds = np.arange(0.0, total, float(spacing))
    if ds.size == 0 or (total - ds[-1]) > 0.25 * spacing:
        ds = np.append(ds, total)

    out = np.empty((ds.size, 2), dtype=float)
    for k, d in enumerate(ds):
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = max(0, min(i, len(seg_len) - 1))
        t = 0.0 if seg_len[i] == 0 else (d - cum[i]) / seg_len[i]
        out[k] = pts[i] * (1.0 - t) + pts[i + 1] * t
    return out
