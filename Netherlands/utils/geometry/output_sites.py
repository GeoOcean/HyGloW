"""Isobath extraction and SnapWave output-site construction.

Used by :class:`utils.snapwave_wrapper.SnapWaveModelWrapper` to build the
``output_sites.txt`` file written into every SnapWave case. Site layout is
configured in the case config JSON → ``simulations.output_sites_config``.

Row order in the output table (SnapWave matches by position):

1. isobath points (depth contours from topobathy)
2. mesh nodes inside named polygons (e.g. Apia)
3. buoy locations from the onshore-waves GeoJSON
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import shapely
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon
from utils.observation import onshore_sites as obs
from utils.config import OUTPUT_SITES_CONFIG, PATHS, resolve

_X_NAMES = ("x", "lon", "longitude")
_Y_NAMES = ("y", "lat", "latitude")

# Columns expected by snapwave_wrapper.SITE_COORDS
OUTPUT_SITE_COLUMNS = (
    "lon",
    "lat",
    "source",
    "source_id",
    "source_label",
    "depth",
    "name",
)


def _xy_names(da: xr.DataArray) -> tuple[str, str]:
    """
    Resolve horizontal coordinate names on a 2D DataArray.

    Parameters
    ----------
    da
        Raster or gridded field with lon/lat (or x/y) coordinates.

    Returns
    -------
    tuple[str, str]
        ``(x_name, y_name)``.

    Raises
    ------
    KeyError
        If no recognised horizontal coordinate pair is found.
    """
    x = next((n for n in _X_NAMES if n in da.coords), None)
    y = next((n for n in _Y_NAMES if n in da.coords), None)
    if x is None or y is None:
        raise KeyError(
            f"Could not find x/y coords in {tuple(da.coords)}; "
            f"expected one of {_X_NAMES} and {_Y_NAMES}."
        )
    return x, y


def _resample_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    """
    Resample a polyline at approximately uniform arc length.

    Parameters
    ----------
    pts
        ``(N, 2)`` vertex array.
    spacing
        Target spacing between output points (same units as *pts*).

    Returns
    -------
    numpy.ndarray
        Resampled ``(M, 2)`` points; empty when the polyline has zero length.
    """
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


def get_isobath_points(
    da: xr.DataArray,
    level: float,
    *,
    spacing: float | None = None,
    aoi_polygon: Polygon | MultiPolygon | None = None,
) -> np.ndarray:
    """
    Extract ``(lon, lat)`` points along a bathymetry isobath.

    Uses Matplotlib contouring on the gridded field (figure is closed
    immediately; no display required).

    Parameters
    ----------
    da
        Topobathy raster with horizontal coordinates and depth values.
    level
        Contour level (negative for depth below sea level, e.g. ``-10``).
    spacing
        Optional resampling spacing in degrees along each contour segment.
    aoi_polygon
        When given, keep only segments with at least one point inside the AoI.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` array of ``[lon, lat]``; empty when no contour is found.

    Raises
    ------
    ValueError
        If array dimensions do not match coordinate lengths.
    """
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
    try:
        cs = ax.contour(x, y, z, levels=[float(level)])
        segments = [np.asarray(p) for p in cs.allsegs[0] if len(p) >= 2]
    finally:
        plt.close(fig)

    if not segments:
        return np.empty((0, 2), dtype=float)

    if aoi_polygon is not None:
        segments = [
            s
            for s in segments
            if shapely.contains_xy(aoi_polygon, s[:, 0], s[:, 1]).any()
        ]

    if spacing is not None and spacing > 0:
        segments = [_resample_polyline(s, spacing) for s in segments]
        segments = [s for s in segments if s.shape[0] > 0]

    return np.vstack(segments) if segments else np.empty((0, 2), dtype=float)


def get_mesh_nodes_in_polygon(
    polygon: Polygon | MultiPolygon,
    mesh: xr.Dataset,
    *,
    wet_only: bool = False,
) -> list[dict[str, Any]]:
    """
    List unstructured-mesh nodes inside a polygon.

    Parameters
    ----------
    polygon
        Area of interest in geographic coordinates.
    mesh
        SnapWave mesh NetCDF with ``mesh2d_node_x/y/z``.
    wet_only
        When ``True``, keep only nodes with negative bed elevation (wet).

    Returns
    -------
    list[dict]
        One dict per node with keys ``lon``, ``lat``, ``depth``, ``node_index``.
    """
    lon = mesh["mesh2d_node_x"].values
    lat = mesh["mesh2d_node_y"].values
    z = mesh["mesh2d_node_z"].values

    mask = shapely.contains_xy(polygon, lon, lat)
    if wet_only:
        mask &= z < 0.0

    idx = np.flatnonzero(mask)
    return [
        {
            "lon": float(lon[i]),
            "lat": float(lat[i]),
            "depth": float(z[i]),
            "node_index": int(i),
        }
        for i in idx
    ]


def compute_output_sites(
    *,
    config: Mapping[str, Any] | None = None,
    topobathy: xr.DataArray | None = None,
    aoi_polygon: Polygon | MultiPolygon | None = None,
) -> pd.DataFrame:
    """
    Build the SnapWave output-sites table from config and static inputs.

    Parameters
    ----------
    config
        Override or extend ``simulations.output_sites_config`` from
        the case config JSON.
    topobathy
        Bathymetry raster; opened from ``paths.topobathy_file`` when omitted.
    aoi_polygon
        AoI geometry for clipping; read from ``paths.aoi_file`` when omitted
        and ``clip_to_aoi`` is enabled.

    Returns
    -------
    pandas.DataFrame
        Columns ``lon``, ``lat``, ``source``, ``source_id``, ``source_label``,
        ``depth``, ``name``. Row order: isobaths → mesh polygons → buoys.

    Raises
    ------
    ValueError
        If no sites are produced or all are clipped by the AoI.
    """
    cfg = {**OUTPUT_SITES_CONFIG, **dict(config or {})}
    rows: list[dict[str, Any]] = []

    clip_to_aoi = bool(cfg.get("clip_to_aoi", True))
    if clip_to_aoi and aoi_polygon is None:
        aoi_polygon = gpd.read_file(resolve(PATHS["aoi_file"])).geometry.iloc[0]

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
                aoi_polygon=aoi_polygon,
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

    poly_specs = cfg.get("polygons") or []
    if poly_specs:
        mesh = xr.open_dataset(resolve(PATHS["mesh_file"]))
        try:
            for spec in poly_specs:
                name = spec.get("name")
                if not name:
                    raise ValueError("Each entry in `polygons` must have a `name`.")
                poly_path = spec.get("file")
                if not poly_path:
                    raise ValueError(
                        f"Polygon {name!r} is missing `file` in output_sites_config."
                    )
                polygon = gpd.read_file(resolve(poly_path)).geometry.iloc[0]
                nodes = get_mesh_nodes_in_polygon(
                    polygon,
                    mesh,
                    wet_only=bool(spec.get("wet_only", False)),
                )
                for node in nodes:
                    i = node["node_index"]
                    rows.append(
                        {
                            "lon": node["lon"],
                            "lat": node["lat"],
                            "source": "mesh",
                            "source_id": f"{name}_{i}",
                            "source_label": name,
                            "depth": node["depth"],
                            "name": None,
                        }
                    )
        finally:
            mesh.close()

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

    if clip_to_aoi:
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
) -> Path:
    """
    Write SnapWave observation coordinates to a text file.

    Parameters
    ----------
    df
        Output-sites table from :func:`compute_output_sites`.
    out_dir
        Case or output directory.
    filename
        Observation file name (``.nc`` sibling is produced by SnapWave).

    Returns
    -------
    Path
        Written text file path.
    """
    path = resolve(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, df[["lon", "lat"]].to_numpy(float), fmt="%.8f")
    return path


__all__ = [
    "OUTPUT_SITE_COLUMNS",
    "compute_output_sites",
    "get_isobath_points",
    "get_mesh_nodes_in_polygon",
    "write_output_sites",
]
