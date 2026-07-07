"""Case map plotting — styling here, paths from :mod:`utils.config`."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rioxarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from utils.config import PATHS, display_title, resolve
from utils.directions import nautical_sector_to_wedge_angles

FIGSIZE = (9, 7)
TOPOBATHY_VMIN, TOPOBATHY_VMAX = -200, 0
TOPOBATHY_CMAP, TOPOBATHY_ALPHA = "terrain", 0.85
AOI_EDGE_COLOR, AOI_LINEWIDTH = "black", 1.2
LAND_FACE_COLOR, LAND_EDGE_COLOR, LAND_ALPHA = "#c4c4c4", "#666666", 0.55
RING_COLOR, RING_LINEWIDTH = "#e377c2", 1.0

HINDCAST_COLORS = {
    "whacs": "#1f77b4",
    "era5": "#ff7f0e",
    "codec": "#2ca02c",
}
DEFAULT_POINT_COLOR = "#17becf"

TYPE_COLORS = {
    "validation": "#d62728",
    "waves": "#1f77b4",
    "spectrum": "#9467bd",
    "calibration": "#ff7f0e",
    "levels": "#2ca02c",
    "ignore": "#7f7f7f",
}
ISOBATH_COLORS = {
    "5m": "#8dd3c7",
    "10m": "#1f78b4",
    "20m": "#33a02c",
    "50m": "#e31a1c",
}
MESH_COLORS = {
    "Apia": "#9467bd",
}
SITE_SOURCE_MARKERS = {
    "isobath": "o",
    "mesh": ".",
    "buoy": "s",
}
POINT_MARKERSIZE, POINT_EDGE_WIDTH = 70, 0.6
POINT_EDGE_COLOR = "white"
ANNOTATE_FONTSIZE, ANNOTATE_OFFSET = 8, (4, 4)
SECTOR_ALPHA = 0.15


def _topobathy():
    return rioxarray.open_rasterio(resolve(PATHS["topobathy_file"])).squeeze(drop=True)


def _aoi() -> gpd.GeoDataFrame:
    return gpd.read_file(resolve(PATHS["aoi_file"]))


def plot_land(land: gpd.GeoDataFrame, ax: Axes, *, label: str = "land") -> None:
    """
    Plot land polygons on an existing axes.

    Parameters
    ----------
    land
        Land geometry in geographic CRS.
    ax
        Target matplotlib axes.
    label
        Legend label.
    """
    land.plot(
        ax=ax,
        facecolor=LAND_FACE_COLOR,
        edgecolor=LAND_EDGE_COLOR,
        alpha=LAND_ALPHA,
        linewidth=0.4,
        label=label,
    )


def plot_ring(
    ring_gdf: gpd.GeoDataFrame, ax: Axes, *, label: str = "buffer ring"
) -> None:
    """
    Plot a buffer-ring boundary as a dashed line.

    Parameters
    ----------
    ring_gdf
        Ring polygon GeoDataFrame.
    ax
        Target matplotlib axes.
    label
        Legend label.
    """
    ring_gdf.boundary.plot(
        ax=ax, color=RING_COLOR, linewidth=RING_LINEWIDTH, linestyle="--", label=label
    )


def plot_base_map(
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_topobathy: bool = True,
    show_aoi: bool = True,
    land: gpd.GeoDataFrame | None = None,
    ring: gpd.GeoDataFrame | None = None,
) -> tuple[Figure, Axes]:
    """
    Draw the standard case-map background.

    Parameters
    ----------
    ax
        Existing axes; created when ``None``.
    title
        Optional map title (defaults to case-aware ``"Case map"``).
    show_topobathy, show_aoi
        Toggle bathymetry raster and AoI outline.
    land, ring
        Optional land and buffer-ring overlays.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes handles.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE) if ax is None else (ax.figure, ax)

    if show_topobathy:
        _topobathy().plot.imshow(
            ax=ax,
            vmin=TOPOBATHY_VMIN,
            vmax=TOPOBATHY_VMAX,
            cmap=TOPOBATHY_CMAP,
            alpha=TOPOBATHY_ALPHA,
        )
    if land is not None:
        plot_land(land, ax)
    if ring is not None:
        plot_ring(ring, ax)
    if show_aoi:
        _aoi().boundary.plot(
            ax=ax, color=AOI_EDGE_COLOR, linewidth=AOI_LINEWIDTH, label="AoI"
        )

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(display_title(title or "Case map"))
    return fig, ax


def plot_points(
    points: gpd.GeoDataFrame,
    ax: Axes,
    *,
    color_col: str = "type",
    label_col: str | None = "instrument_id",
    annotate: bool = True,
    color_map: dict[str, str] | None = None,
) -> None:
    """
    Plot observation or goal points with optional annotations.

    Parameters
    ----------
    points
        Point GeoDataFrame.
    ax
        Target matplotlib axes.
    color_col, label_col
        Columns used for colour grouping and text labels.
    annotate
        Annotate each point when ``True``.
    color_map
        Optional colour lookup by *color_col* value.
    """
    colors = color_map or TYPE_COLORS
    if color_col in points.columns:
        for value, group in points.groupby(color_col):
            group.plot(
                ax=ax,
                color=colors.get(str(value), DEFAULT_POINT_COLOR),
                markersize=POINT_MARKERSIZE,
                edgecolor=POINT_EDGE_COLOR,
                linewidth=POINT_EDGE_WIDTH,
                label=str(value),
            )
    else:
        points.plot(
            ax=ax,
            color=DEFAULT_POINT_COLOR,
            markersize=POINT_MARKERSIZE,
            edgecolor=POINT_EDGE_COLOR,
            linewidth=POINT_EDGE_WIDTH,
        )

    if annotate and label_col and label_col in points.columns:
        for _, row in points.iterrows():
            ax.annotate(
                row[label_col],
                (row.geometry.x, row.geometry.y),
                xytext=ANNOTATE_OFFSET,
                textcoords="offset points",
                fontsize=ANNOTATE_FONTSIZE,
            )
    ax.legend(loc="upper right", fontsize=8)


def plot_sectors(
    targets: gpd.GeoDataFrame,
    goals: dict[str, dict[str, Any]],
    boundary_snaps: dict[str, dict[str, float]],
    ax: Axes,
    *,
    radius_deg: float = 0.35,
) -> None:
    """
    Draw direction wedges at each goal's AoI boundary snap.

    Parameters
    ----------
    targets
        Goal target points with ``goal_id``.
    goals
        Goals dict; ``angles`` is nautical clockwise ``[left, right]`` (see
        :mod:`utils.directions`).
    boundary_snaps
        Hindcast snap lon/lat per goal id.
    ax
        Target matplotlib axes.
    radius_deg
        Wedge radius in degrees.
    """
    for _, row in targets.iterrows():
        key = str(int(row["goal_id"]))
        goal = goals[key]
        left, right = goal["angles"]
        theta1, theta2 = nautical_sector_to_wedge_angles(left, right)
        snap = boundary_snaps[key]
        lon, lat = snap["longitude"], snap["latitude"]
        tgt_lon, tgt_lat = row.geometry.x, row.geometry.y

        ax.plot(
            [tgt_lon, lon],
            [tgt_lat, lat],
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.55,
            zorder=4,
        )
        ax.plot(lon, lat, marker="x", color="black", markersize=5, zorder=5)
        ax.add_patch(
            Wedge(
                (lon, lat),
                radius_deg,
                theta1,
                theta2,
                facecolor="#1f78b4",
                alpha=SECTOR_ALPHA,
                edgecolor="#1f78b4",
                linewidth=0.8,
                zorder=3,
            )
        )


def plot_case_map(
    points: gpd.GeoDataFrame | None = None,
    *,
    title: str | None = None,
    color_col: str = "type",
    label_col: str | None = "instrument_id",
    land: gpd.GeoDataFrame | None = None,
    ring: gpd.GeoDataFrame | None = None,
    color_map: dict[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """
    Full case map with optional point overlay.

    Parameters
    ----------
    points
        Optional points to plot on top of the base map.
    title
        Map title.
    color_col, label_col, color_map
        Passed to :func:`plot_points`.
    land, ring
        Optional land and buffer-ring overlays.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes handles.
    """
    fig, ax = plot_base_map(title=title, land=land, ring=ring)
    if points is not None and not points.empty:
        plot_points(
            points, ax, color_col=color_col, label_col=label_col, color_map=color_map
        )
    fig.tight_layout()
    return fig, ax


def _site_color(source: str, label: str) -> str:
    if source == "isobath":
        return ISOBATH_COLORS.get(str(label), DEFAULT_POINT_COLOR)
    if source == "mesh":
        return MESH_COLORS.get(str(label), "#9467bd")
    if source == "buoy":
        return TYPE_COLORS.get("validation", DEFAULT_POINT_COLOR)
    return DEFAULT_POINT_COLOR


def plot_validation_sites(
    lon: np.ndarray,
    lat: np.ndarray,
    sources: np.ndarray,
    labels: np.ndarray,
    ax: Axes,
    *,
    site_ids: np.ndarray | None = None,
    annotate_buoys: bool = True,
    markersize_scale: float = 1.0,
) -> None:
    """
    Plot validation output sites on an existing map axes.

    Parameters
    ----------
    lon, lat, sources, labels
        Per-site coordinates and metadata arrays.
    ax
        Target matplotlib axes.
    site_ids
        Optional site id array for buoy annotations.
    annotate_buoys
        Label buoy sites when ``True``.
    markersize_scale
        Multiplier on default marker sizes.
    """
    labels = np.asarray(labels, dtype=str)
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    drawn: set[tuple[str, str]] = set()
    for i in range(len(lon)):
        source, label = sources[i], labels[i]
        color = _site_color(source, label)
        marker = SITE_SOURCE_MARKERS.get(source, "o")
        size = POINT_MARKERSIZE * (0.35 if source == "mesh" else 1.0) * markersize_scale
        legend_key = (source, label) if source != "buoy" else ("buoy", "buoy")
        show_label = legend_key not in drawn
        if show_label:
            drawn.add(legend_key)
        legend = (
            label
            if source == "isobath"
            else ("mesh" if source == "mesh" else "buoy")
        )
        ax.scatter(
            lon[i],
            lat[i],
            c=color,
            marker=marker,
            s=size,
            edgecolors=POINT_EDGE_COLOR,
            linewidths=POINT_EDGE_WIDTH,
            label=legend if show_label else None,
            zorder=6 if source == "buoy" else 5,
        )

    if annotate_buoys and site_ids is not None:
        for i, site_id in enumerate(np.asarray(site_ids, dtype=str)):
            if sources[i] != "buoy":
                continue
            name = site_id.removeprefix("buoy_")
            ax.annotate(
                name,
                (lon[i], lat[i]),
                xytext=ANNOTATE_OFFSET,
                textcoords="offset points",
                fontsize=ANNOTATE_FONTSIZE,
                zorder=7,
            )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)


__all__ = [
    "DEFAULT_POINT_COLOR",
    "HINDCAST_COLORS",
    "ISOBATH_COLORS",
    "MESH_COLORS",
    "SITE_SOURCE_MARKERS",
    "plot_base_map",
    "plot_case_map",
    "plot_land",
    "plot_points",
    "plot_ring",
    "plot_sectors",
    "plot_validation_sites",
]
