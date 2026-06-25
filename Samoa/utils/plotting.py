"""Case map plotting — styling here, paths from :mod:`utils.config`."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Wedge
from utils.config import PATHS, resolve

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
POINT_MARKERSIZE, POINT_EDGE_WIDTH = 70, 0.6
POINT_EDGE_COLOR = "white"
ANNOTATE_FONTSIZE, ANNOTATE_OFFSET = 8, (4, 4)
SECTOR_ALPHA = 0.15


def _topobathy():
    return rioxarray.open_rasterio(resolve(PATHS["topobathy_file"])).squeeze(drop=True)


def _aoi() -> gpd.GeoDataFrame:
    return gpd.read_file(resolve(PATHS["aoi_file"]))


def plot_land(land: gpd.GeoDataFrame, ax: Axes, *, label: str = "land") -> None:
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
    if title:
        ax.set_title(title)
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
    """Draw direction wedges at each goal's AoI boundary snap (math deg, CCW from East)."""

    for _, row in targets.iterrows():
        key = str(int(row["goal_id"]))
        goal = goals[key]
        theta1, theta2 = goal["angles"]
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
    fig, ax = plot_base_map(title=title, land=land, ring=ring)
    if points is not None and not points.empty:
        plot_points(
            points, ax, color_col=color_col, label_col=label_col, color_map=color_map
        )
    fig.tight_layout()
    return fig, ax
