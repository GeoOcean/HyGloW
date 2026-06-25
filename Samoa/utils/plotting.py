"""Case map plotting — styling here, paths from :mod:`utils.config`."""

from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import rioxarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from utils.config import PATHS, resolve

FIGSIZE = (9, 7)
TOPOBATHY_VMIN, TOPOBATHY_VMAX = -200, 0
TOPOBATHY_CMAP, TOPOBATHY_ALPHA = "terrain", 0.85
AOI_EDGE_COLOR, AOI_LINEWIDTH = "black", 1.2

TYPE_COLORS = {
    "validation": "#d62728",
    "waves": "#1f77b4",
    "spectrum": "#9467bd",
    "calibration": "#ff7f0e",
    "levels": "#2ca02c",
    "ignore": "#7f7f7f",
}
DEFAULT_POINT_COLOR = "#17becf"
POINT_MARKERSIZE, POINT_EDGE_WIDTH = 70, 0.6
POINT_EDGE_COLOR = "white"
ANNOTATE_FONTSIZE, ANNOTATE_OFFSET = 8, (4, 4)


def _topobathy():
    return rioxarray.open_rasterio(resolve(PATHS["topobathy_file"])).squeeze(drop=True)


def _aoi() -> gpd.GeoDataFrame:
    return gpd.read_file(resolve(PATHS["aoi_file"]))


def plot_base_map(
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_topobathy: bool = True,
    show_aoi: bool = True,
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
) -> None:
    if color_col in points.columns:
        for value, group in points.groupby(color_col):
            group.plot(
                ax=ax,
                color=TYPE_COLORS.get(str(value), DEFAULT_POINT_COLOR),
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
    ax.legend(loc="upper right")


def plot_case_map(
    points: gpd.GeoDataFrame | None = None,
    *,
    title: str | None = None,
    color_col: str = "type",
    label_col: str | None = "instrument_id",
) -> tuple[Figure, Axes]:
    fig, ax = plot_base_map(title=title)
    if points is not None and not points.empty:
        plot_points(points, ax, color_col=color_col, label_col=label_col)
    fig.tight_layout()
    return fig, ax
