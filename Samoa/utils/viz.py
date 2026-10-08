"""Case maps and diagnostics — styling here, paths from :mod:`utils.config`.

- base map of the case (:func:`plot_base_map`: topobathy, isobaths, AoI, mesh,
  satellite — drawn with :class:`bluemath_tk.core.plotting.base_plotting.DefaultStaticPlotting`);
- goals, sectors, observation and validation sites;
- SnapWave pre-run diagnostics (:func:`plot_metamodel_configuration`,
  :func:`plot_dynamic_configuration`);
- finished SnapWave cases (:func:`plot_metamodel_case`, :func:`plot_dynamic_case`):
  :func:`bluemath_tk.wrappers.snapwave.snapwave_plots.plot_case` with this
  module's base map and case titles.

Spectra plots: :mod:`bluemath_tk.core.plotting.spectra`.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from bluemath_tk.core.operations import in_nautical_sector, nautical_sector_to_wedge_angles
from bluemath_tk.core.plotting.base_plotting import (
    DefaultStaticPlotting,
    decimate_raster,
)
from bluemath_tk.waves.hywaves.output_points import smooth_raster
from bluemath_tk.wrappers.snapwave import BOUNDARY_VARS
from bluemath_tk.wrappers.snapwave.snapwave_plots import CasePlotStyle, plot_case
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
from shapely.geometry import MultiPolygon, Polygon

from utils.config import OUTPUT_SITES_CONFIG, PATHS, display_title, resolve

FIGSIZE = (9, 7)
TOPOBATHY_VMIN, TOPOBATHY_VMAX = -200, 0
TOPOBATHY_LAND_VMAX = 300
TOPOBATHY_ALPHA = 0.85
SATELLITE_SOURCE = "arcgis"
ISOBATH_CONTOUR_LEVELS = [-100, -50, -20, -10, -5]
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
    "100m": "#6a3d9a",
}
MESH_COLORS = {
    "Apia": "#9467bd",
}
SITE_SOURCE_MARKERS = {
    "isobath": "o",
    "mesh": ".",
    "buoy": "s",
}
SITE_SOURCE_COLORS = {
    "isobath": "#1f78b4",
    "buoy": "#d62728",
    "mesh": "#9467bd",
}
POINT_MARKERSIZE, POINT_EDGE_WIDTH = 70, 0.6
POINT_EDGE_COLOR = "white"
ANNOTATE_FONTSIZE, ANNOTATE_OFFSET = 8, (4, 4)
SECTOR_ALPHA = 0.15
OUTSIDE_SECTOR_FLAG_PCT = 5.0
DEFAULT_MAX_DIR_MARKS = 200


TOPOBATHY_MAX_PIXELS = 4_000_000  # plenty for any figure size used here


_STATIC = DefaultStaticPlotting()


@lru_cache(maxsize=1)
def _topobathy():
    """Load the case topobathy raster, decimated to a display-sized grid.

    A source GEBCO tiff can be 100M+ pixels (float64) — rendering it at
    native resolution costs several GB per call for a diagnostic figure
    that's a few thousand pixels wide at most (see
    :func:`bluemath_tk.core.plotting.base_plotting.decimate_raster`).
    """
    da = rioxarray.open_rasterio(resolve(PATHS["topobathy_file"])).squeeze(drop=True)
    return decimate_raster(da, TOPOBATHY_MAX_PIXELS).load()


@lru_cache(maxsize=1)
def _aoi() -> gpd.GeoDataFrame:
    """Load the case AoI polygon shapefile (cached)."""
    return gpd.read_file(resolve(PATHS["aoi_file"]))


@lru_cache(maxsize=1)
def _isobath_raster():
    """Topobathy for isobath lines: smoothed like the output sites, if configured.

    Uses the largest ``smooth_m`` of ``simulations.output_sites_config.isobaths``
    so the drawn contours are the ones the isobath sites sit on.
    """
    widths = [s.get("smooth_m") or 0 for s in OUTPUT_SITES_CONFIG.get("isobaths", [])]
    smooth_m = max(widths, default=0)
    if not smooth_m:
        return _topobathy()
    return smooth_raster(_topobathy(), float(smooth_m))


def _isobath_contour_colors() -> list[str]:
    return [ISOBATH_COLORS[f"{abs(level)}m"] for level in ISOBATH_CONTOUR_LEVELS]


def _plot_isobaths(ax: Axes) -> None:
    """Draw labelled isobath contour lines over the topobathy raster.

    Contours of :func:`_isobath_raster` (smoothed like the output sites). Each
    level also gets a legend entry, picked up by whichever ``ax.legend()``
    call the caller makes afterwards.
    """
    _STATIC.plot_isodepths(
        ax,
        _isobath_raster(),
        ISOBATH_CONTOUR_LEVELS,
        legend=True,
        colors=_isobath_contour_colors(),
        linewidths=0.6,
        alpha=0.9,
        zorder=2,
        labels_kwargs={"fmt": lambda v: f"{abs(int(v))}m", "fontsize": 6},
    )


def _satellite_area(pad: float = 0.05) -> tuple[float, float, float, float]:
    """AoI bounding box as ``(lon_min, lon_max, lat_min, lat_max)``, padded."""
    minx, miny, maxx, maxy = _aoi().total_bounds
    return (minx - pad, maxx + pad, miny - pad, maxy + pad)


def _plot_satellite(
    ax: Axes,
    *,
    source: str = SATELLITE_SOURCE,
    area: tuple[float, float, float, float] | None = None,
) -> None:
    """Fetch and draw a satellite/imagery basemap; requires a cartopy GeoAxes.

    ``area`` should match the axes' final displayed extent (not e.g. the
    whole-case AoI) so tiles are fetched at a zoom level with enough detail
    for the actual viewport — it defaults to the AoI bounds, but callers
    that crop the view further (e.g. to an enclosure) should pass their own
    tighter bounds. Tiles are cached per ``(source, area)`` by bluemath.
    """
    if not isinstance(ax, GeoAxes):
        raise TypeError(
            "show_satellite=True requires an axes created with "
            "projection=ccrs.PlateCarree() (pass ax=None to let plot_base_map "
            "create one automatically)"
        )
    area = area if area is not None else _satellite_area()
    _STATIC.plot_satellite(ax, area=area, source=source, zorder=0)


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


def plot_mesh(
    mesh_file: str | Path,
    ax: Axes,
    *,
    color: str = "#fbbf24",
    linewidth: float = 0.3,
    alpha: float = 0.55,
) -> None:
    """
    Draw a UGRID mesh's edges as a thin line collection.

    A case mesh runs into the hundreds of thousands of edges — cheap enough
    for matplotlib to draw once as a static figure (unlike an interactive
    web map, which is where that edge count actually becomes a problem).
    At a whole-domain zoom, individual triangles are far below one pixel —
    every pixel in the mesh's footprint gets touched by some edge, so the
    whole footprint reads as a solid tint rather than a visible texture
    regardless of styling. A bold, saturated tint embraces that rather than
    fighting it: it needs to read clearly as "the mesh covers this area"
    even scaled down to GUI-thumbnail size, which a fainter/thinner style
    (tried first) didn't — it all but disappeared against a busy satellite
    photo at typical viewing size, correct up close but useless at a glance.

    Parameters
    ----------
    mesh_file
        Path to the mesh netCDF (UGRID conventions: ``mesh2d_node_x/y`` in
        degrees, already WGS84 lon/lat — no reprojection needed here; and
        ``mesh2d_edge_nodes`` with a ``start_index`` attribute, typically 1).
    ax
        Target matplotlib axes.
    color, linewidth, alpha
        Line style — bold/saturated so the coverage footprint reads clearly
        at a glance (see above).
    """
    _STATIC.plot_ugrid_mesh(
        ax, resolve(mesh_file), color=color, linewidth=linewidth, alpha=alpha
    )


def plot_base_map(
    *,
    ax: Axes | None = None,
    title: str | None = None,
    show_topobathy: bool | None = None,
    show_isobaths: bool = True,
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
    satellite_area: tuple[float, float, float, float] | None = None,
    show_aoi: bool = True,
    land: gpd.GeoDataFrame | None = None,
    ring: gpd.GeoDataFrame | None = None,
) -> tuple[Figure, Axes]:
    """
    Draw the standard case-map background.

    Parameters
    ----------
    ax
        Existing axes; created when ``None``. Must already be a cartopy
        ``GeoAxes`` (``projection=ccrs.PlateCarree()``) if the caller passes
        ``ax`` and wants ``show_satellite=True``.
    title
        Optional map title (defaults to case-aware ``"Case map"``).
    show_topobathy
        Draw the topobathy raster. Defaults to ``not show_satellite`` — i.e.
        bathymetry is the background unless a satellite image is requested,
        in which case the satellite image is the background on its own. Pass
        ``True`` explicitly together with ``show_satellite=True`` to combine
        both: land is left transparent so the satellite photo shows through,
        and only the ocean is coloured by depth.
    show_isobaths
        Draw labelled isobath contour lines (with a legend entry per level,
        picked up by any ``ax.legend()`` call afterwards). Independent of
        ``show_topobathy`` — the lines read fine over a satellite image too.
    show_satellite, satellite_source, satellite_area
        Draw a satellite/aerial imagery basemap, fetched via
        ``bluemath_tk``. ``satellite_source`` is one of ``"arcgis"``,
        ``"eox"``, ``"osm"``, etc. — see
        :func:`bluemath_tk.core.plotting.satellite.get_satellite_image`.
        ``satellite_area`` overrides the fetch bounds (default: the AoI);
        pass a tighter box matching the axes' final displayed extent when
        the caller crops the view further, so tiles are fetched at a
        matching zoom level.
    show_aoi
        Toggle the AoI outline.
    land, ring
        Optional land and buffer-ring overlays.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes handles.
    """
    if show_topobathy is None:
        show_topobathy = not show_satellite

    if ax is None:
        subplot_kw = {"projection": ccrs.PlateCarree()} if show_satellite else {}
        fig, ax = plt.subplots(figsize=FIGSIZE, subplot_kw=subplot_kw)
    else:
        fig = ax.figure

    if show_satellite:
        _plot_satellite(ax, source=satellite_source, area=satellite_area)
    if show_topobathy:
        # With a satellite photo underneath, only the ocean is coloured
        # (ocean_only), rather than washing the whole raster (land included)
        # semi-transparently over the photo.
        _STATIC.plot_bathymetry(
            ax,
            _topobathy(),
            area=None,
            value_range=(TOPOBATHY_VMIN, TOPOBATHY_LAND_VMAX),
            ocean_only=show_satellite,
            method="imshow",
            alpha=TOPOBATHY_ALPHA,
            add_colorbar=not show_satellite,
        )
    if show_isobaths:
        # Independent of show_topobathy: thin contour lines read fine as a
        # depth-context overlay on a satellite image too, unlike the filled
        # raster (which would just wash out the photo underneath).
        _plot_isobaths(ax)
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
        :mod:`bluemath_tk.core.operations`).
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
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
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
    show_satellite, satellite_source
        Passed to :func:`plot_base_map`.

    Returns
    -------
    tuple[Figure, Axes]
        Figure and axes handles.
    """
    fig, ax = plot_base_map(
        title=title,
        land=land,
        ring=ring,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
    )
    if points is not None and not points.empty:
        plot_points(
            points, ax, color_col=color_col, label_col=label_col, color_map=color_map
        )
    fig.tight_layout()
    return fig, ax


def _site_color(source: str, label: str) -> str:
    """Colour for one output site, keyed by its source type and label."""
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
    sources = np.asarray(sources, dtype=str)

    # Color/marker/size are a pure function of (source, label), so one dense
    # mesh polygon (thousands of sites, all same label) collapses to a single
    # scatter() call instead of one call per point — each ax.scatter() call
    # triggers a full O(n) axes autoscale, so per-point calls were O(n^2).
    drawn: set[tuple[str, str]] = set()
    for source, label in sorted(set(zip(sources, labels, strict=True))):
        mask = (sources == source) & (labels == label)
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
            lon[mask],
            lat[mask],
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


def _goal_colors(goal_ids: list) -> dict:
    """Perceptually-uniform colour per goal, ordered along the AoI boundary.

    Goals are numbered sequentially around the domain, so colour doubles as a
    position cue. A qualitative palette (e.g. tab10) repeats hues past 10
    goals and breaks that mapping; a continuous colormap sampled by rank
    never repeats, at any goal count.
    """
    n = max(len(goal_ids), 1)
    cmap = plt.get_cmap("viridis")
    positions = np.linspace(0.08, 0.92, n) if n > 1 else [0.5]
    return {gid: cmap(pos) for gid, pos in zip(goal_ids, positions, strict=True)}


def _wave_from_arrowprops(
    lon: float,
    lat: float,
    dir_deg: float,
    *,
    scale_deg: float,
    color: str,
    lw: float = 0.9,
    alpha: float = 0.75,
):
    """Return arrow props and endpoints for a wave-from direction arrow."""
    theta = np.deg2rad(float(dir_deg))
    dlon = np.sin(theta) * scale_deg
    dlat = np.cos(theta) * scale_deg
    return dict(
        arrowstyle="-|>",
        color=color,
        lw=lw,
        alpha=alpha,
        shrinkA=0,
        shrinkB=2,
        mutation_scale=8,
    ), (lon + dlon, lat + dlat), (lon, lat)


def _sector_spans(left: float, right: float) -> list[tuple[float, float]]:
    """Return one or two (start, end) spans for a nautical sector on [0, 360)."""
    left_n = float(np.mod(left, 360.0))
    right_n = float(np.mod(right, 360.0))
    if left < 0 or left_n > right_n:
        return [(left_n, 360.0), (0.0, right_n)]
    return [(left_n, right_n)]


def _circular_intervals(dirs: np.ndarray) -> list[tuple[float, float]]:
    """One or two (start, end) spans on [0, 360) tightly bounding *dirs*.

    Plain ``[min(dirs), max(dirs)]`` is wrong whenever the data wraps
    through 0 (a goal sector like ``[-5, 100]`` puts real samples near both
    355-360 and 0-100, nothing in between) — it spans the whole empty
    middle instead of jumping the gap. This finds the single largest gap
    between consecutive sorted values (including the wraparound gap) and
    treats everything else as the populated arc, splitting into two spans
    if that arc crosses 0 — mirroring how :func:`_sector_spans` splits the
    *allowed* sector itself.
    """
    vals = np.sort(np.mod(np.asarray(dirs, dtype=float), 360.0))
    if vals.size <= 1:
        v = float(vals[0]) if vals.size else 0.0
        return [(v, v)]

    gaps = np.diff(vals)
    wrap_gap = vals[0] + 360.0 - vals[-1]
    largest_idx = int(np.argmax(np.append(gaps, wrap_gap)))

    if largest_idx == len(gaps):
        # Largest gap is the wraparound gap itself: data doesn't wrap.
        return [(float(vals[0]), float(vals[-1]))]

    start, end = float(vals[largest_idx + 1]), float(vals[largest_idx])
    if start <= end:
        return [(start, end)]
    return [(start, 360.0), (0.0, end)]


def _offshore_bulk_long(offshore: xr.Dataset) -> pd.DataFrame:
    """Melt ``(time, goal)`` offshore bulk forcing to one row per timestep."""
    if not isinstance(offshore, xr.Dataset):
        raise TypeError(f"offshore must be xarray.Dataset, got {type(offshore)!r}")
    missing = [v for v in BOUNDARY_VARS if v not in offshore]
    if missing:
        raise KeyError(f"offshore missing variables: {missing}")
    if "goal" not in offshore.dims:
        raise KeyError(f"offshore must have a 'goal' dimension; got {dict(offshore.sizes)}")

    df = offshore[list(BOUNDARY_VARS)].to_dataframe().reset_index()
    df["goal"] = df["goal"].astype(int)
    return df


def _frac_dirs_outside_sector(
    dirs: np.ndarray, left: float, right: float
) -> float:
    """Share of directions outside a nautical propagation sector."""
    inside = in_nautical_sector(dirs, left, right)
    return float(1.0 - np.mean(inside))


def _dense_dir_mode(cases: pd.DataFrame, max_dir_marks: int) -> bool:
    """Whether any goal has more samples than fit as individual tick marks.

    Shared by :func:`_plot_goal_dir_range_bars` (to pick the render style)
    and :func:`plot_dynamic_configuration` (to match the info-panel legend
    to whichever style actually got drawn).
    """
    return bool((cases.groupby("goal").size() > max_dir_marks).any())


def _direction_density_strip(
    ax: Axes,
    y: float,
    dirs: np.ndarray,
    color: Any,
    bracket_spans: list[tuple[float, float]],
    *,
    nbins: int = 120,
    height: float = 0.62,
) -> None:
    """Background heat-strip: one hue, alpha-encoded circular density of *dirs*.

    Sequential-by-alpha keeps the row tied to the goal's own colour (matches
    the sector/range bars) while showing the actual shape of the direction
    distribution — far more legible than hundreds of overlapping tick marks
    once a goal has more than a couple hundred samples (e.g. hourly
    reanalysis over a year). The fill always covers the full 0-360 range —
    density can and does spill past *bracket_spans* (e.g. hindcast events
    outside the goal's configured sector); that spillover is the point, not
    a rendering bug.
    """
    counts, _ = np.histogram(np.mod(dirs, 360.0), bins=nbins, range=(0, 360))
    peak = counts.max()
    density = counts / peak if peak else counts.astype(float)
    rgba = np.zeros((1, nbins, 4))
    rgba[0, :, :3] = mcolors.to_rgb(color)
    rgba[0, :, 3] = np.clip(density, 0.0, 1.0) * 0.85
    ax.imshow(
        rgba,
        extent=(0, 360, y - height / 2, y + height / 2),
        aspect="auto",
        # "nearest" keeps each bin a sharp block; "bilinear" softens bin
        # edges into a gradient, which reads as fuzzy/hard-to-bound —
        # exactly what makes the interval hard to pin down at a glance.
        interpolation="nearest",
        zorder=2,
    )

    # The density fill alone doesn't give a crisp edge — near-uniform
    # coverage (e.g. an LHS design deliberately filling its sector) looks
    # like a smear with no obvious boundary, and a same-hue outline just
    # blends into the fill it's supposed to delimit. A darker shade of the
    # same colour outlines *bracket_spans* with real contrast, whatever
    # shape the fill takes.
    edge_color = tuple(c * 0.55 for c in mcolors.to_rgb(color))
    for start, end in bracket_spans:
        ax.barh(
            y,
            end - start,
            left=start,
            height=height,
            fill=False,
            edgecolor=edge_color,
            linewidth=1.4,
            zorder=3,
        )


def _plot_goal_dir_range_bars(
    ax: Axes,
    cases: pd.DataFrame,
    goals_config: dict[str, dict[str, Any]],
    colors: dict[int, Any],
    *,
    title: str = "Sampled direction ranges per goal",
    max_dir_marks: int = DEFAULT_MAX_DIR_MARKS,
    annotate_tp: bool = True,
    bracket_source: str = "observed",
) -> None:
    """Horizontal bars: propagation sector (light) and observed dir range.

    Up to ``max_dir_marks`` samples per goal (e.g. a modest LHS design),
    every observed direction is drawn as its own tick mark plus a solid
    range bar — a true design-of-experiment view. Past that count, a
    random subsample of ticks would misrepresent the population it was
    thinned from, so the row switches to a density strip instead (see
    :func:`_direction_density_strip`), which stays honest at any sample
    size.

    ``bracket_source`` picks what the crisp range bar/outline bounds:

    - ``"observed"`` (LHS/metamodel) — the actual sampled interval via
      :func:`_circular_intervals`, verifying the design fills its sector.
    - ``"sector"`` (dynamic/reanalysis) — the goal's *configured* sector
      via :func:`_sector_spans`, a fixed reference; real hindcast density
      is free to (and often does) spill outside it, which is exactly the
      "% outside sector" the info panel already reports — the crisp box
      makes that spillover visible instead of chasing the data's own
      extent, which for scattered reanalysis directions isn't a
      meaningful interval to bound in the first place.
    """
    goal_ids = sorted(cases["goal"].unique())
    y_positions = np.arange(len(goal_ids))
    dense_mode = _dense_dir_mode(cases, max_dir_marks)

    for y, goal_id in enumerate(goal_ids):
        goal_key = str(goal_id)
        left, right = goals_config[goal_key]["angles"]
        color = colors[goal_id]
        sub = cases[cases["goal"] == goal_id]
        dirs = sub["dir"].to_numpy(float)

        for x0, x1 in _sector_spans(left, right):
            ax.barh(
                y,
                x1 - x0,
                left=x0,
                height=0.55,
                color=color,
                alpha=0.18,
                edgecolor=color,
                linewidth=0.6,
            )

        bracket_spans = (
            _sector_spans(left, right)
            if bracket_source == "sector"
            else _circular_intervals(dirs)
        )

        if dense_mode:
            _direction_density_strip(ax, y, dirs, color, bracket_spans)
        else:
            for start, end in bracket_spans:
                ax.barh(
                    y,
                    end - start,
                    left=start,
                    height=0.28,
                    color=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                )
            # dense_mode is False only when every goal's count already fits
            # under max_dir_marks, so dirs never needs subsampling here.
            ax.scatter(
                dirs,
                np.full(len(dirs), y),
                marker="|",
                s=90,
                color=color,
                linewidths=1.2,
                zorder=4,
            )
        if annotate_tp:
            ax.text(
                362,
                y,
                f"tp {sub['tp'].min():.1f}-{sub['tp'].max():.1f}s",
                va="center",
                ha="left",
                fontsize=8,
                color="#333333",
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"G{gid}" for gid in goal_ids])
    ax.set_xlim(0, 395)
    ax.set_xticks(np.arange(0, 361, 45))
    ax.set_xlabel("wave-from direction (° nautical)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()


def _configuration_figure(
    cases: pd.DataFrame,
    *,
    show_satellite: bool = False,
) -> tuple[Figure, Axes, Axes, Axes, list[int], dict[int, Any]]:
    """Shared figure/gridspec skeleton for the config diagnostics: map, bar panel, info panel.

    The bar panel's height grows with the number of goals so each row keeps
    a constant height — a fixed panel height would cram rows and their
    labels together on cases with many goals (e.g. a long coastline with
    25+ goals) while leaving cases with few goals sparse.
    """
    goal_ids = sorted(cases["goal"].unique())
    colors = _goal_colors(goal_ids)

    map_h = 6.5
    bars_h = max(2.2, len(goal_ids) * 0.22 + 0.9)
    fig = plt.figure(figsize=(15, map_h + bars_h), constrained_layout=True)
    gs = GridSpec(
        2, 2, figure=fig, width_ratios=[2.1, 1.0], height_ratios=[map_h, bars_h]
    )
    map_kw = {"projection": ccrs.PlateCarree()} if show_satellite else {}
    ax_map = fig.add_subplot(gs[0, 0], **map_kw)
    ax_bars = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis("off")
    return fig, ax_map, ax_bars, ax_info, goal_ids, colors


def _boxes_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + margin < bx0 or bx1 + margin < ax0 or ay1 + margin < by0 or by1 + margin < ay0
    )


def _label_candidate_ring(radius: float) -> list[tuple[tuple[float, float], str, str]]:
    """8 compass-direction offsets at a fixed point radius, with matching ha/va."""
    dirs = [
        (1, 1, "left", "bottom"),
        (1, -1, "left", "top"),
        (-1, 1, "right", "bottom"),
        (-1, -1, "right", "top"),
        (0, 1, "center", "bottom"),
        (0, -1, "center", "top"),
        (1, 0, "left", "center"),
        (-1, 0, "right", "center"),
    ]
    out = []
    for ux, uy, ha, va in dirs:
        norm = (ux**2 + uy**2) ** 0.5 or 1
        out.append(((radius * ux / norm, radius * uy / norm), ha, va))
    return out


# Candidate (offset points, ha, va) tried near-to-far for each goal label, as
# rings of 8 compass directions at growing radius. Kept dense at short range
# (few goals never need more than the first ring) and reaching out to ~55pt
# so a tight cluster of goals (e.g. several sectors sharing one headland)
# still finds a free spot instead of stacking labels on top of each other.
_LABEL_CANDIDATES: list[tuple[tuple[float, float], str, str]] = [
    c for radius in (6, 11, 16, 22, 28, 34, 42, 55) for c in _label_candidate_ring(radius)
]


def _place_goal_label(
    ax: Axes,
    renderer,
    lon: float,
    lat: float,
    text: str,
    *,
    fontsize: float,
    color: str,
    halo: list,
    placed_boxes: list[tuple[float, float, float, float]],
) -> None:
    """Annotate a goal label at the first candidate slot that clears earlier labels.

    Measures each candidate's real rendered bbox (via ``renderer``) rather
    than a guessed text size, so it works for any font/DPI. On a long,
    tightly packed coastline every close slot can be taken before all goals
    are placed — the last-tried candidate is kept as a guaranteed fallback,
    with a thin leader line back to its point so the larger offset stays
    unambiguous.
    """
    inv = ax.transData.inverted()
    # Fixed padding (not a fraction of label width) so short labels like
    # "G15"/"G17" still get real breathing room next to each other.
    dpi = ax.figure.dpi
    px0, py0 = inv.transform((0, 0))
    px1, _ = inv.transform((3.5 * dpi / 72.0, 0))
    pad = abs(px1 - px0)
    chosen = _LABEL_CANDIDATES[0]
    chosen_box = None
    for candidate in _LABEL_CANDIDATES:
        (dx_pt, dy_pt), ha, va = candidate
        probe = ax.annotate(
            text,
            (lon, lat),
            xytext=(dx_pt, dy_pt),
            textcoords="offset points",
            fontsize=fontsize,
            ha=ha,
            va=va,
        )
        bbox = probe.get_window_extent(renderer)
        x0, y0 = inv.transform((bbox.x0, bbox.y0))
        x1, y1 = inv.transform((bbox.x1, bbox.y1))
        probe.remove()
        box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        chosen, chosen_box = candidate, box
        if not any(_boxes_overlap(box, other, margin=pad) for other in placed_boxes):
            break
    placed_boxes.append(chosen_box)

    (dx_pt, dy_pt), ha, va = chosen
    leader = (dx_pt**2 + dy_pt**2) ** 0.5 > 9
    ax.annotate(
        text,
        (lon, lat),
        xytext=(dx_pt, dy_pt),
        textcoords="offset points",
        fontsize=fontsize,
        color="black",
        weight="bold",
        ha=ha,
        va=va,
        zorder=7,
        path_effects=halo,
        arrowprops=(
            dict(arrowstyle="-", color=color, lw=0.7, alpha=0.75, shrinkA=0, shrinkB=3)
            if leader
            else None
        ),
    )


def _plot_configuration_map(
    ax_map: Axes,
    *,
    enclosure_polygon: Polygon | MultiPolygon,
    goals_config: dict[str, dict[str, Any]],
    cases: pd.DataFrame,
    colors: dict[int, Any],
    output_sites_df: pd.DataFrame,
    sector_radius_deg: float,
    arrow_scale_deg: float,
    max_arrows_per_goal: int,
    show_topobathy: bool | None,
    title: str,
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
) -> None:
    """Shared base map + enclosure + goal wedges/arrows + output sites for config diagnostics."""
    # Axis limits end up fixed by the enclosure alone (below), so the
    # satellite fetch — if requested — uses that same tight box rather than
    # the whole-case AoI, or tiles come in at a zoom level too coarse for
    # this close-up view.
    enclosure_gdf = gpd.GeoDataFrame(geometry=[enclosure_polygon], crs="EPSG:4326")
    minx, miny, maxx, maxy = enclosure_gdf.total_bounds
    pad = 0.08
    satellite_area = (minx - pad, maxx + pad, miny - pad, maxy + pad)

    plot_base_map(
        ax=ax_map,
        title=title,
        show_topobathy=show_topobathy,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
        satellite_area=satellite_area,
    )
    enclosure_gdf.boundary.plot(
        ax=ax_map,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label="enclosure",
    )

    # Axis limits are fixed by the enclosure alone, so they can be set (and
    # locked in with a draw) before the goal loop — label placement below
    # needs the real data<->pixel scale to measure text boxes accurately.
    ax_map.set_xlim(minx - pad, maxx + pad)
    ax_map.set_ylim(miny - pad, maxy + pad)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.figure.canvas.draw()
    renderer = ax_map.figure.canvas.get_renderer()
    placed_label_boxes: list[tuple[float, float, float, float]] = []

    # Many goals along a short/folded coastline sit close together — shrink
    # markers/labels so a cluster of nearby goals (e.g. several sectors
    # sharing one corner) has room to fan out instead of stacking up.
    n_goals = len(goals_config)
    crowded = n_goals > 15
    star_ms = 9 if crowded else 12
    label_fontsize = 7.5 if crowded else 9
    label_halo = [pe.withStroke(linewidth=2.2, foreground="white")]

    for goal_key in sorted(goals_config, key=int):
        vals = goals_config[goal_key]
        lon, lat = vals["lon"], vals["lat"]
        left, right = vals["angles"]
        color = colors.get(int(goal_key), DEFAULT_POINT_COLOR)
        theta1, theta2 = nautical_sector_to_wedge_angles(left, right)
        ax_map.add_patch(
            Wedge(
                (lon, lat),
                sector_radius_deg,
                theta1,
                theta2,
                facecolor=color,
                edgecolor=color,
                alpha=SECTOR_ALPHA,
                linewidth=0.8,
                zorder=4,
            )
        )
        ax_map.plot(lon, lat, marker="*", color=color, markersize=star_ms, zorder=6)
        _place_goal_label(
            ax_map,
            renderer,
            lon,
            lat,
            f"G{goal_key}",
            fontsize=label_fontsize,
            color=color,
            halo=label_halo,
            placed_boxes=placed_label_boxes,
        )

        sub = cases[cases["goal"] == int(goal_key)]
        if len(sub) > max_arrows_per_goal:
            sub = sub.sample(max_arrows_per_goal, random_state=0)
        for _, row in sub.iterrows():
            props, xytext, xy = _wave_from_arrowprops(
                lon,
                lat,
                row["dir"],
                scale_deg=arrow_scale_deg,
                color=color,
                alpha=0.55,
            )
            ax_map.annotate("", xy=xy, xytext=xytext, arrowprops=props, zorder=5)

    if not output_sites_df.empty:
        sites_gdf = gpd.GeoDataFrame(
            output_sites_df,
            geometry=gpd.points_from_xy(
                output_sites_df["lon"], output_sites_df["lat"]
            ),
            crs="EPSG:4326",
        )
        for source, group in sites_gdf.groupby("source"):
            marker = SITE_SOURCE_MARKERS.get(str(source), "o")
            group.plot(
                ax=ax_map,
                marker=marker,
                color=SITE_SOURCE_COLORS.get(str(source), DEFAULT_POINT_COLOR),
                markersize=18 if source == "buoy" else 10,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.85,
                label=f"sites: {source}",
                zorder=3,
            )

    # "best" avoids the plotted AoI/enclosure/site artists; goal labels are
    # drawn afterwards on this same axes and aren't considered, but this is
    # still strictly better than a fixed corner on domains where the goal
    # chain runs through upper-right.
    ax_map.legend(loc="best", fontsize=7, framealpha=0.9)


def _configuration_map_figure(
    *,
    enclosure_polygon: Polygon | MultiPolygon,
    goals_config: dict[str, dict[str, Any]],
    cases: pd.DataFrame,
    output_sites_df: pd.DataFrame,
    sector_radius_deg: float,
    arrow_scale_deg: float,
    max_arrows_per_goal: int,
    show_topobathy: bool | None,
    title: str,
    figsize: tuple[float, float],
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
) -> Figure:
    """Standalone, larger map panel only — no bar/info panels competing for space.

    Reuses :func:`_plot_configuration_map` as-is, just on its own bigger
    axes, so goal labels and the domain layout are actually inspectable on
    a case with many goals instead of being squeezed into the left column
    of the combined diagnostic.
    """
    colors = _goal_colors(sorted(cases["goal"].unique()))
    subplot_kw = {"projection": ccrs.PlateCarree()} if show_satellite else {}
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
    _plot_configuration_map(
        ax,
        enclosure_polygon=enclosure_polygon,
        goals_config=goals_config,
        cases=cases,
        colors=colors,
        output_sites_df=output_sites_df,
        sector_radius_deg=sector_radius_deg,
        arrow_scale_deg=arrow_scale_deg,
        max_arrows_per_goal=max_arrows_per_goal,
        show_topobathy=show_topobathy,
        title=title,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
    )
    fig.tight_layout()
    return fig


def plot_metamodel_configuration_map(
    *,
    enclosure_polygon: Polygon | MultiPolygon,
    goals_config: dict[str, dict[str, Any]],
    cases_df: pd.DataFrame,
    output_sites_df: pd.DataFrame,
    sector_radius_deg: float = 0.35,
    arrow_scale_deg: float = 0.15,
    max_arrows_per_goal: int = 25,
    show_topobathy: bool | None = None,
    title: str = "Metamodel configuration map",
    figsize: tuple[float, float] = (12, 11),
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
) -> Figure:
    """Bigger, standalone map-only companion to :func:`plot_metamodel_configuration`."""
    cases = cases_df.copy()
    cases["goal"] = cases["goal"].astype(int)
    return _configuration_map_figure(
        enclosure_polygon=enclosure_polygon,
        goals_config=goals_config,
        cases=cases,
        output_sites_df=output_sites_df,
        sector_radius_deg=sector_radius_deg,
        arrow_scale_deg=arrow_scale_deg,
        max_arrows_per_goal=max_arrows_per_goal,
        show_topobathy=show_topobathy,
        title=title,
        figsize=figsize,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
    )


def plot_dynamic_configuration_map(
    *,
    enclosure_polygon: Polygon | MultiPolygon,
    goals_config: dict[str, dict[str, Any]],
    offshore: xr.Dataset,
    output_sites_df: pd.DataFrame,
    sector_radius_deg: float = 0.35,
    arrow_scale_deg: float = 0.15,
    max_arrows_per_goal: int = 25,
    show_topobathy: bool | None = None,
    title: str = "Dynamic configuration map",
    figsize: tuple[float, float] = (12, 11),
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
) -> Figure:
    """Bigger, standalone map-only companion to :func:`plot_dynamic_configuration`."""
    cases = _offshore_bulk_long(offshore)
    return _configuration_map_figure(
        enclosure_polygon=enclosure_polygon,
        goals_config=goals_config,
        cases=cases,
        output_sites_df=output_sites_df,
        sector_radius_deg=sector_radius_deg,
        arrow_scale_deg=arrow_scale_deg,
        max_arrows_per_goal=max_arrows_per_goal,
        show_topobathy=show_topobathy,
        title=title,
        figsize=figsize,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
    )


def plot_metamodel_configuration(
    *,
    aoi_gdf: gpd.GeoDataFrame,
    enclosure_polygon: Polygon | MultiPolygon,
    goals_config: dict[str, dict[str, Any]],
    cases_df: pd.DataFrame,
    output_sites_df: pd.DataFrame,
    bounds: dict[str, list[float]] | None = None,
    num_samples_per_goal: int | None = None,
    mesh_file: str | None = None,
    enclosure_buffer_m: float | None = None,
    sector_radius_deg: float = 0.35,
    arrow_scale_deg: float = 0.15,
    max_arrows_per_goal: int = 25,
    show_topobathy: bool | None = None,
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
    title: str = "Metamodel pre-run diagnostic",
) -> Figure:
    """
    Multi-panel diagnostic before building or running metamodel SnapWave cases.

    Layout: map and direction-range bars on the left; text summary on the right.

    Parameters
    ----------
    aoi_gdf, enclosure_polygon, goals_config, output_sites_df
        Case geometry and goal configuration (see
        :class:`utils.simulation.SnapWaveCaseSetup`).
    cases_df
        LHS cases table with ``goal``, ``tp``, ``dir``, ``spr``, ``wl`` columns.
    bounds, num_samples_per_goal, mesh_file, enclosure_buffer_m
        Optional metadata surfaced in the text summary panel only.
    sector_radius_deg, arrow_scale_deg, max_arrows_per_goal
        Map styling controls for sectors and direction arrows.
    show_topobathy
        Draw bathymetry under the map panel.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    KeyError
        If *cases_df* is missing a required column.
    """
    required = {"goal", "tp", "dir", "spr", "wl"}
    missing = required - set(cases_df.columns)
    if missing:
        raise KeyError(f"cases_df missing columns: {sorted(missing)}")

    cases = cases_df.copy()
    cases["goal"] = cases["goal"].astype(int)

    fig, ax_map, ax_bars, ax_info, goal_ids, colors = _configuration_figure(
        cases, show_satellite=show_satellite
    )
    _plot_configuration_map(
        ax_map,
        enclosure_polygon=enclosure_polygon,
        goals_config=goals_config,
        cases=cases,
        colors=colors,
        output_sites_df=output_sites_df,
        sector_radius_deg=sector_radius_deg,
        arrow_scale_deg=arrow_scale_deg,
        max_arrows_per_goal=max_arrows_per_goal,
        show_topobathy=show_topobathy,
        title=title,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
    )
    _plot_goal_dir_range_bars(ax_bars, cases, goals_config, colors)
    dense_mode = _dense_dir_mode(cases, DEFAULT_MAX_DIR_MARKS)

    n_cases = len(cases)
    n_goals = len(goal_ids)
    lines = [
        display_title("Metamodel summary"),
        "",
        f"Goals            : {n_goals}",
        f"Total cases      : {n_cases}",
        f"Samples / goal   : {num_samples_per_goal or (n_cases // n_goals if n_goals else 0)}",
        f"Output sites     : {len(output_sites_df)}",
    ]
    if enclosure_buffer_m is not None:
        lines.append(f"Enclosure buffer : {enclosure_buffer_m / 1000:g} km")
    if mesh_file:
        lines.append(f"Mesh             : {Path(mesh_file).name}")
    if bounds:
        lines.extend(
            [
                "",
                "Global bounds (dir from goal sectors):",
                f"  tp  : {bounds['tp']}",
                f"  spr : {bounds['spr']}",
                f"  wl  : {bounds['wl']}",
            ]
        )
    lines.extend(["", "Goal sectors [left, right] (nautical):"])
    for goal_key in sorted(goals_config, key=int):
        ang = goals_config[goal_key]["angles"]
        n_goal_cases = int((cases["goal"] == int(goal_key)).sum())
        lines.append(f"  G{goal_key}: {ang}  ({n_goal_cases} cases)")
    lines.extend(
        [
            "",
            "Map legend:",
            "  shaded wedge = propagation sector",
            "  arrows = sampled wave-from directions",
            "  dashed red = SnapWave enclosure",
            "",
            "Bar panel:",
            "  light bar = allowed sector",
        ]
        + (
            [
                "  shaded strip = density of sampled directions",
                "                 (darker = more common)",
                "  crisp outline = actual sampled interval",
            ]
            if dense_mode
            else [
                "  solid bar = sampled dir range",
                "  ticks = individual LHS dirs",
            ]
        )
    )
    ax_info.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        transform=ax_info.transAxes,
    )

    return fig


def plot_dynamic_configuration(
    *,
    aoi_gdf: gpd.GeoDataFrame,
    enclosure_polygon: Polygon | MultiPolygon,
    goals_config: dict[str, dict[str, Any]],
    offshore: xr.Dataset,
    output_sites_df: pd.DataFrame,
    offshore_name: str | None = None,
    mesh_file: str | None = None,
    enclosure_buffer_m: float | None = None,
    sector_radius_deg: float = 0.35,
    arrow_scale_deg: float = 0.15,
    max_arrows_per_goal: int = 25,
    max_dir_marks: int = DEFAULT_MAX_DIR_MARKS,
    show_topobathy: bool | None = None,
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
    maps_to_save: int | None = None,
    title: str = "Dynamic pre-run diagnostic",
) -> Figure:
    """
    Multi-panel diagnostic before building or running dynamic SnapWave cases.

    Shows propagation sectors from the goals JSON, observed bulk forcing ranges
    from offshore reanalysis (``hs``, ``tp``, ``dir``, ``spr``), and highlights
    when wave-from directions fall outside the nominal goal sectors.

    Layout: map and direction-range bars on the left; text summary on the right.

    Parameters
    ----------
    aoi_gdf, enclosure_polygon, goals_config, output_sites_df
        Case geometry and goal configuration (see
        :class:`utils.simulation.SnapWaveCaseSetup`).
    offshore
        Processed offshore dataset with dims ``(time, goal)`` and bulk
        variables ``hs``, ``tp``, ``dir``, ``spr`` (see
        :func:`bluemath_tk.wrappers.snapwave.boundary_time_series`).
    offshore_name, mesh_file, enclosure_buffer_m, maps_to_save
        Optional metadata surfaced in the text summary panel only.
    sector_radius_deg, arrow_scale_deg, max_arrows_per_goal
        Map styling controls for sectors and direction arrows.
    max_dir_marks
        Per-goal sample count above which the direction-range bars switch
        from individual ticks to a density strip (see
        :func:`_dense_dir_mode`).
    show_topobathy
        Draw bathymetry under the map panel.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cases = _offshore_bulk_long(offshore)
    n_time = int(offshore.sizes.get("time", len(cases) // max(cases["goal"].nunique(), 1)))

    fig, ax_map, ax_bars, ax_info, goal_ids, colors = _configuration_figure(
        cases, show_satellite=show_satellite
    )
    _plot_configuration_map(
        ax_map,
        enclosure_polygon=enclosure_polygon,
        goals_config=goals_config,
        cases=cases,
        colors=colors,
        output_sites_df=output_sites_df,
        sector_radius_deg=sector_radius_deg,
        arrow_scale_deg=arrow_scale_deg,
        max_arrows_per_goal=max_arrows_per_goal,
        show_topobathy=show_topobathy,
        title=title,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
    )
    _plot_goal_dir_range_bars(
        ax_bars,
        cases,
        goals_config,
        colors,
        title="Observed direction ranges per goal (reanalysis)",
        max_dir_marks=max_dir_marks,
        bracket_source="sector",
    )
    dense_mode = _dense_dir_mode(cases, max_dir_marks)

    time_vals = pd.to_datetime(offshore["time"].values) if "time" in offshore else None
    lines = [
        display_title("Dynamic summary"),
        "",
        f"Offshore source   : {offshore_name or 'processed offshore waves'}",
        f"Goals per case    : {len(goal_ids)}",
        f"Timesteps         : {n_time}",
        f"Total cases       : {n_time}",
        f"Output sites      : {len(output_sites_df)}",
    ]
    if time_vals is not None and len(time_vals):
        lines.append(
            f"Period            : {time_vals[0]:%Y-%m-%d %H:%M} → "
            f"{time_vals[-1]:%Y-%m-%d %H:%M}"
        )
    if enclosure_buffer_m is not None:
        lines.append(f"Enclosure buffer  : {enclosure_buffer_m / 1000:g} km")
    if mesh_file:
        lines.append(f"Mesh              : {Path(mesh_file).name}")
    if maps_to_save is not None:
        lines.append(f"Maps to save      : {maps_to_save} / {n_time}")

    lines.extend(
        [
            "",
            "Global bulk ranges (all goals, all timesteps):",
            f"  hs  : [{cases['hs'].min():.3f}, {cases['hs'].max():.3f}] m",
            f"  tp  : [{cases['tp'].min():.2f}, {cases['tp'].max():.2f}] s",
            f"  dir : [{cases['dir'].min():.1f}, {cases['dir'].max():.1f}]°",
            f"  spr : [{cases['spr'].min():.1f}, {cases['spr'].max():.1f}]°",
            "",
            "Per goal — sector vs observed dir (nautical):",
        ]
    )
    for goal_key in sorted(goals_config, key=int):
        ang = goals_config[goal_key]["angles"]
        sub = cases[cases["goal"] == int(goal_key)]
        dirs = sub["dir"].to_numpy(float)
        outside = _frac_dirs_outside_sector(dirs, ang[0], ang[1]) * 100.0
        # Real records almost always have a sliver of directions outside the
        # nominal sector (a boundary snap, a rare cross-swell); flagging at
        # >0% would mark every goal and make the flag noise. Reserve it for
        # sectors that miss a meaningful share of observed directions.
        flag = "  ! outside sector" if outside > OUTSIDE_SECTOR_FLAG_PCT else ""
        lines.append(
            f"  G{goal_key}: sector {ang}  "
            f"dir [{dirs.min():.0f}, {dirs.max():.0f}]°  "
            f"({outside:.1f}% outside){flag}"
        )
        lines.append(
            f"         hs [{sub['hs'].min():.2f}, {sub['hs'].max():.2f}] m  "
            f"tp [{sub['tp'].min():.1f}, {sub['tp'].max():.1f}] s  "
            f"spr [{sub['spr'].min():.0f}, {sub['spr'].max():.0f}]°"
        )

    lines.extend(
        [
            "",
            "Map legend:",
            "  shaded wedge = propagation sector (goals JSON)",
            "  arrows = subsampled reanalysis wave-from directions",
            "  dashed red = SnapWave enclosure",
            "",
            "Bar panel:",
            "  light bar = allowed sector",
        ]
        + (
            [
                "  shaded strip = density of observed directions",
                "                 (darker = more common)",
                "  crisp outline = goal's configured sector",
                "                 (density may extend past it — see",
                "                 the % outside sector below)",
            ]
            if dense_mode
            else [
                "  solid bar = goal's configured sector",
                "  ticks = subsampled timesteps (if many)",
            ]
        )
    )
    ax_info.text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        transform=ax_info.transAxes,
    )

    return fig


# ---------------------------------------------------------------------------
# Finished SnapWave cases (bluemath_tk plot_case + case base map and titles)
# ---------------------------------------------------------------------------

def infer_goal_from_path(case_dir: str | Path) -> int | None:
    """
    Infer the goal id from a metamodel case path (``.../<goal>/<case_num>``).

    Parameters
    ----------
    case_dir
        Case working directory. The goal id is the last two-digit path
        component (e.g. ``.../03/0012`` -> ``3``).

    Returns
    -------
    int | None
        Goal id, or ``None`` if no two-digit path component is found
        (e.g. a dynamic case path, which has no goal subfolder).
    """
    for part in reversed(Path(case_dir).parts):
        if re.fullmatch(r"\d{2}", part):
            return int(part)
    return None


def _case_background(show_satellite: bool, satellite_source: str):
    """HyGloW basemap for :func:`plot_case` panels.

    Domain panel: satellite only under SnapWave map output (the mesh covers
    the water, so land shows the photo), else the full case base map. Site
    panels: satellite with ocean-only topobathy, or nothing.
    """

    def background(ax: Axes, extent, panel: str, has_map: bool) -> None:
        common = dict(
            ax=ax,
            satellite_source=satellite_source,
            satellite_area=extent,
        )
        if panel == "domain" and not has_map:
            plot_base_map(
                title="domain overview",
                show_topobathy=True,
                show_aoi=True,
                show_satellite=show_satellite,
                **common,
            )
        elif show_satellite:
            plot_base_map(
                show_topobathy=panel == "site",
                show_isobaths=False,
                show_aoi=False,
                show_satellite=True,
                **common,
            )

    return background


def _plot(
    case_dir: str | Path,
    *,
    mode: str,
    title: str | None,
    show_satellite: bool,
    satellite_source: str,
    show: bool,
    **kwargs,
) -> Figure:
    """Shared body of :func:`plot_metamodel_case` / :func:`plot_dynamic_case`."""
    name = Path(case_dir).name
    if title is None:
        goal = infer_goal_from_path(case_dir) if mode == "metamodel" else None
        label = f" · G{goal}" if goal is not None else ""
        title = display_title(f"SnapWave {mode}{label} · case {name}")
    fig = plot_case(
        case_dir,
        mode=mode,
        title=title,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
        background=_case_background(show_satellite, satellite_source),
        **kwargs,
    )
    if show:
        plt.show()
    return fig


def plot_metamodel_case(
    case_dir: str | Path,
    *,
    time_index: int = 0,
    show_map: bool = True,
    map_variable: str = "hs",
    style: CasePlotStyle | None = None,
    title: str | None = None,
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
    show: bool = True,
) -> Figure:
    """
    Plot a finished metamodel SnapWave case from its folder.

    Full-width domain map (mesh or base map, with the active boundary node
    and its direction arrow) on top; 2x2 output-site panels for
    ``style.top_vars`` below.

    Parameters
    ----------
    case_dir
        Case working directory, e.g. ``<output_dir>/<goal>/<case_num>``.
    time_index
        Forcing/output timestep to plot.
    show_map
        Draw the domain mesh map when available; otherwise falls back to
        the standard case base map.
    map_variable
        Variable shaded on the domain map mesh.
    style
        Plot styling config; defaults to :data:`DEFAULT_CASE_PLOT_STYLE`.
    title
        Optional title override (otherwise built from case metadata).
    show_satellite, satellite_source
        Draw a satellite basemap under the panels.
    show
        Call ``plt.show()`` when ``True``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return _plot(
        case_dir,
        mode="metamodel",
        title=title,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
        show=show,
        time_index=time_index,
        show_map=show_map,
        map_variable=map_variable,
        style=style,
    )


def plot_dynamic_case(
    case_dir: str | Path,
    *,
    time_index: int = 0,
    show_map: bool = True,
    map_variable: str = "hs",
    style: CasePlotStyle | None = None,
    title: str | None = None,
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
    show: bool = True,
) -> Figure:
    """
    Plot a finished dynamic SnapWave case from its folder.

    Full-width domain map (mesh or base map, with boundary nodes coloured/
    sized by forcing ``hs`` and a direction arrow per active node) on top;
    2x2 output-site panels for ``style.top_vars`` below.

    Parameters
    ----------
    case_dir
        Case working directory.
    time_index
        Forcing/output timestep to plot.
    show_map
        Draw the domain mesh map when available; otherwise falls back to
        the standard case base map.
    map_variable
        Variable shaded on the domain map mesh.
    style
        Plot styling config; defaults to :data:`DEFAULT_CASE_PLOT_STYLE`.
    title
        Optional title override (otherwise built from case metadata).
    show_satellite, satellite_source
        Draw a satellite basemap under the panels.
    show
        Call ``plt.show()`` when ``True``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return _plot(
        case_dir,
        mode="dynamic",
        title=title,
        show_satellite=show_satellite,
        satellite_source=satellite_source,
        show=show,
        time_index=time_index,
        show_map=show_map,
        map_variable=map_variable,
        style=style,
    )


__all__ = [
    "DEFAULT_POINT_COLOR",
    "HINDCAST_COLORS",
    "ISOBATH_COLORS",
    "MESH_COLORS",
    "SITE_SOURCE_MARKERS",
    "infer_goal_from_path",
    "plot_base_map",
    "plot_case_map",
    "plot_dynamic_case",
    "plot_dynamic_configuration",
    "plot_dynamic_configuration_map",
    "plot_land",
    "plot_metamodel_case",
    "plot_metamodel_configuration",
    "plot_metamodel_configuration_map",
    "plot_points",
    "plot_ring",
    "plot_sectors",
    "plot_validation_sites",
]
