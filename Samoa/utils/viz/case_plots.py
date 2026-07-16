"""Case map plotting — styling here, paths from :mod:`utils.config`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
from shapely.geometry import MultiPolygon, Polygon
from utils.config import PATHS, display_title, resolve
from utils.directions import (
    in_nautical_sector,
    nautical_sector_to_wedge_angles,
    normalize_nautical,
)
from utils.simulation.offshore_forcing import BOUNDARY_VARS

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
SITE_SOURCE_COLORS = {
    "isobath": "#1f78b4",
    "buoy": "#d62728",
    "mesh": "#9467bd",
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


def _goal_colors(goal_ids: list) -> dict:
    cmap = plt.get_cmap("tab10")
    return {gid: cmap(i % 10) for i, gid in enumerate(goal_ids)}


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
    left_n = float(normalize_nautical(left))
    right_n = float(normalize_nautical(right))
    if left < 0 or left_n > right_n:
        return [(left_n, 360.0), (0.0, right_n)]
    return [(left_n, right_n)]


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


def _plot_goal_dir_range_bars(
    ax: Axes,
    cases: pd.DataFrame,
    goals_config: dict[str, dict[str, Any]],
    colors: dict[int, Any],
    *,
    title: str = "Sampled direction ranges per goal",
    max_dir_marks: int = 200,
    annotate_tp: bool = True,
) -> None:
    """Horizontal bars: propagation sector (light) and observed dir range (solid)."""
    goal_ids = sorted(cases["goal"].unique())
    y_positions = np.arange(len(goal_ids))

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

        dmin, dmax = float(dirs.min()), float(dirs.max())
        ax.barh(
            y,
            dmax - dmin,
            left=dmin,
            height=0.28,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        mark_dirs = dirs
        if len(mark_dirs) > max_dir_marks:
            mark_dirs = np.random.default_rng(0).choice(
                mark_dirs, size=max_dir_marks, replace=False
            )
        ax.scatter(
            mark_dirs,
            np.full(len(mark_dirs), y),
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
    show_topobathy: bool = True,
    title: str = "Metamodel pre-run diagnostic",
) -> Figure:
    """
    Multi-panel diagnostic before building or running metamodel SnapWave cases.

    Layout: map and direction-range bars on the left; text summary on the right.
    """
    required = {"goal", "tp", "dir", "spr", "wl"}
    missing = required - set(cases_df.columns)
    if missing:
        raise KeyError(f"cases_df missing columns: {sorted(missing)}")

    cases = cases_df.copy()
    cases["goal"] = cases["goal"].astype(int)
    goal_ids = sorted(cases["goal"].unique())
    colors = _goal_colors(goal_ids)

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.1, 1.0], height_ratios=[1.5, 0.75])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis("off")

    plot_base_map(ax=ax_map, title=title, show_topobathy=show_topobathy)
    gpd.GeoDataFrame(geometry=[enclosure_polygon], crs="EPSG:4326").boundary.plot(
        ax=ax_map,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label="enclosure",
    )

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
        ax_map.plot(lon, lat, marker="*", color=color, markersize=12, zorder=6)
        ax_map.text(
            lon,
            lat,
            f" G{goal_key}",
            fontsize=9,
            color="black",
            weight="bold",
            zorder=7,
            va="bottom",
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

    minx, miny, maxx, maxy = gpd.GeoDataFrame(
        geometry=[enclosure_polygon], crs="EPSG:4326"
    ).total_bounds
    pad = 0.08
    ax_map.set_xlim(minx - pad, maxx + pad)
    ax_map.set_ylim(miny - pad, maxy + pad)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="upper right", fontsize=7, framealpha=0.9)

    _plot_goal_dir_range_bars(ax_bars, cases, goals_config, colors)

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
            "  solid bar = sampled dir range",
            "  ticks = individual LHS dirs",
        ]
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
    max_dir_marks: int = 200,
    show_topobathy: bool = True,
    maps_to_save: int | None = None,
    title: str = "Dynamic pre-run diagnostic",
) -> Figure:
    """
    Multi-panel diagnostic before building or running dynamic SnapWave cases.

    Shows propagation sectors from the goals JSON, observed bulk forcing ranges
    from offshore reanalysis (``hs``, ``tp``, ``dir``, ``spr``), and highlights
    when wave-from directions fall outside the nominal goal sectors.

    Layout: map and direction-range bars on the left; text summary on the right.
    """
    cases = _offshore_bulk_long(offshore)
    goal_ids = sorted(cases["goal"].unique())
    colors = _goal_colors(goal_ids)
    n_time = int(offshore.sizes.get("time", len(cases) // max(len(goal_ids), 1)))

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.1, 1.0], height_ratios=[1.5, 0.75])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_bars = fig.add_subplot(gs[1, 0])
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis("off")

    plot_base_map(ax=ax_map, title=title, show_topobathy=show_topobathy)
    gpd.GeoDataFrame(geometry=[enclosure_polygon], crs="EPSG:4326").boundary.plot(
        ax=ax_map,
        color="#d62728",
        linestyle="--",
        linewidth=1.4,
        label="enclosure",
    )

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
        ax_map.plot(lon, lat, marker="*", color=color, markersize=12, zorder=6)
        ax_map.text(
            lon,
            lat,
            f" G{goal_key}",
            fontsize=9,
            color="black",
            weight="bold",
            zorder=7,
            va="bottom",
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

    minx, miny, maxx, maxy = gpd.GeoDataFrame(
        geometry=[enclosure_polygon], crs="EPSG:4326"
    ).total_bounds
    pad = 0.08
    ax_map.set_xlim(minx - pad, maxx + pad)
    ax_map.set_ylim(miny - pad, maxy + pad)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.legend(loc="upper right", fontsize=7, framealpha=0.9)

    _plot_goal_dir_range_bars(
        ax_bars,
        cases,
        goals_config,
        colors,
        title="Observed direction ranges per goal (reanalysis)",
        max_dir_marks=max_dir_marks,
    )

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
        flag = "  ! outside sector" if outside > 0 else ""
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
            "  solid bar = observed dir range",
            "  ticks = subsampled timesteps (if many)",
        ]
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


__all__ = [
    "DEFAULT_POINT_COLOR",
    "HINDCAST_COLORS",
    "ISOBATH_COLORS",
    "MESH_COLORS",
    "SITE_SOURCE_MARKERS",
    "plot_base_map",
    "plot_case_map",
    "plot_dynamic_configuration",
    "plot_land",
    "plot_metamodel_configuration",
    "plot_points",
    "plot_ring",
    "plot_sectors",
    "plot_validation_sites",
]
