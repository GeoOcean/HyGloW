"""Case maps and diagnostics: the case's paths and palettes on top of bluemath_tk plots.

- base map of the case (:func:`plot_base_map`: topobathy, isobaths, AoI,
  satellite; drawn with :class:`bluemath_tk.core.plotting.base_plotting.DefaultStaticPlotting`);
- observations, goals and output sites (:func:`plot_case_map`,
  :func:`plot_sectors`, :func:`plot_output_sites`, :func:`plot_site_depths`,
  :func:`plot_validation_sites`);
- pre-run diagnostics (:func:`plot_metamodel_configuration`,
  :func:`plot_dynamic_configuration`), built on
  :mod:`bluemath_tk.waves.hywaves.plots`;
- finished model cases (:func:`plot_metamodel_case`, :func:`plot_dynamic_case`);
- the offshore calibration (:func:`plot_satellite_pairs`, :func:`plot_calibration_metrics`).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from bluemath_tk.core.operations import in_nautical_sector, nautical_sector_to_wedge_angles
from bluemath_tk.core.plotting.base_plotting import DefaultStaticPlotting, decimate_raster
from bluemath_tk.core.plotting.cases import CasePlotStyle
from bluemath_tk.waves.hywaves.output_points import smooth_raster
from bluemath_tk.waves.hywaves.plots import goal_colors, plot_direction_ranges, plot_goal_sectors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Wedge
from shapely.geometry.base import BaseGeometry

from utils.config import OUTPUT_SITES_CONFIG, PATHS, display_title, resolve

FIGSIZE = (9, 7)
TOPOBATHY_VMIN, TOPOBATHY_VMAX = -200, 0
TOPOBATHY_LAND_VMAX = 300
TOPOBATHY_MAX_PIXELS = 4_000_000
SATELLITE_SOURCE = "arcgis"
ISOBATH_CONTOUR_LEVELS = [-100, -50, -20, -10, -5]

HINDCAST_COLORS = {"whacs": "#1f77b4", "era5": "#ff7f0e", "codec": "#2ca02c"}
DEFAULT_POINT_COLOR = "#17becf"
TYPE_COLORS = {
    "validation": "#d62728",
    "waves": "#1f77b4",
    "spectrum": "#9467bd",
    "calibration": "#ff7f0e",
    "levels": "#2ca02c",
    "ignore": "#7f7f7f",
}
ISOBATH_COLORS = {"5m": "#8dd3c7", "10m": "#1f78b4", "20m": "#33a02c", "50m": "#e31a1c", "100m": "#6a3d9a"}
MESH_COLOR = "#9467bd"
SITE_SOURCE_MARKERS = {"isobath": "o", "mesh": ".", "buoy": "s"}
SITE_SOURCE_COLORS = {"isobath": "#1f78b4", "buoy": "#d62728", "mesh": MESH_COLOR}
#: Marker colour of the output sites dropped by the mesh check, by status.
DROPPED_SITE_COLORS = {"outside_mesh": "#d62728", "can_dry": "#ff7f0e"}
POINT_MARKERSIZE, POINT_EDGE_WIDTH, POINT_EDGE_COLOR = 70, 0.6, "white"

_STATIC = DefaultStaticPlotting()


# ---------------------------------------------------------------------------
# Base map
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _topobathy() -> xr.DataArray:
    """The case topobathy raster, decimated to a display-sized grid."""
    da = rioxarray.open_rasterio(resolve(PATHS["topobathy_file"])).squeeze(drop=True)
    return decimate_raster(da, TOPOBATHY_MAX_PIXELS).load()


@lru_cache(maxsize=1)
def _aoi() -> gpd.GeoDataFrame:
    return gpd.read_file(resolve(PATHS["aoi_file"]))


@lru_cache(maxsize=1)
def _isobath_raster() -> xr.DataArray:
    """Topobathy smoothed with the largest isobath ``smooth_m``: the lines the sites sit on."""
    smooth_m = max((s.get("smooth_m") or 0 for s in OUTPUT_SITES_CONFIG.get("isobaths", [])), default=0)
    return smooth_raster(_topobathy(), float(smooth_m)) if smooth_m else _topobathy()


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
) -> tuple[Figure, Axes]:
    """
    The case map background.

    Parameters
    ----------
    ax
        Existing axes (a cartopy ``GeoAxes`` for the satellite image); created
        when ``None``.
    title
        Map title (case name prefixed).
    show_topobathy
        Depth raster; default ``not show_satellite``. With a satellite image
        too, only the ocean is coloured.
    show_isobaths
        Labelled depth contours (:data:`ISOBATH_CONTOUR_LEVELS`).
    show_satellite, satellite_source, satellite_area
        Satellite image (``"arcgis"``, ``"eox"``, ``"osm"``, ...) over
        *satellite_area* (``lon_min, lon_max, lat_min, lat_max``; default the
        AoI): pass the final view of the axes so tiles come at its zoom.
    show_aoi
        AoI outline.

    Returns
    -------
    tuple[Figure, Axes]
    """
    if show_topobathy is None:
        show_topobathy = not show_satellite
    if ax is None:
        fig, ax = plt.subplots(
            figsize=FIGSIZE, subplot_kw={"projection": ccrs.PlateCarree()} if show_satellite else {}
        )
    fig = ax.figure

    if show_satellite:
        if satellite_area is None:
            minx, miny, maxx, maxy = _aoi().total_bounds
            satellite_area = (minx - 0.05, maxx + 0.05, miny - 0.05, maxy + 0.05)
        _STATIC.plot_satellite(ax, area=satellite_area, source=satellite_source, zorder=0)
    if show_topobathy:
        _STATIC.plot_bathymetry(
            ax, _topobathy(), area=None, value_range=(TOPOBATHY_VMIN, TOPOBATHY_LAND_VMAX),
            ocean_only=show_satellite, method="imshow", alpha=0.85, add_colorbar=not show_satellite,
        )
    if show_isobaths:
        _STATIC.plot_isodepths(
            ax, _isobath_raster(), ISOBATH_CONTOUR_LEVELS, legend=True,
            colors=[ISOBATH_COLORS[f"{abs(level)}m"] for level in ISOBATH_CONTOUR_LEVELS],
            linewidths=0.6, alpha=0.9, zorder=2,
            labels_kwargs={"fmt": lambda v: f"{abs(int(v))}m", "fontsize": 6},
        )
    if show_aoi:
        _aoi().boundary.plot(ax=ax, color="black", linewidth=1.2, label="AoI")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(display_title(title or "Case map"))
    return fig, ax


def plot_mesh(mesh_file: str | Path, ax: Axes, *, color: str = "#fbbf24", linewidth: float = 0.3,
              alpha: float = 0.55) -> None:
    """The edges of a UGRID mesh: a solid tint over its footprint at domain zoom."""
    _STATIC.plot_ugrid_mesh(ax, resolve(mesh_file), color=color, linewidth=linewidth, alpha=alpha)


# ---------------------------------------------------------------------------
# Observations, goals and output sites
# ---------------------------------------------------------------------------


def plot_points(
    points: gpd.GeoDataFrame,
    ax: Axes,
    *,
    color_col: str = "type",
    label_col: str | None = "instrument_id",
    color_map: dict[str, str] | None = None,
) -> None:
    """Points coloured by *color_col* (default :data:`TYPE_COLORS`), labelled by *label_col*."""
    colors = color_map or TYPE_COLORS
    groups = points.groupby(color_col) if color_col in points else [(None, points)]
    for value, group in groups:
        group.plot(
            ax=ax, color=colors.get(str(value), DEFAULT_POINT_COLOR), markersize=POINT_MARKERSIZE,
            edgecolor=POINT_EDGE_COLOR, linewidth=POINT_EDGE_WIDTH, label=None if value is None else str(value),
        )
    if label_col in points:
        for _, row in points.iterrows():
            ax.annotate(row[label_col], (row.geometry.x, row.geometry.y), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
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
    Each goal's sector as a wedge at its hindcast snap, linked to its target.

    Parameters
    ----------
    targets
        Goal targets (``goal_id``, point geometry).
    goals
        Goals JSON mapping (``angles`` nautical ``[left, right]``).
    boundary_snaps
        ``{goal_id: {"longitude", "latitude"}}``.
    ax
        Map axes.
    radius_deg
        Wedge radius (degrees).
    """
    for _, row in targets.iterrows():
        key = str(int(row["goal_id"]))
        lon, lat = boundary_snaps[key]["longitude"], boundary_snaps[key]["latitude"]
        ax.plot([row.geometry.x, lon], [row.geometry.y, lat], color="black", ls="--", lw=0.8, alpha=0.55, zorder=4)
        ax.plot(lon, lat, marker="x", color="black", markersize=5, zorder=5)
        theta1, theta2 = nautical_sector_to_wedge_angles(*goals[key]["angles"])
        ax.add_patch(Wedge((lon, lat), radius_deg, theta1, theta2, facecolor="#1f78b4", alpha=0.15,
                           edgecolor="#1f78b4", linewidth=0.8, zorder=3))


def plot_case_map(
    points: gpd.GeoDataFrame | None = None,
    *,
    title: str | None = None,
    color_col: str = "type",
    label_col: str | None = "instrument_id",
    color_map: dict[str, str] | None = None,
    show_satellite: bool = False,
    satellite_source: str = SATELLITE_SOURCE,
) -> tuple[Figure, Axes]:
    """The case base map with optional points (:func:`plot_points`)."""
    fig, ax = plot_base_map(title=title, show_satellite=show_satellite, satellite_source=satellite_source)
    if points is not None and not points.empty:
        plot_points(points, ax, color_col=color_col, label_col=label_col, color_map=color_map)
    fig.tight_layout()
    return fig, ax


def site_group_colors() -> dict[str, str]:
    """Colour per site group name: isobaths by depth, mesh polygons, else default."""
    polygons = {p["name"]: MESH_COLOR for p in OUTPUT_SITES_CONFIG.get("polygons", [])}
    return {**ISOBATH_COLORS, **polygons}


def _site_color(source: str, label: str) -> str:
    if source == "buoy":
        return SITE_SOURCE_COLORS["buoy"]
    return site_group_colors().get(str(label), SITE_SOURCE_COLORS.get(source, DEFAULT_POINT_COLOR))


def plot_validation_sites(sites: xr.Dataset, ax: Axes, *, annotate_buoys: bool = True) -> None:
    """
    Output sites on a map, one colour per group (isobath, polygon) and buoys as squares.

    Parameters
    ----------
    sites
        Dataset on ``sites`` with the site metadata coordinates.
    ax
        Map axes.
    annotate_buoys
        Write the buoy names.
    """
    sources = sites["site_source"].values.astype(str)
    labels = sites["site_label"].values.astype(str)
    lon, lat = sites["site_longitude"].values, sites["site_latitude"].values
    for source, label in sorted(set(zip(sources, labels))):
        mask = (sources == source) & (labels == label)
        ax.scatter(
            lon[mask], lat[mask], c=_site_color(source, label), marker=SITE_SOURCE_MARKERS.get(source, "o"),
            s=POINT_MARKERSIZE * (0.35 if source == "mesh" else 1.0), edgecolors=POINT_EDGE_COLOR,
            linewidths=POINT_EDGE_WIDTH, zorder=6 if source == "buoy" else 5,
            label="buoy" if source == "buoy" else label,
        )
    if annotate_buoys:
        for site, source, x, y in zip(sites.sites.values, sources, lon, lat):
            if source == "buoy":
                ax.annotate(str(site).removeprefix("buoy_"), (x, y), xytext=(4, 4),
                            textcoords="offset points", fontsize=8, zorder=7)
    handles, names = ax.get_legend_handles_labels()
    unique = dict(zip(names, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower left", fontsize=8, framealpha=0.9)


def plot_output_sites(
    sites: pd.DataFrame,
    *,
    title: str | None = "Output sites on the model mesh",
    show_satellite: bool = False,
    cmap: str = "viridis",
) -> tuple[Figure, Axes]:
    """
    Output sites coloured by mesh depth, with the dropped ones marked.

    Parameters
    ----------
    sites
        The output-sites table with every site (``output_sites(all_sites=True)``).
    title, show_satellite
        Passed to :func:`plot_base_map`.
    cmap
        Colormap of ``mesh_depth``.
    """
    fig, ax = plot_base_map(title=title, show_isobaths=False, show_satellite=show_satellite)
    ok = sites[sites["status"] == "ok"]
    norm = mcolors.Normalize(0.0, float(np.nanmax(ok["mesh_depth"])))
    for source, group in ok.groupby("source"):
        sc = ax.scatter(
            group["lon"], group["lat"], c=group["mesh_depth"], cmap=cmap, norm=norm,
            marker=SITE_SOURCE_MARKERS.get(source, "o"), s=POINT_MARKERSIZE * (0.35 if source == "mesh" else 0.6),
            edgecolors=POINT_EDGE_COLOR, linewidths=POINT_EDGE_WIDTH, label=f"{source} ({len(group)})", zorder=6,
        )
    for status, group in sites[sites["status"] != "ok"].groupby("status"):
        ax.scatter(group["lon"], group["lat"], c=DROPPED_SITE_COLORS.get(status, "k"), marker="x",
                   s=POINT_MARKERSIZE * 0.5, label=f"{status} ({len(group)})", zorder=7)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="mesh depth (m)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    return fig, ax


def plot_site_depths(sites: pd.DataFrame) -> tuple[Figure, Axes]:
    """Nominal depth (contour level, buoy measured depth) against mesh depth of the ``ok`` sites."""
    ok = sites[(sites["status"] == "ok") & sites["nominal_depth"].notna()]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for (source, label), group in ok.groupby(["source", "source_label"]):
        if source != "buoy":
            ax.scatter(group["nominal_depth"], group["mesh_depth"], s=14, alpha=0.6,
                       color=_site_color(source, label), label=f"{label} ({len(group)})")
    buoys = ok[ok["source"] == "buoy"]
    ax.scatter(buoys["nominal_depth"], buoys["mesh_depth"], s=30, marker="s", color=SITE_SOURCE_COLORS["buoy"],
               edgecolors="k", linewidths=0.4, label=f"buoys ({len(buoys)})")
    top = float(np.nanmax(ok[["nominal_depth", "mesh_depth"]].to_numpy())) * 1.05
    ax.plot([0, top], [0, top], color="0.5", lw=0.8, ls="--")
    ax.set(xlim=(0, top), ylim=(0, top), aspect="equal",
           xlabel="nominal depth (m): contour level / buoy measured", ylabel="mesh depth (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Pre-run diagnostics
# ---------------------------------------------------------------------------


def _configuration_figure(
    *,
    model_boundary: BaseGeometry,
    model_boundary_label: str,
    goals_config: dict[str, dict[str, Any]],
    cases: pd.DataFrame,
    output_sites_df: pd.DataFrame,
    bracket: str,
    bars_title: str,
    show_topobathy: bool | None,
    show_satellite: bool,
    title: str,
    lines: list[str],
) -> Figure:
    """
    Map (model boundary, goal sectors with the case directions, sites), direction bars, text panel.

    *lines* is the text panel, before the legend added here.
    """
    goal_ids = sorted(cases["goal"].unique())
    colors = goal_colors(goal_ids)
    bars_h = max(2.2, len(goal_ids) * 0.22 + 0.9)
    fig = plt.figure(figsize=(15, 6.5 + bars_h), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.1, 1.0], height_ratios=[6.5, bars_h])
    ax_map = fig.add_subplot(gs[0, 0], **({"projection": ccrs.PlateCarree()} if show_satellite else {}))
    ax_info = fig.add_subplot(gs[:, 1])
    ax_info.axis("off")

    # the view: model boundary, goals and sites; the satellite tiles match it
    boundary = gpd.GeoDataFrame(geometry=[model_boundary], crs="EPSG:4326")
    polygon = model_boundary.geom_type.endswith("Polygon")
    lons = [*boundary.total_bounds[[0, 2]], *(g["lon"] for g in goals_config.values()), *output_sites_df["lon"]]
    lats = [*boundary.total_bounds[[1, 3]], *(g["lat"] for g in goals_config.values()), *output_sites_df["lat"]]
    pad = 0.08 if polygon else 0.25
    view = (min(lons) - pad, max(lons) + pad, min(lats) - pad, max(lats) + pad)
    plot_base_map(ax=ax_map, title=title, show_topobathy=show_topobathy, show_satellite=show_satellite,
                  satellite_area=view)
    (boundary.boundary if polygon else boundary).plot(ax=ax_map, color="#d62728", linestyle="--", linewidth=1.4,
                                                      label=model_boundary_label)
    for source, group in output_sites_df.groupby("source"):
        ax_map.scatter(group["lon"], group["lat"], marker=SITE_SOURCE_MARKERS.get(source, "o"),
                       color=SITE_SOURCE_COLORS.get(source, DEFAULT_POINT_COLOR), s=18 if source == "buoy" else 10,
                       edgecolors="white", linewidths=0.4, alpha=0.85, label=f"sites: {source}", zorder=3)
    ax_map.set_xlim(view[0], view[1])
    ax_map.set_ylim(view[2], view[3])
    ax_map.set_aspect("equal", adjustable="box")
    plot_goal_sectors(ax_map, goals_config, colors, directions=cases)
    ax_map.legend(loc="best", fontsize=7, framealpha=0.9)

    dense = plot_direction_ranges(fig.add_subplot(gs[1, 0]), cases, goals_config, colors,
                                  bracket=bracket, title=bars_title)
    lines += [
        "",
        "Map: shaded wedge = goal sector, arrows = wave-from directions,",
        f"     dashed red = {model_boundary_label}",
        "Bars: light = sector; "
        + ("shaded strip = direction density, outline = " if dense else "solid bar + ticks = ")
        + ("sector" if bracket == "sector" else "sampled range"),
    ]
    ax_info.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace",
                 transform=ax_info.transAxes)
    return fig


def plot_metamodel_configuration(
    *,
    model_boundary: BaseGeometry,
    model_boundary_label: str = "enclosure",
    goals_config: dict[str, dict[str, Any]],
    cases_df: pd.DataFrame,
    output_sites_df: pd.DataFrame,
    bounds: dict[str, list[float]] | None = None,
    num_samples_per_goal: int | None = None,
    max_steepness: float | None = None,
    mesh_file: str | None = None,
    enclosure_buffer_m: float | None = None,
    show_topobathy: bool | None = None,
    show_satellite: bool = False,
    title: str = "Metamodel pre-run diagnostic",
) -> Figure:
    """
    Diagnostic of the metamodel cases before running them.

    Parameters
    ----------
    model_boundary, model_boundary_label, goals_config, output_sites_df
        Case geometry (:class:`utils.simulation.HyWavesWrapperSetup`).
    cases_df
        Cases with ``goal``, ``hs``, ``tp``, ``dir``, ``spr``, ``wl``.
    bounds, num_samples_per_goal, max_steepness, mesh_file, enclosure_buffer_m
        Shown in the text panel.
    show_topobathy, show_satellite
        Map background (:func:`plot_base_map`).
    title
        Figure title.
    """
    cases = cases_df.astype({"goal": int})
    per_goal = cases.groupby("goal").size()
    lines = [
        display_title("Metamodel summary"), "",
        f"Goals            : {len(per_goal)}",
        f"Cases            : {len(cases)} ({per_goal.mean():.0f} / goal kept of {num_samples_per_goal or '-'})",
        f"Output sites     : {len(output_sites_df)}",
    ]
    if enclosure_buffer_m is not None:
        lines.append(f"Enclosure buffer : {enclosure_buffer_m / 1000:g} km")
    if mesh_file:
        lines.append(f"Mesh             : {Path(mesh_file).name}")
    if bounds:
        lines += ["", "Bounds (dir from the goal sectors):"] + [f"  {k:<4}: {v}" for k, v in bounds.items()]
    if max_steepness is not None:
        lines.append(f"Max steepness    : {max_steepness:g} (hs / L0)")
    lines += ["", "Goal sectors [left, right] (nautical):"] + [
        f"  G{k}: {goals_config[k]['angles']}  ({per_goal.get(int(k), 0)} cases)" for k in sorted(goals_config, key=int)
    ]
    return _configuration_figure(
        model_boundary=model_boundary, model_boundary_label=model_boundary_label, goals_config=goals_config,
        cases=cases, output_sites_df=output_sites_df, bracket="observed",
        bars_title="Sampled direction ranges per goal", show_topobathy=show_topobathy,
        show_satellite=show_satellite, title=title, lines=lines,
    )


def _forcing_summary_lines(
    times: pd.DatetimeIndex,
    wind: xr.Dataset | None,
    water_level: xr.DataArray | None,
    wind_name: str | None,
    water_level_name: str | None,
) -> list[str]:
    """Text-panel lines on the wind and water level given to the dynamic cases."""
    from bluemath_tk.waves.wind import wind_speed_direction

    lines: list[str] = []
    if wind is not None:
        speed = wind_speed_direction(wind["u10"].values, wind["v10"].values)[0]
        wt = pd.DatetimeIndex(wind["time"].values)
        covered = ((times >= wt.min()) & (times <= wt.max())).mean() * 100
        p50, p95 = np.nanpercentile(speed, [50, 95])
        lines += [
            "", f"Wind              : {wind_name or 'u10/v10'}, covers {covered:.0f}% of case times",
            f"  u10 P50/P95/max : {p50:.1f} / {p95:.1f} / {np.nanmax(speed):.1f} m/s",
            f"  u10 > 15 m/s    : {np.mean(speed > 15) * 100:.1f}% of grid-hours",
        ]
    if water_level is not None:
        wl = water_level.reindex(time=times)
        missing = float(wl.isnull().all([d for d in wl.dims if d != "time"]).mean()) * 100
        lines += [
            "", f"Water level       : {water_level_name or 'per goal'}"
            + (f", missing {missing:.0f}% of case times (wl = 0)" if missing else ""),
            f"  range           : [{float(wl.min()):.2f}, {float(wl.max()):.2f}] m",
        ]
    return lines


def plot_dynamic_configuration(
    *,
    model_boundary: BaseGeometry,
    model_boundary_label: str = "enclosure",
    goals_config: dict[str, dict[str, Any]],
    offshore: xr.Dataset,
    output_sites_df: pd.DataFrame,
    offshore_name: str | None = None,
    mesh_file: str | None = None,
    enclosure_buffer_m: float | None = None,
    maps_to_save: int | None = None,
    wind: xr.Dataset | None = None,
    water_level: xr.DataArray | None = None,
    wind_name: str | None = None,
    water_level_name: str | None = None,
    show_topobathy: bool | None = None,
    show_satellite: bool = False,
    title: str = "Dynamic pre-run diagnostic",
) -> Figure:
    """
    Diagnostic of the dynamic cases before running them: goal sectors against
    the offshore hindcast directions (share outside each sector), bulk
    ranges, and the wind and water level given to the cases.

    Parameters
    ----------
    model_boundary, model_boundary_label, goals_config, output_sites_df
        Case geometry (:class:`utils.simulation.HyWavesWrapperSetup`).
    offshore
        ``hs``, ``tp``, ``dir``, ``spr`` on ``(time, goal)``.
    offshore_name, mesh_file, enclosure_buffer_m, maps_to_save
        Shown in the text panel.
    wind, water_level, wind_name, water_level_name
        Forcing of the cases, summarised in the text panel.
    show_topobathy, show_satellite, title
        Map background and figure title.
    """
    cases = offshore[["hs", "tp", "dir", "spr"]].to_dataframe().reset_index().astype({"goal": int})
    times = pd.DatetimeIndex(offshore["time"].values)
    lines = [
        display_title("Dynamic summary"), "",
        f"Offshore source   : {offshore_name or 'offshore waves'}",
        f"Goals per case    : {cases['goal'].nunique()}",
        f"Cases (time steps): {len(times)}  {times[0]:%Y-%m-%d %H:%M} → {times[-1]:%Y-%m-%d %H:%M}",
        f"Output sites      : {len(output_sites_df)}",
    ]
    if enclosure_buffer_m is not None:
        lines.append(f"Enclosure buffer  : {enclosure_buffer_m / 1000:g} km")
    if mesh_file:
        lines.append(f"Mesh              : {Path(mesh_file).name}")
    if maps_to_save is not None:
        lines.append(f"Maps to save      : {maps_to_save} / {len(times)}")
    lines += ["", "Bulk ranges (all goals):"] + [
        f"  {v:<4}: [{cases[v].min():.2f}, {cases[v].max():.2f}]" for v in ("hs", "tp", "dir", "spr")
    ]
    lines += _forcing_summary_lines(times, wind, water_level, wind_name, water_level_name)
    lines += ["", "Per goal: sector, share of directions outside (! > 5 %):"]
    for key in sorted(goals_config, key=int):
        angles = goals_config[key]["angles"]
        dirs = cases.loc[cases["goal"] == int(key), "dir"].to_numpy(float)
        outside = (1.0 - np.mean(in_nautical_sector(dirs, *angles))) * 100
        lines.append(f"  G{key}: {angles}  {outside:4.1f}% outside{'  !' if outside > 5 else ''}")
    return _configuration_figure(
        model_boundary=model_boundary, model_boundary_label=model_boundary_label, goals_config=goals_config,
        cases=cases, output_sites_df=output_sites_df, bracket="sector",
        bars_title="Offshore hindcast directions per goal", show_topobathy=show_topobathy,
        show_satellite=show_satellite, title=title, lines=lines,
    )


# ---------------------------------------------------------------------------
# Finished model cases
# ---------------------------------------------------------------------------


def _case_title(case_dir: Path, mode: str, swan: bool) -> str:
    """``SnapWave metamodel · G3 · case 0012`` (the goal is the two-digit parent folder)."""
    goal = next((int(p) for p in reversed(case_dir.parts) if re.fullmatch(r"\d{2}", p)), None)
    label = f" · G{goal}" if mode == "metamodel" and goal is not None else ""
    return display_title(f"{'SWAN' if swan else 'SnapWave'} {mode}{label} · case {case_dir.name}")


def _plot_case(
    case_dir: str | Path,
    mode: str,
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
    """A finished SWAN (``INPUT`` file) or SnapWave case, with the case base map behind it."""
    case_dir = Path(case_dir)
    swan = (case_dir / "INPUT").is_file()
    title = title or _case_title(case_dir, mode, swan)
    if swan:
        from bluemath_tk.wrappers.swan.swan_plots import plot_case

        fig = plot_case(case_dir, mode=mode, style=style, title=title, show_satellite=show_satellite,
                        satellite_source=satellite_source, map_variable=map_variable)
    else:
        from bluemath_tk.wrappers.snapwave.snapwave_plots import plot_case

        def background(ax: Axes, extent, panel: str, has_map: bool) -> None:
            # domain panel without model map: the full base map; else satellite only
            if panel == "domain" and not has_map:
                plot_base_map(ax=ax, title="domain overview", show_topobathy=True, show_satellite=show_satellite,
                              satellite_source=satellite_source, satellite_area=extent)
            elif show_satellite:
                plot_base_map(ax=ax, show_topobathy=panel == "site", show_isobaths=False, show_aoi=False,
                              show_satellite=True, satellite_source=satellite_source, satellite_area=extent)

        fig = plot_case(case_dir, mode=mode, title=title, show_satellite=show_satellite,
                        satellite_source=satellite_source, background=background, time_index=time_index,
                        show_map=show_map, map_variable=map_variable, style=style)
    if show:
        plt.show()
    return fig


def plot_metamodel_case(case_dir: str | Path, **kwargs: Any) -> Figure:
    """
    A finished metamodel case: domain map with the active node, output-site panels.

    *kwargs*: ``time_index``, ``show_map``, ``map_variable``, ``style``
    (:class:`bluemath_tk.core.plotting.cases.CasePlotStyle`), ``title``,
    ``show_satellite``, ``satellite_source``, ``show``.
    """
    return _plot_case(case_dir, "metamodel", **kwargs)


def plot_dynamic_case(case_dir: str | Path, **kwargs: Any) -> Figure:
    """A finished dynamic case and its wind; *kwargs* as for :func:`plot_metamodel_case`."""
    return _plot_case(case_dir, "dynamic", **kwargs)


# ---------------------------------------------------------------------------
# Offshore hindcast vs satellite altimetry (notebook 07b, utils.calibration)
# ---------------------------------------------------------------------------


def plot_satellite_pairs(
    pairs: pd.DataFrame,
    nodes: pd.DataFrame,
    radius_km: float,
    *,
    title: str = "Altimeter passes around the goal nodes",
    show_topobathy: bool | None = False,
) -> Figure:
    """
    Altimeter passes used for each goal (map) and how many (bars).

    Parameters
    ----------
    pairs
        :func:`utils.calibration.satellite_pairs`.
    nodes
        :func:`utils.calibration.goal_nodes` (hindcast node per goal).
    radius_km
        Search radius, drawn around each node.
    title, show_topobathy
        Map title and background.
    """
    goal_ids = list(nodes.index)
    colors = goal_colors(goal_ids)
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    plot_base_map(ax=ax, title=title, show_topobathy=show_topobathy)
    theta = np.linspace(0, 2 * np.pi, 120)
    dlat = np.rad2deg(radius_km / 6371.0)
    for gid, node in nodes.iterrows():
        sub = pairs[pairs["goal"] == str(gid)]
        ax.scatter(sub["sat_lon"], sub["sat_lat"], s=2, color=colors[gid], alpha=0.25, zorder=3)
        dlon = dlat / np.cos(np.deg2rad(node["lat"]))
        ax.plot(node["lon"] + dlon * np.sin(theta), node["lat"] + dlat * np.cos(theta), color=colors[gid], lw=1.0, zorder=4)
        ax.plot(node["lon"], node["lat"], "*", color="black", ms=11, mec="white", zorder=5)
        ax.annotate(f"G{gid}", (node["lon"], node["lat"]), xytext=(4, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", zorder=6)
    lons, lats = np.r_[pairs["sat_lon"], nodes["lon"]], np.r_[pairs["sat_lat"], nodes["lat"]]
    ax.set_xlim(np.nanmin(lons) - 0.2, np.nanmax(lons) + 0.2)
    ax.set_ylim(np.nanmin(lats) - 0.2, np.nanmax(lats) + 0.2)

    ax_bar = fig.add_subplot(gs[0, 1])
    counts = pairs.groupby("goal").size().reindex([str(g) for g in goal_ids]).fillna(0)
    ax_bar.barh([f"G{g}" for g in goal_ids], counts.to_numpy(), color=[colors[g] for g in goal_ids])
    ax_bar.invert_yaxis()
    years = pd.to_datetime(pairs["time"])
    ax_bar.set(xlabel="satellite passes (one median Hs each)",
               title=f"{years.min():%Y}–{years.max():%Y} · radius {radius_km:g} km")
    ax_bar.grid(axis="x", alpha=0.3)
    return fig


def plot_calibration_metrics(metrics: pd.DataFrame, *, subset: str = "test", title: str | None = None) -> Figure:
    """
    Hindcast vs satellite per goal, raw and calibrated: bias, RMSE and slope.

    Parameters
    ----------
    metrics
        :func:`utils.calibration.calibration_metrics` (bias = hindcast − satellite).
    subset
        ``"test"`` (passes not used in the fit), ``"train"`` or ``"all"``.
    title
        Figure title.
    """
    table = metrics[metrics["set"] == subset]
    goals = sorted(table["goal"].astype(str).unique(), key=int)
    x = np.arange(len(goals))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric, ref in zip(axes, ("bias", "rmse", "slope"), (0.0, None, 1.0)):
        for k, (stage, color) in enumerate((("raw", "0.6"), ("calibrated", "tab:blue"))):
            vals = table[table["stage"] == stage].set_index("goal")[metric].reindex(goals).to_numpy()
            ax.bar(x + (k - 0.5) * 0.38, vals, width=0.38, color=color, label=stage)
        if ref is not None:
            ax.axhline(ref, color="k", lw=0.8)
        ax.set_xticks(x, [f"G{g}" for g in goals])
        ax.set_title(f"{metric} ({subset} passes)" + (" (m)" if metric != "slope" else ""))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend()
    n = int(table[table["stage"] == "raw"]["n"].sum())
    fig.suptitle(title or f"Hindcast vs satellite Hs per goal · {n} {subset} passes")
    fig.tight_layout()
    return fig
