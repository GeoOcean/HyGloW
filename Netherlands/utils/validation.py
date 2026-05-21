"""Validation plotting helpers (Hs/Tm/ThetaM time series, map, scatter and metrics)."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping, Sequence

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.plotting.old_scatter import validation_scatter
from matplotlib.gridspec import GridSpec

DEFAULT_VAR_MAP: dict[str, dict[str, str]] = {
    "observation": {"Hs": "Hm0", "Tm": "Tm-10", "ThetaM": "Th0"},
    "metamodel": {"Hs": "hs", "Tm": "tm01", "ThetaM": "dm"},
    "dynamic": {"Hs": "point_hm0", "Tm": "point_tp", "ThetaM": "point_wavdir"},
    "era5": {"Hs": "swh", "Tm": "mwp", "ThetaM": "mwd"},
    "whacs": {"Hs": "hs", "Tm": "t01", "ThetaM": "dir"},
}

DEFAULT_MODEL_STYLE: dict[str, dict[str, Any]] = {
    "observation": {"color": "darkblue", "label": "Buoy", "lw": 1.0, "alpha": 1.0},
    "metamodel": {"color": "darkred", "label": "Metamodel", "lw": 0.9, "alpha": 0.9},
    "dynamic": {"color": "darkgreen", "label": "Dynamic", "lw": 0.9, "alpha": 0.9},
    "era5": {"color": "orange", "label": "ERA5", "lw": 0.9, "alpha": 0.9},
    "whacs": {"color": "purple", "label": "WHACS", "lw": 0.9, "alpha": 0.9},
}

DEFAULT_DEPTH_LEVELS: list[float] = [-40, -20, -10, 0]
DEFAULT_MAP_EXTENT: list[float] = [2.4, 7.1, 51.3, 54.1]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


# def coincident_non_values(
#     array1: np.ndarray, array2: np.ndarray
# ) -> tuple[np.ndarray, np.ndarray]:
#     array1 = np.asarray(array1).astype(float)
#     array2 = np.asarray(array2).astype(float)
#     mask = ~np.isnan(array1) & ~np.isnan(array2)
#     return array1[mask], array2[mask]


def _theta_to_360(values: np.ndarray) -> np.ndarray:
    return np.mod(np.asarray(values, dtype=float), 360.0)


def _circular_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest signed angular difference (a - b) wrapped to [-180, 180]."""

    diff = (
        np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0
    ) % 360.0 - 180.0
    return diff


def _compute_metrics(
    obs: np.ndarray, model: np.ndarray, circular: bool = False
) -> dict[str, float]:
    """Compute basic validation metrics for paired (observation, model) arrays."""

    if circular:
        obs_mod = _theta_to_360(obs)
        model_mod = _theta_to_360(model)
        x, y = coincident_non_values(obs_mod, model_mod)
        if x.size == 0:
            return {"n": 0, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
        diff = _circular_difference(y, x)
        rmse = float(np.sqrt(np.mean(diff**2)))
        bias = float(np.mean(diff))
        # Circular correlation (Fisher–Lee).
        x_rad = np.deg2rad(x)
        y_rad = np.deg2rad(y)
        x_mean = np.arctan2(np.mean(np.sin(x_rad)), np.mean(np.cos(x_rad)))
        y_mean = np.arctan2(np.mean(np.sin(y_rad)), np.mean(np.cos(y_rad)))
        num = np.sum(np.sin(x_rad - x_mean) * np.sin(y_rad - y_mean))
        den = np.sqrt(
            np.sum(np.sin(x_rad - x_mean) ** 2) * np.sum(np.sin(y_rad - y_mean) ** 2)
        )
        corr = float(num / den) if den > 0 else np.nan
    else:
        x, y = coincident_non_values(obs, model)
        if x.size == 0:
            return {"n": 0, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
        rmse = float(np.sqrt(np.mean((y - x) ** 2)))
        bias = float(np.mean(y - x))
        if x.size > 1 and np.std(x) > 0 and np.std(y) > 0:
            corr = float(np.corrcoef(x, y)[0, 1])
        else:
            corr = np.nan
    return {"n": int(x.size), "rmse": rmse, "bias": bias, "corr": corr}


# def _density_scatter(
#     ax: plt.Axes,
#     x: np.ndarray,
#     y: np.ndarray,
#     cmap: str = "jet",
#     point_size: int = 8,
# ) -> None:
#     """Scatter plot coloured by point density (histogram-based, robust)."""

#     if x.size == 0:
#         return
#     bins = max(20, int(np.sqrt(x.size) / 2))
#     hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
#     x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, hist.shape[0] - 1)
#     y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, hist.shape[1] - 1)
#     density = hist[x_idx, y_idx]
#     order = np.argsort(density)
#     ax.scatter(
#         x[order],
#         y[order],
#         c=density[order],
#         cmap=cmap,
#         s=point_size,
#         edgecolor="none",
#     )


# def _plot_validation_scatter(
#     ax: plt.Axes,
#     x: np.ndarray,
#     y: np.ndarray,
#     xlabel: str,
#     ylabel: str,
#     title: str = "",
#     circular: bool = False,
# ) -> None:
#     """Density scatter with 1:1 line and a small metrics annotation."""

#     x_arr, y_arr = coincident_non_values(x, y)
#     if x_arr.size == 0:
#         ax.set_title(f"{title}\n(no overlapping non-NaN values)")
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         return

#     _density_scatter(ax, x_arr, y_arr)

#     lim_lo = float(min(np.nanmin(x_arr), np.nanmin(y_arr)))
#     lim_hi = float(max(np.nanmax(x_arr), np.nanmax(y_arr)))
#     ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", lw=0.8, alpha=0.6)
#     ax.set_xlim(lim_lo, lim_hi)
#     ax.set_ylim(lim_lo, lim_hi)
#     ax.set_aspect("equal", adjustable="box")
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.grid(True, alpha=0.3)
#     if title:
#         ax.set_title(title, fontsize=10)

#     metrics = _compute_metrics(x_arr, y_arr, circular=circular)
#     text = (
#         f"n = {metrics['n']}\n"
#         f"RMSE = {metrics['rmse']:.2f}\n"
#         f"BIAS = {metrics['bias']:.2f}\n"
#         f"corr = {metrics['corr']:.2f}"
#     )
#     ax.text(
#         0.03,
#         0.97,
#         text,
#         ha="left",
#         va="top",
#         transform=ax.transAxes,
#         fontsize=8,
#         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
#     )


def coincident_non_values(array1, array2):
    non_nan_values = np.where(~np.isnan(array1) & ~np.isnan(array2))[0]
    return array1[non_nan_values], array2[non_nan_values]


def _plot_validation_scatter(ax, x, y, xlabel, ylabel, title):
    x_vals, y_vals = coincident_non_values(x, y)
    if x_vals.size == 0:
        ax.set_title(f"{title}\n(no overlapping non-NaN values)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    validation_scatter(
        axs=ax,
        x=x_vals,
        y=y_vals,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        cmap="rainbow",
    )


# ---------------------------------------------------------------------------
# Bathymetry / mesh helpers
# ---------------------------------------------------------------------------


def _split_quads_to_triangles(face_nodes: np.ndarray) -> np.ndarray:
    """Convert FM quad/triangle face_nodes (1-indexed, NaN-padded) into a (n, 3) int array."""

    n0 = face_nodes[:, 0]
    n1 = face_nodes[:, 1]
    n2 = face_nodes[:, 2]
    n3 = face_nodes[:, 3] if face_nodes.shape[1] > 3 else np.full_like(n0, np.nan)

    tri1 = np.stack([n0, n1, n2], axis=1)
    keep1 = ~np.isnan(tri1).any(axis=1)
    tri1 = tri1[keep1].astype(np.int64) - 1

    tri2 = np.stack([n0, n2, n3], axis=1)
    keep2 = ~np.isnan(tri2).any(axis=1)
    tri2 = tri2[keep2].astype(np.int64) - 1

    return np.vstack([tri1, tri2])


@functools.lru_cache(maxsize=4)
def _load_mesh_arrays(path: str) -> dict[str, np.ndarray]:
    ds = xr.open_dataset(path)
    return {
        "x": ds.mesh2d_node_x.values,
        "y": ds.mesh2d_node_y.values,
        "z": ds.mesh2d_node_z.values,
        "triangles": _split_quads_to_triangles(ds.mesh2d_face_nodes.values),
    }


def load_case_mesh(
    path: str | Path, extent: Sequence[float] | None = None
) -> tuple[mtri.Triangulation, np.ndarray]:
    """Load an FM mesh, optionally masking triangles outside ``extent``.

    Triangles touching nodes with NaN depths are always masked so the result can
    be passed straight into ``ax.tricontour``.
    """

    mesh = _load_mesh_arrays(str(path))
    triangulation = mtri.Triangulation(
        mesh["x"], mesh["y"], triangles=mesh["triangles"]
    )
    depth = mesh["z"]

    bad_depth = np.isnan(depth)
    triangle_has_nan = bad_depth[mesh["triangles"]].any(axis=1)

    mask = triangle_has_nan.copy()
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        tx = mesh["x"][mesh["triangles"]]
        ty = mesh["y"][mesh["triangles"]]
        in_extent = (
            (tx.max(axis=1) >= lon_min)
            & (tx.min(axis=1) <= lon_max)
            & (ty.max(axis=1) >= lat_min)
            & (ty.min(axis=1) <= lat_max)
        )
        mask = mask | ~in_extent
    triangulation.set_mask(mask)

    safe_depth = np.where(bad_depth, 0.0, depth)
    return triangulation, safe_depth


# ---------------------------------------------------------------------------
# Data utilities (xarray helpers)
# ---------------------------------------------------------------------------


def _get_data_array(dataset, var_name, selector=None):
    """Return a DataArray for ``var_name``, optionally subsetted via xarray .isel."""

    data = dataset[var_name]
    if selector:
        data = data.isel(**selector)
    return data


def _aligned_values(
    obs: xr.DataArray, model: xr.DataArray, time_dim: str = "time"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align two time-series DataArrays on the time dim and return ``(t, x_obs, y_model)``."""

    if time_dim in obs.dims and time_dim in model.dims:
        a, b = xr.align(obs, model, join="inner")
        return a[time_dim].values, np.asarray(a.values), np.asarray(b.values)
    n = min(obs.size, model.size)
    return np.arange(n), np.asarray(obs.values)[:n], np.asarray(model.values)[:n]


def _scalar(value) -> Any:
    """Return a python scalar from a numpy/xarray 0-d array."""

    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    flat = arr.reshape(-1)
    return flat[0].item() if flat.size else None


def _buoys_to_dataframe(buoys) -> pd.DataFrame:
    """Coerce a buoy collection (DataFrame or xr.Dataset with 'buoy' dim) to DataFrame."""

    if buoys is None:
        return pd.DataFrame()
    if isinstance(buoys, pd.DataFrame):
        return buoys.copy()
    if isinstance(buoys, (xr.Dataset, xr.DataArray)):
        out: dict[str, np.ndarray] = {}
        for name in ("buoy", "lon", "lat", "closest_goal", "depth"):
            if name in buoys.coords or (
                isinstance(buoys, xr.Dataset) and name in buoys.data_vars
            ):
                values = np.asarray(buoys[name].values)
                if values.ndim > 1:
                    with np.errstate(all="ignore"):
                        values = np.nanmean(values, axis=tuple(range(values.ndim - 1)))
                out[name] = values
        return pd.DataFrame(out)
    return pd.DataFrame(buoys)


def _goals_to_dataframe(goals) -> pd.DataFrame:
    if goals is None:
        return pd.DataFrame()
    if isinstance(goals, pd.DataFrame):
        df = goals.copy()
    elif isinstance(goals, (xr.Dataset, xr.DataArray)):
        df = pd.DataFrame(
            {name: np.asarray(goals[name].values) for name in goals.coords}
        )
    else:
        df = pd.DataFrame(goals)
    if "lon" in df.columns and "longitude" not in df.columns:
        df = df.rename(columns={"lon": "longitude"})
    if "lat" in df.columns and "latitude" not in df.columns:
        df = df.rename(columns={"lat": "latitude"})
    return df


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


def plot_map(
    ax,
    buoys=None,
    goals=None,
    extra_points: Sequence[Mapping[str, Any]] | None = None,
    highlight_buoy: int | str | None = None,
    highlight_goal: int | str | None = None,
    bathymetry_path: str | Path | None = None,
    depth_levels: Sequence[float] | None = None,
    extent: Sequence[float] | None = None,
    title: str | None = None,
    show_coastlines: bool = True,
    show_gridlines: bool = True,
    legend: bool = True,
) -> None:
    """Plot a validation map with buoys, goal points, optional extras and depth contours.

    Parameters
    ----------
    ax : cartopy.GeoAxes
        Cartopy axes (created with ``projection=ccrs.PlateCarree()``).
    buoys : pandas.DataFrame | xarray.Dataset
        Buoy table with at least ``lon``, ``lat`` (and ideally ``buoy``).
    goals : pandas.DataFrame | xarray.Dataset
        Goal table with ``longitude``, ``latitude`` (and ``goal``).
    extra_points : list[dict], optional
        Additional point layers; each entry: ``{"df", "label", "color", "marker", "lon_col", "lat_col"}``.
    highlight_buoy, highlight_goal : id, optional
        Buoy / goal id to mark with a star / cross.
    bathymetry_path : path-like, optional
        D-Flow FM ``*_net.nc`` mesh used to draw depth contours.
    depth_levels : sequence[float], optional
        Contour levels (negative = below sea level).
    extent : [lon_min, lon_max, lat_min, lat_max], optional
    """

    if extent is None:
        extent = DEFAULT_MAP_EXTENT
    if depth_levels is None:
        depth_levels = DEFAULT_DEPTH_LEVELS

    ax.set_extent(extent, crs=ccrs.PlateCarree())

    if show_coastlines:
        ax.add_feature(cfeature.OCEAN, facecolor="#dceaf2", zorder=0)
        ax.add_feature(cfeature.LAND, facecolor="#f0e7d8", zorder=1)
        ax.add_feature(cfeature.COASTLINE, lw=0.6, zorder=4)
        ax.add_feature(cfeature.BORDERS, lw=0.3, ls="--", zorder=4)

    if show_gridlines:
        gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4, zorder=5)
        gl.top_labels = False
        gl.right_labels = False

    if bathymetry_path is not None:
        try:
            triangulation, depth = load_case_mesh(bathymetry_path, extent=extent)
            ct = ax.tricontour(
                triangulation,
                depth,
                levels=list(depth_levels),
                colors="steelblue",
                linewidths=0.5,
                alpha=0.7,
                zorder=2,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(
                ct,
                ct.levels,
                inline=True,
                fontsize=7,
                fmt={lev: f"{int(round(-lev))} m" for lev in ct.levels},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[plot_map] Skipping depth contours from {bathymetry_path}: {exc}")

    buoys_df = _buoys_to_dataframe(buoys)
    if not buoys_df.empty and {"lon", "lat"}.issubset(buoys_df.columns):
        ax.scatter(
            buoys_df["lon"],
            buoys_df["lat"],
            s=40,
            marker="o",
            facecolor="white",
            edgecolor="darkblue",
            lw=1.0,
            label="Buoys",
            zorder=10,
            transform=ccrs.PlateCarree(),
        )
        if highlight_buoy is not None and "buoy" in buoys_df.columns:
            sub = buoys_df[buoys_df["buoy"].astype(str) == str(highlight_buoy)]
            if not sub.empty:
                ax.scatter(
                    sub["lon"],
                    sub["lat"],
                    s=220,
                    marker="*",
                    color="red",
                    edgecolor="black",
                    lw=0.8,
                    label=f"Buoy {highlight_buoy}",
                    zorder=12,
                    transform=ccrs.PlateCarree(),
                )
                depth_str = ""
                if "depth" in sub.columns and np.isfinite(sub["depth"].iloc[0]):
                    depth_str = f"  ({-float(sub['depth'].iloc[0]):.0f} m)"
                ax.text(
                    float(sub["lon"].iloc[0]) + 0.05,
                    float(sub["lat"].iloc[0]) + 0.05,
                    f"{highlight_buoy}{depth_str}",
                    fontsize=8,
                    color="darkred",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                    transform=ccrs.PlateCarree(),
                    zorder=13,
                )

    goals_df = _goals_to_dataframe(goals)
    if not goals_df.empty and {"longitude", "latitude"}.issubset(goals_df.columns):
        ax.scatter(
            goals_df["longitude"],
            goals_df["latitude"],
            s=55,
            marker="s",
            facecolor="orange",
            edgecolor="darkorange",
            lw=0.8,
            alpha=0.85,
            label="Goals",
            zorder=10,
            transform=ccrs.PlateCarree(),
        )
        if highlight_goal is not None and "goal" in goals_df.columns:
            sub = goals_df[goals_df["goal"].astype(int) == int(highlight_goal)]
            if not sub.empty:
                ax.scatter(
                    sub["longitude"],
                    sub["latitude"],
                    s=180,
                    marker="P",
                    color="purple",
                    edgecolor="black",
                    lw=0.8,
                    label=f"Goal {int(highlight_goal)}",
                    zorder=12,
                    transform=ccrs.PlateCarree(),
                )

    if extra_points:
        for item in extra_points:
            df = item.get("df")
            if df is None or len(df) == 0:
                continue
            lon_col = item.get("lon_col") or (
                "lon"
                if "lon" in df.columns
                else "longitude"
                if "longitude" in df.columns
                else "station_x_coordinate"
            )
            lat_col = item.get("lat_col") or (
                "lat"
                if "lat" in df.columns
                else "latitude"
                if "latitude" in df.columns
                else "station_y_coordinate"
            )
            ax.scatter(
                df[lon_col],
                df[lat_col],
                s=item.get("size", 35),
                marker=item.get("marker", "D"),
                color=item.get("color", "green"),
                alpha=item.get("alpha", 0.7),
                label=item.get("label"),
                zorder=9,
                transform=ccrs.PlateCarree(),
            )

    if title is not None:
        ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# All-models validation figure
# ---------------------------------------------------------------------------


def _extract_var(dataset, var_map: Mapping[str, str], key: str) -> xr.DataArray | None:
    """Return ``dataset[var_map[key]]`` if the variable exists, else ``None``."""

    if dataset is None:
        return None
    name = var_map.get(key)
    if not name or name not in getattr(dataset, "variables", {}):
        return None
    return dataset[name]


def validate_all_models(
    observation_data,
    metamodel_data,
    dynamic_data,
    era5_data,
    whacs_data,
    *,
    var_map: Mapping[str, Mapping[str, str]] | None = None,
    buoys=None,
    goals=None,
    extra_points: Sequence[Mapping[str, Any]] | None = None,
    bathymetry_path: str | Path | None = None,
    map_extent: Sequence[float] | None = None,
    time_window: tuple | None = None,
    figsize: tuple[float, float] = (16, 11),
    title: str | None = None,
) -> plt.Figure:
    """Validate all models for a single buoy in one figure.

    The figure layout follows the original ``plot_buoy_validation_figure`` style:
    four observation-vs-model scatter panels (top-left), a map (top-right) and a
    full-width Hs time-series at the bottom with all five datasets on it.

    Parameters
    ----------
    observation_data : xarray.Dataset
        Single buoy time-series with at least ``Hm0`` and lon/lat scalars.
    metamodel_data, dynamic_data, era5_data, whacs_data : xarray.Dataset
        Single point/site selection of each model dataset.
    var_map : dict, optional
        Override of ``DEFAULT_VAR_MAP`` (per dataset name: ``Hs``/``Tm``/``ThetaM``).
    buoys, goals, extra_points, bathymetry_path, map_extent
        Forwarded to :func:`plot_map`.
    """

    vmap = {**DEFAULT_VAR_MAP, **(var_map or {})}

    datasets = {
        "observation": observation_data,
        "metamodel": metamodel_data,
        "dynamic": dynamic_data,
        "era5": era5_data,
        "whacs": whacs_data,
    }

    obs_hs = _extract_var(observation_data, vmap["observation"], "Hs")
    if obs_hs is None:
        raise ValueError(
            "observation_data must contain an Hs variable (default name 'Hm0')."
        )

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=3, ncols=4, figure=fig, hspace=0.35, wspace=0.35)

    ax_ts = fig.add_subplot(gs[2, :])
    ax_map = fig.add_subplot(gs[:2, 2:], projection=ccrs.PlateCarree())
    scatter_axes = {
        "metamodel": fig.add_subplot(gs[0, 0]),
        "dynamic": fig.add_subplot(gs[0, 1]),
        "era5": fig.add_subplot(gs[1, 0]),
        "whacs": fig.add_subplot(gs[1, 1]),
    }

    obs_style = DEFAULT_MODEL_STYLE["observation"]
    ax_ts.plot(
        obs_hs[obs_hs.dims[0]].values if obs_hs.dims else np.arange(obs_hs.size),
        obs_hs.values,
        color=obs_style["color"],
        lw=obs_style["lw"],
        label=obs_style["label"],
        alpha=obs_style["alpha"],
    )
    for key in ("metamodel", "dynamic", "era5", "whacs"):
        model_hs = _extract_var(datasets[key], vmap[key], "Hs")
        if model_hs is None:
            continue
        style = DEFAULT_MODEL_STYLE[key]
        _t, _x, _y = _aligned_values(obs_hs, model_hs)
        ax_ts.plot(
            _t,
            _y,
            color=style["color"],
            lw=style["lw"],
            alpha=style["alpha"],
            label=style["label"],
        )

    ax_ts.legend(loc="upper right", ncol=5, fontsize=9)
    if time_window is not None:
        ax_ts.set_xlim(time_window)
    ax_ts.set_title("")
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Hs [m]")
    ax_ts.grid(True, alpha=0.3)

    for key, ax in scatter_axes.items():
        model_hs = _extract_var(datasets[key], vmap[key], "Hs")
        if model_hs is None:
            ax.set_title(f"{DEFAULT_MODEL_STYLE[key]['label']} (no data)")
            continue
        _, obs_vals, mod_vals = _aligned_values(obs_hs, model_hs)
        _plot_validation_scatter(
            ax,
            obs_vals,
            mod_vals,
            "Observations [m]",
            f"{DEFAULT_MODEL_STYLE[key]['label']} [m]",
            title=f"Hs - {DEFAULT_MODEL_STYLE[key]['label']}",
        )

    buoy_id = (
        _scalar(observation_data["buoy"]) if "buoy" in observation_data.coords else None
    )
    closest_goal = (
        _scalar(observation_data["closest_goal"])
        if "closest_goal" in observation_data.coords
        else None
    )
    plot_map(
        ax_map,
        buoys=buoys,
        goals=goals,
        extra_points=extra_points,
        highlight_buoy=buoy_id,
        highlight_goal=closest_goal,
        bathymetry_path=bathymetry_path,
        extent=map_extent,
        title=None if title else f"Buoy {buoy_id} / Goal {closest_goal}",
    )

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hs/Tm/ThetaM series + map + info
# ---------------------------------------------------------------------------


def _metrics_panel(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    title: str = "Metrics",
) -> None:
    """Render a small metrics table in a regular axes."""

    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")
    if not rows:
        ax.text(0.0, 0.5, "(no metrics)", fontsize=9)
        return

    headers = ["Var", "n", "RMSE", "BIAS", "corr"]
    cells = []
    for r in rows:
        cells.append(
            [
                r["var"],
                f"{r['metrics']['n']}",
                f"{r['metrics']['rmse']:.2f}",
                f"{r['metrics']['bias']:.2f}",
                f"{r['metrics']['corr']:.2f}",
            ]
        )

    table = ax.table(
        cellText=cells,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)


def _info_panel(
    ax: plt.Axes, rows: Sequence[tuple[str, str]], title: str = "Info"
) -> None:
    """Render free-form key/value information in a regular axes."""

    ax.axis("off")
    ax.set_title(title, fontsize=10, loc="left")
    if not rows:
        ax.text(0.0, 0.5, "(no info)", fontsize=9)
        return
    text_lines = [f"{key:<14s}  {value}" for key, value in rows]
    ax.text(
        0.0,
        0.95,
        "\n".join(text_lines),
        fontsize=9,
        family="monospace",
        va="top",
        ha="left",
    )


def _series_for_compare(
    dataset_a,
    dataset_b,
    var_map_a: Mapping[str, str],
    var_map_b: Mapping[str, str],
    selector_a: Mapping[str, Any] | None,
    selector_b: Mapping[str, Any] | None,
    time_dim: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Extract and align Hs/Tm/ThetaM time-series for two datasets."""

    required = ("Hs", "Tm", "ThetaM")
    for key in required:
        if key not in var_map_a or key not in var_map_b:
            raise ValueError(
                f"Missing variable mapping for '{key}'. Required keys: {required}."
            )

    series_a = {
        key: _get_data_array(dataset_a, var_map_a[key], selector=selector_a)
        for key in required
    }
    series_b = {
        key: _get_data_array(dataset_b, var_map_b[key], selector=selector_b)
        for key in required
    }

    if time_dim in series_a["Hs"].dims and time_dim in series_b["Hs"].dims:
        aligned_a = xr.Dataset(series_a)
        aligned_b = xr.Dataset(series_b)
        aligned_a, aligned_b = xr.align(aligned_a, aligned_b, join="inner")
        x_values = aligned_a[time_dim].values
        values_a = {key: aligned_a[key].values for key in required}
        values_b = {key: aligned_b[key].values for key in required}
    else:
        n = min(series_a["Hs"].size, series_b["Hs"].size)
        x_values = np.arange(n)
        values_a = {key: np.asarray(series_a[key].values)[:n] for key in required}
        values_b = {key: np.asarray(series_b[key].values)[:n] for key in required}

    values_a["ThetaM"] = _theta_to_360(values_a["ThetaM"])
    values_b["ThetaM"] = _theta_to_360(values_b["ThetaM"])
    return {"time": x_values, "a": values_a, "b": values_b}


def plot_series_validation_with_map(
    dataset_a,
    dataset_b,
    var_map_a: Mapping[str, str],
    var_map_b: Mapping[str, str],
    *,
    selector_a: Mapping[str, Any] | None = None,
    selector_b: Mapping[str, Any] | None = None,
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
    color_a: str = "blue",
    color_b: str = "red",
    time_dim: str = "time",
    title: str | None = None,
    figsize: tuple[float, float] = (16, 10),
    # Map kwargs
    buoys=None,
    goals=None,
    extra_points: Sequence[Mapping[str, Any]] | None = None,
    bathymetry_path: str | Path | None = None,
    map_extent: Sequence[float] | None = None,
    highlight_buoy: int | str | None = None,
    highlight_goal: int | str | None = None,
    # Extra info under the map
    info: Sequence[tuple[str, str]] | None = None,
    observation_data=None,
) -> plt.Figure:
    """Hs/Tm/ThetaM time-series compare with a map and metrics/info panel.

    Left column (3 rows): Hs, Tm, ThetaM time-series (``dataset_a`` vs ``dataset_b``).
    Right column (3 rows): map (rows 0-1), metrics table (row 2), info text (row 2).

    Parameters
    ----------
    dataset_a, dataset_b : xarray.Dataset
        Datasets to compare.
    var_map_a, var_map_b : dict
        Per-dataset variable mapping with keys ``Hs``, ``Tm``, ``ThetaM``.
    selector_a, selector_b : dict, optional
        xarray .isel selectors to pick one site per dataset.
    info : sequence[tuple[str, str]], optional
        Extra (label, value) pairs to render in the info box. If ``observation_data``
        has lon/lat/depth/closest_goal scalars, defaults are auto-populated.
    observation_data : xarray.Dataset, optional
        Optional reference for auto-filling the info box and highlighting the map.
    """

    extracted = _series_for_compare(
        dataset_a, dataset_b, var_map_a, var_map_b, selector_a, selector_b, time_dim
    )
    x_values = extracted["time"]
    values_a, values_b = extracted["a"], extracted["b"]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        nrows=3,
        ncols=2,
        figure=fig,
        width_ratios=[2.2, 1.3],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.35,
        wspace=0.25,
    )

    ax_hs = fig.add_subplot(gs[0, 0])
    ax_tp = fig.add_subplot(gs[1, 0], sharex=ax_hs)
    ax_th = fig.add_subplot(gs[2, 0], sharex=ax_hs)

    for ax, key, ylabel in (
        (ax_hs, "Hs", "Hs [m]"),
        (ax_tp, "Tm", "Tm [s]"),
        (ax_th, "ThetaM", "ThetaM [deg]"),
    ):
        if key == "ThetaM":
            ax.scatter(
                x_values,
                values_a[key],
                color=color_a,
                label=label_a,
                s=1,
            )
            ax.scatter(
                x_values,
                values_b[key],
                color=color_b,
                label=label_b,
                s=1,
            )
        else:
            ax.plot(x_values, values_a[key], color=color_a, lw=0.9, label=label_a)
            ax.plot(
                x_values,
                values_b[key],
                color=color_b,
                lw=0.9,
                alpha=0.85,
                label=label_b,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    ax_th.set_xlabel("Time" if isinstance(x_values[0], np.datetime64) else "Index")
    ax_th.set_ylim(0, 360)
    ax_th.set_yticks([0, 90, 180, 270, 360])
    ax_th.set_yticklabels(["N", "E", "S", "W", "N"])
    ax_hs.legend(loc="upper right", ncol=2, fontsize=9)

    # Right column: map (rows 0-1), metrics (row 2 first half), info (row 2 second half)
    ax_map = fig.add_subplot(gs[0:2, 1], projection=ccrs.PlateCarree())
    bottom_right_gs = gs[2, 1].subgridspec(1, 2, wspace=0.05)
    ax_metrics = fig.add_subplot(bottom_right_gs[0, 0])
    ax_info = fig.add_subplot(bottom_right_gs[0, 1])

    # Resolve highlight defaults from observation_data if provided.
    if observation_data is not None:
        if highlight_buoy is None and "buoy" in observation_data.coords:
            highlight_buoy = _scalar(observation_data["buoy"])
        if highlight_goal is None and "closest_goal" in observation_data.coords:
            highlight_goal = _scalar(observation_data["closest_goal"])

    plot_map(
        ax_map,
        buoys=buoys,
        goals=goals,
        extra_points=extra_points,
        highlight_buoy=highlight_buoy,
        highlight_goal=highlight_goal,
        bathymetry_path=bathymetry_path,
        extent=map_extent,
        title=None,
    )

    metric_rows = [
        {"var": "Hs", "metrics": _compute_metrics(values_a["Hs"], values_b["Hs"])},
        {"var": "Tm", "metrics": _compute_metrics(values_a["Tm"], values_b["Tm"])},
        {
            "var": "ThetaM",
            "metrics": _compute_metrics(
                values_a["ThetaM"], values_b["ThetaM"], circular=True
            ),
        },
    ]
    _metrics_panel(ax_metrics, metric_rows, title=f"{label_b} vs {label_a}")

    info_rows: list[tuple[str, str]] = []
    if observation_data is not None:
        if "buoy" in observation_data.coords:
            info_rows.append(("Buoy", f"{_scalar(observation_data['buoy'])}"))
        if "lon" in observation_data.coords and "lat" in observation_data.coords:
            lon = _scalar(observation_data["lon"])
            lat = _scalar(observation_data["lat"])
            info_rows.append(("Position", f"{lon:.3f}°E, {lat:.3f}°N"))
        if "depth" in observation_data.coords or (
            isinstance(observation_data, xr.Dataset)
            and "depth" in observation_data.data_vars
        ):
            depth_val = observation_data["depth"]
            if depth_val.ndim > 0:
                with np.errstate(all="ignore"):
                    depth_val = float(np.nanmean(depth_val.values))
            else:
                depth_val = _scalar(depth_val)
            if depth_val is not None and np.isfinite(depth_val):
                info_rows.append(("Depth", f"{-float(depth_val):.1f} m"))
        if "closest_goal" in observation_data.coords:
            info_rows.append(
                ("Closest goal", f"{_scalar(observation_data['closest_goal'])}")
            )
    if info:
        info_rows.extend(info)
    if not info_rows:
        info_rows = [("(no info)", "")]
    _info_panel(ax_info, info_rows, title="Site info")

    if title is not None:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig
