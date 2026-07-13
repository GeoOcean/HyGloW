"""Post-run SnapWave case diagnostics read directly from a case folder.

Layout: top row ``hs``, ``tp``, ``dir`` site panels; bottom row full-width domain
map. ``spr`` is omitted from the default layout (see :data:`TOP_VARS`) to keep a
compact 2×3 grid; forcing files still list ``spr`` for metamodel cases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from shapely.geometry import Polygon

from utils.config import display_title
from utils.viz import case_plots as cp

WAVE_VAR_ALIASES: dict[str, tuple[str, ...]] = {
    "hs": ("point_hm0", "hm0", "Hs", "hm"),
    "tp": ("point_tp", "tp", "Tm"),
    "dir": ("point_wavdir", "wd", "wavdir", "dir", "ThetaM"),
}

FORCING_FILES = {
    "hs": "hs.txt",
    "tp": "tp.txt",
    "dir": "dir.txt",
    "spr": "spr.txt",
    "wl": "wl.txt",
}

# Default site panels for case diagnostics (spr omitted — 3-panel + map layout).
TOP_VARS = ("hs", "tp", "dir")


@dataclass
class CasePlotStyle:
    """Styling config for SnapWave case diagnostic plots."""

    cmaps: dict[str, str] = field(
        default_factory=lambda: {
            "hs": "viridis",
            "tp": "plasma",
            "dir": "twilight_shifted",
        }
    )
    metamodel_cmaps: dict[str, str] = field(
        default_factory=lambda: {
            "hs": "Blues_r",
            "tp": "plasma",
            "dir": "twilight_shifted",
        }
    )
    metamodel_vlims: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"hs": (0.0, 1.0)}
    )
    site_marker_size: float = 28
    top_vars: tuple[str, ...] = TOP_VARS

    def cmap_for(self, var: str, *, mode: str = "dynamic") -> str:
        if mode == "metamodel":
            return self.metamodel_cmaps.get(var, self.cmaps.get(var, "viridis"))
        return self.cmaps.get(var, "viridis")


DEFAULT_CASE_PLOT_STYLE = CasePlotStyle()


@dataclass
class CaseContext:
    """Paths and metadata discovered inside a SnapWave case directory."""

    case_dir: Path
    snapwave_inp: dict[str, str]
    gridfile: Path | None
    map_file: Path | None
    sites_file: Path | None
    boundary_file: Path
    enclosure_file: Path
    goal_id: int | None


def resolve_wave_variable(
    ds: xr.Dataset,
    name: str,
    *,
    time_index: int = 0,
    site_dim: str | None = None,
) -> tuple[str, xr.DataArray]:
    """Resolve canonical wave variable inside a SnapWave output dataset.

    Supports ``hs``, ``tp``, ``dir`` for site/map panels. ``spr`` is available
    in forcing files but not in the default :data:`TOP_VARS` layout.
    """
    key = name.strip().lower()
    if key not in WAVE_VAR_ALIASES:
        raise KeyError(
            f"Unknown canonical variable {name!r}; expected one of "
            f"{sorted(WAVE_VAR_ALIASES)}"
        )

    candidates = (key, *WAVE_VAR_ALIASES[key])
    var_name = next((c for c in candidates if c in ds.data_vars), None)
    if var_name is None:
        raise KeyError(
            f"Could not resolve {name!r} in dataset; tried {candidates}. "
            f"Available: {list(ds.data_vars)}"
        )

    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=time_index)
    if da.ndim > 1:
        extra = [d for d in da.dims if d != "time"]
        if not (site_dim and site_dim in da.dims) and len(extra) == 1:
            da = da.squeeze(extra[0], drop=True)
    return var_name, da


def _parse_snapwave_inp(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        entries[key.strip()] = value.strip()
    return entries


def _read_xy_table(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            rows.append([float(parts[0]), float(parts[1])])
    if not rows:
        raise ValueError(f"No coordinates found in {path}")
    return np.asarray(rows, dtype=float)


def read_enclosure_polygon(case_dir: str | Path) -> Polygon:
    coords = _read_xy_table(Path(case_dir) / "enclosure.txt")
    if coords.ndim == 1:
        coords = coords.reshape(1, 2)
    return Polygon(coords)


def read_boundary_nodes(case_dir: str | Path) -> np.ndarray:
    return _read_xy_table(Path(case_dir) / "boundary.txt")


def read_forcing_table(path: str | Path) -> pd.DataFrame:
    data = np.loadtxt(path, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    columns = [f"b{i}" for i in range(data.shape[1] - 1)]
    return pd.DataFrame(data[:, 1:], index=data[:, 0], columns=columns)


def infer_goal_from_path(case_dir: str | Path) -> int | None:
    for part in reversed(Path(case_dir).parts):
        if re.fullmatch(r"\d{2}", part):
            return int(part)
    return None


def _resolve_map_file(case_dir: Path, inp: dict[str, str]) -> Path | None:
    """Return domain map NetCDF for a case directory.

    Uses ``map_file`` from ``snapwave.inp`` when set and present; otherwise
    falls back to ``output_map.nc`` in the case folder.
    """
    map_name = (inp.get("map_file") or "").strip()
    if map_name:
        candidate = case_dir / map_name
        if candidate.exists():
            return candidate
    default = case_dir / "output_map.nc"
    return default if default.exists() else None


def load_case_context(case_dir: str | Path) -> CaseContext:
    case_dir = Path(case_dir)
    inp = _parse_snapwave_inp(case_dir / "snapwave.inp")
    grid = Path(inp["gridfile"]) if inp.get("gridfile") else None
    sites_name = inp.get("his_file", "output_sites.nc")
    sites_file = case_dir / sites_name if sites_name else None
    return CaseContext(
        case_dir=case_dir,
        snapwave_inp=inp,
        gridfile=grid if grid and grid.exists() else None,
        map_file=_resolve_map_file(case_dir, inp),
        sites_file=sites_file if sites_file and sites_file.exists() else None,
        boundary_file=case_dir / "boundary.txt",
        enclosure_file=case_dir / "enclosure.txt",
        goal_id=infer_goal_from_path(case_dir),
    )


def _split_quads_to_triangles(face_nodes: np.ndarray) -> np.ndarray:
    n0, n1, n2 = face_nodes[:, 0], face_nodes[:, 1], face_nodes[:, 2]
    n3 = face_nodes[:, 3] if face_nodes.shape[1] > 3 else np.full_like(n0, np.nan)

    tri1 = np.stack([n0, n1, n2], axis=1)
    tri1 = tri1[~np.isnan(tri1).any(axis=1)].astype(np.int64) - 1

    tri2 = np.stack([n0, n2, n3], axis=1)
    tri2 = tri2[~np.isnan(tri2).any(axis=1)].astype(np.int64) - 1
    return tri1 if tri2.size == 0 else np.vstack([tri1, tri2])


def _map_triangulation(map_ds: xr.Dataset) -> mtri.Triangulation:
    x = map_ds["mesh2d_node_x"].values.astype(float)
    y = map_ds["mesh2d_node_y"].values.astype(float)
    triangles = _split_quads_to_triangles(map_ds["mesh2d_face_nodes"].values)
    tri = mtri.Triangulation(x, y, triangles)
    if "mesh2d_node_z" in map_ds:
        bad = np.isnan(map_ds["mesh2d_node_z"].values.astype(float))
        tri.set_mask(bad[triangles].any(axis=1))
    return tri


def _forcing_row(case_dir: Path, var: str, time_index: int) -> tuple[float, np.ndarray]:
    table = read_forcing_table(case_dir / FORCING_FILES[var])
    idx = min(time_index, len(table) - 1)
    return float(table.index[idx]), table.iloc[idx].to_numpy(float)


def _active_boundary_index(hs_values: np.ndarray) -> int:
    idx = int(np.argmax(hs_values))
    if hs_values[idx] <= 0:
        raise ValueError("No active boundary found (hs row has no positive entry)")
    return idx


def _finite_vmin_vmax(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return tuple(np.nanpercentile(finite, [2, 98]))  # type: ignore[return-value]


def _expand_tiny_range(vmin: float, vmax: float) -> tuple[float, float]:
    """Avoid degenerate colorbars when a field is nearly constant."""
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return 0.0, 1.0
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    span = vmax - vmin
    scale = max(abs(vmin), abs(vmax), 1.0)
    if span < max(1e-6, 1e-4 * scale):
        mid = 0.5 * (vmin + vmax)
        pad = max(0.5, 0.05 * abs(mid)) if mid != 0 else 0.5
        return mid - pad, mid + pad
    return vmin, vmax


def _resolve_plot_scaling(
    var: str,
    values: np.ndarray,
    *,
    mode: str,
    style: CasePlotStyle,
) -> tuple[float, float, str]:
    """Return vmin, vmax, and cmap for one plotted field."""
    if mode == "metamodel" and var in style.metamodel_vlims:
        vmin, vmax = style.metamodel_vlims[var]
        return vmin, vmax, style.cmap_for(var, mode=mode)

    vmin, vmax = _finite_vmin_vmax(values)
    vmin, vmax = _expand_tiny_range(vmin, vmax)
    return vmin, vmax, style.cmap_for(var, mode=mode)


def _add_colorbar(
    ax: Axes, mappable, vmin: float, vmax: float, *, fraction: float
) -> None:
    cb = plt.colorbar(mappable, ax=ax, fraction=fraction, pad=0.02)
    span = vmax - vmin
    if span <= 0:
        cb.set_ticks([vmin])
    elif span < 1.0:
        cb.locator = mticker.MaxNLocator(nbins=5)
        cb.update_ticks()
    else:
        cb.locator = mticker.MaxNLocator(nbins=6)
        cb.update_ticks()
    cb.formatter = mticker.FormatStrFormatter("%.2f")
    cb.update_ticks()


def _build_case_title(
    ctx: CaseContext,
    *,
    mode: str,
    time_index: int,
    has_map: bool,
    user_title: str | None,
) -> tuple[str, str]:
    """Return main title and subtitle with forcing summary."""
    goal = f"G{ctx.goal_id}" if ctx.goal_id is not None else "G?"
    main = user_title or display_title(
        f"SnapWave {mode} · {goal} · case {ctx.case_dir.name}"
    )

    parts: list[str] = []
    if mode == "metamodel":
        _, hs_vals = _forcing_row(ctx.case_dir, "hs", time_index)
        active = _active_boundary_index(hs_vals) + 1
        parts.append(f"hs=1 @bnd{active}")
        for var in ("tp", "dir", "spr", "wl"):
            if (ctx.case_dir / FORCING_FILES[var]).exists():
                _, vals = _forcing_row(ctx.case_dir, var, time_index)
                unit = "°" if var in {"dir", "spr"} else ("m" if var == "wl" else "s")
                parts.append(f"{var}={vals[0]:.3g}{unit}")
    else:
        for var in ("hs", "tp", "dir", "spr", "wl"):
            if (ctx.case_dir / FORCING_FILES[var]).exists():
                t, vals = _forcing_row(ctx.case_dir, var, time_index)
                parts.append(
                    f"{var}=[{np.nanmin(vals):.3g}, {np.nanmax(vals):.3g}] @t={t:g}s"
                )

    parts.append("map=yes" if has_map else "map=no")
    subtitle = "  ·  ".join(parts)
    return main, subtitle


def _plot_site_field(
    ax: Axes,
    sites_ds: xr.Dataset,
    var: str,
    *,
    time_index: int,
    style: CasePlotStyle,
    extent: tuple[float, float, float, float],
    mode: str,
) -> None:
    _, values = resolve_wave_variable(sites_ds, var, time_index=time_index)
    lon = sites_ds["station_x"].values.astype(float)
    lat = sites_ds["station_y"].values.astype(float)
    vals = np.asarray(values.values, dtype=float)
    finite = np.isfinite(vals)
    vmin, vmax, cmap = _resolve_plot_scaling(var, vals, mode=mode, style=style)
    sc = ax.scatter(
        lon[finite],
        lat[finite],
        c=vals[finite],
        s=style.site_marker_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.25,
    )
    ax.set_title(var, fontsize=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=8)
    _add_colorbar(ax, sc, vmin, vmax, fraction=0.05)


def _plot_map_tripcolor(
    ax: Axes,
    map_ds: xr.Dataset,
    var: str,
    *,
    time_index: int,
    style: CasePlotStyle,
    mode: str,
):
    tri = _map_triangulation(map_ds)
    _, values = resolve_wave_variable(
        map_ds, var, time_index=time_index, site_dim="nmesh2d_node"
    )
    vals = np.asarray(values.values, dtype=float)
    if not np.isfinite(vals).any():
        ax.text(0.5, 0.5, "No finite map values", ha="center", va="center")
        return None
    vmin, vmax, cmap = _resolve_plot_scaling(var, vals, mode=mode, style=style)
    mesh = ax.tripcolor(
        tri,
        vals,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
        zorder=1,
    )
    _add_colorbar(ax, mesh, vmin, vmax, fraction=0.025)
    return mesh


def _fit_map_extent(
    ax: Axes, extent: tuple[float, float, float, float]
) -> None:
    """Fill the full-width map panel while keeping equal geographic scale."""
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="datalim")


def _draw_enclosure(ax: Axes, enclosure: Polygon) -> None:
    gpd.GeoDataFrame(geometry=[enclosure], crs="EPSG:4326").boundary.plot(
        ax=ax,
        color="#d62728",
        linestyle="--",
        linewidth=1.3,
        label="enclosure",
    )


def _draw_boundary_nodes(
    ax: Axes,
    boundary: np.ndarray,
    *,
    active_index: int | None = None,
    colors: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
) -> None:
    ax.scatter(
        boundary[:, 0],
        boundary[:, 1],
        c=colors if colors is not None else "#888888",
        s=sizes if sizes is not None else 28,
        edgecolors="white",
        linewidths=0.4,
        zorder=5,
        label="boundary nodes",
    )
    if active_index is not None:
        ax.scatter(
            boundary[active_index, 0],
            boundary[active_index, 1],
            c="#d62728",
            s=120,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="active boundary",
        )


def _draw_dir_arrow(
    ax: Axes,
    lon: float,
    lat: float,
    dir_deg: float,
    *,
    extent: tuple[float, float, float, float],
) -> None:
    theta = np.deg2rad(float(dir_deg))
    length = 0.08 * max(extent[1] - extent[0], extent[3] - extent[2])
    ax.annotate(
        "",
        xy=(lon, lat),
        xytext=(lon + np.sin(theta) * length, lat + np.cos(theta) * length),
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=2.0,
            alpha=0.95,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=22,
        ),
        zorder=7,
    )


def _plot_domain_panel(
    ax: Axes,
    ctx: CaseContext,
    *,
    map_ds: xr.Dataset | None,
    sites_ds: xr.Dataset | None,
    time_index: int,
    mode: str,
    map_variable: str,
    style: CasePlotStyle,
    extent: tuple[float, float, float, float],
) -> None:
    enclosure = read_enclosure_polygon(ctx.case_dir)
    boundary = read_boundary_nodes(ctx.case_dir)

    if map_ds is not None:
        _plot_map_tripcolor(
            ax, map_ds, map_variable, time_index=time_index, style=style, mode=mode
        )
        ax.set_title(f"domain map · {map_variable}", fontsize=11)
    else:
        cp.plot_base_map(
            ax=ax, title="domain overview", show_topobathy=True, show_aoi=True
        )

    _draw_enclosure(ax, enclosure)

    if mode == "metamodel":
        _, hs_vals = _forcing_row(ctx.case_dir, "hs", time_index)
        active = _active_boundary_index(hs_vals)
        _draw_boundary_nodes(ax, boundary, active_index=active)
        _, dir_vals = _forcing_row(ctx.case_dir, "dir", time_index)
        _draw_dir_arrow(
            ax,
            boundary[active, 0],
            boundary[active, 1],
            dir_vals[active],
            extent=extent,
        )
    else:
        _, hs_vals = _forcing_row(ctx.case_dir, "hs", time_index)
        hs_norm = hs_vals / np.nanmax(hs_vals) if np.nanmax(hs_vals) > 0 else hs_vals
        _draw_boundary_nodes(
            ax,
            boundary,
            colors=plt.cm.viridis(hs_norm),
            sizes=20 + 80 * hs_norm,
        )
        _, dir_vals = _forcing_row(ctx.case_dir, "dir", time_index)
        for i, (lon, lat) in enumerate(boundary):
            if hs_vals[i] > 0:
                _draw_dir_arrow(ax, lon, lat, dir_vals[i], extent=extent)

    if sites_ds is not None:
        ax.scatter(
            sites_ds["station_x"].values,
            sites_ds["station_y"].values,
            s=8,
            c="white",
            edgecolors="black",
            linewidths=0.2,
            alpha=0.7,
            label="output sites",
            zorder=8,
        )

    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def _plot_case_figure(
    case_dir: str | Path,
    *,
    mode: str,
    time_index: int = 0,
    show_map: bool = True,
    map_variable: str = "hs",
    style: CasePlotStyle | None = None,
    title: str | None = None,
) -> Figure:
    style = style or DEFAULT_CASE_PLOT_STYLE
    ctx = load_case_context(case_dir)
    sites_ds = xr.open_dataset(ctx.sites_file) if ctx.sites_file else None
    map_ds = xr.open_dataset(ctx.map_file) if show_map and ctx.map_file else None

    extent = _extent_from_enclosure(read_enclosure_polygon(ctx.case_dir))
    main_title, subtitle = _build_case_title(
        ctx,
        mode=mode,
        time_index=time_index,
        has_map=map_ds is not None,
        user_title=title,
    )

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.2], hspace=0.3, wspace=0.22)
    axes_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_map = fig.add_subplot(gs[1, :])

    if sites_ds is None:
        for ax in axes_top:
            ax.text(0.5, 0.5, "output_sites.nc missing", ha="center", va="center")
    else:
        for ax, var in zip(axes_top, style.top_vars):
            _plot_site_field(
                ax,
                sites_ds,
                var,
                time_index=time_index,
                style=style,
                extent=extent,
                mode=mode,
            )

    _plot_domain_panel(
        ax_map,
        ctx,
        map_ds=map_ds,
        sites_ds=sites_ds,
        time_index=time_index,
        mode=mode,
        map_variable=map_variable,
        style=style,
        extent=extent,
    )

    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)
    fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=10, color="#444444")
    fig.subplots_adjust(top=0.90)
    _fit_map_extent(ax_map, extent)

    if sites_ds is not None:
        sites_ds.close()
    if map_ds is not None:
        map_ds.close()
    return fig


def _extent_from_enclosure(
    enclosure: Polygon, pad: float = 0.08
) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = enclosure.bounds
    return minx - pad, maxx + pad, miny - pad, maxy + pad


def plot_metamodel_case(
    case_dir: str | Path,
    *,
    time_index: int = 0,
    show_map: bool = True,
    map_variable: str = "hs",
    style: CasePlotStyle | None = None,
    title: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot a finished metamodel SnapWave case from its folder."""
    fig = _plot_case_figure(
        case_dir,
        mode="metamodel",
        time_index=time_index,
        show_map=show_map,
        map_variable=map_variable,
        style=style,
        title=title,
    )
    if show:
        plt.show()
    return fig


def plot_dynamic_case(
    case_dir: str | Path,
    *,
    time_index: int = 0,
    show_map: bool = True,
    map_variable: str = "hs",
    style: CasePlotStyle | None = None,
    title: str | None = None,
    show: bool = True,
) -> Figure:
    """Plot a finished dynamic SnapWave case from its folder."""
    fig = _plot_case_figure(
        case_dir,
        mode="dynamic",
        time_index=time_index,
        show_map=show_map,
        map_variable=map_variable,
        style=style,
        title=title,
    )
    if show:
        plt.show()
    return fig


__all__ = [
    "DEFAULT_CASE_PLOT_STYLE",
    "CaseContext",
    "CasePlotStyle",
    "FORCING_FILES",
    "TOP_VARS",
    "WAVE_VAR_ALIASES",
    "infer_goal_from_path",
    "load_case_context",
    "plot_dynamic_case",
    "plot_metamodel_case",
    "read_boundary_nodes",
    "read_enclosure_polygon",
    "read_forcing_table",
    "resolve_wave_variable",
]
