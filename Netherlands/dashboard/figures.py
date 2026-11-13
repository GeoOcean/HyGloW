"""Plotly figures shared by the pages: site maps with wind and goal arrows, the 2013 timeline, series."""

from __future__ import annotations

import base64
import io
from functools import lru_cache

import matplotlib
import matplotlib.image
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

import data

MAP_CENTER = {"lat": 52.55, "lon": 4.4}
MAP_ZOOM = 5.5
COLORSCALES = {"hs": "Viridis", "tp": "Plasma", "dir": "Twilight", "spr": "Cividis", "u10": "Blues"}
DIVERGING = "RdBu_r"
FONT = dict(family="Lato, Helvetica Neue, Helvetica, Arial, sans-serif", size=12)


def base_layout(fig: go.Figure, height: int = 420, title: str | None = None, **kwargs) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10), font=FONT,
                      title=dict(text=title, x=0.01, font=dict(size=14)) if title else None,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(font=dict(family=FONT["family"], size=12)), **kwargs)
    return fig


def empty(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#777"), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return base_layout(fig, height)


def site_map(values: np.ndarray, columns: np.ndarray, *, title: str = "", colorscale: str = "Viridis",
             zmin: float | None = None, zmax: float | None = None, colorbar: str = "", height: int = 520,
             diverging: bool = False, size: float = 6, extra: list[go.BaseTraceType] | None = None,
             under: list[go.BaseTraceType] | None = None, layers: list[dict] | None = None,
             counts: np.ndarray | None = None, uirevision: str = "map") -> go.Figure:
    """
    One value per site (cube *columns*) on a basemap; diverging scales are symmetric about 0.

    *under* traces go below the sites, *extra* above; *layers* are map image layers (wind);
    *counts* (pairs per site) are shown on hover. Clicks carry ``customdata = [value, column]``.
    """
    table = data.sites().iloc[columns]
    v = np.asarray(values, float)
    n = np.full(len(v), np.nan) if counts is None else np.asarray(counts, float)
    if diverging:
        lim = float(np.nanquantile(np.abs(v), 0.98)) if np.isfinite(v).any() else 1.0
        zmin, zmax, colorscale = -lim, lim, DIVERGING
    if zmin is None and zmax is None and np.isfinite(v).any():
        zmin, zmax = float(np.nanquantile(v, 0.01)), float(np.nanquantile(v, 0.99))
    buoy = (table["group"] == "buoy").to_numpy()
    text = (table["site"] + " · " + table["group"] + " · " + table["depth"].round(1).astype(str) + " m").to_numpy()
    if counts is not None:
        text = np.array([f"{t} · n {int(c):,}" if np.isfinite(c) else t for t, c in zip(text, n)])
    scale = dict(colorscale=colorscale, cmin=zmin, cmax=zmax)
    traces = []
    empty_ = ~np.isfinite(v)
    if empty_.any():  # sites without a value: faint, so the coast stays readable
        traces.append(go.Scattermap(lat=table["lat"][empty_], lon=table["lon"][empty_], mode="markers", text=text[empty_],
                                    marker=dict(size=size * 0.6, color="#b8bec6", opacity=0.5), showlegend=False,
                                    customdata=np.c_[v[empty_], columns[empty_]],
                                    hovertemplate="%{text}<br>no value<extra></extra>"))
    buoy = buoy & ~empty_
    if (~buoy & ~empty_).any():
        sel = ~buoy & ~empty_
        traces.append(go.Scattermap(lat=table["lat"][sel], lon=table["lon"][sel], mode="markers", text=text[sel],
                                    marker=dict(size=size, color=v[sel], **scale,
                                                colorbar=dict(title=colorbar, thickness=12, len=0.8)),
                                    customdata=np.c_[v[sel], columns[sel]], showlegend=False,
                                    hovertemplate="%{text}<br>value %{customdata[0]:.3f}<extra></extra>"))
    if buoy.any():  # buoys: bigger, with a black ring, on the same colour scale
        traces.append(go.Scattermap(lat=table["lat"][buoy], lon=table["lon"][buoy], mode="markers", hoverinfo="skip",
                                    marker=dict(size=size * 2.6 + 3, color="black"), showlegend=False))
        traces.append(go.Scattermap(lat=table["lat"][buoy], lon=table["lon"][buoy], mode="markers", text=text[buoy],
                                    marker=dict(size=size * 2.6, color=v[buoy], **scale, showscale=not (~buoy & ~empty_).any(),
                                                colorbar=dict(title=colorbar, thickness=12, len=0.8)),
                                    customdata=np.c_[v[buoy], columns[buoy]], showlegend=False,
                                    hovertemplate="%{text}<br>value %{customdata[0]:.3f}<extra></extra>"))
    fig = go.Figure((under or []) + traces + (extra or []))
    fig.update_layout(map=dict(style="carto-positron", center=MAP_CENTER, zoom=MAP_ZOOM, layers=layers or []),
                      uirevision=uirevision, clickmode="event")
    return base_layout(fig, height, title)


def arrows(lat: np.ndarray, lon: np.ndarray, to_deg: np.ndarray, length: np.ndarray,
           head: float = 0.35, spread: float = 25.0) -> tuple[np.ndarray, np.ndarray]:
    """Line coordinates of arrows (shaft + two barbs) from (lat, lon) towards *to_deg*, *length* in degrees of latitude."""
    def offset(la, lo, ang, ln):
        a = np.deg2rad(ang)
        return la + ln * np.cos(a), lo + ln * np.sin(a) / np.cos(np.deg2rad(la))
    lat1, lon1 = offset(lat, lon, to_deg, length)
    b1 = offset(lat1, lon1, to_deg + 180 - spread, head * length)
    b2 = offset(lat1, lon1, to_deg + 180 + spread, head * length)
    nan = np.full(len(lat), np.nan)
    la = np.c_[lat, lat1, nan, b1[0], lat1, b2[0], nan].ravel()
    lo = np.c_[lon, lon1, nan, b1[1], lon1, b2[1], nan].ravel()
    return la, lo


@lru_cache(maxsize=1)
def _wind_grid():
    import xarray as xr
    ds = xr.open_zarr(data.STORE / "wind_grid.zarr").load()
    return ds["u10"].values, ds["v10"].values, ds["latitude"].values, ds["longitude"].values


WIND_MAX = 25.0
WIND_CMAP = "YlOrRd"


def _wind_png(speed: np.ndarray, lat: np.ndarray, upsample: int = 8) -> str:
    """Wind speed grid as a smooth semi-transparent PNG (data URL), north up."""
    from scipy.ndimage import zoom
    grid = speed[::-1] if lat[0] < lat[-1] else speed
    fine = zoom(grid, upsample, order=1)
    rgba = matplotlib.colormaps[WIND_CMAP](np.clip(fine / WIND_MAX, 0, 1))
    rgba[..., 3] = 0.08 + 0.42 * np.clip(fine / WIND_MAX, 0, 1)
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def wind_layer(index: int, stride: int = 2, colorbar: bool = True) -> tuple[list[dict], list[go.Scattermap]]:
    """ERA5 wind at one hour: a shaded speed layer (map ``layers``) and arrows (towards) on every *stride*-th node."""
    u, v, lat, lon = _wind_grid()
    uu, vv = u[index], v[index]
    speed = np.hypot(uu, vv)
    dl, dn = abs(lat[1] - lat[0]) / 2, abs(lon[1] - lon[0]) / 2
    coords = [[lon.min() - dn, lat.max() + dl], [lon.max() + dn, lat.max() + dl],
              [lon.max() + dn, lat.min() - dl], [lon.min() - dn, lat.min() - dl]]
    layer = dict(sourcetype="image", source=_wind_png(speed, lat), coordinates=coords, below="traces")
    LA, LO = np.meshgrid(lat[::stride], lon[::stride], indexing="ij")
    sp = speed[::stride, ::stride].ravel()
    to = (np.rad2deg(np.arctan2(uu, vv))[::stride, ::stride].ravel()) % 360
    la, lo = arrows(LA.ravel(), LO.ravel(), to, 0.012 * sp)
    traces = [go.Scattermap(lat=la, lon=lo, mode="lines", line=dict(color="rgba(60,60,60,0.85)", width=1.6),
                            hoverinfo="skip", showlegend=False),
              go.Scattermap(lat=LA.ravel(), lon=LO.ravel(), mode="markers", showlegend=False,
                            marker=dict(size=4, color=sp, colorscale=WIND_CMAP, cmin=0, cmax=WIND_MAX, opacity=0.9,
                                        showscale=colorbar,
                                        colorbar=dict(title="ERA5 U10 (m/s)", orientation="h", thickness=8, len=0.45,
                                                      x=0.0, xanchor="left", y=-0.02, yanchor="top")),
                            text=[f"ERA5 {a:.1f} m/s from {(b + 180) % 360:.0f}°" for a, b in zip(sp, to)],
                            hovertemplate="%{text}<extra></extra>")]
    return [layer], traces


def goal_traces(offshore: str, index: int, variable: str = "hs", colorscale: str = "Viridis",
                zmin: float | None = None, zmax: float | None = None, half: float = 0.07) -> list[go.Scattermap]:
    """Goals: a square coloured by the offshore *variable* (the map's scale) and a wave arrow (towards, ∝ Hs)."""
    g = data.goals()
    off = data.offshore()
    ids = g["goal"].to_numpy()
    val = {v: off[f"{offshore}_{v}"].isel(time=index).sel(goal=ids).values for v in data.VARIABLES}
    lat0, lon0 = g["lat"].to_numpy(), g["lon"].to_numpy()
    x = val[variable]
    lo_, hi_ = (zmin if zmin is not None else np.nanmin(x)), (zmax if zmax is not None else np.nanmax(x))
    norm = np.clip((x - lo_) / ((hi_ - lo_) or 1), 0, 1)
    colors = sample_colorscale(colorscale, list(np.nan_to_num(norm)))
    hover = [f"goal {i} ({offshore})<br>Hs {h:.2f} m, Tp {t:.1f} s<br>from {d:.0f}°, spreading {sp:.0f}°"
             for i, h, t, d, sp in zip(ids, val["hs"], val["tp"], val["dir"], val["spr"])]
    return goal_glyphs(lat0, lon0, colors, val["dir"], val["hs"], hover, half=half)


def goal_glyphs(lat: np.ndarray, lon: np.ndarray, colors: list[str], dir_from: np.ndarray, hs: np.ndarray,
                hover: list[str], labels: list[str] | None = None, half: float = 0.07,
                scale: float = 0.05) -> list[go.Scattermap]:
    """Goal squares filled with *colors*, a wave arrow (towards, length ∝ *hs*), optional labels, and a hover."""
    traces = []
    for la, lo, c in zip(lat, lon, colors):
        dlo = half / np.cos(np.deg2rad(la))
        traces.append(go.Scattermap(lat=[la - half, la - half, la + half, la + half, la - half],
                                    lon=[lo - dlo, lo + dlo, lo + dlo, lo - dlo, lo - dlo], mode="lines", fill="toself",
                                    fillcolor=c, line=dict(color="#0b1f4d", width=2), hoverinfo="skip", showlegend=False))
    la, lo = arrows(np.asarray(lat, float), np.asarray(lon, float), (np.asarray(dir_from) + 180) % 360,
                    scale * np.asarray(hs, float))
    traces.append(go.Scattermap(lat=la, lon=lo, mode="lines", line=dict(color="#0b1f4d", width=2.5), hoverinfo="skip",
                                showlegend=False))
    traces.append(go.Scattermap(lat=lat, lon=lon, mode="markers+text" if labels else "markers",
                                marker=dict(size=14, opacity=0), text=labels, textposition="top right",
                                textfont=dict(color="#0b1f4d", size=12), customdata=hover, showlegend=False,
                                hovertemplate="%{customdata}<extra></extra>"))
    return traces


def timeline(offshore: str = "WHACS_cal", index: int | None = None, window: tuple[int, int] | None = None,
             height: int = 190) -> go.Figure:
    """Mean offshore Hs at the goals (both sources) and mean ERA5 U10, storms shaded; click to pick an hour."""
    fig = go.Figure()
    for off, color in zip(data.OFFSHORE, ("#9ecae1", "#08519c")):
        c = data.conditions_of(off)
        fig.add_trace(go.Scattergl(x=c["time"], y=c["hs_off"], name=f"{off} Hs (goal mean)", line=dict(color=color, width=1.3)))
    c = data.conditions_of(offshore)
    fig.add_trace(go.Scattergl(x=c["time"], y=c["u10"], name="ERA5 U10 (site mean)", yaxis="y2",
                               line=dict(color="#e6550d", width=0.8), opacity=0.6))
    for s in data.storms().itertuples():
        fig.add_vrect(x0=s.start, x1=s.end, fillcolor="#fdd0a2", opacity=0.35, line_width=0)
    if index is not None:
        fig.add_vline(x=data.times()[index], line=dict(color="black", width=2))
    if window is not None:
        fig.update_xaxes(range=[data.times()[window[0]], data.times()[window[1]]])
    fig.update_layout(yaxis=dict(title="Hs (m)"), yaxis2=dict(title="U10 (m/s)", overlaying="y", side="right",
                                                              showgrid=False),
                      legend=dict(orientation="h", y=1.15, x=0), hovermode="x unified", uirevision="timeline")
    return base_layout(fig, height)


def series(column: int, variable: str, keys: list[str], index: int | None = None,
           window: tuple[int, int] | None = None, height: int = 330) -> go.Figure:
    """Time series of one site for several estimates (buoy as black dots)."""
    t = data.times()
    sl = slice(*window) if window else slice(None)
    fig = go.Figure()
    for key in keys:
        y = data.field(key, variable)[sl, column]
        if not np.isfinite(y).any():
            continue
        meta = data.meta(key)
        if key == "buoy":
            fig.add_trace(go.Scattergl(x=t[sl], y=y, name=meta["label"], mode="markers",
                                       marker=dict(size=3.5, color=meta["color"])))
        else:
            fig.add_trace(go.Scattergl(x=t[sl], y=y, name=meta["label"], mode="lines",
                                       line=dict(color=meta["color"], dash=meta["dash"], width=1.4),
                                       opacity=1.0 if meta["offshore"] == "WHACS_cal" else 0.55))
    if index is not None:
        fig.add_vline(x=t[index], line=dict(color="black", width=1, dash="dot"))
    site = data.sites().iloc[column]
    fig.update_layout(yaxis_title=data.VARIABLE_LABELS[variable], hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=10)))
    return base_layout(fig, height, f"{site['site']} · {site['group']} · {site['depth']:.1f} m")


def density_scatter(r: np.ndarray, e: np.ndarray, variable: str, ref_label: str, est_label: str,
                    height: int = 430) -> go.Figure:
    """2D histogram of estimate vs reference (log counts), 1:1 line and quantile-quantile points."""
    ok = np.isfinite(r) & np.isfinite(e)
    r, e = r[ok], e[ok]
    if r.size == 0:
        return empty("no valid pairs", height)
    hi = float(np.nanquantile(np.r_[r, e], 0.999))
    lo = 0.0 if variable != "dir" else 0.0
    hi = 360.0 if variable == "dir" else hi
    nb = 80
    H, xe, ye = np.histogram2d(r, e, bins=nb, range=[[lo, hi], [lo, hi]])
    z = np.where(H > 0, np.log10(H), np.nan)
    fig = go.Figure(go.Heatmap(x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, z=z.T, colorscale="Blues",
                               colorbar=dict(title="log10 n", thickness=10), hovertemplate="ref %{x:.2f}<br>est %{y:.2f}<br>log10 n %{z:.2f}<extra></extra>"))
    q = np.r_[np.linspace(0.01, 0.99, 50), 0.995, 0.999, 0.9999]
    fig.add_trace(go.Scatter(x=np.quantile(r, q), y=np.quantile(e, q), mode="markers+lines", name="quantiles",
                             marker=dict(size=5, color="#d62728"), line=dict(width=1, color="#d62728")))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="black", dash="dash", width=1),
                             showlegend=False))
    m = data.pooled_metrics(r, e, variable)
    fig.add_annotation(text=f"n {m['n']:,}<br>bias {m['bias']:+.3f}<br>RMSE {m['rmse']:.3f}<br>R² {m['r2']:.2f}",
                       x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False, align="left",
                       bgcolor="rgba(255,255,255,0.85)", font=dict(size=11))
    fig.update_layout(xaxis_title=ref_label, yaxis_title=est_label, legend=dict(x=0.7, y=0.05))
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[lo, hi], scaleanchor="x")
    return base_layout(fig, height)


@lru_cache(maxsize=8)
def _lut(colorscale: str) -> np.ndarray:
    """256 RGB colours of a Plotly colour scale (the image and its colour bar then match exactly)."""
    colors = sample_colorscale(colorscale, list(np.linspace(0, 1, 256)), colortype="rgb")
    return np.array([[float(c) for c in s[s.index("(") + 1:-1].split(",")] for s in colors]).round().astype(np.uint8)


def mesh_png(values: np.ndarray, colorscale: str, zmin: float, zmax: float, nearest: bool = False,
             opacity: float = 0.92) -> str:
    """
    A field on the mesh nodes as a PNG (data URL) on the store raster, north up; outside the mesh and NaN are clear.

    Linear inside each triangle (NaN corners left out), or the *nearest* corner (directions).
    """
    from PIL import Image

    r = data.raster()
    v = values[r["nodes"]]
    finite = np.isfinite(v)
    if nearest:
        px = v[np.arange(len(v)), r["weights"].argmax(1)]
    else:
        w = np.where(finite, r["weights"], 0.0)
        s = w.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            px = np.where(s > 0, (np.where(finite, v, 0.0) * w).sum(1) / s, np.nan)
    rows, cols = r["shape"]
    img = np.zeros((rows * cols, 4), np.uint8)
    ok = np.isfinite(px)
    idx = np.clip(np.round((px[ok] - zmin) / ((zmax - zmin) or 1.0) * 255), 0, 255).astype(int)
    img[r["pixel"][ok], :3] = _lut(colorscale)[idx]
    img[r["pixel"][ok], 3] = int(255 * opacity)
    buf = io.BytesIO()
    Image.fromarray(img.reshape(rows, cols, 4), "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mesh_map(values: np.ndarray, *, title: str = "", colorscale: str = "Viridis", zmin: float = 0.0,
             zmax: float = 1.0, colorbar: str = "", nearest: bool = False, height: int = 480,
             extra: list[go.BaseTraceType] | None = None, uirevision: str = "mesh") -> go.Figure:
    """
    A field on the mesh nodes as an image layer, with the sites on top (invisible; hover shows the value at the
    site's nearest node) carrying the colour bar. *extra* traces go above (goals).
    """
    lon0, lon1, lat0, lat1 = data.raster()["extent"]
    layer = dict(sourcetype="image", source=mesh_png(values, colorscale, zmin, zmax, nearest),
                 coordinates=[[lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]], below="traces")
    table = data.sites()
    at_sites = np.asarray(values, float)[data.site_nodes()]
    text = (table["site"] + " · " + table["group"] + " · " + table["depth"].round(1).astype(str) + " m").to_numpy()
    hover = go.Scattermap(lat=table["lat"], lon=table["lon"], mode="markers", text=text, customdata=at_sites,
                          marker=dict(size=7, opacity=0, color=at_sites, colorscale=colorscale, cmin=zmin, cmax=zmax,
                                      showscale=True, colorbar=dict(title=colorbar, thickness=12, len=0.8)),
                          hovertemplate="%{text}<br>value %{customdata:.3f}<extra></extra>", showlegend=False)
    fig = go.Figure([hover] + (extra or []))
    fig.update_layout(map=dict(style="carto-positron", center=MAP_CENTER, zoom=MAP_ZOOM + 0.6, layers=[layer]),
                      uirevision=uirevision)
    return base_layout(fig, height, title)


#: Isobaths drawn on site maps: depth (m) -> (colour, width).
ISOBATHS = {2.0: ("#9e9ac8", 1.0), 5.0: ("#3f007d", 1.8), 10.0: ("#807dba", 1.2), 20.0: ("#bcbddc", 1.0)}


@lru_cache(maxsize=1)
def _depth_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh depth on the map raster (~400 m, north up): pixel centres ``lon``, ``lat`` and the grid (NaN outside)."""
    r = data.raster()
    values = (data.mesh_depth()[r["nodes"]] * r["weights"]).sum(1)
    rows, cols = r["shape"]
    grid = np.full(rows * cols, np.nan)
    grid[r["pixel"]] = values
    lon0, lon1, lat0, lat1 = r["extent"]
    lon = lon0 + (np.arange(cols) + 0.5) * (lon1 - lon0) / cols
    lat = lat1 - (np.arange(rows) + 0.5) * (lat1 - lat0) / rows
    return lon, lat, grid.reshape(rows, cols)


@lru_cache(maxsize=1)
def isobath_traces() -> list[go.Scattermap]:
    """The :data:`ISOBATHS` traced on the mesh depth (one line trace per depth, in the legend)."""
    from contourpy import contour_generator

    lon, lat, grid = _depth_grid()
    gen = contour_generator(lon, lat, np.ma.masked_invalid(grid))
    traces = []
    for level, (color, width) in ISOBATHS.items():
        lines = gen.lines(level)
        if not lines:
            continue
        xy = np.concatenate([np.vstack([line, [np.nan, np.nan]]) for line in lines])
        traces.append(go.Scattermap(lon=xy[:, 0], lat=xy[:, 1], mode="lines", name=f"{level:g} m (mesh)",
                                    line=dict(color=color, width=width), hoverinfo="skip"))
    return traces


@lru_cache(maxsize=1)
def depth_layer(zmax: float = 30.0) -> dict:
    """Mesh depth as a shaded image layer (grey, darker = deeper; land clear)."""
    depth = np.where(data.mesh_depth() > 0, data.mesh_depth(), np.nan)
    lon0, lon1, lat0, lat1 = data.raster()["extent"]
    return dict(sourcetype="image", source=mesh_png(depth, "Greys", 0.0, zmax, opacity=0.55),
                coordinates=[[lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]], below="traces")


def heatmap(table: pd.DataFrame, title: str, zmid: float | None = 0.0, colorscale: str = DIVERGING,
            fmt: str = "{:+.2f}", height: int = 260, zmax: float | None = None) -> go.Figure:
    """Annotated heatmap of a small table (rows x columns)."""
    z = table.to_numpy(float)
    lim = zmax if zmax is not None else (float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0)
    fig = go.Figure(go.Heatmap(z=z, x=[str(c) for c in table.columns], y=[str(i) for i in table.index],
                               colorscale=colorscale, zmid=zmid, zmin=-lim if zmid is not None else None,
                               zmax=lim, text=[[fmt.format(v) if np.isfinite(v) else "" for v in row] for row in z],
                               texttemplate="%{text}", colorbar=dict(thickness=10)))
    fig.update_yaxes(autorange="reversed")
    return base_layout(fig, height, title)
