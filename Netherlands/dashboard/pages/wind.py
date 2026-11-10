"""Wind: what SnapWave's local wind adds (dynamic with − without wind), where, when, and whether the buoys agree."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/wind", name="Wind", order=4)

SCALES = {"u10": 2.0, "u10dir": 22.5, "hs20": 0.4}


def layout(**_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    top = dbc.Row([
        dbc.Col(ui.dropdown("wd-source", list(data.OFFSHORE), "WHACS", "Offshore source"), md=2),
        dbc.Col(ui.dropdown("wd-quantity", [{"label": "added Hs (m)", "value": "hs"},
                                           {"label": "added energy, Hs² (m²)", "value": "energy"}], "hs", "Quantity"), md=2),
        dbc.Col(ui.labelled("ERA5 U10 at the site, for the per-direction panels (m/s)",
                            dcc.RangeSlider(id="wd-speed", min=0, max=22, step=1, value=[8, 14],
                                            marks={i: str(i) for i in range(0, 23, 2)}, tooltip={"placement": "bottom"})),
                md=8),
    ], className="controls g-2")
    analog = dbc.Row([
        dbc.Col(ui.labelled("U10 (m/s)", dcc.Slider(id="wd-a-u10", min=2, max=20, step=0.5, value=12,
                                                    marks={i: str(i) for i in range(2, 21, 4)},
                                                    tooltip={"placement": "bottom"})), md=3),
        dbc.Col(ui.labelled("wind from (deg)", dcc.Slider(id="wd-a-dir", min=0, max=345, step=15, value=225,
                                                          marks={i: ui.SECTOR_NAMES.get(i, "") for i in range(0, 360, 45)},
                                                          tooltip={"placement": "bottom"})), md=3),
        dbc.Col(ui.labelled("Hs at 20 m without wind (m)", dcc.Slider(id="wd-a-hs", min=0.1, max=4, step=0.1, value=1.5,
                                                                      marks={i: str(i) for i in range(0, 5)},
                                                                      tooltip={"placement": "bottom"})), md=3),
        dbc.Col(ui.labelled("hours", dcc.Slider(id="wd-a-k", min=6, max=96, step=6, value=24,
                                                marks={i: str(i) for i in (6, 24, 48, 96)})), md=3),
    ], className="controls g-2")
    return html.Div([
        ui.header("Wind", "The reconstruction has no wind (hywaves/WIND.md). Here: SnapWave every hour with the ERA5 "
                  "wind minus the same run without it. What the wind adds depends on the sea already there (little on "
                  "top of a grown sea from offshore, everything where the boundary brings nothing) and on the fetch "
                  "inside the domain. The buoys say whether SnapWave adds the right amount."),
        ui.card(None, top),
        dbc.Row([
            dbc.Col(ui.card("By stretch of coast × wind direction at the site (hours in the speed band)",
                            ui.graph(id="wd-heat")), lg=7),
            dbc.Col(ui.card("Against the buoys, by wind direction at the buoy (speed band): run − buoy",
                            ui.graph(id="wd-buoys")), lg=5),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(ui.card("By wind speed at the site, per isobath", ui.graph(id="wd-speedcurve")), lg=5),
            dbc.Col(ui.card("Map of the mean added quantity by wind direction (speed band)",
                            [ui.dropdown("wd-sector", [{"label": f"{ui.SECTOR_NAMES[s]} ({s}°)", "value": s}
                                                       for s in ui.SECTORS], 225, "Wind from"),
                             ui.graph(id="wd-sectormap", config={"scrollZoom": True})]), lg=7),
        ], className="g-2"),
        ui.card("Analogs: the closest hours of 2013 to a situation (ERA5 wind over the domain, Hs at 20 m without wind)", [
            analog, html.Div(id="wd-a-summary", className="fw-semibold my-1"),
            dbc.Row([dbc.Col(ui.graph(id=f"wd-a-{k}", config={"scrollZoom": True}), lg=3) for k in ("off", "on", "add", "buoy")],
                    className="g-1"),
            html.Div(id="wd-a-hours", className="note")]),
        _wind_only_card(),
        ui.card("Hours where the run with wind did not converge (NaN in the files)", ui.graph(id="wd-nonconv")),
    ])


#: Site groups of the wind-only section (cube ``group`` values).
WO_GROUPS = {"5 m": "5m", "10 m": "10m", "20 m": "20m", "buoys": "buoy"}
WO_COLORS = {"5 m": "#fdae61", "10 m": "#74add1", "20 m": "#2c7bb6", "buoys": "#111111"}
WO_MAPS = {"wo": "wind-only Hs (m)", "added": "added Hs, with − without wind (m)",
           "over": "plain sum − with wind: √(E_off + E_wo) − Hs_on (m)", "ratio": "damping: added energy / wind-only energy"}


def _wind_only_card():
    if not data.wind_only_available():
        return ui.card("Wind only", html.P("No wind-only run in the store: run wind_only_storms.py and "
                                           "`build_store.py --only wind_only`.", className="note"))
    events = [{"label": "every storm", "value": -1}] + [{"label": r.event, "value": i}
                                                      for i, r in enumerate(data.storms().itertuples())]
    return ui.card("Wind only: SnapWave with no boundary waves, only the ERA5 wind, on the storm hours (Storms page)", [
        html.P("Could the wind be one more term of the sum (e.g. from analogs of the wind field)? If it were additive, "
               "the energy the wind adds to the run without wind would equal the wind-only energy (ratio 1). The wind "
               "grows an existing sea less than calm water, so the ratio falls as the share of the sea already there "
               "grows: the damping a wind term would need. The damped row uses that curve fitted on the other storms "
               "(leave one storm out), per site group.", className="note"),
        dbc.Row([
            dbc.Col(ui.dropdown("wd-wo-event", events, -1, "Storm"), md=4),
            dbc.Col(ui.dropdown("wd-wo-map", [{"label": v, "value": k} for k, v in WO_MAPS.items()], "over", "Map"), md=5),
        ], className="controls g-2"),
        dbc.Row([
            dbc.Col(ui.graph(id="wd-wo-curve"), lg=5),
            dbc.Col(ui.graph(id="wd-wo-map-fig", config={"scrollZoom": True}), lg=7),
        ], className="g-2"),
        ui.loading(html.Div(id="wd-wo-table")),
    ])


def _wo_columns(group: str) -> np.ndarray:
    cols = np.flatnonzero(data.site_mask(group))
    return np.intersect1d(cols, data.buoy_sites()) if group == "buoy" else cols


def _wo_select(pairs: dict[str, np.ndarray], event: int) -> np.ndarray:
    return np.ones(len(pairs["hours"]), bool) if event == -1 else pairs["event"] == event


@callback(Output("wd-wo-curve", "figure"), Output("wd-wo-table", "children"),
          Input("wd-source", "value"), Input("wd-wo-event", "value"))
def wind_only_curve(source, event):
    if not data.wind_only_available():
        return fg.empty("no wind-only run"), None
    fig = go.Figure()
    centres = (data.SHARE_BINS[:-1] + data.SHARE_BINS[1:]) / 2
    rows = []
    for name, group in WO_GROUPS.items():
        cols = _wo_columns(group)
        pairs = data.wind_only_pairs(source, cols)
        sel = _wo_select(pairs, event)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(pairs["e_wo"] > 1e-4, (pairs["e_on"] - pairs["e_off"]) / pairs["e_wo"], np.nan)
        curve = data.damping_curve(pairs["share"][sel].ravel(), ratio[sel].ravel())
        fig.add_trace(go.Scatter(x=centres, y=curve, name=name, mode="lines+markers", line=dict(color=WO_COLORS[name])))
        on = np.sqrt(pairs["e_on"])[sel]
        rows_cols = np.ix_(pairs["hours"], cols)
        estimates = {f"{base} {text}": data.field(f"{kind}_{source}{suffix}", "hs")[rows_cols]
                     for kind, base in (("dyn_off", "SnapWave no wind"), ("mm_spec", "metamodel spectral"))
                     for suffix, text in (("", ""), ("+wind_only", "⊕ wind only"), ("+damped", "⊕ damped wind only"))}
        obs = data.field("buoy", "hs")[pairs["hours"]][:, cols][sel] if group == "buoy" else None
        for label, est in estimates.items():
            m = data.pooled_metrics(on, est[sel], "hs")
            row = {"group": name, "estimate": label.strip(), "vs wind bias": m["bias"], "vs wind RMSE": m["rmse"]}
            if obs is not None:
                b = data.pooled_metrics(obs, est[sel], "hs")
                row.update({"vs buoy bias": b["bias"], "vs buoy RMSE": b["rmse"]})
            rows.append(row)
        if obs is not None:
            b = data.pooled_metrics(obs, on, "hs")
            rows.append({"group": name, "estimate": "SnapWave with wind", "vs buoy bias": b["bias"],
                         "vs buoy RMSE": b["rmse"]})
    fig.add_hline(y=1, line=dict(color="black", dash="dash", width=1))
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.update_layout(xaxis_title="share of the sea already there, E_off / (E_off + E_wind-only)",
                      yaxis_title="added energy / wind-only energy (median)",
                      legend=dict(orientation="h", x=0.99, xanchor="right", y=0.99, yanchor="top",
                                  bgcolor="rgba(255,255,255,0.7)"))
    table = pd.DataFrame(rows)
    fmt = {c: "{:+.2f}" if c.endswith("bias") else "{:.2f}" for c in table.columns if c.startswith("vs")}
    caption = html.P("Hs (m) over the storm hours, reference SnapWave with wind or the buoys. Damped: "
                     "√(E + f(E / (E + E_wo)) E_wo), E the energy without wind of that estimate (the metamodel's for "
                     "the metamodel, as a reconstruction would), f the SnapWave curve of the other storms.",
                     className="small text-muted mb-1")
    return fg.base_layout(fig, 430, "Is the wind additive? (1 = yes)"), [caption, ui.table(table, fmt, index=False)]


@callback(Output("wd-wo-map-fig", "figure"), Input("wd-source", "value"), Input("wd-wo-event", "value"),
          Input("wd-wo-map", "value"))
def wind_only_map(source, event, kind):
    if not data.wind_only_available():
        return fg.empty("no wind-only run")
    cols = np.flatnonzero(data.sites()["source"].isin(["isobath", "mesh"]).to_numpy())
    pairs = data.wind_only_pairs(source, cols)
    sel = _wo_select(pairs, event)
    e_off, e_on, e_wo = (pairs[k][sel] for k in ("e_off", "e_on", "e_wo"))
    with np.errstate(invalid="ignore", divide="ignore"):
        values = {"wo": lambda: np.sqrt(e_wo), "added": lambda: np.sqrt(e_on) - np.sqrt(e_off),
                  "over": lambda: np.sqrt(e_off + e_wo) - np.sqrt(e_on),
                  "ratio": lambda: np.where(e_wo > 1e-4, (e_on - e_off) / e_wo, np.nan)}[kind]()
        mean = np.nanmedian(values, 0) if kind == "ratio" else np.nanmean(values, 0)
    hours = int(np.isfinite(e_wo).any(1).sum())
    stat = "median" if kind == "ratio" else "mean"
    title = f"{stat} {WO_MAPS[kind]} · {hours} hours"
    if kind == "wo":
        return fg.site_map(mean, cols, title=title, colorscale="Magma_r", zmin=0,
                           zmax=float(np.nanquantile(mean, 0.99)), colorbar="m", height=430, uirevision="wd-wo")
    if kind == "ratio":
        return fg.site_map(mean, cols, title=title, colorscale="RdBu_r", zmin=0, zmax=2, colorbar="ratio",
                           height=430, uirevision="wd-wo")
    return fg.site_map(mean, cols, title=title, diverging=True, colorbar="m", height=430, uirevision="wd-wo")


def _added(source: str, quantity: str) -> np.ndarray:
    h0, h1 = data.field(f"dyn_off_{source}", "hs"), data.field(f"dyn_on_{source}", "hs")
    return h1**2 - h0**2 if quantity == "energy" else h1 - h0


@callback(Output("wd-heat", "figure"), Output("wd-buoys", "figure"), Output("wd-speedcurve", "figure"),
          Input("wd-source", "value"), Input("wd-quantity", "value"), Input("wd-speed", "value"))
def per_direction(source, quantity, speed):
    added = _added(source, quantity)
    u10, wdir = data.field("wind", "u10"), data.field("wind", "u10dir")
    band = (u10 >= speed[0]) & (u10 < speed[1])
    wsec = data.sector(wdir)
    table = data.sites()
    regions = list(data.COAST_REGIONS)
    unit = "m²" if quantity == "energy" else "m"
    figs = []
    rows = []
    for label in ("5m", "10m", "20m"):
        cols = np.flatnonzero((table["group"] == label).to_numpy())
        a = np.where(band[:, cols], added[:, cols], np.nan)
        rc = pd.Categorical(table["region"].to_numpy()[cols], categories=regions).codes
        code = (np.broadcast_to(rc, a.shape) * 8 + np.clip(wsec[:, cols], 0, None) // 45).ravel()
        ok = np.isfinite(a).ravel()
        s = np.bincount(code[ok], weights=a.ravel()[ok], minlength=24)
        n = np.bincount(code[ok], minlength=24)
        with np.errstate(invalid="ignore"):
            t = pd.DataFrame((s / n).reshape(3, 8), index=[f"{label} · {r}" for r in regions],
                             columns=[ui.SECTOR_NAMES[x] for x in ui.SECTORS])
        rows.append(t)
    heat = pd.concat(rows)
    zmax = float(np.nanmax(heat.to_numpy())) if np.isfinite(heat.to_numpy()).any() else 1
    fheat = fg.heatmap(heat, f"mean added {'energy' if quantity == 'energy' else 'Hs'} ({unit}), U10 {speed[0]}-{speed[1]} m/s",
                       zmid=None, colorscale="Magma_r", fmt="{:.2f}", height=430, zmax=zmax)
    fheat.data[0].zmin = 0
    # buoys
    cols = data.buoy_sites()
    obs = data.field("buoy", "hs")[:, cols]
    bband = band[:, cols]
    bsec = wsec[:, cols]
    brows = {}
    for off in data.OFFSHORE:
        e0 = data.field(f"dyn_off_{off}", "hs")[:, cols] - obs
        e1 = data.field(f"dyn_on_{off}", "hs")[:, cols] - obs
        both = np.isfinite(e0) & np.isfinite(e1) & bband
        for name, e in (("no wind", e0), ("wind", e1)):
            brows[f"{off} · {name}"] = [np.nanmean(np.where(both & (bsec == s), e, np.nan)) for s in ui.SECTORS]
    bt = pd.DataFrame(brows, index=[ui.SECTOR_NAMES[s] for s in ui.SECTORS]).T
    fbuoy = fg.heatmap(bt, f"mean Hs error at the buoys (m), U10 {speed[0]}-{speed[1]} m/s", height=430)
    # speed curve
    fcurve = go.Figure()
    bins = np.arange(0, 24, 2)
    sb = np.digitize(u10, bins) - 1
    for label, color in (("5m", "#fdae61"), ("10m", "#74add1"), ("20m", "#2c7bb6")):
        cols = np.flatnonzero((table["group"] == label).to_numpy())
        a = added[:, cols].ravel()
        b = sb[:, cols].ravel()
        ok = np.isfinite(a)
        s = np.bincount(b[ok], weights=a[ok], minlength=len(bins))
        n = np.bincount(b[ok], minlength=len(bins))
        with np.errstate(invalid="ignore"):
            fcurve.add_trace(go.Scatter(x=bins + 1, y=np.where(n > 2000, s / n, np.nan), name=label, mode="lines+markers",
                                        line=dict(color=color)))
    fcurve.update_layout(xaxis_title="ERA5 U10 at the site (m/s)", yaxis_title=f"mean added ({unit})",
                         legend=dict(orientation="h", y=1.1))
    return fheat, fbuoy, fg.base_layout(fcurve, 420)


@callback(Output("wd-sectormap", "figure"), Input("wd-source", "value"), Input("wd-quantity", "value"),
          Input("wd-speed", "value"), Input("wd-sector", "value"))
def sector_map(source, quantity, speed, sector):
    added = _added(source, quantity)
    u10 = data.field("wind", "u10")
    sel = (u10 >= speed[0]) & (u10 < speed[1]) & (data.sector(data.field("wind", "u10dir")) == sector)
    cols = np.flatnonzero(data.sites()["source"].isin(["isobath", "mesh"]).to_numpy())
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(sel[:, cols], added[:, cols], np.nan), 0)
    hours = int(sel.any(1).sum())
    return fg.site_map(mean, cols, title=f"wind from {ui.SECTOR_NAMES[sector]}, {speed[0]}-{speed[1]} m/s · {hours} hours",
                       colorscale="Magma_r", zmin=0, zmax=float(np.nanquantile(mean, 0.99)) if np.isfinite(mean).any() else 1,
                       colorbar="m²" if quantity == "energy" else "m", height=430, uirevision="wd-sector")


@callback(Output("wd-a-summary", "children"), Output("wd-a-off", "figure"), Output("wd-a-on", "figure"),
          Output("wd-a-add", "figure"), Output("wd-a-buoy", "figure"), Output("wd-a-hours", "children"),
          Input("wd-source", "value"), Input("wd-a-u10", "value"), Input("wd-a-dir", "value"), Input("wd-a-hs", "value"),
          Input("wd-a-k", "value"))
def analogs(source, u10, wdir, hs20, k):
    c = data.conditions_of(source)
    c = c[c["wind_converged"]]
    d2 = ((c["u10"] - u10) / SCALES["u10"]) ** 2 + ((((c["u10dir"] - wdir) + 180) % 360 - 180) / SCALES["u10dir"]) ** 2 \
        + ((c["hs20"] - hs20) / SCALES["hs20"]) ** 2
    pick = c.assign(distance=np.sqrt(d2)).nsmallest(k, "distance")
    hours = pick.index.to_numpy()
    cols = np.flatnonzero(data.sites()["source"].isin(["isobath", "mesh"]).to_numpy())
    with np.errstate(invalid="ignore"):
        h0 = np.nanmean(data.field(f"dyn_off_{source}", "hs")[hours][:, cols], 0)
        h1 = np.nanmean(data.field(f"dyn_on_{source}", "hs")[hours][:, cols], 0)
    vmax = float(np.nanquantile(h1, 0.99))
    mid = int(hours[np.argmin(pick["distance"].to_numpy())])
    layers, arrows = fg.wind_layer(mid, colorbar=False)
    f0 = fg.site_map(h0, cols, title="Hs without wind", zmin=0, zmax=vmax, colorbar="m", height=420, under=arrows,
                     layers=layers, uirevision="an")
    f1 = fg.site_map(h1, cols, title="Hs with wind", zmin=0, zmax=vmax, colorbar="m", height=420, under=arrows,
                     layers=layers, uirevision="an")
    fa = fg.site_map(h1 - h0, cols, title="added Hs", diverging=True, colorbar="m", height=420, uirevision="an")
    bc = data.buoy_sites()
    obs = data.field("buoy", "hs")[hours][:, bc]
    with np.errstate(invalid="ignore"):
        e1 = np.nanmean(data.field(f"dyn_on_{source}", "hs")[hours][:, bc] - obs, 0)
        e0 = np.nanmean(data.field(f"dyn_off_{source}", "hs")[hours][:, bc] - obs, 0)
    fb = fg.site_map(e1, bc, title=f"buoys: with wind − buoy (median {np.nanmedian(e1):+.2f}; no wind {np.nanmedian(e0):+.2f})",
                     diverging=True, colorbar="m", size=6, height=420, uirevision="an")
    circ = lambda x: float(np.rad2deg(np.arctan2(np.sin(np.deg2rad(x)).mean(), np.cos(np.deg2rad(x)).mean())) % 360)
    summary = (f"{len(pick)} hours: U10 {pick['u10'].mean():.1f} m/s (sd {pick['u10'].std():.1f}) from {circ(pick['u10dir']):.0f}°, "
               f"Hs at 20 m without wind {pick['hs20'].mean():.2f} m from {circ(pick['dir20']):.0f}° · months "
               f"{sorted(set(pick['time'].dt.month))} · asked {u10} m/s from {wdir}°, {hs20} m")
    links = [html.Span("Open in the Explorer: ")] + [
        dcc.Link(f"{t:%d %b %H:%M}  ", href=f"/explorer?t={t:%Y-%m-%dT%H:%M}&a=dyn_on_{source}&b=dyn_off_{source}")
        for t in pick["time"].sort_values()]
    return summary, f0, f1, fa, fb, links


@callback(Output("wd-nonconv", "figure"), Input("wd-source", "value"))
def nonconverged(source):
    c = data.conditions_of(source)
    bad = ~c["wind_converged"]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=c.loc[~bad, "u10"], y=c.loc[~bad, "hs_off"], mode="markers", name="converged",
                               marker=dict(size=3, color="#c9ced6")))
    fig.add_trace(go.Scattergl(x=c.loc[bad, "u10"], y=c.loc[bad, "hs_off"], mode="markers", name=f"not converged ({int(bad.sum())})",
                               marker=dict(size=5, color="#d62728"), text=c.loc[bad, "time"].dt.strftime("%d %b %H:%M"),
                               hovertemplate="%{text}<br>U10 %{x:.1f}, Hs %{y:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="ERA5 U10, site mean (m/s)", yaxis_title="offshore Hs, goal mean (m)",
                      legend=dict(orientation="h", y=1.1))
    return fg.base_layout(fig, 330)
