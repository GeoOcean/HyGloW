"""Overview: the case on a map, the headline numbers, and 2013 at a glance."""

from __future__ import annotations

from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/", name="Overview", order=0)


@lru_cache(maxsize=1)
def headline() -> dict:
    """Median-over-buoys Hs skill of every estimate, and the error budget of both sources."""
    cols = data.buoy_sites()
    skill = {}
    for key in data.MODELS:
        m = data.site_metrics("buoy", key, "hs", columns=cols)
        skill[key] = m[m["n"] >= 100][["bias", "rmse", "r2"]].median()
    budgets = {off: data.budget(off)[1] for off in data.OFFSHORE}
    q = {}
    for key in ("mm_spec_WHACS", "mm_spec_WHACS_cal"):
        m = data.site_metrics("buoy", key, "hs", columns=cols, quantile=0.999)
        q[key] = float(m.loc[m["n"] >= 100, "qbias"].median())
    return {"skill": skill, "budget": budgets, "p999": q}


def site_overview_map() -> go.Figure:
    table = data.sites()
    colors = {"5m": "#fdae61", "10m": "#abd9e9", "20m": "#2c7bb6", "DeenHaag": "#7b3294", "buoy": "black"}
    traces = []
    for group, color in colors.items():
        sel = table["group"] == group
        size = 11 if group == "buoy" else 5
        has_obs = np.isin(np.flatnonzero(sel), data.buoy_sites()) if group == "buoy" else None
        traces.append(go.Scattermap(lat=table.loc[sel, "lat"], lon=table.loc[sel, "lon"], mode="markers", name=group,
                                    marker=dict(size=size, color=color if group != "buoy" else
                                                np.where(has_obs, "black", "#bbbbbb")),
                                    text=table.loc[sel, "site"] + " · " + table.loc[sel, "depth"].round(1).astype(str) + " m",
                                    hovertemplate="%{text}<extra>" + group + "</extra>"))
    g = data.goals()
    traces.append(go.Scattermap(lat=g["lat"], lon=g["lon"], mode="markers+text", name="goals",
                                marker=dict(size=14, color="#0b3d91"), text=[f"goal {i}" for i in g["goal"]],
                                textposition="top right", textfont=dict(color="#0b3d91")))
    fig = go.Figure(traces)
    fig.update_layout(map=dict(style="carto-positron", center=fg.MAP_CENTER, zoom=6.0),
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"))
    return fg.base_layout(fig, 560)


def layout(**_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    h = headline()
    t = h["skill"]
    b = h["budget"]
    sites = data.sites()
    n_iso = int((sites["source"] == "isobath").sum())
    kpis = dbc.Row([
        dbc.Col(ui.kpi("Linear sum vs dynamic (method)", f"{b['WHACS_cal'].loc['method (bulk − dynamic no wind)', 'all']:+.3f} m",
                       "mean Hs, buoys, WHACS_cal"), md=3),
        dbc.Col(ui.kpi("Missing wind", f"{b['WHACS_cal'].loc['missing wind (no wind − wind)', 'all']:+.2f} m",
                       f"top 1 %: {b['WHACS_cal'].loc['missing wind (no wind − wind)', 'p99-100']:+.2f} m", "#d62728"), md=3),
        dbc.Col(ui.kpi("Spectral metamodel − buoy", f"{b['WHACS_cal'].loc['total (spectral − buoy)', 'all']:+.2f} m",
                       f"RMSE {t['mm_spec_WHACS_cal']['rmse']:.2f} m · R² {t['mm_spec_WHACS_cal']['r2']:.2f} (median of buoys)"),
                md=3),
        dbc.Col(ui.kpi("Calibration, storm tail", f"{h['p999']['mm_spec_WHACS']:+.2f} → {h['p999']['mm_spec_WHACS_cal']:+.2f} m",
                       "p99.9 bias at the buoys, WHACS → WHACS_cal", "#2ca02c"), md=3),
    ], className="g-2 mb-3")
    rows = []
    for key in data.MODELS:
        s = t[key]
        rows.append({"estimate": data.label(key), "bias (m)": s["bias"], "RMSE (m)": s["rmse"], "R²": s["r2"]})
    import pandas as pd
    skill = pd.DataFrame(rows).set_index("estimate")
    budget = pd.concat({off: b[off].drop(index="pairs")["all"] for off in data.OFFSHORE}, axis=1).rename_axis("part")
    return html.Div([
        ui.header("Netherlands 2013: every estimate, every site, every hour",
                  f"10 offshore goals, {n_iso:,} isobath sites (5, 10, 20 m), 141 DeenHaag mesh nodes and "
                  f"{len(data.buoy_sites())} buoys with data. Estimates: the offshore hindcast (WHACS, and WHACS_cal "
                  "calibrated against satellites), the HyWaves linear sum (MDA 600, CODEC water level; bulk and "
                  "spectral) and the hourly SnapWave runs with and without ERA5 wind (SWAN too, for WHACS_cal). "
                  "Metrics are estimate first: "
                  "bias = estimate − reference."),
        kpis,
        dbc.Row([
            dbc.Col(ui.card("The case", dcc.Graph(figure=site_overview_map(), config={"scrollZoom": True})), lg=6),
            dbc.Col([
                ui.card("Hs at the buoys, median over buoys", ui.table(skill, {"bias (m)": "{:+.3f}", "RMSE (m)": "{:.3f}",
                                                                              "R²": "{:.2f}"})),
                ui.card("Error budget at the buoys: mean Hs error of the spectral metamodel, split (m)",
                        ui.table(budget, {c: "{:+.3f}" for c in budget.columns})),
            ], lg=6),
        ]),
        ui.card("2013 · click a time to open it in the Explorer",
                dcc.Graph(id="ov-timeline", figure=fg.timeline(height=230))),
        dcc.Location(id="ov-go", refresh="callback-nav"),
    ])


@callback(Output("ov-go", "href"), Input("ov-timeline", "clickData"), prevent_initial_call=True)
def open_hour(click):
    t = str(click["points"][0]["x"])[:16].replace(" ", "T")
    return f"/explorer?t={t}"
