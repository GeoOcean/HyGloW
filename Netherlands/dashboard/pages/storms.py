"""Storms of 2013: the peaks of every estimate at every buoy, each event hour by hour, and the wind-only test."""

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

dash.register_page(__name__, path="/storms", name="Storms", order=5)


def layout(**_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    storms = data.storms()
    options = [{"label": f"{r.event} · offshore Hs {r.hs_off_peak:.1f} m (max goal {r.hs_off_max_peak:.1f})", "value": i}
               for i, r in enumerate(storms.itertuples())]
    buoys = data.sites().iloc[data.buoy_sites()]["site"].tolist()
    views = [{"label": "every estimate", "value": "all"}]
    if data.wind_only_available():
        views += [{"label": f"wind-only test · {off}", "value": off} for off in reversed(data.OFFSHORE)]
    return html.Div([
        ui.header("Storms", "Events where the mean offshore Hs at the goals (WHACS_cal) exceeds its 2013 p99, 48 h apart, "
                  "with 36 h either side. For each: the peak Hs of every estimate against the buoy's peak (within the "
                  "event, not at the same hour), and the series at any buoy."),
        dbc.Row([
            dbc.Col(ui.dropdown("st-event", options, 0, "Event"), md=5),
            dbc.Col(ui.dropdown("st-buoy", [{"label": b.replace("buoy_", ""), "value": b} for b in buoys],
                                "buoy_Europlatform" if "buoy_Europlatform" in buoys else buoys[0], "Buoy"), md=2),
            dbc.Col(ui.dropdown("st-view", views, "all", "Estimates"), md=3),
            dbc.Col(html.Div(id="st-link", className="mt-4"), md=2),
        ], className="controls g-2"),
        ui.card(None, ui.graph(id="st-timeline")),
        dbc.Row([
            dbc.Col(ui.card("Peak Hs error at each buoy (estimate peak − buoy peak)", ui.graph(id="st-peaks")), lg=6),
            dbc.Col(ui.card(None, ui.graph(id="st-series")), lg=6),
        ], className="g-2"),
        html.Div(id="st-wind-only"),
        ui.card("Peak errors per buoy (m)", ui.loading(html.Div(id="st-table"))),
        ui.card("All events", ui.table(storms[["event", "start", "end", "hs_off_peak", "hs_off_max_peak", "u10_peak",
                                                "u10dir_peak", "dir_off_peak"]].rename(columns=lambda c: c.replace("_", " ")),
                                       {"hs off peak": "{:.2f}", "hs off max peak": "{:.2f}", "u10 peak": "{:.1f}",
                                        "u10dir peak": "{:.0f}", "dir off peak": "{:.0f}"}, index=False)),
    ])


def _window(i: int) -> tuple[int, int, pd.Series]:
    s = data.storms().iloc[i]
    return data.time_index(s["start"]), data.time_index(s["end"]) + 1, s


def _keys(view: str) -> list[str]:
    return data.MODELS if view == "all" else data.wind_only_keys(view)


@callback(Output("st-timeline", "figure"), Output("st-peaks", "figure"), Output("st-table", "children"),
          Output("st-link", "children"), Input("st-event", "value"), Input("st-view", "value"))
def event(i, view):
    a, b, s = _window(i)
    keys = _keys(view)
    ftime = fg.timeline(index=data.time_index(s["peak"]), window=(a, b - 1), height=200)
    cols = data.buoy_sites()
    obs = data.field("buoy", "hs")[a:b][:, cols]
    has = np.isfinite(obs).sum(0) >= 0.5 * (b - a)
    peak_obs = np.where(has, np.nanmax(np.where(np.isfinite(obs), obs, -np.inf), 0), np.nan)
    rows = {}
    for key in keys:
        est = data.field(key, "hs")[a:b][:, cols]
        with np.errstate(invalid="ignore"):
            rows[data.label(key)] = np.nanmax(np.where(np.isfinite(est), est, -np.inf), 0) - peak_obs
    table = pd.DataFrame(rows, index=[n.replace("buoy_", "") for n in data.sites()["site"].to_numpy()[cols]])
    table = table[has]
    fig = go.Figure()
    for key in keys:
        meta = data.meta(key)
        fig.add_trace(go.Box(y=table[meta["label"]], name=meta["label"].replace(" · ", "<br>"), marker_color=meta["color"],
                             boxpoints="all", jitter=0.4, pointpos=0, line_width=1,
                             text=table.index, hovertemplate="%{text}: %{y:+.2f} m<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(showlegend=False, yaxis_title="peak Hs error (m)")
    med = table.median().rename("median").to_frame().T
    tab = ui.table(pd.concat([med, table.round(2)]).rename_axis("buoy"), {c: "{:+.2f}" for c in table.columns})
    link = dcc.Link("Open the peak in the Explorer →", href=f"/explorer?t={s['peak']:%Y-%m-%dT%H:%M}")
    return ftime, fg.base_layout(fig, 430, f"{int(has.sum())} buoys with data"), tab, link


@callback(Output("st-series", "figure"), Input("st-event", "value"), Input("st-buoy", "value"),
          Input("st-view", "value"))
def series(i, buoy, view):
    a, b, s = _window(i)
    col = int(np.flatnonzero(data.sites()["site"].to_numpy() == buoy)[0])
    keys = ["buoy", *_keys(view)]
    return fg.series(col, "hs", keys, index=data.time_index(s["peak"]), window=(a, b), height=430)


#: Site groups of the additivity table.
_TEST_GROUPS = {"5 m": "5m", "10 m": "10m", "20 m": "20m", "buoys": "buoy"}


@callback(Output("st-wind-only", "children"), Input("st-event", "value"), Input("st-view", "value"))
def wind_only(i, view):
    """Every hour of the event: each estimate against SnapWave with wind (is the wind additive?) and the buoys."""
    if view == "all":
        return None
    a, b, _ = _window(i)
    on = f"dyn_on_{view}"
    valid = np.isfinite(data.field("wind_only", "hs")[a:b])
    rows = []
    for key in (k for k in data.wind_only_keys(view) if k != on):
        est = data.field(key, "hs")[a:b]
        row = {"estimate": data.label(key)}
        for name, group in _TEST_GROUPS.items():
            m = data.site_mask(group)
            ref = np.where(valid, data.field(on, "hs")[a:b], np.nan)[:, m]
            met = data.pooled_metrics(ref, est[:, m], "hs")
            row[f"{name} bias"], row[f"{name} RMSE"] = met["bias"], met["rmse"]
        cols = data.buoy_sites()
        met = data.pooled_metrics(np.where(valid, data.field("buoy", "hs")[a:b], np.nan)[:, cols], est[:, cols], "hs")
        row["vs buoy bias"], row["vs buoy RMSE"] = met["bias"], met["rmse"]
        rows.append(row)
    table = pd.DataFrame(rows)
    n_hours = int(valid.any(1).sum())
    note = html.P(f"{n_hours} of {b - a} event hours with a converged wind-only run (pairs only there). "
                  "⊕ = energy sum, Hs = √(Hs² + Hs wind-only²). Additive wind: SnapWave no wind ⊕ wind only = "
                  "SnapWave wind (bias 0); a positive bias is the interaction (the wind grows an existing sea "
                  "less than calm water). Damped: the wind-only energy times f(share of the sea already there), "
                  "f learned on the SnapWave pairs of the other storms (Wind page). The metamodel rows are what "
                  "the reconstruction would give.",
                  className="small text-muted mb-1")
    fmt = {c: "{:+.2f}" if c.endswith("bias") else "{:.2f}" for c in table.columns if c != "estimate"}
    return ui.card(f"Wind-only test: Hs over the event hours (m), reference SnapWave wind ({view}) or the buoys",
                   [note, ui.table(table, fmt, index=False)])
