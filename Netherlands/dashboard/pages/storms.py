"""Storms of 2013: the peaks of every estimate at every buoy, and each event hour by hour."""

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
    return html.Div([
        ui.header("Storms", "Events where the mean offshore Hs at the goals (WHACS_cal) exceeds its 2013 p99, 48 h apart, "
                  "with 36 h either side. For each: the peak Hs of every estimate against the buoy's peak (within the "
                  "event, not at the same hour), and the series at any buoy."),
        dbc.Row([
            dbc.Col(ui.dropdown("st-event", options, 0, "Event"), md=6),
            dbc.Col(ui.dropdown("st-buoy", [{"label": b.replace("buoy_", ""), "value": b} for b in buoys],
                                "buoy_Europlatform" if "buoy_Europlatform" in buoys else buoys[0], "Buoy"), md=3),
            dbc.Col(html.Div(id="st-link", className="mt-4"), md=3),
        ], className="controls g-2"),
        ui.card(None, ui.graph(id="st-timeline")),
        dbc.Row([
            dbc.Col(ui.card("Peak Hs error at each buoy (estimate peak − buoy peak)", ui.graph(id="st-peaks")), lg=6),
            dbc.Col(ui.card(None, ui.graph(id="st-series")), lg=6),
        ], className="g-2"),
        ui.card("Peak errors per buoy (m)", ui.loading(html.Div(id="st-table"))),
        ui.card("All events", ui.table(storms[["event", "start", "end", "hs_off_peak", "hs_off_max_peak", "u10_peak",
                                                "u10dir_peak", "dir_off_peak"]].rename(columns=lambda c: c.replace("_", " ")),
                                       {"hs off peak": "{:.2f}", "hs off max peak": "{:.2f}", "u10 peak": "{:.1f}",
                                        "u10dir peak": "{:.0f}", "dir off peak": "{:.0f}"}, index=False)),
    ])


def _window(i: int) -> tuple[int, int, pd.Series]:
    s = data.storms().iloc[i]
    return data.time_index(s["start"]), data.time_index(s["end"]) + 1, s


@callback(Output("st-timeline", "figure"), Output("st-peaks", "figure"), Output("st-table", "children"),
          Output("st-link", "children"), Input("st-event", "value"))
def event(i):
    a, b, s = _window(i)
    ftime = fg.timeline(index=data.time_index(s["peak"]), window=(a, b - 1), height=200)
    cols = data.buoy_sites()
    obs = data.field("buoy", "hs")[a:b][:, cols]
    has = np.isfinite(obs).sum(0) >= 0.5 * (b - a)
    peak_obs = np.where(has, np.nanmax(np.where(np.isfinite(obs), obs, -np.inf), 0), np.nan)
    rows = {}
    for key in data.MODELS:
        est = data.field(key, "hs")[a:b][:, cols]
        with np.errstate(invalid="ignore"):
            rows[data.label(key)] = np.nanmax(np.where(np.isfinite(est), est, -np.inf), 0) - peak_obs
    table = pd.DataFrame(rows, index=[n.replace("buoy_", "") for n in data.sites()["site"].to_numpy()[cols]])
    table = table[has]
    fig = go.Figure()
    for key in data.MODELS:
        meta = data.ESTIMATES[key]
        fig.add_trace(go.Box(y=table[meta["label"]], name=meta["label"].replace(" · ", "<br>"), marker_color=meta["color"],
                             boxpoints="all", jitter=0.4, pointpos=0, line_width=1,
                             text=table.index, hovertemplate="%{text}: %{y:+.2f} m<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(showlegend=False, yaxis_title="peak Hs error (m)")
    med = table.median().rename("median").to_frame().T
    tab = ui.table(pd.concat([med, table.round(2)]).rename_axis("buoy"), {c: "{:+.2f}" for c in table.columns})
    link = dcc.Link("Open the peak in the Explorer →", href=f"/explorer?t={s['peak']:%Y-%m-%dT%H:%M}")
    return ftime, fg.base_layout(fig, 430, f"{int(has.sum())} buoys with data"), tab, link


@callback(Output("st-series", "figure"), Input("st-event", "value"), Input("st-buoy", "value"))
def series(i, buoy):
    a, b, s = _window(i)
    col = int(np.flatnonzero(data.sites()["site"].to_numpy() == buoy)[0])
    return fg.series(col, "hs", list(data.ESTIMATES), index=data.time_index(s["peak"]), window=(a, b), height=430)
