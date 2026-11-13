"""Error budget at the buoys: where the Hs error of the product comes from, per buoy and per condition."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/budget", name="Error budget", order=9)

PARTS = [name for name, _, _ in data.BUDGET]


def layout(**_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    left = ui.card("Hours", [
        ui.dropdown("bd-source", list(data.OFFSHORE), "WHACS_cal", "Offshore source"),
        ui.condition_filters("bd"),
        ui.dropdown("bd-part", PARTS + ["total (spectral − buoy)"], "missing wind (no wind − wind)", "Part on the map"),
    ])
    return html.Div([
        ui.header("Error budget at the buoys",
                  "The mean Hs error of the product (spectral metamodel − buoy) is the sum of four parts: partitions "
                  "(spectral − bulk metamodel), method (bulk metamodel − dynamic run without wind, like for like), "
                  "missing wind (dynamic without − with wind) and model + forcing (dynamic with wind − buoy). Only "
                  "pairs valid in every estimate count, so the parts add up. Quantile classes are each buoy's own; a "
                  "class picked by the observation regresses to the mean, so read storm skill on the Compare page "
                  "(p99 bias), not here."),
        dbc.Row([
            dbc.Col(left, lg=3),
            dbc.Col([
                html.Div(id="bd-summary", className="fw-semibold mb-2"),
                dbc.Row([dbc.Col(ui.card("By Hs quantile class (each buoy's own)", ui.graph(id="bd-classes")), lg=5),
                         dbc.Col(ui.card(None, ui.graph(id="bd-map", config={"scrollZoom": True})), lg=7)], className="g-2"),
                ui.card("Per buoy, south to north", ui.graph(id="bd-buoys")),
            ], lg=9),
        ], className="g-2"),
    ])


def _bars(table, x, height, title=None):
    fig = go.Figure()
    for (name, _, _), color in zip(data.BUDGET, data.BUDGET_COLORS):
        fig.add_trace(go.Bar(x=x, y=table[name], name=name, marker_color=color))
    fig.add_trace(go.Scatter(x=x, y=table["total (spectral − buoy)"], mode="markers", name="total (spectral − buoy)",
                             marker=dict(symbol="diamond", size=10, color="black")))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(barmode="relative", yaxis_title="mean Hs error (m)", legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
    return fg.base_layout(fig, height, title)


@callback(Output("bd-summary", "children"), Output("bd-classes", "figure"), Output("bd-map", "figure"),
          Output("bd-buoys", "figure"), Input("bd-source", "value"), *ui.filter_inputs("bd"), Input("bd-part", "value"))
def update(source, months, u10, u10dir, hsoff, diroff, part):
    mask = ui.mask_from_filters(source, months, u10, u10dir, hsoff, diroff)
    if not mask.any():
        e = fg.empty("no hours match the filters")
        return "0 hours", e, e, e
    per_buoy, by_class = data.budget(source, np.flatnonzero(mask))
    t = by_class.T
    fclass = _bars(t, [f"{c}<br>n={int(n):,}" for c, n in zip(t.index, t["pairs"])], 430)
    cols = data.buoy_sites()
    fmap = fg.site_map(per_buoy[part].to_numpy(), cols, title=f"{part} (m)", diverging=True, colorbar="m", size=6,
                       height=430, uirevision="bd")
    order = np.argsort(data.sites()["lat"].to_numpy()[cols])
    pb = per_buoy.iloc[order]
    fbuoys = _bars(pb, [s.replace("buoy_", "") for s in pb.index], 420)
    summary = f"{int(mask.sum()):,} hours · {int(by_class.loc['pairs', 'all']):,} buoy-hour pairs valid in every estimate · {source}"
    return summary, fclass, fmap, fbuoys
