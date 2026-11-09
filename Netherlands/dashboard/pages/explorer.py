"""Explorer: any two estimates at any hour of 2013, their difference, and the series of any site."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/explorer", name="Explorer", order=1)

WINDOWS = {"±3 days": 72, "±15 days": 360, "year": None}
DEFAULT_SITE = "buoy_Europlatform"


def _column(site: str | None) -> int:
    table = data.sites()
    hit = np.flatnonzero(table["site"].to_numpy() == (site or DEFAULT_SITE))
    return int(hit[0]) if hit.size else int(data.buoy_sites()[0])


def layout(t: str | None = None, a: str | None = None, b: str | None = None, v: str | None = None,
           site: str | None = None, **_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    index = data.time_index(t or "2013-12-05 18:00")
    a = a if a in data.ESTIMATES else "dyn_on_WHACS_cal"
    b = b if b in data.ESTIMATES else "mm_spec_WHACS_cal"
    v = v if v in data.VARIABLES else "hs"
    n = len(data.times())
    marks = {int(i): m for i, m in zip(np.searchsorted(data.times(), pd.date_range("2013-01-01", periods=12, freq="MS")),
                                       ui.MONTHS.values())}
    controls = dbc.Row([
        dbc.Col(ui.variable_dropdown("ex-var", v), md=2),
        dbc.Col(ui.estimate_dropdown("ex-a", a, "Map A"), md=3),
        dbc.Col(ui.estimate_dropdown("ex-b", b, "Map B (difference = A − B)"), md=3),
        dbc.Col(ui.group_dropdown("ex-group"), md=2),
        dbc.Col(ui.labelled("Layers", dcc.Checklist(id="ex-layers", options=[{"label": " ERA5 wind", "value": "wind"},
                                                                              {"label": " goals", "value": "goals"},
                                                                              {"label": " same scale all year", "value": "fixed"}],
                                                     value=["wind", "goals"], inputStyle={"margin-right": "4px"})), md=2),
    ], className="controls g-2")
    clock = dbc.Row([
        dbc.Col(dbc.ButtonGroup([dbc.Button("−1 d", id="ex-m24", size="sm", outline=True, color="secondary"),
                                 dbc.Button("−1 h", id="ex-m1", size="sm", outline=True, color="secondary"),
                                 dbc.Button("▶ play", id="ex-play", size="sm", color="primary"),
                                 dbc.Button("+1 h", id="ex-p1", size="sm", outline=True, color="secondary"),
                                 dbc.Button("+1 d", id="ex-p24", size="sm", outline=True, color="secondary")]), width="auto"),
        dbc.Col(dcc.Input(id="ex-time", type="text", value=str(data.times()[index])[:16], debounce=True,
                          style={"width": "150px"}), width="auto"),
        dbc.Col(html.Div(id="ex-label", className="fw-semibold"), width="auto"),
        dbc.Col(dcc.Clipboard(id="ex-share", title="copy a link to this view", style={"fontSize": 18}), width="auto"),
        dbc.Col(html.Div(id="ex-url", className="note"), width=True),
    ], className="g-2 align-items-center mb-1")
    return html.Div([
        ui.header("Explorer", "Two estimates at one hour, side by side, and their difference. Squares: the offshore "
                  "sea at each goal, coloured on the map's scale, with an arrow towards where the waves go (length ∝ Hs). "
                  "Shading and grey arrows: ERA5 wind speed and direction (towards). Click a site on any map for its "
                  "series; click the timeline to jump; play steps one hour per second."),
        ui.card(None, [controls, clock,
                       dcc.Slider(id="ex-hour", min=0, max=n - 1, step=1, value=index, marks=marks, updatemode="mouseup",
                                  allow_direct_input=False),
                       ui.graph(id="ex-timeline", config={"displayModeBar": False})]),
        dbc.Row([dbc.Col(ui.card(None, ui.graph(id=f"ex-map-{k}", config={"scrollZoom": True})), lg=4)
                 for k in ("a", "b", "d")], className="g-2"),
        dbc.Row([
            dbc.Col(ui.card("Series at the selected site", [
                dbc.Row([
                    dbc.Col(ui.labelled("Window", dbc.RadioItems(id="ex-window", options=list(WINDOWS), value="±3 days",
                                                                 inline=True)), width="auto"),
                    dbc.Col(ui.labelled("Estimates", dcc.Dropdown(id="ex-keys", options=data.estimate_options(),
                                                                  value=["buoy", "offshore_WHACS_cal", "mm_spec_WHACS_cal",
                                                                         "mm_bulk_WHACS_cal", "dyn_off_WHACS_cal",
                                                                         "dyn_on_WHACS_cal", "swan_on_WHACS_cal"],
                                                                  multi=True)), width=True),
                ], className="controls"),
                ui.graph(id="ex-series")]), lg=8),
            dbc.Col(ui.card("Skill at this site, 2013", ui.loading(html.Div(id="ex-site-table"))), lg=4),
        ], className="g-2"),
        dcc.Store(id="ex-site", data=_column(site)),
        dcc.Interval(id="ex-tick", interval=1000, disabled=True),
    ])


@callback(Output("ex-hour", "value"), Output("ex-time", "value"),
          Input("ex-m24", "n_clicks"), Input("ex-m1", "n_clicks"), Input("ex-p1", "n_clicks"), Input("ex-p24", "n_clicks"),
          Input("ex-tick", "n_intervals"), Input("ex-timeline", "clickData"), Input("ex-time", "value"),
          Input("ex-hour", "value"), prevent_initial_call=True)
def move(m24, m1, p1, p24, tick, click, typed, value):
    trigger = ctx.triggered_id
    n = len(data.times())
    step = {"ex-m24": -24, "ex-m1": -1, "ex-p1": 1, "ex-p24": 24, "ex-tick": 1}.get(trigger)
    if step is not None:
        value = int(np.clip(value + step, 0, n - 1))
    elif trigger == "ex-timeline" and click:
        value = data.time_index(click["points"][0]["x"])
    elif trigger == "ex-time":
        try:
            value = data.time_index(typed)
        except (ValueError, TypeError):
            return no_update, no_update
    return value, str(data.times()[value])[:16]


@callback(Output("ex-tick", "disabled"), Output("ex-play", "children"), Input("ex-play", "n_clicks"),
          State("ex-tick", "disabled"), prevent_initial_call=True)
def play(_, disabled):
    return (False, "⏸ pause") if disabled else (True, "▶ play")


@callback(Output("ex-timeline", "figure"), Input("ex-hour", "value"))
def timeline(index):
    return fg.timeline(index=index, height=170)


@callback(Output("ex-map-a", "figure"), Output("ex-map-b", "figure"), Output("ex-map-d", "figure"),
          Output("ex-label", "children"), Output("ex-url", "children"), Output("ex-share", "content"),
          Input("ex-hour", "value"), Input("ex-var", "value"), Input("ex-a", "value"), Input("ex-b", "value"),
          Input("ex-group", "value"), Input("ex-layers", "value"), State("ex-site", "data"))
def maps(index, var, a, b, group, layers, column):
    cols = np.flatnonzero(data.site_mask(data.GROUPS[group]))
    va, vb = data.hour(a, var, index)[cols], data.hour(b, var, index)[cols]
    if var == "dir":
        zmin, zmax = 0, 360
    elif "fixed" in layers:
        both = np.r_[data.field(a, var)[:, cols].ravel(), data.field(b, var)[:, cols].ravel()]
        zmin, zmax = 0, float(np.nanquantile(both, 0.999))
    else:
        both = np.r_[va, vb]
        zmin, zmax = (float(np.nanmin(both)), float(np.nanmax(both))) if np.isfinite(both).any() else (0, 1)
    cs = fg.COLORSCALES[var]
    unit = data.VARIABLE_LABELS[var]
    goals = []
    if "goals" in layers:
        goals = fg.goal_traces(data.ESTIMATES[a]["offshore"] or "WHACS_cal", index, var, cs, zmin, zmax)
    wind_layers, wind_arrows = fg.wind_layer(index) if "wind" in layers else ([], [])
    common = dict(colorscale=cs, zmin=zmin, zmax=zmax, colorbar=unit, under=wind_arrows, layers=wind_layers,
                  extra=goals, uirevision="ex")
    fa = fg.site_map(va, cols, title=f"A · {data.label(a)}", **common)
    fb = fg.site_map(vb, cols, title=f"B · {data.label(b)}", **common)
    d = data.difference(va, vb, var)
    fd = fg.site_map(d, cols, title=f"A − B · mean {np.nanmean(d):+.2f}, |max| {np.nanmax(np.abs(d)) if np.isfinite(d).any() else np.nan:.2f}",
                     diverging=True, colorbar=unit, under=wind_arrows, layers=wind_layers, uirevision="ex")
    c = data.conditions_of(data.ESTIMATES[a]["offshore"] or "WHACS_cal").iloc[index]
    label = (f"{data.times()[index]:%a %d %b %Y %H:%M} · offshore Hs {c['hs_off']:.2f} m (max {c['hs_off_max']:.2f}) "
             f"from {c['dir_off']:.0f}° · ERA5 {c['u10']:.1f} m/s from {c['u10dir']:.0f}° · WL {c['wl']:+.2f} m"
             + ("" if c["wind_converged"] else " · ⚠ run with wind did not converge"))
    url = f"/explorer?t={data.times()[index]:%Y-%m-%dT%H:%M}&a={a}&b={b}&v={var}&site={data.sites()['site'].iloc[column]}"
    return fa, fb, fd, label, url, url


@callback(Output("ex-site", "data"), Input("ex-map-a", "clickData"), Input("ex-map-b", "clickData"),
          Input("ex-map-d", "clickData"), prevent_initial_call=True)
def pick(*clicks):
    click = clicks[["ex-map-a", "ex-map-b", "ex-map-d"].index(ctx.triggered_id)]
    point = click["points"][0] if click and click.get("points") else None
    if not point:
        return no_update
    custom = point.get("customdata")
    if isinstance(custom, list) and len(custom) == 2 and custom[1] is not None:  # a site: [value, column]
        return int(custom[1])
    if "lat" not in point:
        return no_update
    table = data.sites()  # a click on a goal, an arrow or the shading: the nearest site
    d = (table["lat"] - point["lat"]) ** 2 + ((table["lon"] - point["lon"]) * np.cos(np.deg2rad(point["lat"]))) ** 2
    return int(d.to_numpy().argmin())


@callback(Output("ex-series", "figure"), Output("ex-site-table", "children"), Input("ex-site", "data"),
          Input("ex-var", "value"), Input("ex-window", "value"), Input("ex-keys", "value"), Input("ex-hour", "value"))
def series(column, var, window, keys, index):
    span = WINDOWS[window]
    n = len(data.times())
    win = None if span is None else (max(0, index - span), min(n, index + span))
    fig = fg.series(column, var, keys or [], index=index, window=win)
    has_buoy = np.isfinite(data.field("buoy", var)[:, column]).any()
    reference = "buoy" if has_buoy else "dyn_on_WHACS_cal"
    rows = {}
    for key in data.MODELS:
        if key == reference:
            continue
        r, e = data.field(reference, var)[:, column], data.field(key, var)[:, column]
        rows[data.label(key)] = data.pooled_metrics(r, e, var)
    table = pd.DataFrame(rows).T.rename_axis("estimate")
    return fig, html.Div([html.Div(f"reference: {data.label(reference)}", className="note mb-1"),
                          ui.table(table, {"n": "{:,.0f}", "bias": "{:+.3f}", "rmse": "{:.3f}", "r2": "{:.2f}"})])
