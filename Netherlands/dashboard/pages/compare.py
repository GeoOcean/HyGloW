"""Compare: any estimate against any reference, over any subset of hours (filters or box selection)."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/compare", name="Compare", order=2)

METRICS = {"bias": "bias (estimate − reference)", "rmse": "RMSE", "r2": "R²", "qbias99": "p99 bias",
           "qbias999": "p99.9 bias", "ref_mean": "mean of the reference", "est_mean": "mean of the estimate"}
PRESETS = {
    "everything": {},
    "storms: offshore Hs > 4 m": {"hsoff": [4, 8]},
    "SW wind ≥ 10 m/s": {"u10": [10, 22], "u10dir": [225]},
    "NW wind ≥ 10 m/s": {"u10": [10, 22], "u10dir": [315]},
    "offshore wind (E-SE) ≥ 8 m/s": {"u10": [8, 22], "u10dir": [90, 135]},
    "calm: U10 < 5 m/s": {"u10": [0, 5]},
    "winter (Dec-Feb)": {"months": [12, 1, 2]},
}


def layout(ref: str | None = None, est: str | None = None, v: str | None = None, **_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    ref = ref if ref in data.ESTIMATES else "dyn_off_WHACS_cal"
    est = est if est in data.ESTIMATES else "mm_bulk_WHACS_cal"
    left = ui.card("What and when", [
        ui.estimate_dropdown("cp-ref", ref, "Reference (x)"),
        ui.estimate_dropdown("cp-est", est, "Estimate (y)"),
        dbc.Row([dbc.Col(ui.variable_dropdown("cp-var", v if v in data.VARIABLES else "hs")),
                 dbc.Col(ui.group_dropdown("cp-group", "all sites"))], className="g-1"),
        ui.dropdown("cp-metric", [{"label": v, "value": k} for k, v in METRICS.items()], "bias", "Map metric"),
        html.Hr(className="my-2"),
        ui.dropdown("cp-preset", list(PRESETS), "everything", "Preset"),
        ui.condition_filters("cp"),
        html.Div("Box- or lasso-select hours on the scatter plots below; double-click to clear.", className="note"),
        ui.graph(id="cp-xf-off", config={"displayModeBar": False}),
        ui.graph(id="cp-xf-wind", config={"displayModeBar": False}),
    ])
    right = html.Div([
        html.Div(id="cp-summary", className="fw-semibold mb-2"),
        dbc.Row([dbc.Col(ui.card(None, ui.graph(id="cp-map", config={"scrollZoom": True})), lg=7),
                 dbc.Col(ui.card(None, ui.graph(id="cp-scatter")), lg=5)], className="g-2"),
        dbc.Row([dbc.Col(ui.card(None, ui.graph(id="cp-heat")), lg=7),
                 dbc.Col(ui.card("Per site group (pooled pairs)", ui.loading(html.Div(id="cp-table"))), lg=5)], className="g-2"),
        ui.card("Mean difference in time (sites of the group), selected hours highlighted", ui.graph(id="cp-time")),
    ])
    return html.Div([
        ui.header("Compare", "Any estimate against any reference, on any subset of hours. The conditions come from the "
                  "estimate's offshore source (WHACS_cal for the buoys). Every panel uses the same hours."),
        dbc.Row([dbc.Col(left, lg=3), dbc.Col(right, lg=9)], className="g-2"),
        dcc.Store(id="cp-hours"),
    ])


def _source(est: str, ref: str) -> str:
    return data.ESTIMATES[est]["offshore"] or data.ESTIMATES[ref]["offshore"] or "WHACS_cal"


@callback(Output("cp-months", "value"), Output("cp-u10", "value"), Output("cp-u10dir", "value"),
          Output("cp-hsoff", "value"), Output("cp-diroff", "value"), Input("cp-preset", "value"),
          prevent_initial_call=True)
def preset(name):
    p = PRESETS[name]
    return (p.get("months", []), p.get("u10", [0, 22]), p.get("u10dir", ui.SECTORS), p.get("hsoff", [0, 8]),
            p.get("diroff", ui.SECTORS))


@callback(Output("cp-xf-off", "figure"), Output("cp-xf-wind", "figure"),
          Input("cp-est", "value"), Input("cp-ref", "value"), *ui.filter_inputs("cp"))
def condition_plots(est, ref, months, u10, u10dir, hsoff, diroff):
    """Every hour of the source; red where the filters keep it. Redrawn only when the filters change, so a box
    selection on them stays."""
    source = _source(est, ref)
    mask = ui.mask_from_filters(source, months, u10, u10dir, hsoff, diroff)
    c = data.conditions_of(source)
    figs = []
    for x, y, xt, yt in (("dir_off", "hs_off", "waves from (deg)", "offshore Hs (m)"),
                         ("u10dir", "u10", "wind from (deg)", "U10 (m/s)")):
        fig = go.Figure(go.Scattergl(x=c[x], y=c[y], mode="markers",
                                     marker=dict(size=3, color=np.where(mask, "#d62728", "#c9ced6")),
                                     selected=dict(marker=dict(color="#08519c", size=4)),
                                     unselected=dict(marker=dict(opacity=0.35)),
                                     text=c["time"].dt.strftime("%d %b %H:%M"), hovertemplate="%{text}<extra></extra>"))
        fig.update_layout(dragmode="select", xaxis=dict(title=xt, range=[0, 360], dtick=90), yaxis=dict(title=yt),
                          showlegend=False)
        figs.append(fg.base_layout(fig, 230))
    return figs


@callback(Output("cp-hours", "data"), Input("cp-est", "value"), Input("cp-ref", "value"), *ui.filter_inputs("cp"),
          Input("cp-xf-off", "selectedData"), Input("cp-xf-wind", "selectedData"))
def hours(est, ref, months, u10, u10dir, hsoff, diroff, sel_off, sel_wind):
    """Hours kept by the filters and by any box selection on the condition plots (the point index is the hour)."""
    mask = ui.mask_from_filters(_source(est, ref), months, u10, u10dir, hsoff, diroff)
    for sel in (sel_off, sel_wind):
        if sel and sel.get("points"):
            keep = np.zeros(len(mask), bool)
            keep[[p.get("pointIndex", p.get("pointNumber")) for p in sel["points"]]] = True
            mask &= keep
    return np.flatnonzero(mask).tolist()


@callback(Output("cp-summary", "children"), Output("cp-map", "figure"), Output("cp-scatter", "figure"),
          Output("cp-heat", "figure"), Output("cp-table", "children"), Output("cp-time", "figure"),
          Input("cp-hours", "data"), Input("cp-ref", "value"), Input("cp-est", "value"), Input("cp-var", "value"),
          Input("cp-group", "value"), Input("cp-metric", "value"))
def update(hours, ref, est, var, group, metric):
    hours = np.asarray(hours or [], int)
    if ref == est:
        e = fg.empty("pick two different estimates")
        return "", e, e, e, "", e
    if hours.size == 0:
        e = fg.empty("no hours match the filters")
        return "0 hours", e, e, e, "", e
    cols = np.flatnonzero(data.site_mask(data.GROUPS[group]))
    q = {"qbias99": 0.99, "qbias999": 0.999}.get(metric)
    every = hours.size == len(data.times())
    Rh = data.field(ref, var) if every else data.field(ref, var)[hours]
    Eh = data.field(est, var) if every else data.field(est, var)[hours]
    R, E = Rh[:, cols], Eh[:, cols]
    m = data.array_metrics(R, E, var, q)
    key = "qbias" if q else metric
    values = np.where(m["n"] >= 3, m[key].to_numpy(), np.nan)  # sparse sites still show; hover gives n
    diverging = key in ("bias", "qbias")
    cmap = {"rmse": "Magma_r", "r2": "Viridis", "ref_mean": fg.COLORSCALES[var], "est_mean": fg.COLORSCALES[var]}.get(key)
    fmap = fg.site_map(values, cols, title=f"{METRICS[metric]} · {data.VARIABLE_LABELS[var]}", diverging=diverging,
                       colorscale=cmap or "Viridis", zmin=0 if key == "rmse" else (None if key != "r2" else 0),
                       zmax=None if key != "r2" else 1, colorbar=data.VARIABLE_LABELS[var] if key != "r2" else "R²",
                       height=520, counts=m["n"].to_numpy(), uirevision="cp")
    fscatter = fg.density_scatter(R.ravel(), E.ravel(), var, data.label(ref), data.label(est), height=520)
    # difference by stretch of coast x wind sector at the site
    D = data.difference(E, R, var)
    wsec = data.sector((data.field("wind", "u10dir") if every else data.field("wind", "u10dir")[hours])[:, cols])
    regions = list(data.COAST_REGIONS)
    rcode = pd.Categorical(data.sites()["region"].to_numpy()[cols], categories=regions).codes
    code = (np.broadcast_to(rcode, D.shape) * 8 + np.clip(wsec, 0, None) // 45).ravel()
    ok = np.isfinite(D).ravel() & (wsec.ravel() >= 0)
    sums = np.bincount(code[ok], weights=D.ravel()[ok], minlength=24)
    counts = np.bincount(code[ok], minlength=24)
    with np.errstate(invalid="ignore"):
        heat = pd.DataFrame((sums / counts).reshape(3, 8), index=regions,
                            columns=[ui.SECTOR_NAMES[s] for s in ui.SECTORS])
    fheat = fg.heatmap(heat, f"mean difference by stretch of coast × ERA5 wind direction at the site · {group}",
                       height=280)
    rows = {}
    for name, g in data.GROUPS.items():
        if g is None:
            continue
        gc = np.flatnonzero(data.site_mask(g))
        rr, ee = Rh[:, gc], Eh[:, gc]
        rows[name] = data.pooled_metrics(rr.ravel(), ee.ravel(), var)
    tab = ui.table(pd.DataFrame(rows).T.rename_axis("sites"), {"n": "{:,.0f}", "bias": "{:+.3f}", "rmse": "{:.3f}",
                                                                "r2": "{:.2f}"})
    t = data.times()
    with np.errstate(invalid="ignore"):
        full = np.nanmean(data.difference(data.field(est, var)[:, cols], data.field(ref, var)[:, cols], var), 1)
    ftime = go.Figure()
    ftime.add_trace(go.Scattergl(x=t, y=full, mode="lines", line=dict(color="#c9ced6", width=1), name="all hours"))
    ftime.add_trace(go.Scattergl(x=t[hours], y=full[hours], mode="markers", marker=dict(size=3, color="#d62728"),
                                 name="selected"))
    ftime.add_hline(y=0, line=dict(color="black", width=1))
    ftime.update_layout(yaxis_title=f"mean {var} difference", legend=dict(orientation="h", y=1.1))
    summary = (f"{hours.size:,} hours selected · {int(m['n'].sum()):,} pairs · {len(cols):,} sites · "
               f"{data.label(est)} against {data.label(ref)}")
    return summary, fmap, fscatter, fheat, tab, fg.base_layout(ftime, 220)
