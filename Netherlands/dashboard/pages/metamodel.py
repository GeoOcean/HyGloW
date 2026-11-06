"""Metamodel: the training cases of each goal, their fields, holdout skill vs MDA size, the linear sum."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, ctx, dcc, html, no_update

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/metamodel", name="Metamodel", order=6)

ROLE_COLORS = {"train": "#1f77b4", "test": "#ff7f0e", "unused": "#d0d4da"}
#: Trace order of the input plots (a click gives the trace and the point index, not the customdata).
ROLES = ("unused", "test", "train")


def layout(**_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    cases = data.cases()
    goals = sorted(cases["goal"].unique())
    hold = data.holdout()
    controls = dbc.Row([
        dbc.Col(ui.dropdown("mm-goal", [int(g) for g in goals], int(goals[0]), "Goal"), md=2),
        dbc.Col(ui.dropdown("mm-field", [{"label": "Hs", "value": "hs"}, {"label": "Dir", "value": "dir"},
                                        {"label": "Hs / offshore Hs", "value": "ratio"}], "ratio", "Case field"), md=2),
    ], className="controls g-2")
    sweep = dbc.Row([
        dbc.Col(ui.dropdown("mm-set", sorted(hold["metamodel"].unique()), "isobaths", "Set"), md=2),
        dbc.Col(ui.dropdown("mm-label", sorted(hold["site_label"].astype(str).unique()), "5m", "Site group"), md=2),
        dbc.Col(ui.dropdown("mm-var", sorted(hold["variable"].unique()), "hs", "Variable"), md=2),
        dbc.Col(ui.dropdown("mm-metric", ["rmse", "bias", "r2", "si"], "rmse", "Metric"), md=2),
    ], className="controls g-2")
    ls = data.linear_sum_metrics()
    ls_table = (ls.pivot_table(index=["offshore_name", "site_label"], columns="variable", values=["bias", "rmse", "r2"])
                if not ls.empty else pd.DataFrame())
    if not ls_table.empty:
        ls_table.columns = [f"{v} {m}" for m, v in ls_table.columns]
        ls_table = ls_table[sorted(ls_table.columns)]
    n_train = int((cases["role"] == "train").sum() / len(goals))
    return html.Div([
        ui.header("Metamodel", f"Per goal: the {len(cases) // len(goals)} SnapWave cases kept from the LHS (steep seas "
                  f"dropped), the {n_train} MDA training cases (blue) and the holdout cases (orange); click a case to "
                  "see its field. Then the holdout skill of each goal against the MDA size, and the linear sum of all "
                  "goals against the dynamic runs without wind (2013)."),
        ui.card(None, controls),
        dbc.Row([
            dbc.Col(ui.card("Training cases (MDA inputs)", [ui.graph(id="mm-hsdir"), ui.graph(id="mm-hstp")]), lg=5),
            dbc.Col(ui.card(None, [ui.graph(id="mm-case", config={"scrollZoom": True}),
                                   html.Div(id="mm-case-info", className="note")]), lg=7),
        ], className="g-2"),
        ui.card("Holdout skill against MDA size (test cases of each goal)", [sweep, ui.graph(id="mm-sweep")]),
        ui.card("Linear sum of all goals vs dynamic run without wind, 2013 (bulk metamodel, MDA 600)",
                ui.table(ls_table, {c: ("{:+.3f}" if "bias" in c else "{:.3f}") for c in ls_table.columns})
                if not ls_table.empty else html.Div("no linear-sum metrics in outputs/metamodel_validation/")),
        dcc.Store(id="mm-case-id"),
    ])


@callback(Output("mm-hsdir", "figure"), Output("mm-hstp", "figure"), Input("mm-goal", "value"))
def inputs(goal):
    c = data.cases()
    c = c[c["goal"] == goal]
    figs = []
    for x, y, xt in (("dir", "hs", "dir (deg, from)"), ("tp", "hs", "Tp (s)")):
        fig = go.Figure()
        for role in ROLES:
            s = c[c["role"] == role]
            fig.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=f"{role} ({len(s)})",
                                       customdata=s["case_num"],
                                       marker=dict(size=6 if role != "unused" else 4, color=ROLE_COLORS[role],
                                                   line=dict(width=0.5, color="white")),
                                       hovertemplate="case %{customdata}<br>" + f"{x} %{{x:.1f}}, Hs %{{y:.2f}}<extra></extra>"))
        fig.update_layout(xaxis_title=xt, yaxis_title="Hs (m)", legend=dict(orientation="h", y=1.12), clickmode="event")
        figs.append(fg.base_layout(fig, 300))
    return figs


@callback(Output("mm-case-id", "data"), Input("mm-hsdir", "clickData"), Input("mm-hstp", "clickData"),
          Input("mm-goal", "value"))
def pick(c1, c2, goal):
    click = {"mm-hsdir": c1, "mm-hstp": c2}.get(ctx.triggered_id)
    c = data.cases()
    c = c[c["goal"] == goal]
    if click and click.get("points"):
        point = click["points"][0]
        cases = c[c["role"] == ROLES[point["curveNumber"]]]
        return int(cases.iloc[point.get("pointIndex", point.get("pointNumber"))]["case_num"])
    return int(c[c["role"] == "train"].sort_values("hs").iloc[-1]["case_num"])


@callback(Output("mm-case", "figure"), Output("mm-case-info", "children"), Input("mm-case-id", "data"),
          Input("mm-goal", "value"), Input("mm-field", "value"))
def case_map(case, goal, field):
    var = "dir" if field == "dir" else "hs"
    values, sites, removed = data.case_fields(goal, var)
    c = data.cases()
    c = c[c["goal"] == goal].reset_index(drop=True)  # rows in the order of the goal's case file
    hit = np.flatnonzero(c["case_num"].to_numpy() == case)
    if hit.size == 0:  # the case of the previous goal: the new one arrives next
        return no_update, no_update
    row = c.iloc[hit[0]]
    table = data.sites()
    pos = pd.Series(np.arange(len(sites)), index=sites)
    cols = np.flatnonzero(table["site"].isin(sites).to_numpy())
    v = values[hit[0]][pos.loc[table["site"].to_numpy()[cols]].to_numpy()]
    if field == "ratio":
        v = v / row["hs"]
    g = data.goals().set_index("goal").loc[goal]
    cs = {"hs": "Viridis", "ratio": "Viridis", "dir": "Twilight"}[field]
    zmin, zmax = {"dir": (0, 360), "ratio": (0, 1.2)}.get(field, (0, float(np.nanmax(v)) if np.isfinite(v).any() else 1))
    goal_value = {"hs": row["hs"], "ratio": 1.0, "dir": row["dir"]}[field]  # the forcing on the map's scale
    color = fg.sample_colorscale(cs, [float(np.clip((goal_value - zmin) / ((zmax - zmin) or 1), 0, 1))])
    extra = fg.goal_glyphs([g["lat"]], [g["lon"]], color, [row["dir"]], [row["hs"]],
                           [f"goal {goal} · case {case}<br>Hs {row['hs']:.2f} m, Tp {row['tp']:.1f} s, "
                            f"from {row['dir']:.0f}°"], labels=[f"goal {goal}"])
    fig = fg.site_map(v, cols, title=f"goal {goal} · case {case} ({row['role']})", colorscale=cs, zmin=zmin, zmax=zmax,
                      colorbar={"hs": "m", "ratio": "Hs / Hs offshore", "dir": "deg"}[field], extra=extra, height=560,
                      uirevision="mm")
    info = (f"forcing: Hs {row['hs']:.2f} m, Tp {row['tp']:.1f} s, from {row['dir']:.0f}°, spreading {row['spr']:.0f}°, "
            f"water level {row['wl']:+.2f} m · {len(removed)} sites removed for this goal by the site filter")
    return fig, info


@callback(Output("mm-sweep", "figure"), Input("mm-set", "value"), Input("mm-label", "value"), Input("mm-var", "value"),
          Input("mm-metric", "value"))
def sweep(mset, label, var, metric):
    h = data.holdout()
    h = h[(h["metamodel"] == mset) & (h["site_label"].astype(str) == label) & (h["variable"] == var)]
    if h.empty:
        return fg.empty("no holdout metrics for this choice")
    fig = go.Figure()
    for goal, g in h.groupby("goal"):
        g = g.sort_values("mda_candidate")
        fig.add_trace(go.Scatter(x=g["mda_candidate"], y=g[metric], mode="lines+markers", name=f"goal {goal}"))
    med = h.groupby("mda_candidate")[metric].median()
    fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines", name="median", line=dict(color="black", width=4)))
    fig.update_layout(xaxis_title="MDA size (training cases)", yaxis_title=f"{metric} ({var})", hovermode="x unified")
    return fg.base_layout(fig, 380)
