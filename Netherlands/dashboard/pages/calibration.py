"""Calibration: the satellite fit per goal, the tail map of all goals, the deep-buoy check, and the change nearshore."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, html

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/calibration", name="Calibration", order=7)

#: The three stages of a calibrated hindcast: column in the passes table, label, colour.
STAGES = {"hs_raw": ("raw", "#8c8c8c"), "hs_directional": ("after the directional fit", "#1f77b4"),
          "hs_mapped": ("after the tail map (= _cal)", "#d62728")}
TAIL_P = 0.999


def layout(**_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    goals = sorted(data.goals()["goal"].astype(int))
    controls = dbc.Row([
        dbc.Col(ui.dropdown("cb-hindcast", ["WHACS", "ERA5"], "WHACS", "Hindcast"), md=2),
        dbc.Col(ui.dropdown("cb-goal", goals, goals[0], "Goal (top row)"), md=2),
        dbc.Col(ui.labelled("Stage of the passes", dbc.RadioItems(
            id="cb-stage", options=[{"label": v[0], "value": k} for k, v in STAGES.items()], value="hs_raw",
            inline=True)), md=8),
    ], className="controls g-2")
    return html.Div([
        ui.header("Calibration", "The offshore hindcast is calibrated against satellite altimetry (IMOS) in two steps. "
                  "1 · Directional fit, per goal: an Hs factor for each 22.5° sector, wind sea and swell apart "
                  "(Mínguez et al. 2011, energy fit, no intercept), on 75 % of the goal's passes; sectors with too "
                  "few passes stay at 1. 2 · Tail map, one for all goals: a quantile mapping fitted on the passes of "
                  "every goal pooled (one goal has too few storm passes), with a constant offset beyond p99.9. The "
                  "_cal hindcast is both steps. The map is fitted on every pass, so the passes cannot judge it: the deep "
                  "buoys can."),
        ui.card(None, controls),
        dbc.Row([
            dbc.Col(ui.card("Satellite passes at this goal", ui.graph(id="cb-pairs")), lg=4),
            dbc.Col(ui.card("Step 1 · Hs factor per sector at this goal", ui.graph(id="cb-coef")), lg=4),
            dbc.Col(ui.card("The whole record at this goal, 1979-2023", ui.graph(id="cb-record")), lg=4),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(ui.card("Step 2 · the tail map: passes of all goals, Q-Q against the satellite",
                            ui.graph(id="cb-map")), lg=5),
            dbc.Col(ui.card("Step 1 metrics at this goal (train passes fit it; test passes never seen)",
                            ui.loading(html.Div(id="cb-metrics"))), lg=7),
        ], className="g-2"),
        ui.card("Independent check: deep buoys near a goal node, 1979-2023 (no buoy is used in either step)", dbc.Row([
            dbc.Col([ui.loading(html.Div(id="cb-buoy-summary")), ui.loading(html.Div(id="cb-buoy-table"))], lg=7),
            dbc.Col([ui.dropdown("cb-buoy", [], None, "Buoy"), ui.graph(id="cb-buoy-qq")], lg=5),
        ], className="g-2")),
        ui.card("What the calibration changes in 2013 (WHACS_cal − WHACS), by quantile, from the goals to the coast",
                [dbc.Row([dbc.Col(ui.graph(id="cb-change"), lg=6),
                          dbc.Col(ui.graph(id="cb-change-map", config={"scrollZoom": True}), lg=6)], className="g-1"),
                 ui.dropdown("cb-q", [0.5, 0.9, 0.99, 0.999], 0.99, "Quantile on the map")]),
    ])


def _stats(reference: pd.Series, estimate: pd.Series) -> str:
    ok = reference.notna() & estimate.notna()
    reference, estimate = reference[ok], estimate[ok]
    d = estimate - reference
    q = np.quantile(estimate, 0.99) - np.quantile(reference, 0.99)
    return f"bias {d.mean():+.3f} · RMSE {np.sqrt((d ** 2).mean()):.3f} · p99 {q:+.2f} m"


@callback(Output("cb-pairs", "figure"), Output("cb-coef", "figure"), Output("cb-record", "figure"),
          Output("cb-metrics", "children"), Input("cb-hindcast", "value"), Input("cb-goal", "value"),
          Input("cb-stage", "value"))
def per_goal(hindcast, goal, stage):
    passes = data.calibration_table(hindcast, "passes")
    p = passes[passes["goal"].astype(int) == goal] if not passes.empty else passes
    if p.empty:
        fpairs = fg.empty("no passes")
    else:
        fpairs = go.Figure(go.Scattergl(x=p["sat_hs"], y=p[stage], mode="markers",
                                        marker=dict(size=4, color=STAGES[stage][1], opacity=0.6),
                                        hovertemplate="satellite %{x:.2f}, hindcast %{y:.2f}<extra></extra>"))
        hi = float(np.nanmax(np.r_[p["sat_hs"], p[stage]]))
        fpairs.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color="black", dash="dash")))
        text = "<br>".join(f"<span style='color:{c}'>{lab}</span>: {_stats(p['sat_hs'], p[k])}"
                           for k, (lab, c) in STAGES.items())
        fpairs.add_annotation(text=text, x=0.02, y=0.98, xref="paper", yref="paper", showarrow=False, align="left",
                              bgcolor="rgba(255,255,255,0.85)", font=dict(size=10))
        fpairs.update_layout(xaxis_title="satellite Hs (m)", yaxis_title=f"{hindcast} Hs, {STAGES[stage][0]} (m)",
                             showlegend=False)
        fpairs = fg.base_layout(fpairs, 380, f"{len(p):,} passes (hindcast − satellite)")
    coef = data.calibration_table(hindcast, "coefficients")
    c = coef[coef["goal"] == goal] if not coef.empty else coef
    fcoef = go.Figure()
    for fam, color in (("sea", "#d62728"), ("swell", "#1f77b4")):
        for fitted in (True, False):
            s = c[(c["family"] == fam) & (c["fitted"] == fitted)]
            if s.empty:
                continue
            fcoef.add_trace(go.Barpolar(r=s["factor"], theta=s["center"], marker_color=color if fitted else "#cfd4da",
                                        opacity=0.65, name=f"{fam}" + ("" if fitted else " (not fitted: 1)"),
                                        customdata=s["n"],
                                        hovertemplate=f"{fam} %{{theta}}°: × %{{r:.3f}} (%{{customdata}} passes)<extra></extra>"))
    top = max(1.3, float(c["factor"].max()) if not c.empty else 1.3)
    fcoef.update_layout(polar=dict(angularaxis=dict(direction="clockwise", rotation=90),
                                   radialaxis=dict(range=[0.7, top])), legend=dict(orientation="h", font=dict(size=10)))
    q = data.calibration_quantiles()
    qg = q[(q["hindcast"] == hindcast) & (q["goal"] == goal)]
    raw = qg[qg["source"] == hindcast].set_index("p")["hs"]
    cal = qg[qg["source"] == f"{hindcast}_cal"].set_index("p")["hs"]
    frec = go.Figure()
    if not raw.empty and not cal.empty:
        frec.add_trace(go.Scatter(x=raw.values, y=cal.values, mode="lines+markers", marker=dict(size=4, color="#d62728"),
                                  text=[f"p{100 * x:g}" for x in raw.index],
                                  hovertemplate="%{text}: %{x:.2f} → %{y:.2f} m<extra></extra>"))
        hi = float(np.nanmax(cal.values))
        frec.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color="black", dash="dash")))
        p999 = raw.loc[raw.index >= TAIL_P - 1e-9].iloc[0]
        frec.add_vline(x=p999, line=dict(color="#999", dash="dot"), annotation_text="p99.9", annotation_position="top")
    frec.update_layout(xaxis_title=f"{hindcast} Hs quantile (m)", yaxis_title=f"{hindcast}_cal Hs quantile (m)",
                       showlegend=False)
    metrics = data.calibration_table(hindcast, "metrics")
    m = metrics[metrics["goal"] == goal].drop(columns=["goal"]) if not metrics.empty else metrics
    m = m.replace({"stage": {"calibrated": "directional"}})
    table = ui.table(m, {"bias": "{:+.3f}", "rmse": "{:.3f}", "si": "{:.3f}", "r2": "{:.3f}", "slope": "{:.3f}"},
                     index=False)
    note = html.Div("Metrics of step 1 only (the tail map is fitted on every pass). Bias = hindcast − satellite.",
                    className="note mt-1")
    return (fpairs, fg.base_layout(fcoef, 380), fg.base_layout(frec, 380, "quantile of the calibrated vs raw record"),
            [table, note])


@callback(Output("cb-map", "figure"), Input("cb-hindcast", "value"))
def tail_map(hindcast):
    passes = data.calibration_table(hindcast, "passes")
    if passes.empty:
        return fg.empty("no passes")
    passes = passes.dropna(subset=["sat_hs", *STAGES])  # the same passes for every stage
    probs = np.r_[np.linspace(0.01, 0.99, 99), 0.995, 0.998, TAIL_P]
    sat = np.quantile(passes["sat_hs"], probs)
    fig = go.Figure()
    for k, (lab, color) in STAGES.items():
        est = np.quantile(passes[k], probs)
        fig.add_trace(go.Scatter(x=sat, y=est, mode="lines+markers", name=lab, marker=dict(size=4, color=color),
                                 line=dict(color=color), text=[f"p{100 * x:g}" for x in probs],
                                 hovertemplate="%{text}: satellite %{x:.2f}, hindcast %{y:.2f} m<extra></extra>"))
    hi = float(max(sat.max(), np.quantile(passes["hs_mapped"], TAIL_P)))
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color="black", dash="dash"),
                             showlegend=False))
    fig.add_vline(x=sat[-1], line=dict(color="#999", dash="dot"), annotation_text="p99.9: beyond, a constant offset",
                  annotation_position="top left")
    fig.update_layout(xaxis_title="satellite Hs quantile (m)", yaxis_title="hindcast Hs quantile (m)",
                      legend=dict(orientation="h", y=-0.2, font=dict(size=10)))
    return fg.base_layout(fig, 430, f"{len(passes):,} passes, {passes['goal'].nunique()} goals")


@callback(Output("cb-buoy-summary", "children"), Output("cb-buoy-table", "children"), Output("cb-buoy", "options"),
          Output("cb-buoy", "value"), Input("cb-hindcast", "value"))
def buoys(hindcast):
    m = data.calibration_table(hindcast, "buoy_metrics")
    sites = data.calibration_table(hindcast, "buoy_sites")
    if m.empty:
        return html.Div("no buoy check for this hindcast"), None, [], None
    cols = ["n", "bias", "rmse", "r2", "q50", "q99", "q99.9"]
    fmt = {"n": "{:,.0f}", "bias": "{:+.3f}", "rmse": "{:.3f}", "r2": "{:.3f}", "q50": "{:+.2f}", "q99": "{:+.2f}",
           "q99.9": "{:+.2f}"}
    summary = m.groupby("estimate")[cols].median().rename_axis("median of the buoys")
    per = m.pivot(index="site", columns="estimate", values=["bias", "rmse", "q99.9"])
    per.columns = [f"{v} {e}" for v, e in per.columns]
    per = per.join(sites.set_index("site")[["goal", "distance_km", "depth"]]).rename(index=lambda s: s.replace("buoy_", ""))
    per = per[["goal", "distance_km", "depth"] + [c for c in per.columns if c not in ("goal", "distance_km", "depth")]]
    per_fmt = {c: ("{:+.2f}" if ("bias" in c or "q99" in c) else "{:.2f}") for c in per.columns} | {
        "goal": "{:.0f}", "distance_km": "{:.0f}", "depth": "{:.0f}"}
    note = html.Div(f"{len(sites)} buoys deeper than 24 m within 50 km of their closest goal node; the hindcast at that "
                    "node, no propagation. q = hindcast − buoy Hs at that quantile, on the hours both have (quantiles, "
                    "not bins of observed Hs: those regress to the mean).", className="note mb-1")
    options = [{"label": s.replace("buoy_", ""), "value": s} for s in sites["site"]]
    return ([note, ui.table(summary, fmt)], ui.table(per.rename_axis("buoy"), per_fmt),
            options, options[0]["value"] if options else None)


@callback(Output("cb-buoy-qq", "figure"), Input("cb-hindcast", "value"), Input("cb-buoy", "value"))
def buoy_qq(hindcast, site):
    q = data.calibration_table(hindcast, "buoy_quantiles")
    if q.empty or site is None:
        return fg.empty("no buoy quantiles")
    q = q[q["site"] == site]
    fig = go.Figure()
    for (name, g), color in zip(q.groupby("estimate", sort=False), ("#8c8c8c", "#d62728")):
        fig.add_trace(go.Scatter(x=g["buoy"], y=g["hindcast"], mode="lines+markers", name=name, marker=dict(size=4),
                                 line=dict(color=color), text=[f"p{100 * x:g}" for x in g["p"]],
                                 hovertemplate="%{text}: buoy %{x:.2f}, hindcast %{y:.2f} m<extra></extra>"))
    hi = float(np.nanmax(q[["buoy", "hindcast"]].to_numpy()))
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color="black", dash="dash"),
                             showlegend=False))
    fig.update_layout(xaxis_title="buoy Hs quantile (m)", yaxis_title="hindcast Hs quantile (m)",
                      legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"))
    return fg.base_layout(fig, 400, f"{site.replace('buoy_', '')}: Q-Q, 1979-2023 hours with both")


@callback(Output("cb-change", "figure"), Output("cb-change-map", "figure"), Input("cb-q", "value"))
def change(q):
    probs = [0.5, 0.9, 0.99, 0.999]
    off = data.offshore()
    goal_change = [float(np.nanmedian(np.nanquantile(off["WHACS_cal_hs"].values, p, 0)
                                      - np.nanquantile(off["WHACS_hs"].values, p, 0))) for p in probs]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[f"p{100 * p:g}" for p in probs], y=goal_change, mode="lines+markers",
                             name="goals (offshore)", line=dict(color="black", width=3)))
    for label, color in (("20m", "#2c7bb6"), ("10m", "#74add1"), ("5m", "#fdae61")):
        cols = data.site_mask(label)
        a = data.field("mm_spec_WHACS", "hs")[:, cols]
        b = data.field("mm_spec_WHACS_cal", "hs")[:, cols]
        y = [float(np.nanmedian(np.nanquantile(b, p, 0) - np.nanquantile(a, p, 0))) for p in probs]
        fig.add_trace(go.Scatter(x=[f"p{100 * p:g}" for p in probs], y=y, mode="lines+markers",
                                 name=f"{label} (spectral metamodel)", line=dict(color=color)))
    fig.update_layout(yaxis_title="median change of the Hs quantile (m)", legend=dict(orientation="h", y=1.12))
    cols = np.flatnonzero(data.sites()["source"].isin(["isobath", "mesh"]).to_numpy())
    a = data.field("mm_spec_WHACS", "hs")[:, cols]
    b = data.field("mm_spec_WHACS_cal", "hs")[:, cols]
    d = np.nanquantile(b, q, 0) - np.nanquantile(a, q, 0)
    fmap = fg.site_map(d, cols, title=f"change of the p{100 * q:g} Hs, spectral metamodel (m)", diverging=True,
                       colorbar="m", height=420, uirevision="cb")
    return fg.base_layout(fig, 420), fmap
