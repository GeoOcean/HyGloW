"""Numerical tests: how the SWAN and SnapWave templates were chosen (numerical_tests/), interactively."""

from __future__ import annotations

import base64
import re

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/numerical-tests", name="Numerical tests", order=8)


def _readme_sections() -> dict[str, str]:
    path = data.numerical_tests_dir() / "README.md"
    if not path.exists():
        return {}
    parts = re.split(r"^## ", path.read_text(), flags=re.M)
    return {p.splitlines()[0].strip(): "\n".join(p.splitlines()[1:]) for p in parts[1:]}


def _csv(name: str) -> pd.DataFrame:
    path = data.numerical_tests_dir() / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def layout(**_):
    if not data.available() or not data.numerical_tests_dir().exists():
        return dbc.Alert("store/numerical_tests is missing: run `python build_store.py --only numerical_tests`.", color="danger")
    sections = _readme_sections()
    scores = _csv("buoy_scores.csv")
    experiments = list(dict.fromkeys(scores["experiment"])) if not scores.empty else []
    ratios = _csv("metamodel_ratios.csv")
    figure = data.numerical_tests_dir() / "figures" / "snapwave_enclosure.png"
    img = (html.Img(src="data:image/png;base64," + base64.b64encode(figure.read_bytes()).decode(), style={"width": "100%"})
           if figure.exists() else html.Div())
    conv = _csv("snapwave_convergence_2013.csv")
    unstable = _csv("snapwave_unstable_hours.csv")
    return html.Div([
        ui.header("Numerical tests", "Why the SWAN and SnapWave templates look the way they do: every option tested on the "
                  "Netherlands case, against 27 buoys where possible (39 test hours: 8 calm, 20 January-February storm, "
                  "11 Xaver; 120 other hours to check), and the two models on the same goal cases. Source: "
                  "Netherlands/numerical_tests/."),
        ui.card("Decisions", dcc.Markdown(sections.get("Decisions", ""), className="markdown")),
        dbc.Row([
            dbc.Col(ui.card("Buoy scores of each experiment (Hs; bias = model − buoy)", [
                dbc.Row([dbc.Col(ui.dropdown("nt-exp", experiments, experiments[0] if experiments else None, "Experiment")),
                         dbc.Col(ui.dropdown("nt-metric", ["rmse", "bias"], "rmse", "Metric"))], className="controls g-2"),
                ui.graph(id="nt-scores"), html.Div(id="nt-scores-note", className="note")]), lg=7),
            dbc.Col(ui.card("Goal cases: Hs ratio to the reference model by steepness", [
                ui.dropdown("nt-ratio-exp", list(dict.fromkeys(ratios["experiment"])) if not ratios.empty else [],
                            ratios["experiment"].iloc[0] if not ratios.empty else None, "Experiment"),
                ui.graph(id="nt-ratios")]), lg=5),
        ], className="g-2"),
        dbc.Row([
            dbc.Col(ui.card("SnapWave boundary enclosure: AoI rectangle vs open boundary", img), lg=6),
            dbc.Col([ui.card("SnapWave convergence over 2013 (final error of each hour)", ui.table(conv, index=False)),
                     ui.card("Unstable hours, by variant", ui.table(unstable, index=False))], lg=6),
        ], className="g-2"),
        ui.card("What was learned, in order", dcc.Markdown(sections.get("What was learned, in order", ""), className="markdown")),
        dbc.Row([
            dbc.Col(ui.card("Test design", dcc.Markdown(sections.get("Test design", ""), className="markdown")), lg=6),
            dbc.Col(ui.card("Open points", dcc.Markdown(sections.get("Open points", ""), className="markdown")), lg=6),
        ], className="g-2"),
    ])


@callback(Output("nt-scores", "figure"), Output("nt-scores-note", "children"), Input("nt-exp", "value"),
          Input("nt-metric", "value"))
def scores(experiment, metric):
    s = _csv("buoy_scores.csv")
    s = s[s["experiment"] == experiment]
    if s.empty:
        return fg.empty("no scores"), ""
    fig = px.bar(s, x="group", y=metric, color="label", barmode="group", hover_data=["n", "variant", "source"])
    fig.update_layout(legend=dict(orientation="h", y=-0.3, font=dict(size=10)), xaxis_title="", yaxis_title=f"{metric} (m)")
    return fg.base_layout(fig, 470), f"{s['hours'].iloc[0]} · groups: calm / storm / xaver hours, or observed Hs and depth classes"


@callback(Output("nt-ratios", "figure"), Input("nt-ratio-exp", "value"))
def ratios(experiment):
    r = _csv("metamodel_ratios.csv")
    r = r[r["experiment"] == experiment]
    if r.empty:
        return fg.empty("no ratios")
    fig = px.line(r, x="steepness", y="ratio", color="variant", line_dash="where", markers=True,
                  hover_data=["reference"])
    fig.add_hline(y=1, line=dict(color="black", width=1))
    fig.update_layout(legend=dict(orientation="h", y=-0.35, font=dict(size=10)),
                      yaxis_title=f"Hs ratio to {r['reference'].iloc[0]}")
    return fg.base_layout(fig, 470)
