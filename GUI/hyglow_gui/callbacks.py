import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html, no_update
import dash_bootstrap_components as dbc

from .layout import case_panel
from .loaders import (
    list_figures,
    load_all_metrics,
    load_aoi,
    load_case,
    load_case_config,
    load_goals,
    load_observations,
    mda_candidates,
    production_mda,
    region_bounds,
)
from .map_figure import build_map

CHART_TEMPLATE = "plotly_dark"
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(15, 23, 42, 0)",
    plot_bgcolor="rgba(15, 23, 42, 0.35)",
    font=dict(color="#e2e8f0"),
    margin=dict(l=48, r=24, t=48, b=48),
)


def register_callbacks(app):
    @app.callback(
        Output("case-sidebar", "children"),
        Output("mda-select", "options"),
        Output("mda-select", "value"),
        Output("goal-select", "options"),
        Output("goal-select", "value"),
        Output("figures-store", "data"),
        Output("region-store", "data"),
        Input("case-select", "value"),
    )
    def on_case_change(case_id):
        case = load_case(case_id)
        config = load_case_config(case)
        metrics = load_all_metrics(case)
        goals = load_goals(case, config)
        observations = load_observations(case, config)
        aoi = load_aoi(case, config)
        candidates = mda_candidates(case, config, metrics)
        default_mda = production_mda(candidates)

        goal_options = [{"label": f"Goal {g}", "value": g} for g in sorted(goals["goal"].unique(), key=int)]
        if not metrics.empty:
            metric_goals = sorted(metrics["goal"].unique(), key=lambda g: int(g))
            goal_options = [{"label": f"Goal {g}", "value": g} for g in metric_goals]

        return (
            case_panel(case, config),
            [{"label": str(n), "value": n} for n in candidates],
            default_mda,
            goal_options,
            goal_options[0]["value"] if goal_options else None,
            [p.name for p in list_figures(case)],
            region_bounds(case, config, goals, observations, aoi),
        )

    @app.callback(
        Output("goal-select", "value", allow_duplicate=True),
        Input("domain-map", "clickData"),
        prevent_initial_call=True,
    )
    def map_select_goal(click_data):
        if not click_data or not click_data.get("points"):
            return no_update
        point = click_data["points"][0]
        custom = point.get("customdata")
        if custom is not None:
            return str(custom)
        return no_update

    @app.callback(
        Output("domain-map", "figure"),
        Input("case-select", "value"),
        Input("goal-select", "value"),
        State("region-store", "data"),
    )
    def update_map(case_id, goal_id, bounds):
        case = load_case(case_id)
        config = load_case_config(case)
        goals = load_goals(case, config)
        buoys = load_observations(case, config)
        aoi = load_aoi(case, config)
        return build_map(
            goals,
            buoys,
            bounds or {"lon": [-180, 180], "lat": [-90, 90]},
            selected_goal=goal_id,
            aoi=aoi,
        )

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "active_tab"),
        Input("case-select", "value"),
        Input("goal-select", "value"),
        Input("mda-select", "value"),
        State("figures-store", "data"),
    )
    def render_tab(tab, case_id, goal_id, mda_n, figure_names):
        metrics = load_all_metrics(load_case(case_id))
        if metrics.empty:
            return dbc.Alert(
                "No validation_metrics.csv found under outputs/goal_*/. Run val.sweep() in the case first.",
                color="warning",
                className="drawer-alert",
            )

        if goal_id is None or mda_n is None:
            return dbc.Alert("Select a goal and MDA size.", color="secondary", className="drawer-alert")

        subset = metrics[(metrics["goal"] == str(goal_id)) & (metrics["mda_candidate"] == mda_n)]

        if tab == "tab-metrics":
            fig = px.bar(
                subset,
                x="group",
                y="rmse",
                color="variable",
                barmode="group",
                title=f"RMSE by site group — goal {goal_id}, MDA={mda_n}",
                template=CHART_TEMPLATE,
                color_discrete_sequence=["#22d3ee", "#a78bfa", "#34d399", "#fb923c"],
            )
            fig.update_layout(**CHART_LAYOUT)
            table = dash_table.DataTable(
                data=subset.to_dict("records"),
                columns=[{"name": c, "id": c} for c in subset.columns],
                page_size=8,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#1e293b",
                    "color": "#e2e8f0",
                    "border": "1px solid #334155",
                },
                style_cell={
                    "backgroundColor": "#0f172a",
                    "color": "#cbd5e1",
                    "border": "1px solid #334155",
                    "fontSize": "0.85rem",
                },
            )
            return html.Div([dcc.Graph(figure=fig, config={"displayModeBar": False}), table], className="drawer-charts")

        if tab == "tab-sweep":
            sweep = metrics[
                (metrics["goal"] == str(goal_id))
                & (metrics["variable"] == "hs")
                & (metrics["group"] == "10m")
            ]
            if sweep.empty:
                sweep = metrics[(metrics["goal"] == str(goal_id)) & (metrics["variable"] == "hs")]
            fig = px.line(
                sweep,
                x="mda_candidate",
                y="r2",
                markers=True,
                title=f"Holdout R² (Hs) vs MDA size — goal {goal_id}",
                template=CHART_TEMPLATE,
            )
            fig.update_traces(line=dict(color="#22d3ee", width=3), marker=dict(size=10))
            fig.update_layout(**CHART_LAYOUT)
            return html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}), className="drawer-charts")

        cards = [
            html.Div(
                [
                    html.Div(name.replace("_", " ").replace(".png", ""), className="gallery-caption"),
                    html.Img(src=f"/case-figures/{case_id}/{name}", className="gallery-img"),
                ],
                className="gallery-item",
            )
            for name in (figure_names or [])
        ]
        if not cards:
            return dbc.Alert("No course figures found in outputs/course_figures/.", color="secondary", className="drawer-alert")
        return html.Div(cards, className="gallery-grid")
