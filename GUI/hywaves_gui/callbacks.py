import plotly.express as px
from dash import Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc

from .colors import CHART_LAYOUT, CHART_TEMPLATE, CONTOUR_COLORS, MODEL_COLORS
from .layout import case_panel, legend_rows
from .loaders import (
    available_buoy_figures,
    book_available,
    case_mda_candidate,
    load_all_metrics,
    load_aoi,
    load_buoy_validation,
    load_case,
    load_case_config,
    load_goals,
    load_isobaths,
    load_metamodel_validation,
    load_observations,
    load_output_sites,
    mda_candidates,
    production_mda,
    region_bounds,
)
from .map_figure import build_map

VARIABLE_ORDER = ["Hs", "Tp", "Dir", "Spr"]
MODEL_ORDER = ["Bulk metamodel", "Spectral metamodel", "WHACS (closest goal)"]


def _sort_isobath_groups(groups) -> list[str]:
    """Order contour-band group names ("5m", "10m", ...) by depth, not text."""
    try:
        return sorted(groups, key=lambda g: float(str(g).rstrip("m")))
    except ValueError:
        return sorted(groups)


def _ordered(present, preferred: list[str]) -> list[str]:
    """*preferred* items that are actually present, in that order, then any extras."""
    present = set(present)
    return [v for v in preferred if v in present] + sorted(present - set(preferred))


def _variable_order(labels) -> list[str]:
    return _ordered(labels, VARIABLE_ORDER)


def _effective_mda(case: dict, config: dict, metrics) -> int | None:
    """Fixed MDA train size for the validation tabs: catalog value, else the largest fitted candidate."""
    return case_mda_candidate(case) or production_mda(mda_candidates(case, config, metrics))


def _facet_by_variable(fig):
    """Give each variable's facet its own y-axis and a bare title.

    Hs (metres), Tp (near-zero — preserved by construction), and Dir/Spr
    (degrees) live on wildly different scales; sharing one axis makes the
    smaller ones look like zero error instead of just a different unit.
    """
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(showticklabels=False, title=None)
    return fig


def _render_goals_validation(case: dict):
    """Holdout RMSE spread across every goal, one box per contour × variable."""
    config = load_case_config(case)
    metrics = load_all_metrics(case)
    if metrics.empty:
        return dbc.Alert(
            "No validation_metrics.csv found under outputs/goal_*/. Run val.sweep() for every goal first.",
            color="warning",
            className="drawer-alert",
        )

    mda = _effective_mda(case, config, metrics)
    subset = metrics[(metrics["site_source"] == "isobath") & (metrics["mda_candidate"] == mda)]
    if subset.empty:
        return dbc.Alert(f"No isobath validation rows at MDA={mda}.", color="secondary", className="drawer-alert")

    fig = px.box(
        subset,
        x="group",
        y="rmse",
        color="group",
        points="all",
        facet_col="label",
        facet_col_wrap=4,
        category_orders={
            "label": _variable_order(subset["label"].unique()),
            "group": _sort_isobath_groups(subset["group"].unique()),
        },
        title=f"Holdout RMSE across {subset['goal'].nunique()} goals — MDA={mda}",
        template=CHART_TEMPLATE,
        color_discrete_map=CONTOUR_COLORS,
        labels={"rmse": "RMSE", "group": "Contour"},
    )
    _facet_by_variable(fig)
    fig.update_layout(**CHART_LAYOUT, legend_title_text="Contour")
    return html.Div(dcc.Graph(figure=fig, config={"displayModeBar": False}), className="drawer-charts")


def _render_metamodel_validation(case: dict):
    """Domain-wide metamodel-vs-dynamic accuracy from a single reconstructed period."""
    dyn = load_metamodel_validation(case)
    if dyn.empty:
        return dbc.Alert(
            "No metamodel_validation CSV found under outputs/metamodel_validation/. "
            "Run 09_Validate_Linear_Summation.ipynb (with its save-metrics cell) first.",
            color="warning",
            className="drawer-alert",
        )

    mda = case_mda_candidate(case)
    if mda is not None and "mda_candidate" in dyn.columns:
        dyn = dyn[dyn["mda_candidate"] == mda]

    subset = dyn[dyn["site_source"] == "isobath"]
    if subset.empty:
        return dbc.Alert("No isobath rows in the metamodel validation export.", color="secondary", className="drawer-alert")

    fig = px.bar(
        subset,
        x="group",
        y="rmse",
        color="group",
        facet_col="label",
        facet_col_wrap=4,
        category_orders={
            "label": _variable_order(subset["label"].unique()),
            "group": _sort_isobath_groups(subset["group"].unique()),
        },
        title="Linear summation vs. dynamic SnapWave — domain-wide RMSE",
        template=CHART_TEMPLATE,
        color_discrete_map=CONTOUR_COLORS,
        labels={"rmse": "RMSE", "group": "Contour"},
    )
    _facet_by_variable(fig)
    fig.update_layout(**CHART_LAYOUT, legend_title_text="Contour")

    meta = subset.iloc[0]
    caption = (
        f"Period {meta['period']} · MDA={meta['mda_candidate']} · "
        f"{int(meta['n_sites']):,} sites reconstructed in {meta['reconstruct_seconds']:.2f}s"
    )
    return html.Div(
        [html.Div(caption, className="chart-caption"), dcc.Graph(figure=fig, config={"displayModeBar": False})],
        className="drawer-charts",
    )


def _buoy_section(subset, *, title: str):
    """One boxplot panel (RMSE across every buoy, faceted by variable) for one buoy subset."""
    n_buoys = subset["group"].nunique()
    period = ", ".join(sorted(subset["period"].unique()))
    fig = px.box(
        subset,
        x="model",
        y="rmse",
        color="model",
        points="all",
        facet_col="label",
        facet_col_wrap=4,
        category_orders={
            "label": _variable_order(subset["label"].unique()),
            "model": _ordered(subset["model"].unique(), MODEL_ORDER),
        },
        title=f"{title} — RMSE across {n_buoys} buoys",
        template=CHART_TEMPLATE,
        color_discrete_map=MODEL_COLORS,
        labels={"rmse": "RMSE", "model": "Model"},
    )
    _facet_by_variable(fig)
    fig.update_layout(**CHART_LAYOUT, legend_title_text="Model")
    return html.Div(
        [
            html.Div(f"Period: {period}", className="chart-caption"),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ]
    )


def _render_buoy_validation(case: dict):
    """RMSE boxplots across every buoy, one panel each for offshore and onshore — mirrors Goals Validation."""
    buoys = load_buoy_validation(case)
    if buoys.empty:
        return dbc.Alert(
            "No buoy_validation CSV found under outputs/buoy_validation/. "
            "Run 10a_Validate_Offshore_Buoys.ipynb / 10b_Validate_Onshore_Buoys.ipynb "
            "(with their save-summary cells) first.",
            color="warning",
            className="drawer-alert",
        )

    mda = case_mda_candidate(case)
    if mda is not None and "mda_candidate" in buoys.columns:
        buoys = buoys[buoys["mda_candidate"] == mda]
    if buoys.empty:
        return dbc.Alert(f"No buoy validation rows at MDA={mda}.", color="secondary", className="drawer-alert")

    sections = [
        _buoy_section(subset, title=buoy_type.title() + " buoys")
        for buoy_type in ("offshore", "onshore")
        if not (subset := buoys[buoys["buoy_type"] == buoy_type]).empty
    ]
    if not sections:
        return dbc.Alert("No buoy validation rows to plot.", color="secondary", className="drawer-alert")
    return html.Div(sections, className="drawer-charts")


def register_callbacks(app):
    @app.callback(
        Output("book-link", "href"),
        Output("book-link", "className"),
        Output("book-link", "title"),
        Input("case-select", "value"),
    )
    def on_case_change_book_link(case_id):
        case = load_case(case_id)
        if book_available(case):
            return f"/case-book/{case_id}/index.html", "header-link header-link-book", "Open the rendered notebook report"
        return "#", "header-link header-link-book disabled", "No report built yet for this case"

    @app.callback(
        Output("case-sidebar", "children"),
        Output("legend-body", "children"),
        Input("case-select", "value"),
    )
    def on_case_change(case_id):
        case = load_case(case_id)
        config = load_case_config(case)
        return case_panel(case, config), legend_rows(config)

    @app.callback(
        Output("domain-map", "figure"),
        Input("case-select", "value"),
    )
    def update_map(case_id):
        case = load_case(case_id)
        config = load_case_config(case)
        goals = load_goals(case, config)
        buoys = load_observations(case, config)
        aoi = load_aoi(case, config)
        isobaths = load_isobaths(case, config)
        output_sites = load_output_sites(case, config)
        bounds = region_bounds(case, config, goals, buoys, aoi)
        return build_map(
            goals,
            buoys,
            bounds,
            case_id=case_id,
            aoi=aoi,
            isobaths=isobaths,
            output_sites=output_sites,
        )

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "active_tab"),
        Input("case-select", "value"),
    )
    def render_tab(tab, case_id):
        case = load_case(case_id)

        if tab == "tab-goals":
            return _render_goals_validation(case)

        if tab == "tab-metamodel":
            return _render_metamodel_validation(case)

        return _render_buoy_validation(case)

    @app.callback(
        Output("buoy-modal", "is_open"),
        Output("buoy-modal-title", "children"),
        Output("buoy-modal-body", "children"),
        Input("domain-map", "clickData"),
        State("case-select", "value"),
        prevent_initial_call=True,
    )
    def open_buoy_modal(click_data, case_id):
        if not click_data or not click_data.get("points"):
            return no_update, no_update, no_update
        custom = click_data["points"][0].get("customdata")
        if not custom:
            return no_update, no_update, no_update

        site_id, name = custom[0], custom[1]
        case = load_case(case_id)
        if site_id not in available_buoy_figures(case):
            body = dbc.Alert(
                "No validation summary generated yet for this buoy. Run "
                "10a_Validate_Offshore_Buoys.ipynb / 10b_Validate_Onshore_Buoys.ipynb "
                "(with their save-summary-PNG cell) first.",
                color="secondary",
                className="drawer-alert",
            )
        else:
            body = html.Img(src=f"/buoy-figures/{case_id}/{site_id}.png", className="buoy-modal-img")
        return True, name, body
