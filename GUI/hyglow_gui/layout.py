import dash_bootstrap_components as dbc
from dash import dcc, html

from .loaders import (
    list_figures,
    load_aoi,
    load_case,
    load_case_config,
    load_catalog,
    load_goals,
    load_observations,
    offshore_names,
    region_bounds,
)


def _author_links(authors: list[dict]) -> list:
    return [
        html.A(a["name"], href=a["github"], target="_blank", className="author-link")
        for a in authors
    ]


def _pipeline_pill_class(value) -> str:
    status = str(value).lower()
    if status == "done":
        return "done"
    if status in ("todo", "pending", "not_started"):
        return "todo"
    # Not "progress" — that class name collides with Bootstrap's built-in
    # .progress (progress-bar) component, which forces height:1rem and
    # overflow:hidden and silently clips the pill's text mid-line.
    return "wip"


def _pipeline_badges(case: dict) -> html.Div:
    pipeline = case.get("pipeline", {})
    return html.Div(
        [
            html.Span(
                key.replace("_", " "),
                className=f"status-pill {_pipeline_pill_class(value)}",
                title=str(value),
            )
            for key, value in pipeline.items()
        ],
        className="status-grid",
    )


def case_panel(case: dict, config: dict) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span("USE CASE", className="panel-kicker"),
                    html.H2(config.get("name", case["id"]), className="panel-title"),
                ],
                className="panel-heading",
            ),
            html.P(config.get("description", ""), className="panel-desc"),
            html.Dl(
                [
                    html.Dt("Institution"),
                    html.Dd(case["institution"]),
                    html.Dt("Contributors"),
                    html.Dd(html.Div(_author_links(case.get("authors", [])))),
                    html.Dt("Offshore forcing"),
                    html.Dd(", ".join(offshore_names(config)) or "—"),
                    html.Dt("Repository"),
                    html.Dd(html.A("HyGloW on GitHub", href=case["repository"], target="_blank")),
                    html.Dt("Last update"),
                    html.Dd(case.get("updated", "—")),
                ],
                className="meta-list",
            ),
            html.Hr(className="panel-divider"),
            html.Span("Pipeline status", className="panel-kicker"),
            _pipeline_badges(case),
        ],
        className="glass-panel case-panel",
    )


def controls_panel(catalog: list[dict]) -> html.Div:
    return html.Div(
        [
            html.Span("INSPECT", className="panel-kicker"),
            html.Label("Use case", className="control-label"),
            dcc.Dropdown(
                id="case-select",
                options=[{"label": c["id"].title(), "value": c["id"]} for c in catalog],
                value=catalog[0]["id"],
                clearable=False,
                className="hyglow-dropdown",
            ),
            html.Label("Goal", className="control-label"),
            dcc.Dropdown(id="goal-select", clearable=False, className="hyglow-dropdown"),
            html.Label("MDA train size", className="control-label"),
            dcc.Dropdown(id="mda-select", clearable=False, className="hyglow-dropdown"),
            html.P("Click a goal on the map to inspect it.", className="map-hint"),
        ],
        className="glass-panel controls-panel",
    )


def legend_panel() -> html.Div:
    return html.Div(
        [
            html.Div([html.Span(className="dot aoi"), " Area of interest"], className="legend-row"),
            html.Div([html.Span(className="dot goal"), " Downscaling goals"], className="legend-row"),
            html.Div([html.Span(className="dot selected"), " Selected goal"], className="legend-row"),
            html.Div([html.Span(className="dot buoy"), " Wave observations"], className="legend-row"),
            html.Div([html.Span(className="dot isobath-line"), " Isobath contours"], className="legend-row"),
            html.Div([html.Span(className="dot site-isobath"), " Isobath output sites"], className="legend-row"),
            html.Div([html.Span(className="dot site-mesh"), " Mesh output sites"], className="legend-row"),
        ],
        className="glass-panel legend-panel",
    )


def results_drawer() -> html.Div:
    return html.Div(
        [
            html.Div(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Validation", tab_id="tab-metrics"),
                        dbc.Tab(label="MDA sweep", tab_id="tab-sweep"),
                        dbc.Tab(label="Gallery", tab_id="tab-gallery"),
                    ],
                    id="main-tabs",
                    active_tab="tab-metrics",
                    className="drawer-tabs",
                ),
                className="drawer-header",
            ),
            html.Div(id="tab-content", className="drawer-body"),
        ],
        className="glass-panel results-drawer",
    )


def gallery_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id="gallery-modal-title"), close_button=True),
            dbc.ModalBody(html.Img(id="gallery-modal-img", className="gallery-modal-img")),
        ],
        id="gallery-modal",
        is_open=False,
        size="xl",
        centered=True,
        className="gallery-modal",
    )


def header() -> html.Div:
    return html.Header(
        html.Div(
            [
                html.Div(
                    [
                        html.H1("HyGloW", className="brand-title"),
                        html.P(
                            "Hybrid Global / Local Wave downscaling — open use-case registry",
                            className="brand-subtitle",
                        ),
                    ],
                    className="brand-block",
                ),
                html.Div(
                    [
                        html.A("GitHub", href="https://github.com/tausiaj/HyGloW", target="_blank", className="header-link"),
                        html.Span("GeoOcean", className="header-tag"),
                    ],
                    className="header-actions",
                ),
            ],
            className="header-inner",
        ),
        className="app-header",
    )


def build_layout() -> html.Div:
    catalog = load_catalog()
    case = load_case(catalog[0]["id"])
    config = load_case_config(case)
    goals = load_goals(case, config)
    observations = load_observations(case, config)
    aoi = load_aoi(case, config)
    bounds = region_bounds(case, config, goals, observations, aoi)
    figures = list_figures(case)

    map_config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["toImage", "lasso2d", "select2d"],
        "displaylogo": False,
    }

    return html.Div(
        [
            header(),
            html.Div(
                [
                    dcc.Graph(id="domain-map", className="basemap", config=map_config),
                    html.Div(case_panel(case, config), id="case-sidebar"),
                    controls_panel(catalog),
                    legend_panel(),
                    results_drawer(),
                ],
                className="map-stage",
            ),
            dcc.Store(id="figures-store", data=[p.name for p in figures]),
            dcc.Store(id="region-store", data=bounds),
            gallery_modal(),
        ],
        className="app-shell",
    )
