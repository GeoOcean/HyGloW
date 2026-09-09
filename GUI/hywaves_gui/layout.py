import dash_bootstrap_components as dbc
from dash import dcc, html

from .loaders import (
    load_case,
    load_case_config,
    load_catalog,
    offshore_names,
)
from .colors import CONTOUR_COLORS, CONTOUR_DEFAULT


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
                options=[{"label": c.get("case_dir", c["id"]), "value": c["id"]} for c in catalog],
                value=catalog[0]["id"],
                clearable=False,
                className="hywaves-dropdown",
            ),
        ],
        className="glass-panel controls-panel",
    )


def legend_rows(config: dict) -> list[html.Div]:
    """Legend rows for the current case — isobath rows are colored to match their map dots/lines."""
    rows = [
        html.Div([html.Span(className="dot aoi"), " Area of interest"], className="legend-row"),
        html.Div([html.Span(className="dot goal"), " Downscaling goals"], className="legend-row"),
        html.Div([html.Span(className="dot buoy"), " Wave observations"], className="legend-row"),
    ]
    isobaths = config.get("simulations", {}).get("output_sites_config", {}).get("isobaths") or []
    for spec in isobaths:
        name = spec.get("name")
        if not name:
            continue
        color = CONTOUR_COLORS.get(name, CONTOUR_DEFAULT)
        rows.append(
            html.Div(
                [html.Span(className="dot", style={"background": color}), f" {name} isobath (contour & sites)"],
                className="legend-row",
            )
        )
    if config.get("simulations", {}).get("output_sites_config", {}).get("polygons"):
        rows.append(html.Div([html.Span(className="dot site-mesh"), " Mesh output sites"], className="legend-row"))
    return rows


def legend_panel(config: dict) -> html.Div:
    return html.Div(html.Div(legend_rows(config), id="legend-body"), className="glass-panel legend-panel")


def results_drawer() -> html.Div:
    return html.Div(
        [
            html.Div(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Goals Validation", tab_id="tab-goals"),
                        dbc.Tab(label="Metamodel Validation", tab_id="tab-metamodel"),
                        dbc.Tab(label="Buoy Validation", tab_id="tab-buoys"),
                    ],
                    id="main-tabs",
                    active_tab="tab-goals",
                    className="drawer-tabs",
                ),
                className="drawer-header",
            ),
            html.Div(id="tab-content", className="drawer-body"),
        ],
        className="glass-panel results-drawer",
    )


def buoy_modal() -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(id="buoy-modal-title"), close_button=True),
            dbc.ModalBody(id="buoy-modal-body"),
        ],
        id="buoy-modal",
        is_open=False,
        size="xl",
        centered=True,
        className="buoy-modal",
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
                        html.A(
                            "📖 Report",
                            id="book-link",
                            href="#",
                            target="_blank",
                            rel="noopener",
                            className="header-link header-link-book",
                        ),
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
                    legend_panel(config),
                    results_drawer(),
                ],
                className="map-stage",
            ),
            buoy_modal(),
        ],
        className="app-shell",
    )
