import dash_bootstrap_components as dbc
from dash import dcc, html

from .colors import CONTOUR_COLORS, CONTOUR_DEFAULT
from .loaders import (
    catalog_by_continent,
    load_case,
    load_case_config,
    load_catalog,
    offshore_names,
)
from .map_figure import CASE_MAP_CONFIG, DEFAULT_SANDBOX_LAYERS, SANDBOX_LAYER_LABELS


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
                    html.Dd(
                        html.A(
                            "HyGloW on GitHub", href=case["repository"], target="_blank"
                        )
                    ),
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
                options=[
                    {"label": c.get("case_dir", c["id"]), "value": c["id"]}
                    for c in catalog
                ],
                value=catalog[0]["id"],
                clearable=False,
                className="hywaves-dropdown",
            ),
        ],
        id="controls-panel",
        className="glass-panel controls-panel",
    )


def legend_rows(config: dict) -> list[html.Div]:
    """Legend rows for the current case — isobath rows are colored to match their map dots/lines."""
    rows = [
        html.Div(
            [html.Span(className="dot aoi"), " Area of interest"],
            className="legend-row",
        ),
        html.Div(
            [html.Span(className="dot goal"), " Downscaling goals"],
            className="legend-row",
        ),
        html.Div(
            [html.Span(className="dot buoy"), " Wave observations"],
            className="legend-row",
        ),
    ]
    isobaths = (
        config.get("simulations", {}).get("output_sites_config", {}).get("isobaths")
        or []
    )
    for spec in isobaths:
        name = spec.get("name")
        if not name:
            continue
        color = CONTOUR_COLORS.get(name, CONTOUR_DEFAULT)
        rows.append(
            html.Div(
                [
                    html.Span(className="dot", style={"background": color}),
                    f" {name} isobath (contour & sites)",
                ],
                className="legend-row",
            )
        )
    if config.get("simulations", {}).get("output_sites_config", {}).get("polygons"):
        rows.append(
            html.Div(
                [html.Span(className="dot site-mesh"), " Mesh output sites"],
                className="legend-row",
            )
        )
    return rows


def legend_panel(config: dict) -> html.Div:
    return html.Div(
        html.Div(legend_rows(config), id="legend-body"),
        id="legend-panel",
        className="glass-panel legend-panel",
    )


def results_drawer() -> html.Div:
    return html.Div(
        [
            html.Div(
                dbc.Tabs(
                    [
                        dbc.Tab(label="Metamodel Validation", tab_id="tab-metamodel"),
                        dbc.Tab(label="Buoy Validation", tab_id="tab-buoys"),
                        dbc.Tab(label="Goals Validation", tab_id="tab-goals"),
                    ],
                    id="main-tabs",
                    active_tab="tab-metamodel",
                    className="drawer-tabs",
                ),
                className="drawer-header",
            ),
            html.Div(id="tab-content", className="drawer-body"),
        ],
        id="results-drawer",
        className="glass-panel results-drawer",
    )


def sandbox_panel() -> html.Div:
    return html.Div(
        [
            html.Span("SANDBOX", className="panel-kicker"),
            html.H2("Draw your own area", className="panel-title"),
            html.P(
                "Use the rectangle or freeform-path tool in the map's toolbar "
                "(top right) to sketch an area of interest anywhere in the "
                "world. Name it, export it, and use the shapefile to "
                "bootstrap a new case in the notebooks.",
                className="panel-desc",
            ),
            html.Label("AoI name", className="control-label"),
            dcc.Input(
                id="aoi-name-input",
                type="text",
                value="MyArea",
                className="hywaves-input",
            ),
            html.Button(
                "⬇ Export as shapefile",
                id="export-aoi-btn",
                n_clicks=0,
                className="sandbox-export-btn",
            ),
            html.Div(id="aoi-export-status"),
            html.Hr(className="panel-divider"),
            html.Label("Reference layers", className="control-label"),
            dcc.Checklist(
                id="sandbox-layers",
                options=[
                    {"label": f" {label}", "value": key}
                    for key, label in SANDBOX_LAYER_LABELS.items()
                ],
                value=list(DEFAULT_SANDBOX_LAYERS),
                className="sandbox-layer-checklist",
                labelClassName="sandbox-layer-label",
            ),
            html.Hr(className="panel-divider"),
            html.Button(
                "‹ Exit sandbox",
                id="exit-sandbox-btn",
                n_clicks=0,
                className="landing-back-btn",
            ),
        ],
        id="sandbox-panel",
        className="glass-panel case-panel",
        style={"display": "none"},
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


def _continent_grid(catalog: list[dict]) -> html.Div:
    groups = catalog_by_continent(catalog)
    cards = []
    for continent, cases in groups.items():
        cards.append(
            html.Button(
                [
                    html.Span(continent, className="landing-card-title"),
                    html.Span(
                        f"{len(cases)} use case{'s' if len(cases) != 1 else ''}",
                        className="landing-card-sub",
                    ),
                ],
                id={"type": "landing-continent-btn", "continent": continent},
                n_clicks=0,
                className="landing-card landing-continent-card",
            )
        )
    return html.Div(
        [
            html.P(
                "Pick a region to explore its downscaling use case.",
                className="landing-intro",
            ),
            html.Div(cards, className="landing-grid"),
            html.Div("or", className="landing-divider"),
            html.Button(
                [
                    html.Span("🧭 Start playing", className="landing-card-title"),
                    html.Span(
                        "Skip the case list — draw your own area of interest "
                        "from scratch on a blank world map.",
                        className="landing-card-sub landing-card-desc",
                    ),
                ],
                id={"type": "start-sandbox-btn", "index": 0},
                n_clicks=0,
                className="landing-card landing-sandbox-card",
            ),
        ]
    )


def _case_grid(catalog: list[dict], continent: str) -> html.Div:
    cases = catalog_by_continent(catalog).get(continent, [])
    cards = []
    for case in cases:
        config = load_case_config(case)
        cards.append(
            html.Button(
                [
                    html.Span(
                        config.get("name", case["id"]), className="landing-card-title"
                    ),
                    html.Span(
                        config.get("description", "") or case["institution"],
                        className="landing-card-sub landing-card-desc",
                    ),
                    _pipeline_badges(case),
                ],
                id={"type": "landing-case-btn", "case": case["id"]},
                n_clicks=0,
                className="landing-card landing-case-card",
            )
        )
    return html.Div(
        [
            html.Button(
                "‹ Back to regions",
                id={"type": "landing-back-btn", "index": 0},
                n_clicks=0,
                className="landing-back-btn",
            ),
            html.P(continent, className="landing-intro landing-continent-heading"),
            html.Div(cards, className="landing-grid"),
        ]
    )


def landing_modal(catalog: list[dict]) -> dbc.Modal:
    return dbc.Modal(
        [
            dbc.ModalHeader(
                dbc.ModalTitle("Welcome to HyGloW"), close_button=True
            ),
            dbc.ModalBody(_continent_grid(catalog), id="landing-modal-body"),
        ],
        id="landing-modal",
        is_open=True,
        size="lg",
        centered=True,
        className="landing-modal",
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
                        html.Button(
                            "🗺️ Browse cases",
                            id="browse-cases-btn",
                            n_clicks=0,
                            className="header-link header-link-browse",
                        ),
                        html.A(
                            "📖 Report",
                            id="book-link",
                            href="#",
                            target="_blank",
                            rel="noopener",
                            className="header-link header-link-book",
                        ),
                        html.A(
                            "GitHub",
                            href="https://github.com/tausiaj/HyGloW",
                            target="_blank",
                            className="header-link",
                        ),
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

    # Case-specific panels start hidden — until the user actually picks a
    # case (landing modal or dropdown), only the bare map should show; the
    # dropdown still technically defaults to catalog[0] under the hood, but
    # nothing should visibly reveal that before it's a real choice.
    hidden = {"display": "none"}
    sidebar = html.Div(case_panel(case, config), id="case-sidebar", style=hidden)
    controls = controls_panel(catalog)
    controls.style = hidden
    legend = legend_panel(config)
    legend.style = hidden
    drawer = results_drawer()
    drawer.style = hidden

    return html.Div(
        [
            dcc.Store(id="app-mode", data="case"),
            dcc.Store(id="case-confirmed", data=False),
            dcc.Store(id="sandbox-aoi-store"),
            dcc.Store(id="sandbox-layer-index-map"),
            dcc.Download(id="aoi-download"),
            header(),
            html.Div(
                [
                    dcc.Graph(
                        id="domain-map", className="basemap", config=CASE_MAP_CONFIG
                    ),
                    sidebar,
                    controls,
                    legend,
                    drawer,
                    sandbox_panel(),
                ],
                className="map-stage",
            ),
            buoy_modal(),
            landing_modal(catalog),
        ],
        className="app-shell",
    )
