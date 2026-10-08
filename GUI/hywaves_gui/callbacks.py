import io
import json
import zipfile

import plotly.express as px
import shapefile
from dash import ALL, ClientsideFunction, Input, Output, Patch, State, ctx, dcc, get_relative_path, html, no_update
import dash_bootstrap_components as dbc

from .colors import CHART_LAYOUT, CHART_TEMPLATE, CONTOUR_COLORS, MODEL_COLORS
from .layout import _case_grid, _continent_grid, case_panel, legend_rows
from .loaders import (
    all_case_aois,
    available_buoy_figures,
    available_gui_figures,
    book_available,
    case_mda_candidate,
    case_options,
    catalog_by_continent,
    load_all_metrics,
    load_aoi,
    load_buoy_validation,
    load_case,
    load_case_config,
    load_catalog,
    load_coastal_buffer_fill,
    load_coastal_buffer_line,
    load_depth200_world,
    load_goals,
    load_isobaths,
    load_metamodel_validation,
    load_observations,
    load_output_sites,
    load_world_coastline,
    mda_candidates,
    production_mda,
    region_bounds,
)
from .map_figure import CASE_MAP_CONFIG, SANDBOX_MAP_CONFIG, build_map, build_sandbox_map

# WGS84 .prj sidecar for exported AoI shapefiles — same CRS every case's own
# AoI_Polygons shapefiles already use.
WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
)

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
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font={"size": 18}))
    fig.update_xaxes(showticklabels=False, title=None)
    return fig


def _render_goals_validation(case: dict):
    """Holdout RMSE spread across every goal, one box per contour × variable."""
    config = load_case_config(case)
    metrics = load_all_metrics(case)
    if metrics.empty:
        return dbc.Alert(
            "No holdout metrics published for this case yet. Run val.sweep() for every goal, "
            "then `python -m hywaves publish`.",
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
            "No metamodel validation published for this case yet. Run "
            "09_Validate_Linear_Summation.ipynb (with its save-metrics cell), then `python -m hywaves publish`.",
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
    # str() each value — a numeric-looking period (e.g. "2020", unlike
    # "2001-2011") round-trips through the CSV as an int64, not a str.
    period = ", ".join(sorted(str(p) for p in subset["period"].unique()))
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
            "No buoy validation published for this case yet. Run "
            "10a_Validate_Offshore_Buoys.ipynb / 10b_Validate_Onshore_Buoys.ipynb "
            "(with their save-summary cells), then `python -m hywaves publish`.",
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


# (filename stem, title, caption) — filenames match what 12_Final_Plots.ipynb
# saves (NewSouthWales and Samoa both run it); available_gui_figures() below
# filters this down to whichever of these a given case actually has.
_FINAL_PLOTS = [
    (
        "fig01_domain",
        "Domain",
        "Satellite imagery, bathymetry, isobaths, the AoI, the full mesh, and goals with their propagation sectors.",
    ),
    (
        "fig02_map_comparison",
        "Reconstruction comparison",
        "Dynamic SnapWave vs. the metamodel at every isobath + buoy site, for a stormy and a calm moment.",
    ),
    (
        "fig03_metamodel",
        "Metamodel training (LHS cases)",
        "Sampled wave directions per goal from the LHS SnapWave cases.",
    ),
    (
        "fig04_spectra_reconstruction",
        "Reconstruction spectra network",
        "Reconstructed spectra along the 5 m isobath, with WHACS partition forcing arrows at the goals.",
    ),
]


def _render_final_plots(case: dict, case_id: str):
    """The case's own 12_Final_Plots.ipynb figures, as clickable thumbnails —
    degrades per-figure via available_gui_figures() rather than failing whole
    for a case that hasn't run that notebook (or ran an older version
    producing fewer figures). Each thumbnail's full-size view is a separate
    callback (open_final_plot_modal) triggered by its pattern-matching id,
    reusing the buoy-modal's enlarge-on-click convention."""
    available = available_gui_figures(case)
    if not available:
        return dbc.Alert(
            "No figures published for this case yet. Run 12_Final_Plots.ipynb, "
            "then `python -m hywaves publish`.",
            color="warning",
            className="drawer-alert",
        )
    cards = [
        html.Div(
            [
                html.Button(
                    html.Img(
                        src=get_relative_path(f"/gui-figures/{case_id}/{name}.png"),
                        className="final-plot-thumb-img",
                    ),
                    id={"type": "final-plot-thumb", "name": name},
                    n_clicks=0,
                    className="final-plot-thumb-btn",
                    title="Click to enlarge",
                ),
                html.H3(title, className="final-plot-title"),
                html.P(caption, className="chart-caption"),
            ],
            className="final-plot-card",
        )
        for name, title, caption in _FINAL_PLOTS
        if name in available
    ]
    if not cards:
        return dbc.Alert("No matching figures found yet.", color="secondary", className="drawer-alert")
    return html.Div(cards, className="final-plot-grid")


def _ensure_clockwise(ring: list[list[float]]) -> list[list[float]]:
    """Shapefile polygon exterior rings must wind clockwise; a hand-drawn ring may not."""
    signed_area2 = sum(
        ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1] for i in range(len(ring) - 1)
    )
    return ring if signed_area2 < 0 else list(reversed(ring))


def _build_aoi_shapefile_zip(ring: list[list[float]], name: str) -> bytes:
    """A sandbox-drawn AoI ring as a zipped shapefile, matching the aoi_file
    format every HyGloW case already uses for its paths.aoi_file."""
    ring = _ensure_clockwise(ring)
    shp_io, shx_io, dbf_io = io.BytesIO(), io.BytesIO(), io.BytesIO()
    writer = shapefile.Writer(shp=shp_io, shx=shx_io, dbf=dbf_io, shapeType=shapefile.POLYGON)
    writer.field("name", "C")
    writer.poly([ring])
    writer.record(name)
    writer.close()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}_AoI.shp", shp_io.getvalue())
        zf.writestr(f"{name}_AoI.shx", shx_io.getvalue())
        zf.writestr(f"{name}_AoI.dbf", dbf_io.getvalue())
        zf.writestr(f"{name}_AoI.prj", WGS84_PRJ)
    return buf.getvalue()


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
            return (
                get_relative_path(f"/case-book/{case_id}/index.html"),
                "header-link header-link-book",
                "Open the rendered notebook report",
            )
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
        Output("domain-map", "figure", allow_duplicate=True),
        Output("domain-map", "config"),
        Output("sandbox-layer-index-map", "data"),
        Input("case-select", "value"),
        Input("app-mode", "data"),
        State("sandbox-layers", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def update_map(case_id, mode, sandbox_layers):
        # `config` is only re-emitted when app-mode is what actually
        # triggered this call (case<->sandbox really does need a different
        # modebar). Re-sending the *same* config object on every case
        # switch — as this used to do unconditionally — makes dcc.Graph
        # treat it as a changed prop and fully remount the Plotly figure
        # (Plotly.newPlot, not the cheap Plotly.react), which restarts the
        # underlying MapLibre GL instance. On a cold cache that remount
        # loses the race with the previous instance's tile loading badly
        # enough that the very first case picked after a hard refresh could
        # render with no basemap — subsequent switches "worked" only
        # because a warm cache made the remount fast enough not to notice.
        # (The automatic page-load firing of this callback, before any real
        # interaction, has ctx.triggered_id None and mode "sandbox" — its
        # own default — so it naturally lands in the branch below with no
        # special-casing needed.)
        mode_changed = ctx.triggered_id == "app-mode"

        if mode == "sandbox":
            fig, layer_indices = build_sandbox_map(
                load_world_coastline(),
                depth200=load_depth200_world(),
                buffer_fill=load_coastal_buffer_fill(),
                buffer_line=load_coastal_buffer_line(),
                case_aois=all_case_aois(load_catalog()),
                visible_layers=set(sandbox_layers or []),
            )
            return fig, (SANDBOX_MAP_CONFIG if mode_changed else no_update), layer_indices

        case = load_case(case_id)
        config = load_case_config(case)
        goals = load_goals(case, config)
        buoys = load_observations(case, config)
        aoi = load_aoi(case, config)
        isobaths = load_isobaths(case, config)
        output_sites = load_output_sites(case, config)
        bounds = region_bounds(case, config, goals, buoys, aoi)
        fig = build_map(
            goals,
            buoys,
            bounds,
            case_id=case_id,
            aoi=aoi,
            isobaths=isobaths,
            output_sites=output_sites,
        )
        return fig, (CASE_MAP_CONFIG if mode_changed else no_update), no_update

    @app.callback(
        Output("domain-map", "figure", allow_duplicate=True),
        Input("sandbox-layers", "value"),
        State("sandbox-layer-index-map", "data"),
        prevent_initial_call=True,
    )
    def toggle_sandbox_layers(active_keys, layer_indices):
        if not layer_indices:
            return no_update
        active = set(active_keys or [])
        patch = Patch()
        for key, indices in layer_indices.items():
            for i in indices:
                patch["data"][i]["visible"] = key in active
        return patch

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

        if tab == "tab-final":
            return _render_final_plots(case, case_id)

        return _render_buoy_validation(case)

    @app.callback(
        Output("final-plot-modal", "is_open"),
        Output("final-plot-modal-title", "children"),
        Output("final-plot-modal-body", "children"),
        Input({"type": "final-plot-thumb", "name": ALL}, "n_clicks"),
        State("case-select", "value"),
        prevent_initial_call=True,
    )
    def open_final_plot_modal(_clicks, case_id):
        # Same mount-vs-click caveat as switch_landing_view/select_case_from_grid:
        # a pattern-matching ALL input also fires when the set of matching
        # thumbnails changes (e.g. switching tabs/cases re-renders them), not
        # just on a real click — only react to a genuine nonzero click.
        fired = [t for t in ctx.triggered if t["value"]]
        if not fired:
            return no_update, no_update, no_update
        prop_id = fired[0]["prop_id"].rsplit(".", 1)[0]
        name = json.loads(prop_id)["name"]
        title = next((t for n, t, _c in _FINAL_PLOTS if n == name), name)
        img = html.Img(src=get_relative_path(f"/gui-figures/{case_id}/{name}.png"), className="buoy-modal-img")
        return True, title, img

    @app.callback(
        Output("buoy-modal", "is_open"),
        Output("buoy-modal-title", "children"),
        Output("buoy-modal-body", "children"),
        Output("landing-modal", "is_open", allow_duplicate=True),
        Output("case-select", "value", allow_duplicate=True),
        Output("case-confirmed", "data", allow_duplicate=True),
        Output("app-mode", "data", allow_duplicate=True),
        Input("domain-map", "clickData"),
        State("case-select", "value"),
        prevent_initial_call=True,
    )
    def handle_domain_map_click(click_data, case_id):
        # The single handler for every domain-map.clickData change — buoy
        # markers (customdata "buoy_...") and case-AoI markers (customdata
        # "case_...") used to each get their own separate callback, all
        # three sharing this one Input. Goals, isobath/mesh-site markers and
        # AoI fills carry no customdata at all, so a click on any of those
        # (e.g. a Goal) was a guaranteed no-op in all three — but Dash still
        # dispatched three separate server round-trips for it, each showing
        # the header's "Updating..." indicator, for a click that visibly did
        # nothing. One combined handler means one no-op round-trip instead
        # of three for that case, and one genuine round-trip instead of two
        # redundant ones for an actual buoy/case click.
        none = (no_update,) * 7
        if not click_data or not click_data.get("points"):
            return none
        custom = click_data["points"][0].get("customdata")
        if not custom:
            return none
        key = str(custom[0])

        if key.startswith("buoy_"):
            name = custom[1]
            case = load_case(case_id)
            if key not in available_buoy_figures(case):
                body = dbc.Alert(
                    "No validation summary generated yet for this buoy. Run "
                    "10a_Validate_Offshore_Buoys.ipynb / 10b_Validate_Onshore_Buoys.ipynb "
                    "(with their save-summary-PNG cell), then `python -m hywaves publish`.",
                    color="secondary",
                    className="drawer-alert",
                )
            else:
                body = html.Img(src=get_relative_path(f"/buoy-figures/{case_id}/{key}.png"), className="buoy-modal-img")
            return True, name, body, no_update, no_update, no_update, no_update

        if key.startswith("case_"):
            clicked_case_id = key[len("case_") :]
            return no_update, no_update, no_update, False, clicked_case_id, True, "case"

        return none

    @app.callback(
        Output("landing-modal-body", "children"),
        Input({"type": "landing-continent-btn", "continent": ALL}, "n_clicks"),
        Input({"type": "landing-back-btn", "index": ALL}, "n_clicks"),
        Input("browse-cases-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def switch_landing_view(_continent_clicks, _back_clicks, _browse_clicks):
        # Pattern-matching ALL inputs also fire when a matching component is
        # freshly mounted (n_clicks=0) rather than actually clicked — e.g. the
        # case grid's back-button appearing when this callback renders it. Act
        # only on a real click (nonzero value); ignore mount-only triggers.
        fired = [t for t in ctx.triggered if t["value"]]
        if not fired:
            return no_update
        catalog = load_catalog()
        prop_id = fired[0]["prop_id"].rsplit(".", 1)[0]
        parsed = json.loads(prop_id) if prop_id.startswith("{") else prop_id
        if isinstance(parsed, dict) and parsed.get("type") == "landing-continent-btn":
            return _case_grid(catalog, parsed["continent"])
        # "Back" or reopening via the header button both reset to the top-level view.
        return _continent_grid(catalog)

    @app.callback(
        Output("landing-modal", "is_open"),
        Input("browse-cases-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def open_landing_modal(_n_clicks):
        # A dedicated callback/output-group just for this one literal-id
        # button, deliberately kept separate from select_case_from_grid
        # below. Both used to live in one callback sharing a single output
        # group — but select_case_from_grid's own Input (landing-case-btn,
        # pattern-matching ALL) also spuriously re-fires whenever the set of
        # matching elements changes structurally (e.g. switch_landing_view
        # replacing the modal body on this very click, which destroys the
        # previous grid's landing-case-btn cards). That gave two dispatches
        # of the *same* callback/output-group per click — dash-renderer
        # discards the older response once a newer request for the same
        # output group is in flight, so the genuine "open" response
        # regularly lost the race to the spurious no-op one, and the modal
        # only opened on the click *after* that (once the stale grid was
        # already gone and nothing spurious fired anymore). Splitting the
        # output groups means the spurious re-fire below can never race
        # this one.
        return True

    @app.callback(
        Output("landing-modal", "is_open", allow_duplicate=True),
        Output("case-select", "value"),
        Output("case-confirmed", "data"),
        Input({"type": "landing-case-btn", "case": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def select_case_from_grid(_case_clicks):
        # Map clicks (case-AoI markers) are handled by handle_domain_map_click
        # instead — see its own docstring for why domain-map.clickData isn't
        # an Input here too.
        # Same mount-vs-click caveat as switch_landing_view above: only react
        # to a genuine click, or every case card populated into the grid
        # would auto-"select" itself (n_clicks=0) the instant it is rendered.
        fired = [t for t in ctx.triggered if t["value"]]
        if not fired:
            return no_update, no_update, no_update
        prop_id = fired[0]["prop_id"].rsplit(".", 1)[0]
        parsed = json.loads(prop_id) if prop_id.startswith("{") else prop_id
        if isinstance(parsed, dict) and parsed.get("type") == "landing-case-btn":
            # Picking a case here is what un-hides the case-specific panels
            # (see toggle_mode_panels) — set directly rather than relying on
            # case-select's value actually changing, which won't fire if the
            # chosen case happens to match the dropdown's current default.
            return False, parsed["case"], True
        return no_update, no_update, no_update

    @app.callback(
        Output("case-confirmed", "data", allow_duplicate=True),
        Input("case-select", "value"),
        prevent_initial_call=True,
    )
    def confirm_case_from_dropdown(_value):
        # Covers picking a case directly from the controls-panel dropdown
        # (once it's visible) rather than the landing modal.
        return True

    @app.callback(
        Output("case-select", "options"),
        Input("case-select", "value"),
    )
    def filter_case_options(case_id):
        # Keeps the dropdown scoped to "other cases in this same continent"
        # — picking a case from a different continent (map click or Browse
        # cases) re-narrows it, since this fires on every case-select value
        # change regardless of how it changed.
        catalog = load_catalog()
        continent = load_case(case_id).get("continent")
        cases = catalog_by_continent(catalog).get(continent, catalog)
        return case_options(cases)

    @app.callback(
        Output("welcome-modal", "is_open"),
        Input("start-discovering-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_welcome_modal(_n_clicks):
        # The header's own X close (dbc's close_button=True) already
        # handles that path — this just covers the primary CTA button.
        return False

    @app.callback(
        Output("app-mode", "data"),
        Input("back-to-landing-btn", "n_clicks"),
        Input({"type": "landing-case-btn", "case": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def set_app_mode(_back_clicks, _case_clicks):
        # Map clicks (case-AoI markers) are handled by handle_domain_map_click
        # instead — see its own docstring for why domain-map.clickData isn't
        # an Input here too.
        fired = [t for t in ctx.triggered if t["value"]]
        if not fired:
            return no_update
        prop_id = fired[0]["prop_id"].rsplit(".", 1)[0]
        if prop_id == "back-to-landing-btn":
            return "sandbox"
        parsed = json.loads(prop_id) if prop_id.startswith("{") else prop_id
        if isinstance(parsed, dict) and parsed.get("type") == "landing-case-btn":
            return "case"
        return no_update

    @app.callback(
        Output("case-sidebar", "style"),
        Output("controls-panel", "style"),
        Output("legend-panel", "style"),
        Output("results-drawer", "style"),
        Output("sandbox-panel", "style"),
        Output("continent-legend", "style"),
        Input("app-mode", "data"),
        Input("case-confirmed", "data"),
    )
    def toggle_mode_panels(mode, confirmed):
        hidden = {"display": "none"}
        if mode == "sandbox":
            return hidden, hidden, hidden, hidden, {}, {}
        # "case" mode: the case-specific panels stay hidden — no spoilers —
        # until a case has actually been picked (landing modal, dropdown,
        # or clicking a case's AoI on the landing map).
        reveal = {} if confirmed else hidden
        return reveal, reveal, reveal, reveal, hidden, hidden

    @app.callback(
        Output("results-drawer", "className"),
        Output("drawer-collapse-btn", "children"),
        Output("drawer-collapse-btn", "title"),
        Input("drawer-collapse-btn", "n_clicks"),
        State("results-drawer", "className"),
        prevent_initial_call=True,
    )
    def toggle_drawer_collapsed(_n_clicks, current_class):
        # Collapsing just hides the tab content (CSS: .collapsed .drawer-body)
        # and shrinks the drawer down to its tab bar — expanded, the drawer
        # takes most of the screen height, right over the map, so it starts
        # collapsed (layout.results_drawer) and toggles from there.
        if "collapsed" in (current_class or "").split():
            return "glass-panel results-drawer", "⌄", "Collapse — frees up the map underneath"
        return "glass-panel results-drawer collapsed", "⌃", "Expand"

    app.clientside_callback(
        ClientsideFunction("map_fixes", "force_repaint"),
        Output("map-repaint-tick", "data"),
        Input("domain-map", "figure"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        ClientsideFunction("sandbox", "extract_aoi"),
        Output("sandbox-aoi-store", "data"),
        Input("domain-map", "relayoutData"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("aoi-download", "data"),
        Output("aoi-export-status", "children"),
        Input("export-aoi-btn", "n_clicks"),
        State("sandbox-aoi-store", "data"),
        State("aoi-name-input", "value"),
        prevent_initial_call=True,
    )
    def export_aoi(_n_clicks, ring, name):
        if not ring:
            return no_update, dbc.Alert(
                "Draw a rectangle or shape on the map first, then export it.",
                color="warning",
                className="drawer-alert",
            )

        safe_name = "".join(c if c.isalnum() else "_" for c in (name or "AoI")).strip("_") or "AoI"
        zip_bytes = _build_aoi_shapefile_zip(ring, safe_name)
        status = dbc.Alert(
            f"Exported {safe_name}_AoI.zip — unzip it into <YourCase>/inputs/AoI_Polygons/ "
            "to use it as that case's aoi_file.",
            color="success",
            className="drawer-alert",
        )
        return dcc.send_bytes(zip_bytes, filename=f"{safe_name}_AoI.zip"), status
