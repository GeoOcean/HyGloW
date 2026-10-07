"""Interactive basemap figure (OpenStreetMap via Plotly)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .colors import (
    AOI_FILL,
    AOI_LINE,
    BASEMAP_BUTTON_STYLE,
    BUOY_COLOR,
    CONTOUR_COLORS,
    CONTOUR_DEFAULT,
    GOAL_COLOR,
    MESH_SITE_COLOR,
    TEXT,
)
from .world_regions import WORLD_REGIONS

MAP_STYLE = "open-street-map"
# No Mapbox token available, so Plotly's token-free named styles (used for
# MAP_STYLE above) top out at street/terrain maps — none of them is actual
# satellite imagery. This raster overlay is the same Esri World Imagery
# source already used for the case-plot notebooks (see
# ``utils.viz.SATELLITE_SOURCE = "arcgis"``), free and tokenless.
SATELLITE_TILE_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
SATELLITE_LAYER = {
    "sourcetype": "raster",
    "source": [SATELLITE_TILE_URL],
    "sourceattribution": "Tiles © Esri",
    "below": "traces",
}

# Fallback only — build_map is always called with case_id in practice.
MAP_UIREVISION = "hywaves-map-v1"

COASTLINE_COLOR = "#334155"
DEPTH200_COLOR = "#facc15"
BUFFER_COLOR = "#fb923c"
SANDBOX_UIREVISION = "hywaves-sandbox-v1"

# Zoom for the world view (the landing page, and sandbox drawing on it),
# centered at (0, 0). There's no way to know a given viewer's actual
# container size from here, so this is a fixed compromise tuned for typical
# desktop widths (~1200-1600px) — wide enough to show the whole world in one
# wrap without much repeat, without cropping the poles too aggressively on
# narrower/mobile viewports either.
WORLD_VIEW_ZOOM = 1.1

CASE_AOI_COLOR = AOI_LINE

# A case's AoI on the landing map is colored like its own continent block
# (WORLD_REGIONS), so "this case lives in that colored region" reads at a
# glance — falls back to CASE_AOI_COLOR for a continent with no matching
# region (shouldn't happen with the current catalog, but cheap to guard).
CONTINENT_COLORS = {region["name"]: region["color"] for region in WORLD_REGIONS}


def _case_aoi_color(continent: str | None) -> str:
    return CONTINENT_COLORS.get(continent, CASE_AOI_COLOR)

# Reference layers on by default on the landing/sandbox world view — the
# rest (dense depth/buffer datasets) are opt-in via the panel's checklist.
# WHACS hindcast coverage isn't offered at all for now — see
# catalog/build_sandbox_layers.py.
DEFAULT_SANDBOX_LAYERS = {"cases", "continents", "coastline"}

SANDBOX_LAYER_LABELS = {
    "cases": "Use cases",
    "continents": "Continents",
    "coastline": "Coastline",
    "depth200": "200m isobath (global)",
    "buffer_fill": "50km coastal buffer (fill)",
    "buffer_line": "50km / 200m boundary (line)",
}

CASE_MAP_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["toImage", "lasso2d", "select2d"],
    "displaylogo": False,
}

SANDBOX_MAP_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["toImage", "lasso2d", "select2d"],
    "modeBarButtonsToAdd": ["drawrect", "drawclosedpath", "eraseshape"],
    "displaylogo": False,
}


def _center(bounds: dict[str, list[float]]) -> dict[str, float]:
    lon0, lon1 = bounds["lon"]
    lat0, lat1 = bounds["lat"]
    return {"lat": (lat0 + lat1) / 2, "lon": (lon0 + lon1) / 2}


def _zoom(bounds: dict[str, list[float]]) -> float:
    lon_span = abs(bounds["lon"][1] - bounds["lon"][0])
    lat_span = abs(bounds["lat"][1] - bounds["lat"][0])
    span = max(lon_span, lat_span)
    if span > 40:
        return 3.5
    if span > 15:
        return 5
    if span > 5:
        return 6
    return 7.5


def _basemap_toggle_menu() -> dict:
    return {
        "type": "buttons",
        "direction": "left",
        "x": 0.5,
        "xanchor": "center",
        "y": 0.99,
        "yanchor": "top",
        "showactive": True,
        "pad": {"t": 4, "b": 4, "l": 8, "r": 8},
        **BASEMAP_BUTTON_STYLE,
        "buttons": [
            {
                "label": "Map",
                "method": "relayout",
                "args": [{"map.style": MAP_STYLE, "map.layers": []}],
            },
            {
                "label": "Satellite",
                "method": "relayout",
                "args": [{"map.style": "white-bg", "map.layers": [SATELLITE_LAYER]}],
            },
        ],
    }


def build_map(
    goals: pd.DataFrame,
    observations: pd.DataFrame,
    bounds: dict[str, list[float]],
    *,
    case_id: str | None = None,
    aoi: list[list[tuple[float, float]]] | None = None,
    isobaths: list[dict] | None = None,
    output_sites: pd.DataFrame | None = None,
) -> go.Figure:
    fig = go.Figure()

    for iso in isobaths or []:
        lon = []
        lat = []
        for seg in iso["segments"]:
            lon += [p[0] for p in seg] + [None]
            lat += [p[1] for p in seg] + [None]
        if not lon:
            continue
        fig.add_trace(
            go.Scattermap(
                lon=lon,
                lat=lat,
                mode="lines",
                name=f"Isobath {iso['name']}",
                line={
                    "color": CONTOUR_COLORS.get(iso["name"], CONTOUR_DEFAULT),
                    "width": 1.5,
                },
                hoverinfo="skip",
            )
        )

    if aoi:
        lon: list[float | None] = []
        lat: list[float | None] = []
        for ring in aoi:
            lon += [p[0] for p in ring] + [ring[0][0], None]
            lat += [p[1] for p in ring] + [ring[0][1], None]
        fig.add_trace(
            go.Scattermap(
                lon=lon,
                lat=lat,
                mode="lines",
                name="AoI",
                fill="toself",
                fillcolor=AOI_FILL,
                line={"color": AOI_LINE, "width": 2.5},
                hoverinfo="skip",
            )
        )

    if output_sites is not None and not output_sites.empty:
        # Isobath sites split further by depth band ("5m", "10m", ...) so
        # each dot matches the color of its own contour line above — mesh
        # sites (no such sub-grouping) stay one flat color.
        for source in output_sites["source"].unique():
            by_source = output_sites[output_sites["source"] == source]
            if source == "isobath":
                for depth, group in by_source.groupby("label"):
                    fig.add_trace(
                        go.Scattermap(
                            lon=group["lon"],
                            lat=group["lat"],
                            mode="markers",
                            name=f"Sites: {depth}",
                            marker={
                                "size": 6,
                                "color": CONTOUR_COLORS.get(str(depth), CONTOUR_DEFAULT),
                                "opacity": 0.55,
                            },
                            text=group["label"],
                            hovertemplate=(
                                f"<b>isobath site</b> {depth}<br>"
                                "lon %{lon:.3f}, lat %{lat:.3f}<extra></extra>"
                            ),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scattermap(
                        lon=by_source["lon"],
                        lat=by_source["lat"],
                        mode="markers",
                        name=f"Sites: {source}",
                        marker={"size": 4, "color": MESH_SITE_COLOR, "opacity": 0.55},
                        text=by_source["label"],
                        hovertemplate=(
                            f"<b>{source} site</b> %{{text}}<br>"
                            "lon %{lon:.3f}, lat %{lat:.3f}<extra></extra>"
                        ),
                    )
                )

    if not goals.empty:
        fig.add_trace(
            go.Scattermap(
                lon=goals["lon"],
                lat=goals["lat"],
                mode="markers",
                name="Goals",
                marker={"size": 13, "color": GOAL_COLOR, "opacity": 0.85},
                text=goals["goal"],
                hovertemplate=(
                    "<b>Goal %{text}</b><br>lon %{lon:.3f}, lat %{lat:.3f}<extra></extra>"
                ),
            )
        )

    if not observations.empty:
        site_ids = [
            iid if str(iid).startswith("buoy_") else f"buoy_{iid}"
            for iid in observations["instrument_id"]
        ]
        customdata = list(zip(site_ids, observations["name"], strict=True))
        # A soft color halo behind the star: Scattermap's built-in "star"
        # icon is a monochrome (black) sprite — `marker.color` has no effect
        # on named icon symbols, only on the plain circle default — so the
        # halo is what actually carries BUOY_COLOR through to the map.
        fig.add_trace(
            go.Scattermap(
                lon=observations["lon"],
                lat=observations["lat"],
                mode="markers",
                name="Observations halo",
                marker={"size": 26, "color": BUOY_COLOR, "opacity": 0.45},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scattermap(
                lon=observations["lon"],
                lat=observations["lat"],
                mode="markers",
                name="Observations",
                marker={"size": 20, "symbol": "star"},
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[1]}</b><br>lon %{lon:.3f}, lat %{lat:.3f}"
                    "<br><i>click for validation summary</i><extra></extra>"
                ),
            )
        )

    center = _center(bounds)
    fig.update_layout(
        map={
            "style": MAP_STYLE,
            "center": center,
            "zoom": _zoom(bounds),
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        # Keyed on case_id (not a constant) so switching cases always
        # re-centers on the new case's bounds. A constant uirevision here
        # previously told Plotly "this is the same view as before" across
        # case switches, so it kept whatever pan/zoom was left over from
        # the last case instead of applying the new center/zoom at all —
        # e.g. switching to NewSouthWales while still looking at wherever
        # you'd panned to in Samoa kept showing Samoa's location.
        uirevision=case_id or MAP_UIREVISION,
        dragmode="pan",
        updatemenus=[_basemap_toggle_menu()],
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _multiline_trace(name: str, segments, color: str, width: float = 1) -> go.Scattermap:
    lon: list[float | None] = []
    lat: list[float | None] = []
    for seg in segments:
        lon += [p[0] for p in seg] + [None]
        lat += [p[1] for p in seg] + [None]
    return go.Scattermap(
        lon=lon,
        lat=lat,
        mode="lines",
        name=name,
        line={"color": color, "width": width},
        hoverinfo="skip",
    )


def _ring_fill_trace(name: str, rings, color: str, *, fill_alpha: float, line_width: float) -> go.Scattermap:
    lon: list[float | None] = []
    lat: list[float | None] = []
    for ring in rings:
        lon += [p[0] for p in ring] + [ring[0][0], None]
        lat += [p[1] for p in ring] + [ring[0][1], None]
    return go.Scattermap(
        lon=lon,
        lat=lat,
        mode="lines",
        name=name,
        fill="toself",
        fillcolor=_hex_to_rgba(color, fill_alpha),
        line={"color": color, "width": line_width},
        hoverinfo="skip",
    )


def build_sandbox_map(
    coastline: list[list[tuple[float, float]]],
    *,
    depth200: list[list[tuple[float, float]]] | None = None,
    buffer_fill: list[list[tuple[float, float]]] | None = None,
    buffer_line: list[list[tuple[float, float]]] | None = None,
    case_aois: list[dict] | None = None,
    visible_layers: set[str] | None = None,
) -> tuple[go.Figure, dict[str, list[int]]]:
    """World view for the landing page (also sandbox mode for drawing a new
    AoI) — hand-drawn continent blocks, every catalog case's real AoI
    (clickable, see callbacks.handle_domain_map_click), plus a set
    of opt-in reference layers (global 200m isobath, 50km coastal buffer),
    each independently toggleable without disturbing any AoI already drawn
    on the map (see callbacks.toggle_sandbox_layers).

    Returns the figure AND a ``{layer_key: [trace_index, ...]}`` map — the
    single source of truth both this function and that callback use to agree
    on which traces belong to which toggle, since a reference layer with no
    data (its ``catalog/build_sandbox_layers.py`` step never run) is simply
    omitted rather than added empty, which would otherwise shift indices.

    The AoI shape itself is drawn client-side (Plotly's ``drawrect``/
    ``drawclosedpath`` modebar tools, wired up in callbacks.py) rather than
    as a trace here.
    """
    visible_layers = DEFAULT_SANDBOX_LAYERS if visible_layers is None else visible_layers
    fig = go.Figure()
    layer_indices: dict[str, list[int]] = {}

    def _add(key: str, trace: go.Scattermap) -> None:
        trace.visible = key in visible_layers
        layer_indices.setdefault(key, []).append(len(fig.data))
        fig.add_trace(trace)

    for region in WORLD_REGIONS:
        for ring in region["rings"]:
            lon = [p[0] for p in ring] + [ring[0][0]]
            lat = [p[1] for p in ring] + [ring[0][1]]
            _add(
                "continents",
                go.Scattermap(
                    lon=lon,
                    lat=lat,
                    mode="lines",
                    name=region["name"],
                    fill="toself",
                    fillcolor=_hex_to_rgba(region["color"], 0.28),
                    line={"color": region["color"], "width": 1},
                    hoverinfo="skip",
                ),
            )

    if coastline:
        _add("coastline", _multiline_trace("Coastline", coastline, COASTLINE_COLOR))

    if depth200:
        _add(
            "depth200",
            _multiline_trace(SANDBOX_LAYER_LABELS["depth200"], depth200, DEPTH200_COLOR),
        )

    if buffer_fill:
        _add(
            "buffer_fill",
            _ring_fill_trace(
                SANDBOX_LAYER_LABELS["buffer_fill"], buffer_fill, BUFFER_COLOR, fill_alpha=0.22, line_width=0.5
            ),
        )

    if buffer_line:
        _add(
            "buffer_line",
            _multiline_trace(SANDBOX_LAYER_LABELS["buffer_line"], buffer_line, BUFFER_COLOR, width=1.5),
        )

    if case_aois:
        # Added last so cases render on top of every other layer — they're
        # the primary click target on this view. Each case's AoI is colored
        # like its own continent block (see _case_aoi_color) so it reads as
        # "this case lives in that region" at a glance.
        for case in case_aois:
            color = _case_aoi_color(case.get("continent"))
            _add(
                "cases",
                _ring_fill_trace(f"AoI: {case['name']}", case["rings"], color, fill_alpha=0.28, line_width=2),
            )
        _add(
            "cases",
            go.Scattermap(
                lon=[c["center"][0] for c in case_aois],
                lat=[c["center"][1] for c in case_aois],
                mode="markers+text",
                name=SANDBOX_LAYER_LABELS["cases"],
                marker={
                    "size": 14,
                    "color": [_case_aoi_color(c.get("continent")) for c in case_aois],
                    "opacity": 0.95,
                },
                text=[c["name"] for c in case_aois],
                textposition="top center",
                textfont={"color": TEXT, "size": 11},
                customdata=[(f"case_{c['case_id']}", c["name"]) for c in case_aois],
                hovertemplate="<b>%{text}</b><br><i>click to open</i><extra></extra>",
            ),
        )

    fig.update_layout(
        map={"style": MAP_STYLE, "center": {"lat": 0, "lon": 0}, "zoom": WORLD_VIEW_ZOOM},
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        uirevision=SANDBOX_UIREVISION,
        # Pan by default — drawing is opt-in via the rectangle/freeform
        # modebar buttons (added in SANDBOX_MAP_CONFIG below), so exploring
        # the map doesn't accidentally drop a shape on every drag.
        dragmode="pan",
        newshape={"line": {"color": GOAL_COLOR, "width": 2.5}, "fillcolor": AOI_FILL},
        updatemenus=[_basemap_toggle_menu()],
    )
    return fig, layer_indices
