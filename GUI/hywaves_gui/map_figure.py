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
)

MAP_STYLE = "open-street-map"
# No Mapbox token available, so Plotly's token-free named styles (used for
# MAP_STYLE above) top out at street/terrain maps — none of them is actual
# satellite imagery. This raster overlay is the same Esri World Imagery
# source already used for the case-plot notebooks (see
# ``utils.viz.case_plots.SATELLITE_SOURCE = "arcgis"``), free and tokenless.
SATELLITE_TILE_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
SATELLITE_LAYER = {
    "sourcetype": "raster",
    "source": [SATELLITE_TILE_URL],
    "sourceattribution": "Tiles © Esri",
    "below": "traces",
}

# Fallback only — build_map is always called with case_id in practice.
MAP_UIREVISION = "hywaves-map-v1"


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
        dragmode="zoom",
        updatemenus=[
            {
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
        ],
    )
    return fig
