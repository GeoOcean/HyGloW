"""Interactive basemap figure (OpenStreetMap via Plotly)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

MAP_STYLE = "open-street-map"

GOAL_COLOR = "#22d3ee"
GOAL_SELECTED = "#fbbf24"
BUOY_COLOR = "#f87171"
AOI_LINE = "#7dd3fc"
AOI_FILL = "rgba(125, 211, 252, 0.22)"
SITE_SOURCE_COLORS = {"isobath": "#38bdf8", "mesh": "#a78bfa"}
# Same hex values as utils.viz.case_plots.ISOBATH_COLORS in the science codebase.
ISOBATH_LINE_COLORS = {"5m": "#8dd3c7", "10m": "#1f78b4", "20m": "#33a02c", "50m": "#e31a1c"}
ISOBATH_LINE_DEFAULT = "#94a3b8"
MAP_UIREVISION = "hyglow-map-v1"


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
    selected_goal: str | None = None,
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
            go.Scattermapbox(
                lon=lon,
                lat=lat,
                mode="lines",
                name=f"Isobath {iso['name']}",
                line={
                    "color": ISOBATH_LINE_COLORS.get(iso["name"], ISOBATH_LINE_DEFAULT),
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
            go.Scattermapbox(
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
        for source, group in output_sites.groupby("source"):
            fig.add_trace(
                go.Scattermapbox(
                    lon=group["lon"],
                    lat=group["lat"],
                    mode="markers",
                    name=f"Sites: {source}",
                    marker={
                        "size": 4 if source == "mesh" else 6,
                        "color": SITE_SOURCE_COLORS.get(str(source), "#94a3b8"),
                        "opacity": 0.55,
                    },
                    text=group["label"],
                    hovertemplate=(
                        f"<b>{source} site</b> %{{text}}<br>"
                        "lon %{lon:.3f}, lat %{lat:.3f}<extra></extra>"
                    ),
                )
            )

    selected = str(selected_goal) if selected_goal is not None else None
    other = goals[goals["goal"] != selected] if selected else goals
    sel = goals[goals["goal"] == selected] if selected else goals.iloc[0:0]

    if not other.empty:
        fig.add_trace(
            go.Scattermapbox(
                lon=other["lon"],
                lat=other["lat"],
                mode="markers",
                name="Goals",
                customdata=other["goal"],
                marker={"size": 13, "color": GOAL_COLOR, "opacity": 0.85},
                hovertemplate=(
                    "<b>Goal %{customdata}</b><br>"
                    "lon %{lon:.3f}, lat %{lat:.3f}<br>"
                    "<extra>Click to inspect</extra>"
                ),
            )
        )

    if not sel.empty:
        fig.add_trace(
            go.Scattermapbox(
                lon=sel["lon"],
                lat=sel["lat"],
                mode="markers",
                name=f"Goal {selected}",
                customdata=sel["goal"],
                marker={"size": 20, "color": GOAL_SELECTED, "opacity": 1.0},
                hovertemplate=(
                    "<b>Goal %{customdata}</b> (selected)<br>"
                    "lon %{lon:.3f}, lat %{lat:.3f}<extra></extra>"
                ),
            )
        )

    if not observations.empty:
        fig.add_trace(
            go.Scattermapbox(
                lon=observations["lon"],
                lat=observations["lat"],
                mode="markers",
                name="Observations",
                marker={"size": 16, "color": BUOY_COLOR},
                text=observations["name"],
                hovertemplate="<b>%{text}</b><br>lon %{lon:.3f}, lat %{lat:.3f}<extra></extra>",
            )
        )

    center = _center(bounds)
    fig.update_layout(
        mapbox={
            "style": MAP_STYLE,
            "center": center,
            "zoom": _zoom(bounds),
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        uirevision=MAP_UIREVISION,
        dragmode="zoom",
    )
    return fig
