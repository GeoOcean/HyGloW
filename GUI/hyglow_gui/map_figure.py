"""Interactive basemap figure (OpenStreetMap via Plotly)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

MAP_STYLE = "open-street-map"

GOAL_COLOR = "#22d3ee"
GOAL_SELECTED = "#fbbf24"
BUOY_COLOR = "#f87171"
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
) -> go.Figure:
    fig = go.Figure()

    if goals.empty:
        fig.update_layout(
            mapbox={"style": MAP_STYLE, "center": _center(bounds), "zoom": _zoom(bounds)},
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            uirevision=MAP_UIREVISION,
        )
        return fig

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
