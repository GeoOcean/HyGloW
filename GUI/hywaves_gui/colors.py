"""Single source of truth for every color used in the GUI's Python-rendered
output — the map (:mod:`map_figure`) and the validation charts
(:mod:`callbacks`). Change a value here and it updates everywhere at once.

This mirrors (but is separate from) the CSS custom properties in
``assets/style.css``'s ``:root`` block, which cover the HTML chrome — panels,
text, pills — around these figures. Keep the two in visual agreement when
you tune either one.
"""

from __future__ import annotations

# --- Base palette — matches assets/style.css :root ------------------------
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ACCENT = "#22d3ee"
DANGER = "#f87171"

# --- Map: goals, buoys, AoI -------------------------------------------------
GOAL_COLOR = ACCENT
BUOY_COLOR = DANGER
AOI_LINE = "#7dd3fc"
AOI_FILL = "rgba(125, 211, 252, 0.22)"
MESH_SITE_COLOR = "#a78bfa"

# --- Contour depth bands -----------------------------------------------------
# Drives BOTH the map (isobath lines + their output-site dots, map_figure.py)
# AND the validation charts (per-contour boxes/bars, callbacks.py) — "5m"
# is always this same teal everywhere in the app. Same hex values as
# ``utils.viz.ISOBATH_COLORS`` in the science codebase, so the
# GUI and the notebook-generated gallery figures agree too.
CONTOUR_COLORS: dict[str, str] = {
    "5m": "#8dd3c7",
    "10m": "#1f78b4",
    "20m": "#33a02c",
    "50m": "#e31a1c",
}
CONTOUR_DEFAULT = MUTED

# --- Buoy-validation models --------------------------------------------------
MODEL_COLORS: dict[str, str] = {
    "Bulk metamodel": "#22d3ee",
    "Spectral metamodel": "#a78bfa",
    "WHACS (closest goal)": "#fb923c",
}

# --- Basemap Map/Satellite toggle buttons -----------------------------------
BASEMAP_BUTTON_STYLE = {
    "bgcolor": "rgba(15, 23, 42, 0.85)",
    "bordercolor": "#334155",
    "font": {"color": TEXT, "size": 11},
}

# --- Plotly chart theme (dark, transparent — sits over a glass panel) ------
CHART_TEMPLATE = "plotly_dark"
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(15, 23, 42, 0)",
    plot_bgcolor="rgba(15, 23, 42, 0.35)",
    font=dict(color=TEXT),
    margin=dict(l=48, r=24, t=48, b=48),
)
