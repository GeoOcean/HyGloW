"""Findings: the conclusions so far (this folder's README, hywaves/WIND.md) and a guided tour into the other pages."""

from __future__ import annotations

import re

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

import components as ui
import data

dash.register_page(__name__, path="/findings", name="Findings", order=9)

TOUR = [
    ("The linear sum works on average",
     "Bulk metamodel vs the dynamic run without wind at every site: like for like, the method error.",
     "/compare?ref=dyn_off_WHACS_cal&est=mm_bulk_WHACS_cal&v=hs"),
    ("… but not in the biggest seas at 5 m",
     "Same comparison; pick the p99.9 bias on the map and the 5 m sites, or the storm preset.",
     "/compare?ref=dyn_off_WHACS_cal&est=mm_bulk_WHACS_cal&v=hs"),
    ("Missing wind is the largest error",
     "The budget at the buoys, split by part; filter by wind direction to see where it matters.", "/budget"),
    ("SnapWave grows too much sea with SW wind",
     "The wind page: added Hs by stretch of coast and direction, and the buoys by wind direction.", "/wind"),
    ("Xaver, hour by hour",
     "Dynamic with wind vs the spectral metamodel at the peak; press play.",
     "/explorer?t=2013-12-05T21:00&a=dyn_on_WHACS_cal&b=mm_spec_WHACS_cal&v=hs&site=buoy_Europlatform"),
    ("Spectral Tp is poor",
     "Spectral metamodel Tp against the dynamic run: far too long near the coast, while Hs is fine.",
     "/compare?ref=dyn_off_WHACS_cal&est=mm_spec_WHACS_cal&v=tp"),
    ("Calibration helps the storm tail", "From the satellite fit to the change at the coast.", "/calibration"),
    ("Storm peaks at every buoy", "Every estimate's peak against the buoy's, per event.", "/storms"),
]


def _section(path, heading: str) -> str:
    text = path.read_text() if path.exists() else ""
    m = re.search(rf"^## {re.escape(heading)}.*?$(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    return m.group(1) if m else ""


def layout(**_):
    readme = data.HERE / "README.md"
    wind = data.CASE.parent / "hywaves" / "WIND.md"
    tour = dbc.Row([dbc.Col(dbc.Card(dbc.CardBody([html.H6(title, className="fw-bold"), html.P(text, className="note mb-2"),
                                                     dcc.Link("open →", href=href)]), className="h-100"), md=3, className="mb-2")
                    for title, text, href in TOUR], className="g-2")
    built = data.manifest()
    return html.Div([
        ui.header("Findings", "What the first iteration (SnapWave, 2013, MDA 600, CODEC) shows, with a link into the "
                  "data for each point. The text is read live from Netherlands/dashboard/README.md and hywaves/WIND.md."),
        ui.card("Guided tour", tour),
        dbc.Row([
            dbc.Col(ui.card("What it shows (dashboard/README.md)",
                            dcc.Markdown(_section(readme, "What it shows"), className="markdown findings")), lg=6),
            dbc.Col(ui.card("Wind: the 2013 dynamic pairs (hywaves/WIND.md)",
                            dcc.Markdown(_section(wind, "The 2013 dynamic pairs"), className="markdown findings")), lg=6),
        ], className="g-2"),
        ui.card("Wind: options for the reconstruction (hywaves/WIND.md)",
                dcc.Markdown(_section(wind, "Options"), className="markdown findings")),
        html.Div("Store built: " + ", ".join(f"{k} {v['built']}" for k, v in built.items()), className="note"),
    ])
