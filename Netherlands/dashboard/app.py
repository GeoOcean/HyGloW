"""
HyGloW Netherlands dashboard: every result of the case for 2013 in one place.

``python app.py [--port 8050]`` from this folder (after ``python build_store.py``);
forward the port and open http://localhost:8050.
"""

from __future__ import annotations

import argparse
import threading
import warnings

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html

import data

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected (sites without data)

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True,
           title="HyGloW · Netherlands")
server = app.server

NAV = [("Overview", "/"), ("Explorer", "/explorer"), ("Compare", "/compare"), ("Error budget", "/budget"),
       ("Wind", "/wind"), ("Storms", "/storms"), ("Metamodel", "/metamodel"), ("Calibration", "/calibration"),
       ("Numerical tests", "/numerical-tests"), ("Findings", "/findings")]

app.layout = html.Div([
    dbc.Navbar(dbc.Container([
        dbc.NavbarBrand([html.Span("HyGloW", className="brand-main"), html.Span(" · Netherlands 2013", className="brand-sub")],
                        href="/"),
        dbc.Nav([dbc.NavItem(dbc.NavLink(name, href=path, active="exact")) for name, path in NAV], navbar=True,
                className="ms-auto flex-wrap"),
    ], fluid=True), color="dark", dark=True, className="mb-3 py-1"),
    dbc.Container(dcc.Loading(dash.page_container, target_components={"_pages_content": "children"}, type="circle",
                              color="#18bc9c", delay_show=300, overlay_style={"visibility": "visible", "opacity": 0.4}),
                  fluid=True, className="pb-5"),
    dcc.Store(id="session", storage_type="session"),
])


def _warm() -> None:
    """Compute the overview's headline numbers at start-up, so the first page opens at once."""
    from pages.overview import headline
    headline()


def main() -> None:
    parser = argparse.ArgumentParser(description="HyGloW Netherlands dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not data.available():
        raise SystemExit("store/ is missing or incomplete: run `python build_store.py` first")
    data.preload()
    threading.Thread(target=_warm, daemon=True).start()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
