import os

import dash
import dash_bootstrap_components as dbc
from flask import send_from_directory
from hywaves_gui.callbacks import register_callbacks
from hywaves_gui.layout import build_layout
from hywaves_gui.loaders import book_dir, buoy_figures_dir, gui_figures_dir, load_case

# URL prefix the app is served under, e.g. "/hyglow/" behind a reverse proxy
# that forwards /hyglow/... unchanged (proxy_pass http://localhost:PORT/hyglow).
# Every Dash route AND the file routes below live under it; links built in
# callbacks go through dash.get_relative_path so they follow it too.
_base = os.environ.get("HYGLOW_URL_BASE", "").strip("/")
URL_BASE = f"/{_base}/" if _base else "/"

app = dash.Dash(
    __name__,
    url_base_pathname=URL_BASE,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "HyGloW Results Explorer"
app.layout = build_layout()
register_callbacks(app)
server = app.server


@app.server.route(f"{URL_BASE}buoy-figures/<case_id>/<path:filename>")
def serve_buoy_figure(case_id: str, filename: str):
    return send_from_directory(buoy_figures_dir(load_case(case_id)), filename)


@app.server.route(f"{URL_BASE}gui-figures/<case_id>/<path:filename>")
def serve_gui_figure(case_id: str, filename: str):
    return send_from_directory(gui_figures_dir(load_case(case_id)), filename)


@app.server.route(f"{URL_BASE}case-book/<case_id>/")
@app.server.route(f"{URL_BASE}case-book/<case_id>/<path:filename>")
def serve_book(case_id: str, filename: str = "index.html"):
    directory = book_dir(load_case(case_id))
    # The book's chapter routes (e.g. "final-plots") are directories on disk
    # holding their own index.html, not flat files — fall back to that for
    # a hard refresh or a nav link opened directly, not just client-side
    # routing from the already-loaded single-page app.
    if not (directory / filename).is_file():
        nested = f"{filename.rstrip('/')}/index.html"
        if (directory / nested).is_file():
            filename = nested
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
