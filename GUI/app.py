import dash
import dash_bootstrap_components as dbc
from flask import send_from_directory
from hywaves_gui.callbacks import register_callbacks
from hywaves_gui.layout import build_layout
from hywaves_gui.loaders import load_case
from hywaves_gui.paths import case_root

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "HyGloW Results Explorer"
app.layout = build_layout()
register_callbacks(app)
server = app.server


@app.server.route("/case-figures/<case_id>/<path:filename>")
def serve_figure(case_id: str, filename: str):
    case = load_case(case_id)
    directory = case_root(case) / "outputs" / "gui_figures"
    return send_from_directory(directory, filename)


@app.server.route("/case-book/<case_id>/")
@app.server.route("/case-book/<case_id>/<path:filename>")
def serve_book(case_id: str, filename: str = "index.html"):
    case = load_case(case_id)
    directory = case_root(case) / "book" / "_build" / "html"
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
