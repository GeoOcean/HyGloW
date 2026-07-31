import dash
import dash_bootstrap_components as dbc
from flask import send_from_directory
from hyglow_gui.callbacks import register_callbacks
from hyglow_gui.layout import build_layout
from hyglow_gui.loaders import load_case
from hyglow_gui.paths import case_root

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
    directory = case_root(case) / "outputs" / "course_figures"
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
