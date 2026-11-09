"""Maps: the dynamic runs (SnapWave, SWAN) on the whole mesh at the hour with maps most similar to a situation."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

import components as ui
import data
import figures as fg

dash.register_page(__name__, path="/maps", name="Maps", order=5)

#: Analog distance: one unit is this far in each quantity (offshore Hs m, wave and wind direction deg, U10 m/s).
SCALES = {"hs_off": 0.5, "dir_off": 22.5, "u10": 2.0, "u10dir": 22.5}
DEFAULT_RUNS = ["dyn_on_WHACS_cal", "swan_on_WHACS_cal"]
#: Hours offered: the most similar first, then the next ones.
N_CLOSEST = 10


def _run_label(key: str) -> str:
    n = len(data.map_times().get(key, []))
    return f"{data.label(key)} ({n} maps)"


def layout(**_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    if not data.maps_available():
        return dbc.Alert("store/maps.zarr is missing: run `python build_store.py --only maps`.", color="danger")
    runs = [k for k in data.DYNAMIC if k in data.map_times()]
    run_options = [{"label": _run_label(k), "value": k} for k in runs]
    direction_marks = {i: ui.SECTOR_NAMES.get(i, "") for i in range(0, 360, 45)}
    controls = dbc.Row([
        dbc.Col(ui.labelled("Runs (analogs come from the hours where every one has a map)",
                            dcc.Dropdown(id="mp-runs", options=run_options, multi=True,
                                         value=[k for k in DEFAULT_RUNS if k in runs])), md=5),
        dbc.Col(ui.dropdown("mp-ref", [], None, "Reference"), md=3),
        dbc.Col(ui.variable_dropdown("mp-var"), md=2),
        dbc.Col(ui.labelled("Show", dbc.RadioItems(id="mp-mode", value="values", options=[
            {"label": "values", "value": "values"}, {"label": "run − reference", "value": "diff"}])), md=2),
    ], className="controls g-2")
    analog = dbc.Row([
        dbc.Col(ui.labelled("Offshore Hs, goal mean (m)", dcc.Slider(id="mp-hs", min=0, max=8, step=0.1, value=2.5,
                                                                     marks={i: str(i) for i in range(0, 9, 2)},
                                                                     tooltip={"placement": "bottom"})), md=3),
        dbc.Col(ui.labelled("Waves from (deg)", dcc.Slider(id="mp-dirw", min=0, max=345, step=15, value=315,
                                                           marks=direction_marks, tooltip={"placement": "bottom"})), md=3),
        dbc.Col(ui.labelled("ERA5 U10, site mean (m/s)", dcc.Slider(id="mp-u10", min=0, max=22, step=0.5, value=10,
                                                                     marks={i: str(i) for i in range(0, 23, 4)},
                                                                     tooltip={"placement": "bottom"})), md=3),
        dbc.Col(ui.labelled("Wind from (deg)", dcc.Slider(id="mp-wdir", min=0, max=345, step=15, value=270,
                                                          marks=direction_marks, tooltip={"placement": "bottom"})), md=3),
    ], className="controls g-2")
    return html.Div([
        ui.header("Maps", "The dynamic runs on the whole mesh, no metamodel. Grid maps were saved at a sample of hours "
                  "(notebook 08): SWAN has them at the hours SnapWave WHACS_cal has them with the same wind setting. "
                  "Describe a situation: the most similar hour where every chosen run has a map is shown (the next "
                  "ones are in the list), as values or as the difference to the reference. Hover a site for the "
                  "value at its nearest mesh node."),
        ui.card(None, [controls, analog, html.Div(id="mp-summary", className="fw-semibold my-1"),
                       dbc.Row([dbc.Col(ui.dropdown("mp-hour", [], None, "Hour (most similar first)"), md=8),
                                dbc.Col(html.Div(id="mp-links", className="note mt-4"), md=4)], className="g-2"),
                       html.Div(id="mp-label", className="note")]),
        ui.loading(html.Div(id="mp-panels")),
        ui.card("Hours with a map in both runs (diagonal: in the run)", html.Div(id="mp-pairs")),
    ])


@callback(Output("mp-ref", "options"), Output("mp-ref", "value"), Input("mp-runs", "value"), State("mp-ref", "value"))
def reference(runs, current):
    runs = runs or []
    options = [{"label": data.label(k), "value": k} for k in runs]
    return options, current if current in runs else (runs[0] if runs else None)


def _offshore(runs: list[str]) -> str:
    """Source of the conditions used for the analogs: the first run's."""
    return data.ESTIMATES[runs[0]]["offshore"] if runs else "WHACS_cal"


def _common_hours(runs: list[str]) -> pd.DatetimeIndex:
    times = [data.map_times()[k] for k in runs]
    common = times[0]
    for t in times[1:]:
        common = common.intersection(t)
    return common.sort_values()


def _hour_text(c: pd.Series) -> str:
    text = (f"offshore Hs {c['hs_off']:.2f} m (max {c['hs_off_max']:.2f}) from {c['dir_off']:.0f}° · ERA5 "
            f"{c['u10']:.1f} m/s from {c['u10dir']:.0f}° · WL {c['wl']:+.2f} m")
    acc = [f"{name} {c[col]:.1f} %" for col, name in (("swan_on_accuracy", "SWAN wind"),
                                                       ("swan_off_accuracy", "SWAN no wind"))
           if col in c and np.isfinite(c[col])]
    return text + (f" · accuracy {', '.join(acc)}" if acc else "")


@callback(Output("mp-hour", "options"), Output("mp-hour", "value"), Output("mp-summary", "children"),
          Output("mp-pairs", "children"),
          Input("mp-runs", "value"), Input("mp-hs", "value"), Input("mp-dirw", "value"), Input("mp-u10", "value"),
          Input("mp-wdir", "value"))
def analogs(runs, hs, dirw, u10, wdir):
    runs = runs or []
    pairs = _pairs_table(runs)
    if not runs:
        return [], None, "Pick at least one run.", pairs
    common = _common_hours(runs)
    if common.empty:
        return [], None, "These runs share no hour with a map: see the table below.", pairs
    c = data.conditions_of(_offshore(runs)).set_index("time").loc[common]
    wrap = lambda x: (x + 180) % 360 - 180
    d2 = (((c["hs_off"] - hs) / SCALES["hs_off"]) ** 2 + (wrap(c["dir_off"] - dirw) / SCALES["dir_off"]) ** 2
          + ((c["u10"] - u10) / SCALES["u10"]) ** 2 + (wrap(c["u10dir"] - wdir) / SCALES["u10dir"]) ** 2)
    pick = c.assign(distance=np.sqrt(d2)).nsmallest(N_CLOSEST, "distance")
    options = [{"label": f"{t:%a %d %b %H:%M} · d {r.distance:.1f} · Hs {r.hs_off:.1f} m from {r.dir_off:.0f}° · "
                         f"U10 {r.u10:.1f} m/s from {r.u10dir:.0f}°", "value": str(t)} for t, r in pick.iterrows()]
    best = pick.iloc[0]
    summary = (f"Most similar of {len(common)} hours with a map in every run: {pick.index[0]:%a %d %b %Y %H:%M}, "
               f"distance {best['distance']:.1f} (1 = {SCALES['hs_off']} m Hs, {SCALES['dir_off']}° direction, "
               f"{SCALES['u10']} m/s U10; {_offshore(runs)} conditions)")
    return options, str(pick.index[0]), summary, pairs


def _pairs_table(runs: list[str]) -> dbc.Table:
    keys = runs or [k for k in data.DYNAMIC if k in data.map_times()]
    times = data.map_times()
    table = pd.DataFrame({data.label(b): [len(times[a].intersection(times[b])) for a in keys] for b in keys},
                         index=[data.label(a) for a in keys]).rename_axis("run")
    return ui.table(table, {c: "{:,}" for c in table.columns})


@callback(Output("mp-panels", "children"), Output("mp-label", "children"), Output("mp-links", "children"),
          Input("mp-hour", "value"), Input("mp-var", "value"), Input("mp-mode", "value"), Input("mp-ref", "value"),
          State("mp-runs", "value"))
def panels(hour, var, mode, ref, runs):
    runs = runs or []
    if not hour or not runs:
        return None, "", ""
    time = pd.Timestamp(hour)
    if time not in _common_hours(runs):
        return None, "", ""
    ref = ref if ref in runs else runs[0]
    values = {k: data.map_field(k, var, time) for k in runs}
    diff = mode == "diff" and len(runs) > 1
    diffs = {k: data.difference(values[k], values[ref], var) for k in runs if k != ref} if diff else {}
    cs, unit = fg.COLORSCALES[var], data.VARIABLE_LABELS[var]
    if var == "dir":
        zmin, zmax = 0.0, 360.0
    else:
        both = np.concatenate([v[np.isfinite(v)] for v in values.values()])
        zmin = 0.0 if var == "hs" else float(np.quantile(both, 0.005))
        zmax = float(np.quantile(both, 0.995)) if both.size else 1.0
    lim = 1.0
    if diffs:
        alld = np.concatenate([np.abs(d[np.isfinite(d)]) for d in diffs.values()])
        lim = float(np.quantile(alld, 0.98)) if alld.size else 1.0
        lim = lim or 1.0
    index = data.time_index(time)
    figures = []
    for key in runs:
        if key in diffs:
            d = diffs[key]
            title = (f"{data.label(key)} − reference · node mean {np.nanmean(d):+.2f}, "
                     f"p95 |d| {np.nanquantile(np.abs(d), 0.95):.2f}")
            fig = fg.mesh_map(d, title=title, colorscale=fg.DIVERGING, zmin=-lim, zmax=lim, colorbar=unit,
                              uirevision="mp")
        else:
            goals = fg.goal_traces(data.ESTIMATES[key]["offshore"], index, var, cs, zmin, zmax)
            title = f"{data.label(key)}{' (reference)' if diff and key == ref else ''}"
            fig = fg.mesh_map(values[key], title=title, colorscale=cs, zmin=zmin, zmax=zmax, colorbar=unit,
                              nearest=var == "dir", extra=goals, uirevision="mp")
        figures.append(fig)
    width = 6 if len(runs) <= 4 else 4
    grid = dbc.Row([dbc.Col(ui.card(None, dcc.Graph(figure=f, config={"scrollZoom": True})), lg=width) for f in figures],
                   className="g-2")
    cond = data.conditions_of(_offshore(runs)).set_index("time")
    label = f"{time:%a %d %b %Y %H:%M} · " + _hour_text(cond.loc[time])
    others = [k for k in runs if k != ref]
    links = dcc.Link("Open this hour in the Explorer →",
                     href=f"/explorer?t={time:%Y-%m-%dT%H:%M}&a={(others or [ref])[0]}&b={ref}&v={var}")
    return grid, label, links
