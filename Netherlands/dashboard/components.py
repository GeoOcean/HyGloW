"""Small layout pieces shared by the pages."""

from __future__ import annotations

import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html

import data

SECTORS = [0, 45, 90, 135, 180, 225, 270, 315]
SECTOR_NAMES = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}
MONTHS = {i: m for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                                       "Dec"], start=1)}


def loading(child) -> dcc.Loading:
    """Spinner over *child* while a callback updates it; the old content stays visible, dimmed."""
    return dcc.Loading(child, type="circle", color="#18bc9c", delay_show=350,
                       overlay_style={"visibility": "visible", "opacity": 0.45})


def graph(**kwargs) -> dcc.Loading:
    """``dcc.Graph`` with a :func:`loading` spinner, blank (no axes) until its first figure arrives."""
    kwargs.setdefault("figure", {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False},
                                            "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)"}})
    return loading(dcc.Graph(**kwargs))


def header(title: str, lead: str) -> html.Div:
    return html.Div([html.H3(title, className="page-title"), html.P(lead, className="page-lead")])


def card(title: str | None, body, **kwargs) -> dbc.Card:
    return dbc.Card([dbc.CardHeader(title) if title else None, dbc.CardBody(body, className="p-2")],
                    className="mb-3", **kwargs)


def kpi(name: str, value: str, sub: str = "", color: str = "#212529") -> dbc.Card:
    return dbc.Card(html.Div([html.Div(name, className="name"), html.Div(value, className="value", style={"color": color}),
                              html.Div(sub, className="sub")], className="kpi"), className="mb-2 h-100")


def labelled(label: str, control) -> html.Div:
    return html.Div([html.Div(label, className="control-label"), control], className="mb-2")


def dropdown(id_, options, value, label: str, **kwargs) -> html.Div:
    if options and not isinstance(options[0], dict):
        options = [{"label": str(o), "value": o} for o in options]
    return labelled(label, dcc.Dropdown(id=id_, options=options, value=value, clearable=False, **kwargs))


def estimate_dropdown(id_, value: str, label: str, include_buoy: bool = True) -> html.Div:
    return dropdown(id_, data.estimate_options(include_buoy), value, label)


def variable_dropdown(id_, value: str = "hs", label: str = "Variable") -> html.Div:
    return dropdown(id_, [{"label": v, "value": k} for k, v in data.VARIABLE_LABELS.items()], value, label)


def group_dropdown(id_, value: str = "all sites", label: str = "Sites") -> html.Div:
    return dropdown(id_, list(data.GROUPS), value, label)


def table(df: pd.DataFrame, fmt: dict[str, str] | None = None, index: bool = True) -> dbc.Table:
    """Bootstrap table of a small DataFrame (*fmt*: column -> format string, default 3 significant)."""
    fmt = fmt or {}
    df = df.reset_index() if index else df
    def cell(c, v):
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                return ""
            default = "{:,.0f}" if float(v).is_integer() and abs(v) >= 100 else "{:.3g}"
            return fmt.get(c, default).format(v)
        if isinstance(v, (int, np.integer)):
            return fmt.get(c, "{:,}").format(v)
        return str(v)
    head = html.Thead(html.Tr([html.Th(str(c)) for c in df.columns]))
    body = html.Tbody([html.Tr([html.Td(cell(c, v)) for c, v in zip(df.columns, row)]) for row in df.itertuples(index=False)])
    return dbc.Table([head, body], bordered=False, hover=True, size="sm", striped=True, className="small-table mb-0")


def condition_filters(prefix: str, offshore: bool = True) -> html.Div:
    """Hour filters: months, ERA5 wind speed and direction, offshore Hs and direction (sectors of 45 deg)."""
    sector_options = [{"label": SECTOR_NAMES[s], "value": s} for s in SECTORS]
    items = [
        labelled("Months", dcc.Dropdown(id=f"{prefix}-months", options=[{"label": v, "value": k} for k, v in MONTHS.items()],
                                        value=[], multi=True, placeholder="all")),
        labelled("ERA5 U10, site mean (m/s)", dcc.RangeSlider(id=f"{prefix}-u10", min=0, max=22, step=0.5, value=[0, 22],
                                                              marks={i: str(i) for i in range(0, 23, 4)},
                                                              tooltip={"placement": "bottom"})),
        labelled("Wind from", dcc.Checklist(id=f"{prefix}-u10dir", options=sector_options, value=SECTORS, inline=True,
                                            inputStyle={"margin-right": "3px", "margin-left": "8px"})),
    ]
    if offshore:
        items += [
            labelled("Offshore Hs, goal mean (m)", dcc.RangeSlider(id=f"{prefix}-hsoff", min=0, max=8, step=0.1, value=[0, 8],
                                                                   marks={i: str(i) for i in range(0, 9)},
                                                                   tooltip={"placement": "bottom"})),
            labelled("Waves from (offshore)", dcc.Checklist(id=f"{prefix}-diroff", options=sector_options, value=SECTORS,
                                                            inline=True,
                                                            inputStyle={"margin-right": "3px", "margin-left": "8px"})),
        ]
    return html.Div(items, className="controls")


def filter_inputs(prefix: str, offshore: bool = True) -> list:
    from dash import Input
    ids = ["months", "u10", "u10dir"] + (["hsoff", "diroff"] if offshore else [])
    return [Input(f"{prefix}-{i}", "value") for i in ids]


def mask_from_filters(source: str, months, u10, u10dir, hsoff=None, diroff=None, selected=None) -> np.ndarray:
    return data.hours_mask(source, months=months or None, u10=tuple(u10) if u10 else None, u10dir=u10dir,
                           hs_off=tuple(hsoff) if hsoff else None, dir_off=diroff, selected=selected)
