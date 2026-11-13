"""Breaking: four findings (SnapWave's depth ceiling, what the metamodel misses, where, a cap) and the details behind them."""

from __future__ import annotations

import warnings
from functools import lru_cache

import breaking as brk
import components as ui
import dash
import dash_bootstrap_components as dbc
import data
import figures as fg
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, callback, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

dash.register_page(__name__, path="/breaking", name="Breaking", order=6)

#: Site groups of the page (cube ``group`` values).
GROUPS = {"5 m": "5m", "10 m": "10m", "20 m": "20m", "buoys": "buoy", "DeenHaag": "DeenHaag"}
GROUP_COLORS = {"5 m": "#d7301f", "10 m": "#fc8d59", "20 m": "#4575b4", "buoys": "#111111", "DeenHaag": "#1a9850"}
THRESHOLDS = [1, 5, 10, 25, 50, 100]
#: Dw (W/m²) above which a site "breaks" in the findings.
BREAKS = 10.0
#: Metamodel Hs: plain, hard cap (None) or smooth cap of power p.
CAPS = {"linear sum": "none", "hard cap": None, "smooth p = 4": 4.0, "smooth p = 6": 6.0, "smooth p = 8": 8.0}
MAP_METRICS = {
    "miss": "% of hours the dynamic run breaks and the goals alone do not",
    "brk": "% of hours the dynamic run breaks (Dw > threshold)",
    "bias_brk": "Hs bias, metamodel − dynamic, when the dynamic run breaks (m)",
    "ratio": "median Σ Dw goals alone / Dw dynamic, when the dynamic run breaks",
    "top": "Hs bias, metamodel − dynamic, at the site's top 0.1 % dynamic Hs (m)",
    "dyn_max": "highest Hs / h, dynamic run",
    "sum_max": "highest Hs / h, metamodel",
    "sat": "highest Hs / h of a goal alone (training cases): the ceiling",
}
#: Finding 1 map: what is mapped, the statistic over the period, the period.
RATIO_WHAT = {"dyn_cap": "dynamic run Hs − cap (m)", "mm_cap": "metamodel Hs − cap (m)",
              "dyn": "dynamic run Hs / h", "mm": "metamodel Hs / h", "ceiling": "the ceiling (Hs / h, training cases)"}
RATIO_STATS = {"mean": "mean", "median": "median", "std": "std", "p99": "p99", "max": "max"}
PERIODS = {"year": "all of 2013", "storm": "one storm", "hour": "one hour of the storm"}
COLORS = {"dyn": "#ff7f0e", "mm": "#1f77b4", "hard": "#7b3294", "smooth": "#1a9850", "ceiling": "#d7301f"}

PHYSICS = r"""
SnapWave solves the stationary energy balance per direction θ, with one dissipation rate for the
whole spectrum:

$$\nabla\cdot(\vec c_g E_\theta) + \partial_\theta (c_\theta E_\theta) = -\frac{E_\theta}{E}\,(D_w + D_f)$$

- **Height**: $H_{rms} = \sqrt{8E/\rho g}$; the output is $H_{m0} = \sqrt2\,H_{rms}$. $T_p$ is one value for the
  whole domain in each run (from the boundary).
- **Breaking** (Baldock et al. 1998):
  $D_w = 0.28\,\alpha\,\dfrac{\rho g}{T_p}\,e^{-(H_{max}/H_{rms})^2}\,(H_{max}^2 + H_{rms}^2)$, with
  $H_{max} = \gamma h$ and $h = z_s - z_b$. The exponential $Q_b$ is the share of breaking waves (Rayleigh
  heights above $H_{max}$): nothing for small waves, then it grows fast.
- **Limiter**: if $H_{rms} > \gamma_{max} h$ the energy is scaled down to it ($H_{m0} \le \sqrt2\,\gamma_{max} h$).
  It does not act here: Baldock saturates the waves well before.
- **Friction**: $D_f = 0.28\,\rho f_w u_{orb}^3$ (`point_df`).
- **`point_dw`** is $D_w$ at the output point on the converged $H$ and $T_p$, in W/m² (energy lost per m² of sea
  surface per second). It is a diagnostic: the same formula on Hm0, Tp and depth, which we already have.
- **How hard**: $E c_g / D_w$ is the distance over which breaking at this rate would remove the whole flux.

**The metamodel has no Dw of its own.** For the page, each goal's contribution (Hs_i, Tp_i predicted by the
metamodel at the site, at that goal's water level) goes through the same formula, and the goals are summed:
Σ Dw_i, "the goals alone". The dynamic run's Dw is the same formula on its own Hs, Tp and depth (not
`point_dw`, which is averaged from the mesh nodes: details, `point_dw` against the formula), so both sides are
computed the same way.
"""

HOUR_GUIDE = """
**How to read Dw on the maps.** Dw is the wave energy breaking removes per second from each m² of sea surface
(W/m²); the colour scale is the same on the first two maps. To give it a size: Hs 3 m with Tp 10 s in 5 m of water
carries about 36 kW per metre of crest towards the coast. Breaking at 10 W/m² takes about 3 % of that every 100 m,
at 100 W/m² about 27 % every 100 m, and at 300 W/m² it is gone within ~120 m: the waves are saturated.

- **Left, dynamic run**: where all the goals together break, and how hard. In a storm it lights up at the
  shallowest sites first.
- **Middle, goals alone**: each goal's share of the sea through the same formula, summed: what the metamodel
  carries. Each goal alone is smaller than the combined sea, and breaking grows exponentially with H, so this map is
  much paler.
- **Right, what it does to Hs**: metamodel − dynamic. Red where the metamodel keeps energy the dynamic run broke.
  Red can also appear where both maps are dark: the dynamic run broke **before** the site (seaward, on a bar) and
  arrives already reduced, while the goals alone never broke on the way.

Grey points: no value (the dynamic run did not converge that hour, or the site is dry).
"""


def _finding(number: int, title: str, text, figure) -> dbc.Row:
    """One finding: its statement on the left, the figure that shows it on the right."""
    return dbc.Row([
        dbc.Col([html.Div(f"{number}", className="finding-number"), html.H5(title, className="fw-bold"),
                 html.Div(text, className="finding-text")], lg=4),
        dbc.Col(figure, lg=8),
    ], className="g-3 py-3 border-bottom")


def _storm_options() -> list[dict]:
    return [{"label": f"{r.event} · offshore Hs {r.hs_off_peak:.1f} m", "value": i}
            for i, r in enumerate(data.storms().itertuples())]


def _xaver() -> int:
    hits = np.flatnonzero(data.storms()["event"].str.contains("Xaver").to_numpy())
    return int(hits[0]) if hits.size else 0


def layout(**_):
    if not data.available():
        return dbc.Alert("store/ is missing: run `python build_store.py`.", color="danger")
    if not (data.breaking_available() and data.goal_breaking_available()):
        return dbc.Alert("No breaking data in store/: run `python build_store.py --only breaking goal_breaking`.",
                         color="warning")
    params = data.breaking_params()
    ptable = pd.DataFrame([
        ("gamma", params["gamma"], "breaker index, Hmax = γ h (Hrms)", "template"),
        ("alpha", params["alpha"], "dissipation coefficient", "default"),
        ("gammax", params["gammax"], "limiter, Hrms ≤ γmax h", "default"),
        ("fw", params["fw"], "friction factor", "default"),
        ("hmin", params["hmin"], "minimum depth (m)", "template"),
    ], columns=["key", "value", "meaning", "from"])
    storms = _storm_options()
    runs = [k for k in data.DYNAMIC if data.DYNAMIC_KINDS[data.ESTIMATES[k]["kind"]][0] == "snapwave"]
    ceiling = {g: float(np.nanmedian(data.saturation()[_group(g)])) for g in ("5 m", "10 m", "20 m")}

    sliders = dbc.Row([
        dbc.Col(ui.labelled("depth h (m)", dcc.Slider(id="br-h", min=1, max=25, step=0.5, value=5,
                                                      marks={i: str(i) for i in (1, 5, 10, 15, 20, 25)},
                                                      tooltip={"placement": "bottom"})), md=6),
        dbc.Col(ui.labelled("Tp (s)", dcc.Slider(id="br-tp", min=4, max=16, step=0.5, value=10,
                                                 marks={i: str(i) for i in range(4, 17, 2)},
                                                 tooltip={"placement": "bottom"})), md=6),
    ], className="controls g-2")
    findings = ui.card("What we found (2013, metamodel = bulk linear sum, MDA 600, against SnapWave every hour; "
                       "both without wind)", [
        _finding(1, "SnapWave caps the waves by depth", [
            html.P(["In shallow water, breaking (Baldock) grows exponentially with the wave height, so at each site "
                    "Hs / h stops growing at a ceiling. We take it from the training cases: per site, the highest "
                    "Hs / h of any case of any goal (offshore Hs up to 10 m). Its median is ",
                    html.B(f"{ceiling['5 m']:.2f} at 5 m, {ceiling['10 m']:.2f} at 10 m, {ceiling['20 m']:.2f} at 20 m"),
                    f", well below the formula's breaker index γ = {params['gamma']:g}."]),
            html.Div(id="br-f1"),
            html.P(["Map: cap = ceiling × h. ", html.B("Hs − cap"), " below 0: room left before the ceiling; above 0: "
                    "over it. The dynamic run should stay at or below 0; what the metamodel has above 0 is what a "
                    "hard cap removes. Over the year the mean and median are calm hours: max, p99, a storm or one "
                    "hour show the breaking."], className="note"),
            dbc.Row([
                dbc.Col(ui.dropdown("br-r-what", [{"label": v, "value": k} for k, v in RATIO_WHAT.items()],
                                    "dyn_cap", "Map of"), md=7),
                dbc.Col(ui.dropdown("br-r-stat", [{"label": v, "value": k} for k, v in RATIO_STATS.items()], "max",
                                    "Statistic over the hours"), md=5),
            ], className="g-2"),
            dbc.Row([
                dbc.Col(ui.dropdown("br-r-period", [{"label": v, "value": k} for k, v in PERIODS.items()], "year",
                                    "Period"), md=5),
                dbc.Col(ui.dropdown("br-r-storm", storms, _xaver(), "Storm"), md=7),
            ], className="g-2"),
            ui.labelled("Hour of the storm (opens at the peak)",
                        dcc.Slider(id="br-r-hour", min=0, max=1, step=1, value=0, tooltip={"placement": "bottom"})),
            dcc.Checklist(id="br-r-opts", value=["isobaths"], className="small", options=[
                {"label": " isobaths of the mesh (2, 5, 10, 20 m)", "value": "isobaths"},
                {"label": " mesh depth (grey, 0-30 m)", "value": "depth"}]),
        ], [ui.graph(id="br-r-map", config={"scrollZoom": True}),
            html.Div("h = mesh depth + the dynamic run's water level, the same for the dynamic run and the metamodel "
                     "(a reconstruction would use the forcing level). Hours with h ≤ 0.5 m or no dynamic value are "
                     "left out.", className="note")]),
        _finding(2, "The metamodel misses part of that breaking", html.Div(id="br-f2"),
                 [ui.graph(id="br-relative"),
                  html.Div("Hs bias, metamodel − dynamic run, against the dynamic run's Hs / h (h = mesh depth + water "
                           "level), every hour of 2013; line = mean, band = 10-90 %. Dotted: the ceiling of each group "
                           "(finding 1).", className="note")]),
        _finding(3, "The error is at the shallow sites, in storms", [
            html.Div(id="br-f3"),
            ui.labelled("Map: Hs of", dbc.RadioItems(
                id="br-f3-cap", value="linear sum", className="small",
                options=[{"label": "the metamodel (linear sum)", "value": "linear sum"},
                         {"label": "the metamodel with the hard cap", "value": "hard cap"},
                         {"label": "the metamodel with the smooth cap, p = 6", "value": "smooth p = 6"}])),
            ui.labelled("Layers", dcc.Checklist(
                id="br-f3-layers", value=["isobaths", "depth"], className="small",
                options=[{"label": " isobaths of the mesh (2, 5, 10, 20 m)", "value": "isobaths"},
                         {"label": " mesh depth (grey, darker = deeper, 0-30 m)", "value": "depth"}])),
            ui.labelled("Mean over the sites (m), same 0.1 % highest hours", ui.loading(html.Div(id="br-f3-means"))),
        ], ui.graph(id="br-f3-map", config={"scrollZoom": True})),
        _finding(4, "Capping Hs at the ceiling removes most of the shallow overshoot", html.Div(id="br-f4"),
                 [ui.dropdown("br-f4-storm", storms, _xaver(), "Storm"), ui.graph(id="br-f4-series")]),
    ])

    details = dbc.Accordion([
        dbc.AccordionItem([dcc.Markdown(PHYSICS, mathjax=True, className="markdown"),
                           ui.table(ptable, {"value": "{:g}"}, index=False),
                           html.H6("The formula", className="mt-3 fw-bold"),
                           html.P("Move the depth and the period to see when breaking starts (Dw, Qb) and how fast it "
                                  "drains the waves; the red line is the ceiling at sites of that depth.",
                                  className="note"),
                           sliders, ui.graph(id="br-curve"), html.Div(id="br-curve-note", className="note")],
                          title="SnapWave's breaking in full, and how the page computes Dw"),
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col(ui.dropdown("br-check-run", [{"label": data.label(k), "value": k} for k in runs],
                                    "dyn_off_WHACS_cal", "Dynamic run"), md=4),
                dbc.Col(ui.dropdown("br-check-storm", storms, 0, "Hour: the peak of"), md=5),
            ], className="controls g-2"),
            dbc.Row([dbc.Col(ui.graph(id="br-check"), lg=7), dbc.Col(html.Div(id="br-check-note", className="note"), lg=5)]),
        ], title="point_dw against the formula on SnapWave's own Hm0, Tp and depth"),
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col(ui.dropdown("br-group", list(GROUPS), "5 m", "Sites (table)"), md=4),
                        dbc.Col(ui.dropdown("br-thr", THRESHOLDS, 10, "Breaking: Dw above (W/m², table and map)"), md=8),
                    ], className="controls g-2"),
                    ui.loading(html.Div(id="br-classes")),
                ], lg=6),
                dbc.Col([
                    ui.dropdown("br-metric", [{"label": v, "value": k} for k, v in MAP_METRICS.items()], "miss",
                                "Per site, over 2013"),
                    ui.graph(id="br-map", config={"scrollZoom": True}),
                    html.Div([html.B("Breaks"), ": the dynamic run's Dw at the site is above the threshold; ",
                              html.B("goals alone do not"), ": the sum of each goal's Dw is not. ",
                              html.B("Σ Dw goals / Dw dynamic"), ": how much of the dynamic run's breaking the goals "
                              "alone reproduce (1 = all, 0.2 = a fifth). ",
                              html.B("Top 0.1 %"), ": the site's ~9 highest dynamic hours."], className="note"),
                ], lg=6),
            ], className="g-3"),
        ], title="Who breaks: the dynamic run and the goals alone, per class and per site"),
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col(ui.dropdown("br-storm", storms, _xaver(), "Storm"), md=4),
                dbc.Col(ui.labelled("hour of the event (opens at the peak)",
                                    dcc.Slider(id="br-hour", min=0, max=1, step=1, value=0,
                                               tooltip={"placement": "bottom"})), md=8),
            ], className="controls g-2"),
            html.Div(id="br-hour-label", className="fw-semibold my-1"),
            dbc.Row([dbc.Col(ui.graph(id=f"br-hmap-{k}", config={"scrollZoom": True}), lg=4)
                     for k in ("dyn", "goals", "bias")], className="g-1"),
            html.Div("Zoom or pan one map and the other two follow.", className="note"),
            dcc.Markdown(HOUR_GUIDE, className="markdown note mt-2"),
        ], title="A storm hour by hour: Dw of the dynamic run, of the goals alone, and the Hs bias"),
        dbc.AccordionItem([
            html.P(["Hs capped at (highest Hs / h of any training case at the site) × (mesh depth + the mean CODEC "
                    "level at the goals). Smooth: (Hs⁻ᵖ + cap⁻ᵖ)^(−1/p), which also lowers seas just under the "
                    "ceiling; ", html.B("p is chosen on the same 2013 hours it is scored on"),
                    ". Top 1 % / 0.1 %: hours where the dynamic Hs is above the group's p99 / p99.9."], className="note"),
            ui.loading(html.Div(id="br-cap")),
        ], title="Every cap variant, every site group"),
    ], start_collapsed=True, always_open=True)

    return html.Div([
        ui.header("Breaking", "The metamodel adds up the energy of goals run one at a time, but breaking is not "
                  "linear: goals that each stay under the breaking limit can add up to a sea above it. Four "
                  "findings, each next to the figure that shows it; the details are folded below."),
        dbc.Row(dbc.Col(ui.dropdown("br-source", list(data.OFFSHORE), "WHACS_cal", "Offshore source"), md=2),
                className="controls"),
        findings,
        html.H5("Details", className="mt-3"),
        details,
        dcc.Store(id="br-view", data={"center": fg.MAP_CENTER, "zoom": fg.MAP_ZOOM}),
    ])


# ---------------------------------------------------------------------------
# Data shared by the findings and the details
# ---------------------------------------------------------------------------


def _group(group: str) -> np.ndarray:
    return (data.sites()["group"] == GROUPS[group]).to_numpy()


@lru_cache(maxsize=4)
def _pairs(source: str) -> dict[str, np.ndarray]:
    """
    ``(time, site)`` arrays, metamodel against the dynamic run without wind: Hs of both (``hm``, ``hd``), the dynamic
    run's depth ``h``, and Dw of the dynamic run (``dwd``) and of the goals alone (``dwg``), both by the formula.
    """
    dyn, mm, goals = f"dyn_off_{source}", f"mm_bulk_{source}", f"goals_{source}"
    hd, hm = data.field(dyn, "hs"), data.field(mm, "hs")
    p = data.breaking_params()
    h = data.run_depth(dyn)
    dwd = brk.dissipation(hd, data.field(dyn, "tp"), h, p["gamma"], p["alpha"]).astype("float32")
    return {"ok": np.isfinite(hd) & np.isfinite(hm) & (hd > 0.05), "hd": hd, "hm": hm, "h": h, "dwd": dwd,
            "dwg": data.field(goals, "dw")}


@lru_cache(maxsize=16)
def _site_metric(source: str, metric: str, thr: float) -> np.ndarray:
    """One value per cube column over 2013 (:data:`MAP_METRICS`)."""
    d = _pairs(source)
    ok = d["ok"]
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "sat":
            return data.saturation()
        if metric in ("dyn_max", "sum_max"):
            ratio = (d["hd"] if metric == "dyn_max" else d["hm"]) / d["h"]
            return np.nanmax(np.where(ok & (d["h"] > 0.5), ratio, np.nan), 0)
        breaks = ok & (d["dwd"] > thr)
        n = ok.sum(0)
        if metric == "brk":
            return np.where(n > 0, 100 * breaks.sum(0) / n, np.nan)
        if metric == "miss":
            return np.where(n > 0, 100 * (breaks & ~(d["dwg"] > thr)).sum(0) / n, np.nan)
        if metric == "bias_brk":
            return np.nanmean(np.where(breaks, d["hm"] - d["hd"], np.nan), 0)
        if metric == "ratio":
            return np.nanmedian(np.where(breaks, d["dwg"] / d["dwd"], np.nan), 0)
        hd = np.where(ok, d["hd"], np.nan)
        top = hd > np.nanquantile(hd, 0.999, 0)
        return np.nanmean(np.where(top, d["hm"] - d["hd"], np.nan), 0)


@lru_cache(maxsize=2)
def _cap_table(source: str) -> pd.DataFrame:
    """Metamodel Hs (plain and capped) against the dynamic run per site group, on all hours and the top 1 / 0.1 %."""
    d = _pairs(source)
    rows = []
    for group in ("5 m", "10 m", "20 m", "buoys"):
        cols = _group(group)
        ok = d["ok"][:, cols]
        ref = d["hd"][:, cols][ok]
        q99, q999 = np.quantile(ref, [0.99, 0.999])
        for name, p in CAPS.items():
            est = d["hm"] if p == "none" else data.capped(f"mm_bulk_{source}", p)
            diff = est[:, cols][ok] - ref
            rows.append({"sites": group, "estimate": name, "bias all": diff.mean(), "RMSE all": np.sqrt((diff**2).mean()),
                         "bias top 1 %": diff[ref > q99].mean(), "RMSE top 1 %": np.sqrt((diff[ref > q99] ** 2).mean()),
                         "bias top 0.1 %": diff[ref > q999].mean(),
                         "RMSE top 0.1 %": np.sqrt((diff[ref > q999] ** 2).mean()),
                         "% hours lowered > 1 cm": 100 * np.mean((d["hm"][:, cols][ok] - est[:, cols][ok]) > 0.01)})
    return pd.DataFrame(rows)


@lru_cache(maxsize=2)
def _missed(source: str) -> dict[str, float]:
    """Median Σ Dw goals / Dw dynamic where the dynamic run breaks, and the 5 m bias near / below the ceiling."""
    d = _pairs(source)
    out = {}
    for group in ("5 m", "10 m"):
        cols = _group(group)
        ok = d["ok"][:, cols]
        dwd, dwg = d["dwd"][:, cols][ok], d["dwg"][:, cols][ok]
        breaks = dwd > BREAKS
        out[group] = float(np.median(dwg[breaks] / dwd[breaks]))
    cols = _group("5 m")
    ok = d["ok"][:, cols] & (d["h"][:, cols] > 0.5)
    ratio = (d["hd"][:, cols] / d["h"][:, cols])[ok]
    bias = (d["hm"][:, cols] - d["hd"][:, cols])[ok]
    out["near"], out["below"] = float(bias[ratio >= 0.35].mean()), float(bias[ratio < 0.3].mean())
    return out


# ---------------------------------------------------------------------------
# Finding 1: the formula
# ---------------------------------------------------------------------------


@callback(Output("br-curve", "figure"), Output("br-curve-note", "children"), Input("br-h", "value"),
          Input("br-tp", "value"))
def curve(h, tp):
    p = data.breaking_params()
    hm0 = np.linspace(0.01, 1.0 * h, 400)
    dw = brk.dissipation(hm0, tp, h, p["gamma"], p["alpha"])
    q = brk.qb(hm0, h, p["gamma"])
    length = brk.decay_length(hm0, tp, h, p["gamma"], p["alpha"]) / 1000
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Dw (W/m²)", "Qb = exp(−(Hmax/Hrms)²)", "E cg / Dw (km, log)"))
    fig.add_trace(go.Scatter(x=hm0, y=dw, line=dict(color="#d7301f", width=2), name="Dw"), 1, 1)
    fig.add_trace(go.Scatter(x=hm0, y=q, line=dict(color="#4575b4", width=2), name="Qb"), 2, 1)
    fig.add_trace(go.Scatter(x=hm0, y=np.where(length < 1e4, length, np.nan), line=dict(color="#1a9850", width=2),
                             name="decay length"), 3, 1)
    fig.update_yaxes(type="log", row=3, col=1, range=[-2, 3])
    marks = [(np.sqrt(2) * p["gamma"] * h, f"Hrms = γh ({np.sqrt(2) * p['gamma']:.2f} h)", "#4575b4"),
             (np.sqrt(2) * p["gammax"] * h, f"limiter ({np.sqrt(2) * p['gammax']:.2f} h)", "#555555")]
    table = data.sites()
    sat = data.saturation()
    near = np.abs(table["depth"].to_numpy() - h) < max(0.5, 0.1 * h)
    note = f"No site with a mesh depth near {h:g} m."
    if near.any() and np.isfinite(sat[near]).any():
        s = float(np.nanmedian(sat[near]))
        marks.append((s * h, f"ceiling ({s:.2f} h)", COLORS["ceiling"]))
        note = f"{int(near.sum())} sites with a mesh depth near {h:g} m: ceiling {s:.2f} h = {s * h:.2f} m."
    for x, text, color in marks:
        if x <= hm0[-1]:
            fig.add_vline(x=x, line=dict(color=color, dash="dot", width=1.5))
            fig.add_annotation(x=x, y=1, yref="paper", text=text, showarrow=False, textangle=-90, xanchor="right",
                               yanchor="top", font=dict(size=10, color=color))
    fig.update_xaxes(title_text="Hm0 (m)", row=3, col=1)
    fig.update_layout(showlegend=False, hovermode="x unified")
    return fg.base_layout(fig, 520), note


# ---------------------------------------------------------------------------
# Finding 1: Hs / h per site
# ---------------------------------------------------------------------------


def _ratio(source: str, what: str, a: int, b: int) -> np.ndarray:
    """
    ``(hour, site)`` values of *what* over hours ``a:b`` (:data:`RATIO_WHAT` but the ceiling): Hs / h, or Hs − cap
    (m) with cap = ceiling × h; h = the dynamic run's depth.
    """
    d = _pairs(source)
    h = d["h"][a:b]
    hs = d["hd"][a:b] if what.startswith("dyn") else d["hm"][a:b]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = hs - data.saturation()[None, :] * h if what.endswith("_cap") else hs / h
    return np.where(d["ok"][a:b] & (h > 0.5), v, np.nan)


@lru_cache(maxsize=64)
def _ratio_stat(source: str, what: str, stat: str, a: int, b: int) -> np.ndarray:
    """Per site, *stat* (:data:`RATIO_STATS`) of Hs / h over hours ``a:b``."""
    x = _ratio(source, what, a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN sites
        return {"mean": np.nanmean, "median": np.nanmedian, "std": np.nanstd, "max": np.nanmax,
                "p99": lambda v, axis: np.nanquantile(v, 0.99, axis=axis)}[stat](x, axis=0)


@callback(Output("br-r-hour", "max"), Output("br-r-hour", "value"), Output("br-r-hour", "marks"),
          Input("br-r-storm", "value"))
def ratio_hours(storm):
    return hours(storm)


@callback(Output("br-r-stat", "disabled"), Output("br-r-storm", "disabled"), Output("br-r-hour", "disabled"),
          Input("br-r-what", "value"), Input("br-r-period", "value"))
def ratio_controls(what, period):
    ceiling = what == "ceiling"
    return ceiling or period == "hour", ceiling or period == "year", ceiling or period != "hour"


@callback(Output("br-r-map", "figure"), Input("br-source", "value"), Input("br-r-what", "value"),
          Input("br-r-stat", "value"), Input("br-r-period", "value"), Input("br-r-storm", "value"),
          Input("br-r-hour", "value"), Input("br-r-opts", "value"))
def ratio_map(source, what, stat, period, storm, hour, opts):
    ceiling = data.saturation()
    s = data.storms().iloc[storm]
    if period == "year":
        a, b, when = 0, len(data.times()), "2013"
    else:
        a, b = data.time_index(s["start"]), data.time_index(s["end"]) + 1
        when = f"{s['event']}"
        if period == "hour":
            a = min(a + int(hour or 0), b - 1)
            b, when = a + 1, f"{s['event']}, {data.times()[a]:%Y-%m-%d %H:%M}"

    def values_of(w: str) -> np.ndarray:
        return _ratio(source, w, a, b)[0] if period == "hour" else _ratio_stat(source, w, stat, a, b)

    if what == "ceiling":
        values, title = ceiling, "the ceiling: highest Hs / h of any training case (any goal)"
    else:
        values = values_of(what)
        title = (f"{source}, {RATIO_WHAT[what]}" + ("" if period == "hour" else f", {RATIO_STATS[stat]}")
                 + f" · {when}")
    if what.endswith("_cap") and (period == "hour" or stat != "std"):
        # symmetric about 0, sized on the overshoot (deep sites have metres of room and would wash it out), and
        # shared by the dynamic run and the metamodel, so the two maps compare
        both = np.r_[values_of("dyn_cap"), values_of("mm_cap")]
        over = both[both > 0]
        lim = max(float(np.quantile(over, 0.98)) if over.size else 0.0, 0.25)
        kw = dict(colorscale=fg.DIVERGING, zmin=-lim, zmax=lim, colorbar="m")
    else:  # from 0 to the dynamic run's (or the ceiling's) range, so switching dynamic / metamodel keeps the scale
        ref = ceiling if what == "ceiling" else values_of("dyn_cap" if what.endswith("_cap") else "dyn")
        kw = dict(colorscale="YlOrRd", zmin=0.0, zmax=float(np.nanquantile(ref, 0.99)) if np.isfinite(ref).any() else 1.0,
                  colorbar="m" if what.endswith("_cap") else "Hs / h")
    fig = fg.site_map(values, np.arange(len(values)), height=600, uirevision="br-r", title=title,
                      under=fg.isobath_traces() if "isobaths" in opts else None,
                      layers=[fg.depth_layer()] if "depth" in opts else None, **kw)
    fig.update_layout(legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,0.8)",
                                  font=dict(size=10)))
    return fig


@lru_cache(maxsize=2)
def _over_ceiling(source: str) -> dict[str, dict[str, float]]:
    """Per isobath: % of sites whose highest 2013 Hs is above the cap, for the dynamic run and the metamodel."""
    n = len(data.times())
    return {w: {g: 100 * float(np.nanmean(_ratio_stat(source, f"{w}_cap", "max", 0, n)[_group(g)] > 0))
                for g in ("5 m", "10 m", "20 m")} for w in ("dyn", "mm")}


# ---------------------------------------------------------------------------
# Findings 2-4
# ---------------------------------------------------------------------------


@callback(Output("br-f1", "children"), Output("br-f2", "children"), Output("br-f3", "children"),
          Output("br-f4", "children"), Input("br-source", "value"))
def findings_text(source):
    over = _over_ceiling(source)
    f1 = html.P(["The dynamic run (all goals together) stays under it: its highest Hs of 2013 is above the cap at ",
                 html.B(f"{over['dyn']['5 m']:.0f} % of the 5 m sites"), f" ({over['dyn']['10 m']:.0f} % at 10 m). "
                 "The metamodel, a sum of goals, goes above it at ",
                 html.B(f"{over['mm']['5 m']:.0f} % of the 5 m sites"),
                 f" ({over['mm']['10 m']:.0f} % at 10 m, {over['mm']['20 m']:.0f} % at 20 m)."])
    m = _missed(source)
    f2 = [
        html.P(["Each goal alone is a smaller sea than all the goals together, and breaking grows exponentially with "
                "the wave height, so the goals alone break much less. Where the dynamic run breaks "
                f"(Dw > {BREAKS:g} W/m²), the goals alone give a median ",
                html.B(f"{m['5 m']:.0%} of its breaking at 5 m"), f" ({m['10 m']:.0%} at 10 m)."]),
        html.P(["The energy that is not broken stays in the sum. So the error grows as the dynamic run approaches "
                "its ceiling: at 5 m the bias is ", html.B(f"{m['near']:+.2f} m when Hs / h ≥ 0.35"),
                f" and {m['below']:+.2f} m below 0.3."]),
    ]
    cap = _cap_table(source).set_index(["sites", "estimate"])
    top = {g: _site_metric(source, "top", BREAKS)[_group(g)] for g in ("5 m", "10 m", "20 m")}
    n = len(data.times())
    deep = float(np.nanmedian(_ratio_stat(source, "dyn", "max", 0, n)[_group("20 m")]))
    f3 = [
        html.P(["Over all hours the metamodel matches the dynamic run (bias "
                f"{cap.loc[('5 m', 'linear sum'), 'bias all']:+.2f} m at 5 m). In each site's 0.1 % highest hours "
                "(~9 hours of 2013) it is too high, by ", html.B(f"{np.nanmean(top['5 m']):+.2f} m at 5 m"),
                f", {np.nanmean(top['10 m']):+.2f} m at 10 m and {np.nanmean(top['20 m']):+.2f} m at 20 m "
                "(mean over the sites)."]),
        html.P(f"The map shows that bias per site; pick a capped metamodel to see what the cap leaves. At 20 m the "
               f"highest Hs / h of the year is {deep:.2f} (median over the sites), far from the ceiling: the bias "
               "left there is not breaking.", className="note"),
    ]
    rows = [{"sites": g, "estimate": e, "bias top 0.1 % (m)": cap.loc[(g, e), "bias top 0.1 %"],
             "bias top 1 % (m)": cap.loc[(g, e), "bias top 1 %"], "bias all (m)": cap.loc[(g, e), "bias all"]}
            for g in ("5 m", "10 m", "20 m") for e in ("linear sum", "hard cap", "smooth p = 6")]
    f4 = [
        html.P(["Cap the metamodel's Hs at each site at ceiling × h (finding 1), with h = mesh depth + the forcing "
                "water level. It needs only the training cases: nothing is fitted on the dynamic runs."]),
        html.P([html.B("Hard cap"), ", min(Hs, cap): changes only the hours above the cap, so the 20 m sites barely "
                "move; it leaves part of the 5 m overshoot, because SnapWave already bends below the ceiling. ",
                html.B("Smooth cap"), " (p = 6): also lowers seas just under the cap, as Baldock does, which removes "
                "more at 5 m but also lowers the 20 m storm hours, where nothing breaks."]),
        ui.table(pd.DataFrame(rows), {c: "{:+.2f}" for c in ("bias top 0.1 % (m)", "bias top 1 % (m)", "bias all (m)")},
                 index=False),
        html.P(["Bias = metamodel − dynamic run. Top 0.1 % / 1 %: hours above the group's p99.9 / p99 of the dynamic "
                "Hs. ", html.B("p = 6 was picked on these same hours"), "; the hard cap has nothing to pick. ",
                "The reconstruction now applies the hard cap (metamodel.reconstruction.depth_cap); the other "
                "pages of this dashboard show the plain linear sum."],
               className="note mt-2"),
    ]
    return f1, f2, f3, f4


@callback(Output("br-relative", "figure"), Input("br-source", "value"))
def relative(source):
    d = _pairs(source)
    sat = data.saturation()
    bins = np.arange(0, 0.625, 0.025)
    centres = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure()
    for name in ("5 m", "10 m", "20 m", "buoys"):
        cols = _group(name)
        ok = d["ok"][:, cols] & (d["h"][:, cols] > 0.5)
        x = (d["hd"][:, cols] / d["h"][:, cols])[ok]
        bias = (d["hm"][:, cols] - d["hd"][:, cols])[ok]
        b = np.digitize(x, bins) - 1
        stats = [(np.mean(bias[b == k]), *np.percentile(bias[b == k], [10, 90]), (b == k).sum())
                 if (b == k).sum() >= 30 else (np.nan, np.nan, np.nan, 0) for k in range(len(centres))]
        mean, lo, hi, n = map(np.array, zip(*stats))
        valid = np.isfinite(mean)  # the band is one closed polygon over the bins with enough pairs
        x_, lo_, hi_ = centres[valid], lo[valid], hi[valid]
        color = GROUP_COLORS[name]
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
        fig.add_trace(go.Scatter(x=np.r_[x_, x_[::-1]], y=np.r_[hi_, lo_[::-1]], fill="toself",
                                 fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.15)", line=dict(width=0), mode="lines",
                                 hoverinfo="skip", showlegend=False, legendgroup=name))
        fig.add_trace(go.Scatter(x=x_, y=mean[valid], mode="lines+markers", name=name, legendgroup=name,
                                 line=dict(color=color, width=2), customdata=n[valid],
                                 hovertemplate=f"{name}: Hs/h %{{x:.3f}}, bias %{{y:+.2f}} m, "
                                               "n %{customdata:,}<extra></extra>"))
        s = np.nanmedian(sat[cols])
        if np.isfinite(s):
            fig.add_vline(x=s, line=dict(color=color, dash="dot", width=1.2))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.update_layout(xaxis_title="Hs / h of the dynamic run", yaxis_title="Hs bias, metamodel − dynamic (m)",
                      legend=dict(orientation="h", y=-0.2, x=0))
    fig = fg.base_layout(fig, 440, f"{source}: the error grows as the dynamic run nears its ceiling")
    fig.update_layout(margin=dict(b=70))
    return fig


@lru_cache(maxsize=8)
def _top_bias(source: str, estimate: str) -> np.ndarray:
    """Per site: mean Hs bias against the dynamic run over the site's 0.1 % highest dynamic hours (:data:`CAPS` name)."""
    d = _pairs(source)
    p = CAPS[estimate]
    hm = d["hm"] if p == "none" else data.capped(f"mm_bulk_{source}", p)
    hd = np.where(d["ok"], d["hd"], np.nan)
    with np.errstate(invalid="ignore"):
        top = hd > np.nanquantile(hd, 0.999, 0)
        return np.nanmean(np.where(top, hm - d["hd"], np.nan), 0)


@callback(Output("br-f3-means", "children"), Input("br-source", "value"))
def top_means(source):
    """Mean of the per-site top-0.1 % bias, per isobath and for the NE corner, for every estimate of the map."""
    table = data.sites()
    groups = {g: (table["group"] == GROUPS[g]).to_numpy() for g in ("5 m", "10 m", "20 m")}
    groups["5 m, NE (east of 6° E)"] = groups["5 m"] & (table["lon"] > 6.0).to_numpy()
    rows = [{"Hs of": e, **{g: float(np.nanmean(_top_bias(source, e)[m])) for g, m in groups.items()}}
            for e in ("linear sum", "hard cap", "smooth p = 6")]
    return ui.table(pd.DataFrame(rows), {g: "{:+.2f}" for g in groups}, index=False)


@callback(Output("br-f3-map", "figure"), Input("br-source", "value"), Input("br-f3-cap", "value"),
          Input("br-f3-layers", "value"))
def top_map(source, estimate, layers):
    values = _top_bias(source, estimate)
    plain = _top_bias(source, "linear sum")
    lim = float(np.nanquantile(np.abs(plain), 0.98))  # the uncapped scale, so the three maps compare
    fig = fg.site_map(values, np.arange(len(values)), colorscale=fg.DIVERGING, zmin=-lim, zmax=lim, colorbar="m",
                      height=600, uirevision="br-f3",
                      title=f"{source}, {estimate}: Hs − dynamic in each site's 0.1 % highest hours",
                      under=fg.isobath_traces() if "isobaths" in layers else None,
                      layers=[fg.depth_layer()] if "depth" in layers else None)
    fig.update_layout(legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,0.8)",
                                  font=dict(size=10)))
    return fig


@callback(Output("br-f4-series", "figure"), Input("br-f4-storm", "value"), Input("br-source", "value"))
def cap_series(storm, source):
    """The 5 m site where the metamodel is highest above the dynamic run at the storm peak, through the storm."""
    s = data.storms().iloc[storm]
    a, b, peak = data.time_index(s["start"]), data.time_index(s["end"]) + 1, data.time_index(s["peak"])
    d = _pairs(source)
    cols = np.flatnonzero(_group("5 m"))
    over = np.where(d["ok"][peak, cols], d["hm"][peak, cols] - d["hd"][peak, cols], -np.inf)
    col = int(cols[np.argmax(over)])
    t = data.times()[a:b]
    cap = data.saturation()[col] * data.forcing_depth(source)[a:b, col]
    site = data.sites().iloc[col]
    fig = go.Figure()
    for name, y, color, dash_ in (
        ("dynamic run (all goals together)", d["hd"][a:b, col], COLORS["dyn"], "solid"),
        ("metamodel (sum of the goals)", d["hm"][a:b, col], COLORS["mm"], "solid"),
        ("metamodel, hard cap", data.capped(f"mm_bulk_{source}", None)[a:b, col], COLORS["hard"], "dash"),
        ("metamodel, smooth cap p = 6", data.capped(f"mm_bulk_{source}", 6.0)[a:b, col], COLORS["smooth"], "dash"),
        ("ceiling × depth", cap, COLORS["ceiling"], "dot"),
    ):
        fig.add_trace(go.Scatter(x=t, y=y, name=name, line=dict(color=color, dash=dash_, width=1.8 if dash_ != "dot" else 1.2)))
    fig.add_vline(x=s["peak"], line=dict(color="black", width=1, dash="dot"))
    fig.update_layout(yaxis_title="Hs (m)", hovermode="x unified", legend=dict(orientation="h", y=-0.15, x=0))
    fig = fg.base_layout(fig, 440, f"{site['site']} (mesh depth {site['depth']:.1f} m): the 5 m site where the "
                                   f"metamodel is highest above the dynamic run at the peak")
    fig.update_layout(margin=dict(b=70))
    return fig


# ---------------------------------------------------------------------------
# Details
# ---------------------------------------------------------------------------


@callback(Output("br-check", "figure"), Output("br-check-note", "children"), Input("br-check-run", "value"),
          Input("br-check-storm", "value"))
def check(run, storm):
    p = data.breaking_params()
    i = data.time_index(data.storms().iloc[storm]["peak"])
    hs, tp, dw = data.field(run, "hs")[i], data.field(run, "tp")[i], data.field(run, "dw")[i]
    if not np.isfinite(dw).any():
        return fg.empty("this run did not converge at this hour"), ""
    est = brk.dissipation(hs, tp, data.run_depth(run)[i], p["gamma"], p["alpha"])
    groups = data.sites()["group"].to_numpy()
    fig = go.Figure()
    for name, group in GROUPS.items():
        m = (groups == group) & np.isfinite(dw) & (dw > 0.01)
        fig.add_trace(go.Scattergl(x=dw[m], y=est[m], mode="markers", name=name,
                                   marker=dict(size=5 if group != "buoy" else 9, color=GROUP_COLORS[name], opacity=0.6)))
    fig.add_trace(go.Scatter(x=[0.01, 2000], y=[0.01, 2000], mode="lines", line=dict(color="black", dash="dash", width=1),
                             showlegend=False))
    fig.update_xaxes(type="log", title="point_dw, SnapWave (W/m²)", range=[-2, 3.3])
    fig.update_yaxes(type="log", title="formula on point_hm0, point_tp, depth + zs (W/m²)", range=[-2, 3.3])
    fig.update_layout(legend=dict(orientation="h", y=1.08))
    rows = []
    for thr in (1, 10, 100):
        m = np.isfinite(dw) & (dw > thr)
        if m.sum():
            r = est[m] / dw[m]
            rows.append({"Dw above (W/m²)": thr, "sites": int(m.sum()), "median formula / SnapWave": np.median(r),
                         "p5": np.percentile(r, 5), "p95": np.percentile(r, 95)})
    note = [html.P(f"{data.times()[i]:%Y-%m-%d %H:%M}, Tp {np.nanmedian(tp):.1f} s (one value for the domain), "
                   f"water level {np.nanmedian(data.field(run, 'zs')[i]):+.2f} m."),
            ui.table(pd.DataFrame(rows), {"median formula / SnapWave": "{:.3f}", "p5": "{:.2f}", "p95": "{:.2f}"},
                     index=False),
            html.P("Points where point_dw is far above the formula: SnapWave computes Dw at the mesh nodes, each with its "
                   "own depth, and writes at an output point the weighted mean of the nodes' Dw (and, separately, of "
                   "their H). When the point's triangle mixes a shallow node that breaks with deeper ones that do not "
                   "(e.g. 0.9 and 7.8 m), the point inherits part of the shallow node's Dw, while the formula on the "
                   "point's mean H and depth sees no breaking. Recomputing Dw at the nodes from the grid maps and "
                   "interpolating the same way reproduces point_dw exactly. So it is not breaking at the point: it "
                   "says a node next to it breaks (steep slopes, shoals, channel edges, coarse mesh). That is why the "
                   "rest of the page uses the formula at the point on both sides.", className="mt-2")]
    return fg.base_layout(fig, 430), note


@callback(Output("br-classes", "children"), Input("br-source", "value"), Input("br-group", "value"),
          Input("br-thr", "value"))
def classes(source, group, thr):
    d = _pairs(source)
    cols = _group(group)
    ok = d["ok"][:, cols]
    hd, hm, h = d["hd"][:, cols][ok], d["hm"][:, cols][ok], d["h"][:, cols][ok]
    dyn, goals = d["dwd"][:, cols][ok] > thr, d["dwg"][:, cols][ok] > thr
    bias = hm - hd
    rows = []
    for name, m in (("both break", dyn & goals), ("dynamic breaks, goals alone do not", dyn & ~goals),
                    ("goals alone break, dynamic does not", ~dyn & goals), ("neither", ~dyn & ~goals)):
        rows.append({"class": name, "pairs": int(m.sum()), "% of pairs": 100 * m.mean(),
                     "mean Hs dynamic (m)": hd[m].mean() if m.any() else np.nan,
                     "mean Hs / h dynamic": (hd[m] / h[m]).mean() if m.any() else np.nan,
                     "bias (m)": bias[m].mean() if m.any() else np.nan,
                     "RMSE (m)": np.sqrt((bias[m] ** 2).mean()) if m.any() else np.nan,
                     "% of the positive error": 100 * np.clip(bias[m], 0, None).sum() / np.clip(bias, 0, None).sum()})
    note = html.P(f"{source}, {group}, every valid hour × site of 2013; 'breaks' = Dw above {thr} W/m². Last column: "
                  "share of the metamodel's total overestimation (positive errors) in each class.",
                  className="note mb-1")
    fmt = {"% of pairs": "{:.2f}", "mean Hs dynamic (m)": "{:.2f}", "mean Hs / h dynamic": "{:.3f}",
           "bias (m)": "{:+.3f}", "RMSE (m)": "{:.3f}", "% of the positive error": "{:.1f}"}
    return [note, ui.table(pd.DataFrame(rows), fmt, index=False)]


@callback(Output("br-map", "figure"), Input("br-source", "value"), Input("br-metric", "value"),
          Input("br-thr", "value"))
def site_map(source, metric, thr):
    values = _site_metric(source, metric, float(thr))
    scale = {"brk": "YlOrRd", "miss": "YlOrRd", "ratio": "Viridis"}.get(metric, "Viridis")
    zmin, zmax = (0, 1) if metric == "ratio" else (None, None)
    unit = {"brk": "%", "miss": "%", "bias_brk": "m", "top": "m", "ratio": "ratio"}.get(metric, "Hs / h")
    return fg.site_map(values, np.arange(len(values)), colorscale=scale, zmin=zmin, zmax=zmax,
                       diverging=metric in ("bias_brk", "top"), colorbar=unit, height=560, uirevision="br-map",
                       title=f"{source}: {MAP_METRICS[metric]}")


@callback(Output("br-hour", "max"), Output("br-hour", "value"), Output("br-hour", "marks"), Input("br-storm", "value"))
def hours(storm):
    s = data.storms().iloc[storm]
    a, b = data.time_index(s["start"]), data.time_index(s["end"])
    marks = {k: f"{data.times()[a + k]:%d %H h}" for k in range(24, b - a, 24)}  # none at the ends (they overflow)
    return b - a, data.time_index(s["peak"]) - a, marks


HOUR_MAPS = ("br-hmap-dyn", "br-hmap-goals", "br-hmap-bias")


@callback(*(Output(m, "figure") for m in HOUR_MAPS), Output("br-hour-label", "children"), Input("br-hour", "value"),
          Input("br-storm", "value"), Input("br-source", "value"), State("br-view", "data"))
def hour_maps(k, storm, source, view):
    i = data.time_index(data.storms().iloc[storm]["start"]) + int(k or 0)
    d = _pairs(source)
    cols = np.arange(d["hd"].shape[1])
    dwd, dwg = d["dwd"][i], d["dwg"][i]
    both = np.r_[dwd, dwg]
    zmax = float(np.nanquantile(both[np.isfinite(both)], 0.99)) if np.isfinite(dwd).any() else 1.0
    kw = dict(colorscale="YlOrRd", zmin=0, zmax=max(zmax, 1.0), colorbar="W/m²", height=470, size=5)
    figs = (fg.site_map(dwd, cols, title="Dw, dynamic run", uirevision="br-h1", **kw),
            fg.site_map(dwg, cols, title="Σ Dw, goals alone", uirevision="br-h2", **kw),
            fg.site_map(np.where(d["ok"][i], d["hm"][i] - d["hd"][i], np.nan), cols, title="Hs, metamodel − dynamic (m)",
                        diverging=True, colorbar="m", height=470, size=5, uirevision="br-h3"))
    for fig in figs:  # open where the user last left the three maps
        fig.update_layout(map=dict(center=view["center"], zoom=view["zoom"]))
    c = data.conditions_of(source).iloc[i]
    label = (f"{data.times()[i]:%Y-%m-%d %H:%M} · {source} offshore Hs {c['hs_off']:.2f} m (goal mean, max "
             f"{c['hs_off_max']:.2f}) from {c['dir_off']:.0f}° · water level {c['wl']:+.2f} m")
    return *figs, label


@callback(Output("br-view", "data"), *(Output(m, "figure", allow_duplicate=True) for m in HOUR_MAPS),
          *(Input(m, "relayoutData") for m in HOUR_MAPS), prevent_initial_call=True)
def sync_view(*relayouts):
    """Zoom or pan on one hour map moves the other two (and the next hours open there)."""
    moved = relayouts[HOUR_MAPS.index(ctx.triggered_id)] or {}
    center, zoom = moved.get("map.center"), moved.get("map.zoom")
    if center is None or zoom is None:
        return no_update, *(no_update for _ in HOUR_MAPS)
    patches = []
    for m in HOUR_MAPS:
        if m == ctx.triggered_id:
            patches.append(no_update)
            continue
        p = Patch()
        p["layout"]["map"]["center"] = center
        p["layout"]["map"]["zoom"] = zoom
        patches.append(p)
    return {"center": center, "zoom": zoom}, *patches


@callback(Output("br-cap", "children"), Input("br-source", "value"))
def cap(source):
    table = _cap_table(source)
    fmt = {c: ("{:+.3f}" if c.startswith("bias") else "{:.3f}") for c in table.columns if c not in ("sites", "estimate")}
    fmt["% hours lowered > 1 cm"] = "{:.1f}"
    return ui.table(table, fmt, index=False)
