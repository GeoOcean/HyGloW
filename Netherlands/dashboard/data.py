"""
Read the dashboard store (``store/``, written by ``build_store.py``) and compute metrics on it.

Arrays of the cube are ``(time, site)`` float32 (8760 x 5156); the first read of
an ``(estimate, variable)`` loads it whole and keeps it (LRU), so maps, series
and metrics after that are numpy slices. Metrics are reference first:
bias = estimate - reference, R² normalised by the reference.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr

HERE = Path(__file__).resolve().parent
CASE = HERE.parent
STORE = HERE / "store"

OFFSHORE = ("WHACS", "WHACS_cal")
VARIABLES = ("hs", "tp", "dir", "spr")
VARIABLE_LABELS = {"hs": "Hs (m)", "tp": "Tp (s)", "dir": "Dir (deg, from)", "spr": "Spreading (deg)"}
#: Stretches of coast, south to north, by the upper latitude of each.
COAST_REGIONS = {"Belgium-Zeeland": 51.95, "Holland": 52.95, "Wadden": 90.0}
GROUPS = {"all sites": None, "5 m": "5m", "10 m": "10m", "20 m": "20m", "buoys": "buoy", "DeenHaag": "DeenHaag"}

_KINDS = {
    "offshore": ("offshore (closest goal)", "#8c8c8c", "dot"),
    "mm_bulk": ("metamodel bulk", "#1f77b4", "dash"),
    "mm_spec": ("metamodel spectral", "#2ca02c", "solid"),
    "dyn_off": ("SnapWave no wind", "#ff7f0e", "dash"),
    "dyn_on": ("SnapWave wind", "#d62728", "solid"),
    "swan_off": ("SWAN no wind", "#c497d6", "dash"),
    "swan_on": ("SWAN wind", "#7b3294", "solid"),
}
#: Dynamic runs of notebook 08 (every hour, all goals): estimate kind -> (model, wind).
DYNAMIC_KINDS = {"dyn_off": ("snapwave", False), "dyn_on": ("snapwave", True),
                 "swan_off": ("swan", False), "swan_on": ("swan", True)}
#: Offshore sources with SWAN dynamic runs (SnapWave has every source).
SWAN_OFFSHORE = ("WHACS_cal",)
#: Every estimate of the cube: key -> label, colour, line dash, kind, offshore source.
ESTIMATES: dict[str, dict[str, str]] = {"buoy": {"label": "Buoy", "color": "#111111", "dash": "solid",
                                                 "kind": "buoy", "offshore": ""}}
for _off in OFFSHORE:
    for _kind, (_label, _color, _dash) in _KINDS.items():
        if _kind.startswith("swan") and _off not in SWAN_OFFSHORE:
            continue
        ESTIMATES[f"{_kind}_{_off}"] = {"label": f"{_off} · {_label}", "color": _color, "dash": _dash,
                                       "kind": _kind, "offshore": _off}
#: The model estimates (everything but the buoys).
MODELS = [k for k in ESTIMATES if k != "buoy"]
#: The dynamic runs (the estimates with grid maps).
DYNAMIC = [k for k in ESTIMATES if ESTIMATES[k]["kind"] in DYNAMIC_KINDS]
#: Error-budget parts at the buoys: (name, minuend, subtrahend) as estimate kinds.
BUDGET = (
    ("partitions (spectral − bulk)", "mm_spec", "mm_bulk"),
    ("method (bulk − dynamic no wind)", "mm_bulk", "dyn_off"),
    ("missing wind (no wind − wind)", "dyn_off", "dyn_on"),
    ("model + forcing (dynamic wind − buoy)", "dyn_on", "buoy"),
)
BUDGET_COLORS = ("#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd")

#: Storm-hours test (Storms page only): SnapWave with no boundary waves, ERA5 wind (``wind_only``), and
#: estimates without wind plus it, summed in energy (Hs = sqrt(Hs² + Hs_wind-only²)) or damped by the share
#: of the sea already there (:func:`damped_wind_only`); Hs only.
WIND_ONLY: dict[str, dict[str, str]] = {"wind_only": {"label": "SnapWave wind only (no waves)", "color": "#e377c2",
                                                      "dash": "dot", "kind": "wind_only", "offshore": "", "base": ""}}
_PLUS_WIND_ONLY = {"mm_bulk": ("#17becf", "#08519c"), "mm_spec": ("#8c564b", "#006d2c"),
                   "dyn_off": ("#bcbd22", "#a63603")}
#: Suffix of the estimates plus wind only -> (label, dash).
_WIND_ONLY_SUFFIX = {"+wind_only": ("⊕ wind only", "dashdot"), "+damped": ("⊕ damped wind only", "solid")}
for _off in OFFSHORE:
    for _kind, _colors in _PLUS_WIND_ONLY.items():
        for (_suffix, (_text, _dash)), _color in zip(_WIND_ONLY_SUFFIX.items(), _colors):
            WIND_ONLY[f"{_kind}_{_off}{_suffix}"] = {"label": f"{_off} · {_KINDS[_kind][0]} {_text}", "color": _color,
                                                     "dash": _dash, "kind": f"{_kind}{_suffix}", "offshore": _off,
                                                     "base": f"{_kind}_{_off}"}


def wind_only_keys(offshore: str) -> list[str]:
    """Estimates of the wind-only test for one source: without wind, plus wind only (plain, damped), with wind, wind only."""
    return [*(f"{k}_{offshore}" for k in _PLUS_WIND_ONLY),
            *(f"{k}_{offshore}{x}" for x in _WIND_ONLY_SUFFIX for k in _PLUS_WIND_ONLY),
            f"dyn_on_{offshore}", "wind_only"]


def meta(key: str) -> dict[str, str]:
    """Label, colour, dash, kind and source of an estimate of :data:`ESTIMATES` or :data:`WIND_ONLY`."""
    return ESTIMATES[key] if key in ESTIMATES else WIND_ONLY[key]


def label(key: str) -> str:
    return meta(key)["label"]


def estimate_options(include_buoy: bool = True) -> list[dict[str, str]]:
    return [{"label": v["label"], "value": k} for k, v in ESTIMATES.items() if include_buoy or k != "buoy"]


# ---------------------------------------------------------------------------
# Store access
# ---------------------------------------------------------------------------


def available() -> bool:
    return (STORE / "cube.zarr").exists() and (STORE / "conditions.parquet").exists()


@lru_cache(maxsize=1)
def _cube() -> zarr.Group:
    return zarr.open_group(STORE / "cube.zarr", mode="r")


@lru_cache(maxsize=1)
def times() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(_cube().attrs["time"])


@lru_cache(maxsize=1)
def sites() -> pd.DataFrame:
    """One row per cube column: ``site, source, label, depth, lon, lat, region, goal``; ``group`` for filters."""
    table = pd.read_parquet(STORE / "sites.parquet").set_index("site").loc[_cube().attrs["sites"]].reset_index()
    table["group"] = np.where(table["source"] == "buoy", "buoy", table["label"])
    table.loc[table["source"] == "mesh", "group"] = "DeenHaag"
    return table


def site_mask(group: str | None) -> np.ndarray:
    """Boolean mask of the cube columns of one :data:`GROUPS` value (None = all)."""
    table = sites()
    return np.ones(len(table), bool) if group is None else (table["group"] == group).to_numpy()


@lru_cache(maxsize=1)
def buoy_sites() -> np.ndarray:
    """Columns of the buoys with observations in 2013."""
    obs = field("buoy", "hs")
    return np.flatnonzero((sites()["group"] == "buoy").to_numpy() & np.isfinite(obs).any(0))


_lock = threading.Lock()


@lru_cache(maxsize=64)
def _load(key: str) -> np.ndarray:
    return _cube()[key][:]


def wind_only_available() -> bool:
    return "wind_only" in _cube()


#: Bins of the share of the sea already there, E_off / (E_off + E_wind-only), for the damping curve.
SHARE_BINS = np.linspace(0, 1, 21)


@lru_cache(maxsize=1)
def storm_events() -> np.ndarray:
    """Event number (0..) of every hour of 2013, -1 outside the storms."""
    event = np.full(len(times()), -1)
    for i, s in enumerate(storms().itertuples()):
        event[time_index(s.start):time_index(s.end) + 1] = i
    return event


def wind_only_pairs(offshore: str, columns: np.ndarray) -> dict[str, np.ndarray]:
    """
    Storm hours with a wind-only run, at *columns*: energies without wind (``e_off``), with wind
    (``e_on``), wind only (``e_wo``), the share of the sea already there and the event of each hour.
    """
    hours = np.flatnonzero(storm_events() >= 0)
    e_off = field(f"dyn_off_{offshore}", "hs")[hours][:, columns].astype(float) ** 2
    e_on = field(f"dyn_on_{offshore}", "hs")[hours][:, columns].astype(float) ** 2
    e_wo = field("wind_only", "hs")[hours][:, columns].astype(float) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        share = e_off / (e_off + e_wo)
    return {"hours": hours, "event": storm_events()[hours], "e_off": e_off, "e_on": e_on, "e_wo": e_wo, "share": share}


def damping_curve(share: np.ndarray, ratio: np.ndarray, min_pairs: int = 50) -> np.ndarray:
    """Median of *ratio* (added energy / wind-only energy) in each :data:`SHARE_BINS` bin (NaN below *min_pairs*)."""
    ok = np.isfinite(share) & np.isfinite(ratio)
    b = np.clip(np.digitize(share[ok], SHARE_BINS) - 1, 0, len(SHARE_BINS) - 2)
    r = ratio[ok]
    return np.array([np.median(r[b == i]) if (b == i).sum() >= min_pairs else np.nan
                     for i in range(len(SHARE_BINS) - 1)])


def damped_wind_only(pairs: dict[str, np.ndarray], e_base: np.ndarray | None = None) -> np.ndarray:
    """
    Hs without wind plus the wind-only energy times the damping of its share, the curve fitted on the
    SnapWave pairs of the other events (leave one event out): ``sqrt(E + f(E / (E + E_wo)) E_wo)``.

    *e_base* is the energy without wind E (e.g. the metamodel's, as in a reconstruction); default the
    SnapWave run without wind of *pairs*.
    """
    e_base = pairs["e_off"] if e_base is None else e_base
    with np.errstate(invalid="ignore", divide="ignore"):
        share_base = e_base / (e_base + pairs["e_wo"])
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (pairs["e_on"] - pairs["e_off"]) / pairs["e_wo"]
    ratio = np.where(pairs["e_wo"] > 1e-4, ratio, np.nan)
    centres = (SHARE_BINS[:-1] + SHARE_BINS[1:]) / 2
    out = np.full(pairs["e_off"].shape, np.nan)
    for ev in np.unique(pairs["event"]):
        test = pairs["event"] == ev
        curve = damping_curve(pairs["share"][~test].ravel(), ratio[~test].ravel())
        ok = np.isfinite(curve)
        if not ok.any():
            continue
        s = share_base[test]
        f = np.interp(np.where(np.isfinite(s), s, 0), centres[ok], curve[ok])
        with np.errstate(invalid="ignore"):
            out[test] = np.sqrt(np.clip(e_base[test] + f * pairs["e_wo"][test], 0, None))
    return out


@lru_cache(maxsize=8)
def _damped(base: str) -> np.ndarray:
    """``(time, site)`` Hs of *base* plus the damped wind only (storm hours; curve per site group)."""
    hs = field(base, "hs")
    out = np.full(hs.shape, np.nan, dtype="float32")
    groups = sites()["group"].to_numpy()
    for group in np.unique(groups):
        cols = np.flatnonzero(groups == group)
        pairs = wind_only_pairs(ESTIMATES[base]["offshore"], cols)
        e_base = hs[pairs["hours"]][:, cols].astype(float) ** 2
        out[np.ix_(pairs["hours"], cols)] = damped_wind_only(pairs, e_base)
    return out


@lru_cache(maxsize=8)
def _plus_wind_only(base: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sqrt(field(base, "hs") ** 2 + field("wind_only", "hs") ** 2)


def field(estimate: str, variable: str) -> np.ndarray:
    """``(time, site)`` array of one estimate (``wind`` for ``u10`` / ``u10dir``; ``...+wind_only|+damped``: Hs only)."""
    if estimate.endswith(tuple(_WIND_ONLY_SUFFIX)):
        if variable != "hs":
            raise ValueError("estimates plus wind only have Hs only")
        base = WIND_ONLY[estimate]["base"]
        return _plus_wind_only(base) if estimate.endswith("+wind_only") else _damped(base)
    with _lock:
        return _load(f"{estimate}/{variable}")


def hour(estimate: str, variable: str, index: int) -> np.ndarray:
    return field(estimate, variable)[index]


def preload(variables: tuple[str, ...] = ("hs",)) -> None:
    """Load the most used arrays in a background thread (the first page loads then answer at once)."""
    def run() -> None:
        for var in variables:
            for key in ESTIMATES:
                field(key, var)
        field("wind", "u10"), field("wind", "u10dir")
        for key in ESTIMATES:
            if key != "buoy":
                field(key, "dir")
    threading.Thread(target=run, daemon=True).start()


@lru_cache(maxsize=1)
def conditions() -> pd.DataFrame:
    return pd.read_parquet(STORE / "conditions.parquet")


def conditions_of(offshore: str) -> pd.DataFrame:
    """Conditions of one source, indexed by hour number (0..8759), with the ``time`` column."""
    c = conditions()
    c = c[c["offshore"] == offshore].reset_index(drop=True)
    return c


@lru_cache(maxsize=1)
def storms() -> pd.DataFrame:
    return pd.read_csv(STORE / "storms.csv", parse_dates=["peak", "start", "end"])


@lru_cache(maxsize=1)
def goals() -> pd.DataFrame:
    return pd.read_parquet(STORE / "goals.parquet")


@lru_cache(maxsize=1)
def offshore() -> xr.Dataset:
    return xr.open_zarr(STORE / "offshore.zarr").load()


@lru_cache(maxsize=1)
def manifest() -> dict:
    path = STORE / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def time_index(value: str | pd.Timestamp | None) -> int:
    """Hour number of a time (nearest), 0 if None or outside 2013."""
    if value is None or value == "":
        return 0
    t = pd.Timestamp(value)
    return int(np.clip(times().get_indexer([t], method="nearest")[0], 0, len(times()) - 1))


# ---------------------------------------------------------------------------
# Grid maps of the dynamic runs (maps.zarr)
# ---------------------------------------------------------------------------


def maps_available() -> bool:
    return (STORE / "maps.zarr").exists()


@lru_cache(maxsize=1)
def _maps() -> zarr.Group:
    return zarr.open_group(STORE / "maps.zarr", mode="r")


@lru_cache(maxsize=1)
def map_times() -> dict[str, pd.DatetimeIndex]:
    """Hours with a map of each dynamic run (the hours where at least two runs have one)."""
    runs = _maps()["runs"]
    return {key: pd.DatetimeIndex(runs[key].attrs["time"]) for key in DYNAMIC if key in runs}


@lru_cache(maxsize=256)
def _map_row(key: str, variable: str, row: int) -> np.ndarray:
    return _maps()[f"runs/{key}/{variable}"][row].astype("float32")


def map_field(key: str, variable: str, time: pd.Timestamp) -> np.ndarray:
    """Values on the mesh nodes of one run at one hour (the hour must have a map)."""
    return _map_row(key, variable, int(map_times()[key].get_loc(time)))


@lru_cache(maxsize=1)
def raster() -> dict:
    """
    Pixels of the map image inside the mesh: flat ``pixel`` index, the 3 ``nodes`` of its triangle and their
    barycentric ``weights``; ``shape`` (rows, cols, north up) and ``extent`` (lon0, lon1, lat0, lat1).
    """
    group = _maps()["raster"]
    return {"pixel": group["pixel"][:], "nodes": group["nodes"][:], "weights": group["weights"][:],
            "shape": tuple(group.attrs["shape"]), "extent": tuple(group.attrs["extent"])}


@lru_cache(maxsize=1)
def site_nodes() -> np.ndarray:
    """Nearest mesh node of every cube site (column order)."""
    return _maps()["site_node"][:]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def difference(estimate: np.ndarray, reference: np.ndarray, variable: str) -> np.ndarray:
    """``estimate - reference``, wrapped to +-180 for directions."""
    d = estimate - reference
    return (d + 180) % 360 - 180 if variable == "dir" else d


def site_metrics(reference: str, estimate: str, variable: str, hours: np.ndarray | None = None,
                 columns: np.ndarray | None = None, quantile: float | None = None) -> pd.DataFrame:
    """
    Metrics per site of *estimate* against *reference* over *hours* (bool or index, default all).

    Columns ``n, bias, rmse, r2`` (r2 NaN for directions), ``ref_mean``, ``est_mean`` and,
    with *quantile*, ``qbias`` = q(estimate) - q(reference) on the common pairs.
    """
    r, e = field(reference, variable), field(estimate, variable)
    if hours is not None:
        r, e = r[hours], e[hours]
    if columns is not None:
        r, e = r[:, columns], e[:, columns]
    return array_metrics(r, e, variable, quantile)


def array_metrics(r: np.ndarray, e: np.ndarray, variable: str, quantile: float | None = None) -> pd.DataFrame:
    """:func:`site_metrics` on two ``(time, site)`` arrays."""
    ok = np.isfinite(r) & np.isfinite(e)
    n = ok.sum(0)
    d = np.where(ok, difference(e, r, variable), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        bias = np.nanmean(d, 0)
        rmse = np.sqrt(np.nanmean(d**2, 0))
        rr = np.where(ok, r, np.nan)
        ss = np.nansum((rr - np.nanmean(rr, 0)) ** 2, 0)
        r2 = 1 - np.nansum(d**2, 0) / ss if variable != "dir" else np.full(len(n), np.nan)
        out = pd.DataFrame({"n": n, "bias": bias, "rmse": rmse, "r2": r2,
                            "ref_mean": np.nanmean(rr, 0), "est_mean": np.nanmean(np.where(ok, e, np.nan), 0)})
        if quantile is not None:
            out["qbias"] = (np.nanquantile(np.where(ok, e, np.nan), quantile, 0)
                            - np.nanquantile(rr, quantile, 0))
    out.loc[out["n"] == 0, ["bias", "rmse", "r2", "ref_mean", "est_mean"]] = np.nan
    return out


def pooled_metrics(r: np.ndarray, e: np.ndarray, variable: str) -> dict[str, float]:
    """Metrics over all valid pairs of two arrays."""
    ok = np.isfinite(r) & np.isfinite(e)
    r, e = r[ok], e[ok]
    if r.size == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "r2": np.nan}
    d = difference(e, r, variable)
    r2 = np.nan if variable == "dir" else 1 - np.sum(d**2) / np.sum((r - r.mean()) ** 2)
    return {"n": int(r.size), "bias": float(d.mean()), "rmse": float(np.sqrt((d**2).mean())), "r2": float(r2)}


def sector(direction: np.ndarray, width: float = 45.0) -> np.ndarray:
    """Centre of the *width* sector of each direction (deg), -1 for NaN."""
    out = (np.round(direction / width) % (360 / width)) * width
    return np.where(np.isfinite(direction), out, -1).astype(int)


def budget(offshore: str, hours: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Error budget of the spectral metamodel at the buoys (Hs, m), pairs valid in every estimate.

    Returns the mean of each :data:`BUDGET` part and the total per buoy, and the same pooled
    over each buoy's own Hs quantile classes (``all``, ``p0-50`` ... ``p99-100``).
    """
    cols = buoy_sites()
    keys = {"buoy": "buoy"} | {k: f"{k}_{offshore}" for k in ("mm_spec", "mm_bulk", "dyn_off", "dyn_on")}
    hs = {k: field(v, "hs")[:, cols] for k, v in keys.items()}
    if hours is not None:
        hs = {k: v[hours] for k, v in hs.items()}
    ok = np.logical_and.reduce([np.isfinite(v) for v in hs.values()])
    hs = {k: np.where(ok, v, np.nan) for k, v in hs.items()}
    parts = {name: hs[a] - hs[b] for name, a, b in BUDGET}
    parts["total (spectral − buoy)"] = hs["mm_spec"] - hs["buoy"]
    per_buoy = pd.DataFrame({k: np.nanmean(v, 0) for k, v in parts.items()}, index=sites()["site"].to_numpy()[cols])
    per_buoy["pairs"] = ok.sum(0)
    rank = pd.DataFrame(hs["buoy"]).rank(pct=True).to_numpy()
    classes = {"all": ok, "p0-50": rank <= 0.5, "p50-90": (rank > 0.5) & (rank <= 0.9),
               "p90-99": (rank > 0.9) & (rank <= 0.99), "p99-100": rank > 0.99}
    by_class = pd.DataFrame({c: {k: float(np.nanmean(np.where(m & ok, v, np.nan))) for k, v in parts.items()}
                             | {"pairs": int((m & ok).sum())} for c, m in classes.items()})
    return per_buoy, by_class


def hours_mask(offshore: str, *, months: list[int] | None = None, u10: tuple[float, float] | None = None,
               u10dir: list[int] | None = None, hs_off: tuple[float, float] | None = None,
               dir_off: list[int] | None = None, selected: list[int] | None = None,
               wind_converged: bool = False) -> np.ndarray:
    """Hours (bool, 8760) of *offshore* matching every given condition (sectors of 45 deg)."""
    c = conditions_of(offshore)
    m = np.ones(len(c), bool)
    if months:
        m &= c["time"].dt.month.isin(months).to_numpy()
    if u10:
        m &= c["u10"].between(*u10).to_numpy()
    if u10dir is not None and len(u10dir) < 8:
        m &= np.isin(sector(c["u10dir"].to_numpy()), u10dir)
    if hs_off:
        m &= c["hs_off"].between(*hs_off).to_numpy()
    if dir_off is not None and len(dir_off) < 8:
        m &= np.isin(sector(c["dir_off"].to_numpy()), dir_off)
    if selected is not None:
        keep = np.zeros(len(c), bool)
        keep[np.asarray(selected, int)] = True
        m &= keep
    if wind_converged:
        m &= c["wind_converged"].to_numpy()
    return m


# ---------------------------------------------------------------------------
# Metamodel, calibration, numerical tests
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def cases() -> pd.DataFrame:
    return pd.read_parquet(STORE / "metamodel" / "cases.parquet")


@lru_cache(maxsize=1)
def holdout() -> pd.DataFrame:
    return pd.read_parquet(STORE / "metamodel" / "holdout.parquet")


@lru_cache(maxsize=24)
def case_fields(goal: int, variable: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Training-case field ``(case, site)`` of one goal, its sites and the goal's removed sites."""
    group = zarr.open_group(STORE / "metamodel" / "case_fields.zarr", mode="r")[str(goal)]
    return group[variable][:], list(group.attrs["sites"]), list(group.attrs["removed_sites"])


def linear_sum_metrics() -> pd.DataFrame:
    paths = sorted((STORE / "metamodel").glob("linear_summation_*.csv"))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def calibration_table(hindcast: str, name: str) -> pd.DataFrame:
    path = STORE / "calibration" / f"{hindcast}_{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def calibration_quantiles() -> pd.DataFrame:
    return pd.read_parquet(STORE / "calibration" / "quantiles.parquet")


def numerical_tests_dir() -> Path:
    return STORE / "numerical_tests"
