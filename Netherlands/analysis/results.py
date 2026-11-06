"""
First-iteration analysis of the Netherlands case (SnapWave, 2013): every estimate on the same sites and hours.

Case-only code, imported by the notebooks of this folder (they put the case
folder on ``sys.path`` for ``utils``). Estimates of one offshore source:

- ``offshore``: the hindcast at each buoy's closest goal;
- ``metamodel bulk`` / ``metamodel spectral``: the linear sum, read from the
  reconstruction cache (MDA :data:`MDA_N`, CODEC water level);
- ``dynamic no wind`` / ``dynamic wind``: the hourly runs of notebook 08.

The dynamic runs force every goal with its bulk sea state, so the bulk metamodel
is their like-for-like counterpart (the method error); the spectral one is the
product. Metrics are reference first (bias = estimate - reference).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from utils import metamodel as mm
from utils import onshore_sites as obs
from utils import simulation as sim
from utils import validation as val
from utils.geometry import attach_site_meta

PERIOD = slice("2013-01-01", "2013-12-31 23:00")
YEAR = "2013"
MDA_N = 600
OFFSHORE = ("WHACS", "WHACS_cal")
VARIABLES = ("hs", "tp", "dir", "spr")
#: Order of the estimates in tables and plots.
ESTIMATES = ("offshore", "metamodel bulk", "metamodel spectral", "dynamic no wind", "dynamic wind")
#: Error-budget parts: (name, minuend, subtrahend); they add up to metamodel spectral - buoy.
BUDGET = (
    ("partitions (spectral - bulk)", "metamodel spectral", "metamodel bulk"),
    ("method (bulk - dynamic no wind)", "metamodel bulk", "dynamic no wind"),
    ("missing wind (no wind - wind)", "dynamic no wind", "dynamic wind"),
    ("model + forcing (dynamic wind - buoy)", "dynamic wind", "buoy"),
)
#: Stretches of coast, south to north, by the upper latitude of each.
COAST_REGIONS = {"Belgium-Zeeland": 51.95, "Holland": 52.95, "Wadden": 90.0}
#: Scales of the analog distance: wind speed (m/s), wind direction (deg), Hs (m), wave direction (deg).
ANALOG_SCALES = {"u10": 2.0, "u10dir": 22.5, "hs20": 0.4, "dir20": 30.0}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def dynamic(offshore: str, wind: bool, sites: Iterable[str] | None = None) -> xr.Dataset | None:
    """
    Dynamic run of *offshore* (wind on or off) on ``(time, sites)``, or None if not postprocessed.

    Hours with Hs outside (0, 15) m (not converged or corrupted) are NaN in every variable.
    """
    path = sim.cases_dir("dynamic", offshore=offshore, wind=wind) / f"Dynamic_Cases_{YEAR}.nc"
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        ds = ds[list(VARIABLES)]
        ds = (ds.sel(sites=list(sites)) if sites is not None else ds).load()
    return ds.where((ds["hs"] > 0) & (ds["hs"] < 15))


def metamodel(offshore: str, metamodel_set: str, spectral: bool, sites: Iterable[str] | None = None) -> xr.Dataset:
    """
    Linear sum of *offshore* from the reconstruction cache (raises if a site-month is missing).

    Parameters
    ----------
    offshore
        ``offshore_waves`` entry.
    metamodel_set
        ``buoys``, ``isobaths`` or ``deenhaag``.
    spectral
        Partitions summed (True) or the bulk sea state.
    sites
        Default every site of the set.
    """
    store = mm.reconstruction_store(MDA_N, metamodel=metamodel_set, spectral=spectral,
                                    offshore_name=offshore, wl="forcing")
    site_ids = mm.reconstruction_sites(metamodel_set, None if sites is None else list(sites))
    return attach_site_meta(store.read(PERIOD, site_ids))


def buoys(variables: Iterable[str] = VARIABLES) -> xr.Dataset:
    """Buoy records of :data:`PERIOD` (the ``ok`` buoys with Hs)."""
    return val.load_buoy_observations(PERIOD, obs.list_snapwave_buoy_sites(), variables=list(variables))


def estimates_at_buoys(offshore: str, buoy: xr.Dataset) -> dict[str, xr.Dataset]:
    """Every estimate of *offshore* at the buoys of *buoy*, on its time axis (missing runs left out)."""
    sites = list(buoy.sites.values)
    off, _ = val.load_offshore_hindcast_at_buoys(PERIOD, sites, offshore_names=[offshore], variables=list(VARIABLES))
    out = {
        "offshore": next(iter(off.values())),
        "metamodel bulk": metamodel(offshore, "buoys", False, sites),
        "metamodel spectral": metamodel(offshore, "buoys", True, sites),
        "dynamic no wind": dynamic(offshore, False, sites),
        "dynamic wind": dynamic(offshore, True, sites),
    }
    return {k: v.reindex(time=buoy.time, sites=buoy.sites) for k, v in out.items() if v is not None}


def wind_at_sites(sites: xr.Dataset) -> xr.Dataset:
    """ERA5 wind (``forcing.wind``) at the sites of *sites* and on its time axis: ``u10`` speed, ``u10dir`` (from)."""
    from bluemath_tk.waves.wind import wind_speed_direction

    from utils.config import open_wind

    wind = open_wind(PERIOD).interp(
        time=sites["time"],
        latitude=xr.DataArray(sites["site_latitude"].values, dims="sites"),
        longitude=xr.DataArray(sites["site_longitude"].values, dims="sites"),
    ).load()
    speed, direction = wind_speed_direction(wind["u10"].values, wind["v10"].values)
    coords = {"time": sites["time"].values, "sites": sites["sites"].values}
    return xr.Dataset({"u10": (("time", "sites"), speed), "u10dir": (("time", "sites"), direction)}, coords=coords)


def hs_rank(field: xr.Dataset) -> xr.DataArray:
    """Per-site rank of Hs over time, in (0, 1] (NaN where Hs is NaN)."""
    ranks = field["hs"].transpose("time", "sites").to_pandas().rank(pct=True)
    return xr.DataArray(ranks, dims=("time", "sites"), coords={"time": field["time"], "sites": field["sites"]})


def by_class(diff: xr.DataArray, classes: xr.DataArray, order: Iterable[str] | None = None) -> pd.DataFrame:
    """Mean, RMSE, p10 / p90 and count of *diff* in each label of *classes* (same shape)."""
    df = pd.DataFrame({"d": diff.values.ravel(), "c": classes.values.ravel()}).dropna()
    out = df.groupby("c")["d"].agg(mean="mean", rmse=lambda d: float(np.sqrt((d**2).mean())),
                                   p10=lambda d: d.quantile(0.1), p90=lambda d: d.quantile(0.9), n="size")
    return out.reindex(list(order)) if order is not None else out


def rank_classes(rank: xr.DataArray, edges: Iterable[float] = (0, 0.5, 0.9, 0.99, 1.0)) -> xr.DataArray:
    """Labels ``p0-50``, ``p50-90``, ... of a :func:`hs_rank` field."""
    edges = list(edges)
    labels = [f"p{100 * a:g}-{100 * b:g}" for a, b in zip(edges[:-1], edges[1:])]
    cut = pd.cut(rank.values.ravel(), edges, labels=labels, include_lowest=True)
    return xr.DataArray(np.asarray(cut, dtype=object).reshape(rank.shape), dims=rank.dims, coords=rank.coords)


def common_valid(fields: Iterable[xr.Dataset], variable: str = "hs") -> xr.DataArray:
    """Mask of the ``(time, sites)`` where *variable* is finite in every field."""
    fields = list(fields)
    mask = fields[0][variable].notnull()
    for field in fields[1:]:
        mask &= field[variable].notnull()
    return mask


def coast_region(sites: xr.Dataset | xr.DataArray) -> xr.DataArray:
    """:data:`COAST_REGIONS` label of each site, from its latitude."""
    names = np.array(list(COAST_REGIONS))
    labels = names[np.searchsorted(list(COAST_REGIONS.values()), sites["site_latitude"].values)]
    return xr.DataArray(labels, dims="sites", coords={"sites": sites["sites"].values})


def sector(direction: xr.DataArray, width: float = 45.0) -> xr.DataArray:
    """Centre (deg, int) of the *width* sector of each direction (NaN -> -1)."""
    centre = (np.round(direction / width) % (360 / width)) * width
    return centre.fillna(-1).astype(int)


def mean_by(values: xr.DataArray, *groups: xr.DataArray) -> pd.Series:
    """Mean of *values* grouped by the labels of *groups* (broadcast against it; NaN values dropped)."""
    arrays = xr.broadcast(values, *groups)
    df = pd.DataFrame({f"g{i}": a.values.ravel() for i, a in enumerate(arrays)}).dropna(subset=["g0"])
    return df.groupby([f"g{i}" for i in range(1, len(arrays))])["g0"].mean()


def hour_conditions(no_wind: xr.Dataset, wind: xr.Dataset, era5: xr.Dataset, labels: xr.DataArray) -> pd.DataFrame:
    """
    One row per hour where both runs have results: the ERA5 wind over the sites and the sea at 20 m without wind.

    Columns ``u10`` (mean speed), ``u10dir`` (direction of the mean wind vector, from),
    ``hs20`` (mean Hs of the 20 m sites) and ``dir20`` (their energy-weighted mean direction).
    """
    rad = np.deg2rad(era5["u10dir"])
    u, v = (era5["u10"] * np.sin(rad)).mean("sites"), (era5["u10"] * np.cos(rad)).mean("sites")
    deep = no_wind.isel(sites=(labels == "20m").values)
    energy = deep["hs"] ** 2
    wrad = np.deg2rad(deep["dir"])
    wu, wv = (energy * np.sin(wrad)).sum("sites"), (energy * np.cos(wrad)).sum("sites")
    out = pd.DataFrame({
        "u10": era5["u10"].mean("sites").to_series(),
        "u10dir": (np.rad2deg(np.arctan2(u, v)) % 360).to_series(),
        "hs20": deep["hs"].mean("sites").to_series(),
        "dir20": (np.rad2deg(np.arctan2(wu, wv)) % 360).to_series(),
    })
    valid = (no_wind["hs"].notnull().any("sites") & wind["hs"].notnull().any("sites")).to_series()
    return out[valid].dropna()


def analog_hours(conditions: pd.DataFrame, k: int = 24, **target: float) -> pd.DataFrame:
    """
    The *k* hours of *conditions* closest to *target* (any of ``u10``, ``u10dir``, ``hs20``, ``dir20``).

    Distance: sum of squared differences scaled by :data:`ANALOG_SCALES` (directions
    wrapped to +-180 deg). Returns their rows with a ``distance`` column, closest first.
    """
    d2 = pd.Series(0.0, index=conditions.index)
    for name, value in target.items():
        diff = conditions[name] - value
        if "dir" in name:
            diff = (diff + 180) % 360 - 180
        d2 += (diff / ANALOG_SCALES[name]) ** 2
    return conditions.assign(distance=np.sqrt(d2)).nsmallest(k, "distance")


def plot_analogs(hours: pd.DataFrame, no_wind: xr.DataArray, wind: xr.DataArray,
                 buoy_error: Mapping[str, xr.DataArray] | None = None, title: str = "") -> Figure:
    """
    Mean over the analog *hours* of Hs without and with wind, the added Hs, and the buoy errors.

    *no_wind* / *wind*: Hs on ``(time, sites)`` with the site coordinates.
    *buoy_error*: ``{"no wind": run - buoy, "wind": run - buoy}`` at the buoys
    (the fourth panel shows the run with wind, the title both medians).
    """
    times = hours.index
    h0, h1 = no_wind.sel(time=times), wind.sel(time=times)
    ok = h0.notnull() & h1.notnull()
    h0, h1 = h0.where(ok).mean("time"), h1.where(ok).mean("time")
    vmax = float(np.nanquantile(h1, 0.99))
    fig, axes = plt.subplots(1, 4 if buoy_error else 3, figsize=(26 if buoy_error else 20, 5.2))
    plot_site_map(h0, "Hs without wind", "m", cmap="viridis", vmin=0, vmax=vmax, ax=axes[0])
    plot_site_map(h1, "Hs with wind", "m", cmap="viridis", vmin=0, vmax=vmax, ax=axes[1])
    plot_site_map(h1 - h0, "added Hs (wind - no wind)", "m", ax=axes[2])
    for ax, h in ((axes[0], h0), (axes[1], h1)):
        _wind_arrow(ax, hours, h)
    if buoy_error:
        err = {k: v.sel(time=times).mean("time") for k, v in buoy_error.items()}
        e1 = err["wind"].dropna("sites")
        lim = max(0.1, float(np.nanmax(np.abs(e1))))
        sc = axes[3].scatter(e1["site_longitude"], e1["site_latitude"], c=e1, cmap="RdBu_r", vmin=-lim, vmax=lim,
                             s=70, edgecolor="k")
        fig.colorbar(sc, ax=axes[3], label="m", shrink=0.85)
        axes[3].set_xlim(axes[2].get_xlim()); axes[3].set_ylim(axes[2].get_ylim())
        axes[3].set_aspect(axes[2].get_aspect())
        axes[3].set_title("buoys: run with wind - buoy\nmedian " + ", ".join(
            f"{k} {float(v.median()):+.2f}" for k, v in err.items()) + " m", fontsize=10)
    c = hours.mean()
    fig.suptitle(f"{title}{len(hours)} hours: U10 {c['u10']:.1f} m/s (sd {hours['u10'].std():.1f}) from "
                 f"{_circmean(hours['u10dir']):.0f} deg, Hs at 20 m without wind {c['hs20']:.2f} m from "
                 f"{_circmean(hours['dir20']):.0f} deg; months {sorted(set(hours.index.month))}", fontsize=11)
    fig.tight_layout()
    return fig


def _circmean(deg: pd.Series) -> float:
    rad = np.deg2rad(deg)
    return float(np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def _wind_arrow(ax: plt.Axes, hours: pd.DataFrame, field: xr.DataArray) -> None:
    """Arrow of the mean wind (towards), in the upper-left corner."""
    to = np.deg2rad(_circmean(hours["u10dir"]) + 180)
    ax.annotate("", xy=(0.12 + 0.07 * np.sin(to), 0.85 + 0.07 * np.cos(to)), xytext=(0.12, 0.85),
                xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 2.5, "color": "crimson"})
    ax.text(0.12, 0.74, "wind", transform=ax.transAxes, ha="center", color="crimson", fontsize=9)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def site_metrics(reference: xr.Dataset, estimate: xr.Dataset, variable: str = "hs", min_n: int = 100) -> pd.DataFrame:
    """Metrics of *estimate* against *reference* per site (pairs valid in both, at least *min_n*)."""
    rows = {}
    for site in reference.sites.values:
        r = reference[variable].sel(sites=site).values
        e = estimate[variable].sel(sites=site).values
        ok = np.isfinite(r) & np.isfinite(e)
        if ok.sum() >= min_n:
            rows[site] = val.compute_metrics(r[ok], e[ok], variable)
    return pd.DataFrame.from_dict(rows, orient="index")


def median_metrics(reference: xr.Dataset, estimates: Mapping[str, xr.Dataset], variable: str = "hs") -> pd.DataFrame:
    """Median over sites of :func:`site_metrics`, one row per estimate."""
    return pd.DataFrame({name: site_metrics(reference, est, variable).median() for name, est in estimates.items()}).T


def quantile_bias(reference: xr.Dataset, estimates: Mapping[str, xr.Dataset], variable: str = "hs",
                  quantiles: Iterable[float] = (0.5, 0.9, 0.99, 0.999)) -> pd.DataFrame:
    """Median over sites of ``q(estimate) - q(reference)`` (the storm tail without regression to the mean)."""
    quantiles = list(quantiles)
    ref_q = reference[variable].quantile(quantiles, dim="time")
    return pd.DataFrame({
        name: (est[variable].quantile(quantiles, dim="time") - ref_q).median("sites").to_series()
        for name, est in estimates.items()
    }).T.rename(columns=lambda q: f"p{100 * q:g}")


def error_budget(buoy: xr.Dataset, estimates: Mapping[str, xr.Dataset], classes: Mapping[str, tuple[float, float]] | None = None) -> pd.DataFrame:
    """
    Mean Hs error of the spectral metamodel at the buoys, split into the :data:`BUDGET` parts.

    Only pairs valid in the buoy and every estimate count, so the parts add up to
    the total. Columns: all pairs, then classes of each buoy's own Hs quantiles
    (``{"p0-50": (0, 0.5), ...}``): the split between parts by class is fair, but a
    class picked by the observation regresses to the mean, so do not read storm
    skill from it (:func:`quantile_bias` does that).
    """
    fields = {"buoy": buoy, **{k: v for k, v in estimates.items() if k != "offshore"}}
    mask = common_valid(fields.values())
    hs = {k: v["hs"].where(mask) for k, v in fields.items()}
    classes = classes or {"p0-50": (0, 0.5), "p50-90": (0.5, 0.9), "p90-99": (0.9, 0.99), "p99-100": (0.99, 1.0)}
    rank = xr.DataArray(hs["buoy"].transpose("time", "sites").to_pandas().rank(pct=True), dims=("time", "sites"))
    columns = {"all": mask} | {name: mask & (rank > lo) & (rank <= hi) for name, (lo, hi) in classes.items()}
    rows = {}
    for name, a, b in BUDGET:
        if a in hs and b in hs:
            rows[name] = {c: float((hs[a] - hs[b]).where(m).mean()) for c, m in columns.items()}
    rows["total (spectral - buoy)"] = {c: float((hs["metamodel spectral"] - hs["buoy"]).where(m).mean()) for c, m in columns.items()}
    rows["pairs"] = {c: int(m.sum()) for c, m in columns.items()}
    return pd.DataFrame(rows).T


def group_metrics(reference: xr.Dataset, estimate: xr.Dataset, variable: str = "hs") -> pd.DataFrame:
    """Metrics pooled over the pairs of each ``site_label`` group (pairs valid in both)."""
    rows = {}
    labels = reference["site_label"].values
    for label in pd.unique(labels):
        sel = labels == label
        r = reference[variable].isel(sites=sel).values.ravel()
        e = estimate[variable].isel(sites=sel).values.ravel()
        ok = np.isfinite(r) & np.isfinite(e)
        rows[label] = {"sites": int(sel.sum()), **val.compute_metrics(r[ok], e[ok], variable)}
    return pd.DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_budget(budget: pd.DataFrame, title: str = "") -> Figure:
    """Stacked bars of the budget parts per column, with the total as a black marker."""
    parts = budget.drop(index=["pairs", "total (spectral - buoy)"])
    fig, ax = plt.subplots(figsize=(1.6 * budget.shape[1] + 3, 4.5))
    x = np.arange(budget.shape[1])
    pos, neg = np.zeros(len(x)), np.zeros(len(x))
    for name, row in parts.iterrows():
        v = row.to_numpy(float)
        base = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=base, width=0.6, label=name)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.plot(x, budget.loc["total (spectral - buoy)"].to_numpy(float), "kD", label="total (spectral - buoy)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{c}\n(n={int(n):,})" for c, n in budget.loc["pairs"].items()])
    ax.set(ylabel="mean Hs error (m)", title=title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_site_map(values: xr.DataArray, title: str, label: str, *, cmap: str = "RdBu_r",
                  vmin: float | None = None, vmax: float | None = None, ax: plt.Axes | None = None) -> Figure:
    """One value per site on a lon/lat scatter (symmetric colour limits for diverging maps)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5.5))
    if vmin is None and vmax is None and cmap in ("RdBu_r", "coolwarm", "PuOr"):
        lim = float(np.nanquantile(np.abs(values), 0.98))
        vmin, vmax = -lim, lim
    sc = ax.scatter(values["site_longitude"], values["site_latitude"], c=values, s=4, cmap=cmap,
                    vmin=vmin, vmax=vmax, rasterized=True)
    ax.figure.colorbar(sc, ax=ax, label=label, shrink=0.85)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(values["site_latitude"].mean()))))
    ax.set_title(title, fontsize=10)
    return ax.figure
