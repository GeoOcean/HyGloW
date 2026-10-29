"""Wind correction of a HyGloW case: fetch table, training pairs, K model.

The metamodel cases run without wind; the reconstruction adds the local wind
sea afterwards (:mod:`bluemath_tk.waves.hywaves.wind_correction`). This module
is the case layer, driven by ``metamodel.wind_correction`` in the case JSON:

- :func:`fetch_table` — fetch and mean depth along it, per output site and
  wind direction, over the model grid (``outputs/wind/fetch_table_{model}.nc``);
- :func:`generate_wind_cases` — the training design: full-domain sea states
  that are (partly) wind sea, each with a wind of ``speed_ratio`` times the
  fully developed speed, blowing ``wind_wave_angle`` off the wave direction;
- :func:`wind_case_wrappers` — the same cases with and without wind
  (dynamic templates, ``simulations.<model>.wind_cases``);
- :func:`training_pairs` / :func:`fit_wind_correction` — ``K = Hs(with wind) /
  Hs(without wind)`` per site and case, and the fitted model
  (``outputs/wind/wind_correction_{model}.pkl``);
- :func:`formula_k` / :func:`k_comparison` — the growth-curve ``K`` next to
  the fitted one and the runs', on held-out cases (notebook 05b);
- :func:`holdout_cases` / :func:`case_comparison` — held-out cases at every
  site: with and without wind, and corrected (notebook 05b);
- :func:`k_response` / :func:`training_range` — ``K`` of the fitted model
  against each input, and where the training cases lie (notebook 05b);
- :func:`reconstruction_inputs` — what :func:`utils.metamodel.reconstruct`
  passes to the spectral sum when the correction is on.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.io import load_model
from bluemath_tk.core.operations import in_nautical_sector
from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.waves.fetch import effective_fetch
from bluemath_tk.waves.fetch import fetch_table as mesh_fetch_table
from bluemath_tk.waves.hywaves.metamodel import fit_mda
from bluemath_tk.waves.hywaves.wind_correction import WindCorrection
from bluemath_tk.waves.steepness import wave_steepness
from bluemath_tk.waves.wind import (
    WIND_SEA_SPREAD,
    WIND_SEA_STEEPNESS,
    equivalent_fetch,
    fully_developed_wind,
    growth_hs,
    wind_sea_weight,
    wind_speed_direction,
)

from utils.config import (
    METAMODEL,
    SIMULATIONS,
    model_config,
    open_wind,
    resolve,
    simulation_model,
)

logger = logging.getLogger("hywaves.wind")

#: Defaults of ``metamodel.wind_correction`` (the case JSON overrides them).
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "wind": "inferred",
    "num_cases": 100,
    "speed_ratio": [0.8, 2.0],
    "wind_wave_angle": [-60.0, 60.0],
    "u10_range": [3.0, 35.0],
    "steepness_range": list(WIND_SEA_STEEPNESS),
    "spr_range": list(WIND_SEA_SPREAD),
    "speed_factor": 1.0,
    "added_spr": 30.0,
    "added_tp": "wind_sea",
    "fetch": {
        "directions_step": 5.0,
        "step_m": 250.0,
        "max_distance_m": 500e3,
        "half_width": 45.0,
        "open_tolerance_m": 5e3,
    },
}

#: Training-design inputs, sampled by LHS before the MDA selection.
DESIGN_VARS = ("hs", "tp", "dir", "spr", "wl", "speed_ratio", "wind_wave_angle")

#: Smallest wind-sea weight of a training case (``K`` is learned on the
#: ``w hs^2`` part of the energy, which gets noisy for small ``w``).
MIN_TRAINING_WEIGHT = 0.5

#: Reference time of the first training case; case ``i`` is ``i`` hours later.
CASES_T0 = pd.Timestamp("2000-01-01")


def config() -> dict[str, Any]:
    """``metamodel.wind_correction`` merged over :data:`DEFAULTS`."""
    user = dict(METAMODEL.get("wind_correction", {}))
    fetch = {**DEFAULTS["fetch"], **user.pop("fetch", {})}
    return {**DEFAULTS, **user, "fetch": fetch}


def enabled() -> bool:
    """Whether the reconstruction adds the wind correction by default."""
    return bool(config()["enabled"])


def _ramps() -> dict[str, tuple[float, float]]:
    cfg = config()
    return {
        "steepness_range": tuple(cfg["steepness_range"]),
        "spr_range": tuple(cfg["spr_range"]),
    }


def output_dir() -> Path:
    """``outputs/wind`` (created)."""
    path = resolve("outputs/wind")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_table_path(model: str | None = None) -> Path:
    """``outputs/wind/fetch_table_{model}.nc``."""
    return output_dir() / f"fetch_table_{simulation_model(model)}.nc"


def fetch_table(
    sites: pd.DataFrame | None = None,
    *,
    model: str | None = None,
    recompute: bool = False,
) -> xr.Dataset:
    """
    Fetch per output site and wind direction over the model grid (cached).

    Rays are walked upwind through the grid triangles (see
    :func:`bluemath_tk.waves.fetch.fetch_table`); the open boundary is SWAN's
    open-boundary node string, or the goals polyline for SnapWave.

    Parameters
    ----------
    sites
        Output sites (``source_id``, ``lon``, ``lat``); default the saved ``ok``
        sites (:func:`utils.geometry.output_sites`). Only used to compute.
    model
        Model whose grid is used; default ``simulations.model``.
    recompute
        Ignore the cached table.

    Returns
    -------
    xarray.Dataset
        On ``(sites, dir)``: ``fetch``, ``fetch_eff`` (Saville), ``mean_depth``
        and ``open_exit``; ``site_depth`` on ``sites``.

    Raises
    ------
    ValueError
        If the cached table is older than the output-sites table (site ids are
        positional, so rebuilt sites may keep their ids at other places).
    """
    from utils.geometry import output_sites, output_sites_path

    model = simulation_model(model)
    path = fetch_table_path(model)
    if path.is_file() and not recompute:
        sites_file = output_sites_path(model)
        if sites_file.is_file() and sites_file.stat().st_mtime > path.stat().st_mtime:
            raise ValueError(
                f"{path.name} is older than {sites_file.name}; "
                "rebuild it with fetch_table(recompute=True)."
            )
        with xr.open_dataset(path) as cached:
            return cached.load()

    if sites is None:
        sites = output_sites(model)
    cfg = config()["fetch"]
    from utils.geometry import model_grid

    grid = model_grid(model)
    open_boundary = grid.open_boundary
    if open_boundary is None:
        from utils.simulation import HyWavesWrapperSetup

        open_boundary = np.asarray(HyWavesWrapperSetup.from_case(model=model).boundary_nodes)
    table = mesh_fetch_table(
        sites[["lon", "lat"]].to_numpy(float),
        grid.x,
        grid.y,
        grid.triangles,
        grid.depth,
        directions=np.arange(0.0, 360.0, float(cfg["directions_step"])),
        step=float(cfg["step_m"]),
        max_distance=float(cfg["max_distance_m"]),
        open_boundary=open_boundary,
        open_tolerance=float(cfg["open_tolerance_m"]),
    )
    table["fetch_eff"] = effective_fetch(table, half_width=float(cfg["half_width"]))
    table = (
        table.rename(point="sites", point_depth="site_depth")
        .assign_coords(sites=sites["source_id"].astype(str).to_numpy())
        .reset_coords("site_depth")
    )
    table.attrs.update(model=model, grid=str(model_config(model)["grid_file"]))
    table.to_netcdf(path)
    outside = int(table["site_depth"].isnull().sum())
    logger.info("Fetch table (%s): %d sites, %d outside the grid -> %s", model, table.sizes["sites"], outside, path)
    return table


# ---------------------------------------------------------------------------
# Training cases
# ---------------------------------------------------------------------------


def generate_wind_cases(goals: dict[str, dict[str, Any]], seed: int = 0) -> pd.DataFrame:
    """
    Training design: full-domain wind seas with a wind, MDA-selected.

    A large LHS of ``hs``, ``tp``, ``spr``, ``wl`` (``simulations.bounds``),
    ``dir`` (within some goal's sector), the wind-speed ratio to the fully
    developed speed and the wind-wave angle is filtered to plausible
    (``max_steepness``) and at least partly wind seas (wind-sea weight > 0);
    MDA picks ``num_cases`` of them. Only cases whose wind-sea weight is at
    least :data:`MIN_TRAINING_WEIGHT` are kept. Each case forces every goal with the same
    sea state.

    Parameters
    ----------
    goals
        Goals mapping (sectors from ``angles``).
    seed
        Seed of the LHS pool.

    Returns
    -------
    pandas.DataFrame
        ``hs, tp, dir, spr, wl, speed_ratio, wind_wave_angle, ws_weight, u10,
        u10dir``, one row per case.
    """
    cfg = config()
    bounds = {
        **SIMULATIONS["bounds"],
        "dir": [0.0, 360.0],
        "speed_ratio": cfg["speed_ratio"],
        "wind_wave_angle": cfg["wind_wave_angle"],
    }
    names = list(DESIGN_VARS)
    pool = LHS(num_dimensions=len(names), seed=seed).generate(
        dimensions_names=names,
        lower_bounds=[bounds[v][0] for v in names],
        upper_bounds=[bounds[v][1] for v in names],
        num_samples=int(cfg["num_cases"]) * 50,
    )
    sectors = [g["angles"] for g in goals.values()]
    seen = np.zeros(len(pool), dtype=bool)
    for left, right in sectors:
        seen |= in_nautical_sector(pool["dir"].to_numpy(), left, right)
    pool["ws_weight"] = wind_sea_weight(pool["hs"], pool["tp"], pool["spr"], **_ramps())
    keep = (
        seen
        & (wave_steepness(pool["hs"], pool["tp"]) <= SIMULATIONS["max_steepness"])
        & (pool["ws_weight"] >= MIN_TRAINING_WEIGHT)
    )
    pool = pool[keep].reset_index(drop=True)
    if len(pool) <= cfg["num_cases"]:
        raise ValueError(f"only {len(pool)} plausible wind-sea samples; widen the bounds")
    _, idx, _ = fit_mda(pool[names], int(cfg["num_cases"]), ["dir", "wind_wave_angle"])
    cases = pool.iloc[idx].reset_index(drop=True)
    u_min, u_max = cfg["u10_range"]
    cases["u10"] = np.clip(cases["speed_ratio"] * fully_developed_wind(cases["hs"], cases["tp"]), u_min, u_max)
    cases["u10dir"] = np.mod(cases["dir"] + cases["wind_wave_angle"], 360.0)
    return cases


def wind_cases_dir(with_wind: bool, model: str | None = None) -> Path:
    """``simulations.<model>.wind_cases`` / ``wind`` or ``no_wind``."""
    root = resolve(model_config(model).get("wind_cases", f"{simulation_model(model)}_CASES_Wind"))
    return root / ("wind" if with_wind else "no_wind")


def case_parameters(cases: pd.DataFrame, n_nodes: int, with_wind: bool) -> dict[str, list]:
    """
    Dynamic-wrapper parameters: every goal with the case sea state.

    Parameters
    ----------
    cases
        Output of :func:`generate_wind_cases`.
    n_nodes
        Number of goals (boundary nodes).
    with_wind
        Uniform ``u10`` / ``u10dir`` from the cases, or ``u10 = 0`` (the
        templates' no-wind settings, as the metamodel cases).

    Returns
    -------
    dict[str, list]
        ``tref`` (one hour apart from :data:`CASES_T0`), ``{var}_nodes``,
        ``u10``, ``u10dir``.
    """
    params: dict[str, list] = {
        "tref": [(CASES_T0 + pd.Timedelta(hours=i)).strftime("%Y%m%d %H%M%S") for i in range(len(cases))],
    }
    for var in ("hs", "tp", "dir", "spr", "wl"):
        params[f"{var}_nodes"] = [[float(v)] * n_nodes for v in cases[var]]
    params["u10"] = [float(u) if with_wind else 0.0 for u in cases["u10"]]
    params["u10dir"] = [float(d) for d in cases["u10dir"]]
    return params


def wind_case_wrappers(setup: Any, cases: pd.DataFrame | None = None, **kwargs: Any) -> dict[str, Any]:
    """
    The training cases with and without wind, as the model's dynamic wrappers.

    Parameters
    ----------
    setup
        :class:`utils.simulation.HyWavesWrapperSetup` of the model.
    cases
        Output of :func:`generate_wind_cases`; written to
        ``<wind_cases>/wind_cases.csv``. Default: read that file back.
    **kwargs
        Passed to ``setup.dynamic_wrapper`` (e.g. ``debug=False``).

    Returns
    -------
    dict
        ``{"wind": wrapper, "no_wind": wrapper}``.
    """
    table = wind_cases_dir(True, setup.model).parent / "wind_cases.csv"
    if cases is None:
        cases = pd.read_csv(table)
    else:
        table.parent.mkdir(parents=True, exist_ok=True)
        cases.to_csv(table, index=False)
    n_nodes = len(setup.boundary_nodes)
    return {
        label: setup.dynamic_wrapper(
            case_parameters(cases, n_nodes, with_wind=label == "wind"),
            output_dir=wind_cases_dir(label == "wind", setup.model),
            **kwargs,
        )
        for label in ("wind", "no_wind")
    }


def postprocessed_path(with_wind: bool, model: str | None = None) -> Path:
    """``<wind_cases>/{wind,no_wind}.nc``: postprocessed training cases."""
    folder = wind_cases_dir(with_wind, model)
    return folder.parent / f"{folder.name}.nc"


def training_pairs(model: str | None = None) -> pd.DataFrame:
    """
    One row per training case and site: Hs without / with wind and the features.

    Reads ``wind_cases.csv`` and the postprocessed runs (:func:`postprocessed_path`),
    and the fetch table at each case's wind direction.

    Parameters
    ----------
    model
        Default ``simulations.model``.

    Returns
    -------
    pandas.DataFrame
        ``case, sites, hs0, hs1, ws_weight, u10, fetch, depth, mean_depth,
        angle`` plus
        ``site_source``/``site_label`` when present.
    """
    model = simulation_model(model)
    cases = pd.read_csv(wind_cases_dir(True, model).parent / "wind_cases.csv")
    with xr.open_dataset(postprocessed_path(False, model)) as a, xr.open_dataset(postprocessed_path(True, model)) as b:
        hs0 = a["hs"].transpose("time", "sites").values
        hs1 = b["hs"].transpose("time", "sites").values
        site_ids = a["sites"].values.astype(str)
        meta = {c: a[c].values for c in ("site_source", "site_label") if c in a.coords}
    table = fetch_table(model=model).sel(sites=site_ids)
    step = 360.0 / table.sizes["dir"]
    j = np.mod(np.rint(cases["u10dir"].to_numpy() / step).astype(int), table.sizes["dir"])
    n_cases, n_sites = hs0.shape
    if n_cases != len(cases):
        raise ValueError(f"{n_cases} postprocessed cases but {len(cases)} in wind_cases.csv")
    pairs = pd.DataFrame(
        {
            "case": np.repeat(np.arange(n_cases), n_sites),
            "sites": np.tile(site_ids, n_cases),
            "hs0": hs0.ravel(),
            "hs1": hs1.ravel(),
            "ws_weight": np.repeat(cases["ws_weight"].to_numpy(), n_sites),
            "u10": np.repeat(cases["u10"].to_numpy(), n_sites),
            "fetch": table["fetch_eff"].values[:, j].T.ravel(),
            "depth": np.tile(table["site_depth"].values, n_cases),
            "mean_depth": table["mean_depth"].values[:, j].T.ravel(),
            "angle": np.repeat(cases["wind_wave_angle"].to_numpy(), n_sites),
        }
    )
    for name, values in meta.items():
        pairs[name] = np.tile(values.astype(str), n_cases)
    return pairs


def correction_path(model: str | None = None) -> Path:
    """``outputs/wind/wind_correction_{model}.pkl``."""
    return output_dir() / f"wind_correction_{simulation_model(model)}.pkl"


def _fit(pairs: pd.DataFrame) -> WindCorrection:
    return WindCorrection().fit(
        pairs["hs0"].to_numpy(), pairs["hs1"].to_numpy(), pairs["u10"].to_numpy(),
        pairs["fetch"].to_numpy(), pairs["depth"].to_numpy(), pairs["mean_depth"].to_numpy(),
        pairs["angle"].to_numpy(), weight=pairs["ws_weight"].to_numpy(),
    )


def _holdout(pairs: pd.DataFrame, test_fraction: float, seed: int) -> pd.Series:
    """Pairs of a random fraction of the cases (all their sites), held out."""
    rng = np.random.default_rng(seed)
    case_ids = pairs["case"].unique()
    n_test = max(1, int(round(test_fraction * len(case_ids))))
    return pairs["case"].isin(rng.choice(case_ids, n_test, replace=False))


def _held_out(pairs: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, WindCorrection]:
    """
    Held-out pairs with the ``K`` of a model fitted on the other cases.

    Adds ``hs_ws`` (wind-sea Hs without wind), ``k_fitted`` and
    ``hs_corrected`` (as in the reconstruction: ``K`` scales the wind-sea
    energy ``w hs^2`` only); returns the model too.
    """
    test = _holdout(pairs, test_fraction, seed)
    held = pairs[test].copy()
    fitted = _fit(pairs[~test])
    w = held["ws_weight"].to_numpy()
    held["hs_ws"] = np.sqrt(w) * held["hs0"]
    held["k_fitted"] = fitted.predict(
        held["hs_ws"].to_numpy(), held["u10"].to_numpy(), held["fetch"].to_numpy(),
        held["depth"].to_numpy(), held["mean_depth"].to_numpy(), held["angle"].to_numpy(),
    )
    held["hs_corrected"] = held["hs0"] * np.sqrt(1.0 + (held["k_fitted"] ** 2 - 1.0) * w)
    return held, fitted


def fit_wind_correction(
    model: str | None = None, test_fraction: float = 0.2, seed: int = 0
) -> tuple[WindCorrection, pd.DataFrame]:
    """
    Fit ``K`` on the training pairs; report it on held-out cases; save it.

    A fraction of the cases (all their sites) is held out to score the fit
    (reference = the run with wind); the saved model is then refitted on
    every case.

    Parameters
    ----------
    model
        Default ``simulations.model``.
    test_fraction
        Fraction of held-out cases for the scores.
    seed
        Seed of the case split.

    Returns
    -------
    tuple[WindCorrection, pandas.DataFrame]
        The model fitted on all cases, and holdout scores (``bias``, ``rmse``
        of Hs, m) per site group, without and with the correction.
    """
    pairs = training_pairs(model).dropna(subset=["hs0", "hs1"])
    held, _ = _held_out(pairs, test_fraction, seed)
    group = "site_label" if "site_label" in held else None
    rows = []
    for name, sub in (held.groupby(group) if group else [("all", held)]):
        for label, col in (("no correction", "hs0"), ("wind correction", "hs_corrected")):
            err = sub[col] - sub["hs1"]
            rows.append({"group": name, "estimate": label, "n": len(sub), "bias": err.mean(), "rmse": float(np.sqrt((err**2).mean()))})
    scores = pd.DataFrame(rows)

    correction = _fit(pairs)
    correction.save_model(str(correction_path(model)))
    logger.info("Wind correction fitted on %d pairs -> %s", correction.n_samples, correction_path(model))
    return correction, scores


def formula_k(
    hs_ws: np.ndarray, u10: np.ndarray, fetch: np.ndarray, depth: np.ndarray | None = None
) -> np.ndarray:
    """
    ``K`` from the Kahma & Calkoen growth curves (the textbook estimate).

    The wind sea ``hs_ws`` is taken as grown from calm over its equivalent
    fetch (:func:`bluemath_tk.waves.wind.equivalent_fetch`), then grown over
    *fetch* more (:func:`bluemath_tk.waves.wind.growth_hs`), capped by the
    fully developed sea and, with *depth*, by the Breugem & Holthuijsen depth
    limit. The formula only adds energy: ``K >= 1``.

    Parameters
    ----------
    hs_ws : np.ndarray
        Wind-sea Hs without wind (m).
    u10 : np.ndarray
        Wind speed (m/s).
    fetch : np.ndarray
        Fetch along the wind (m).
    depth : np.ndarray, optional
        Depth for the depth limit (m); None is deep water.

    Returns
    -------
    np.ndarray
        ``K``, NaN where it is undefined (no wind sea or no wind).
    """
    hs_ws, u10 = np.asarray(hs_ws, dtype=float), np.asarray(u10, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        x0 = equivalent_fetch(hs_ws, u10)
        grown = growth_hs(u10, np.where(np.isfinite(x0), x0 + np.asarray(fetch, dtype=float), 1e12), depth)
        k = np.maximum(grown / hs_ws, 1.0)
    return np.where((hs_ws > 0) & (u10 > 0), k, np.nan)


def k_comparison(model: str | None = None, test_fraction: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """
    ``K`` three ways on held-out training cases: the runs', the fitted model's, the formula's.

    The cases held out are those of :func:`fit_wind_correction` (same
    *test_fraction* and *seed*); the model compared is refitted on the other
    cases, so it never saw them.

    Parameters
    ----------
    model
        Default ``simulations.model``.
    test_fraction, seed
        Held-out cases, as in :func:`fit_wind_correction`.

    Returns
    -------
    pandas.DataFrame
        One row per held-out case and site with a wind sea: ``hs_ws``, ``u10``,
        ``fetch``, ``depth``, ``angle``, ``hs_ws_u2`` (``g Hs_ws / U^2``),
        ``k_runs`` (from the runs with and without wind, the reference),
        ``k_fitted``, ``k_formula_deep`` and ``k_formula_depth`` (depth limit
        at the site; both bounded like the others), plus ``site_label`` when
        present.
    """
    pairs = training_pairs(model).dropna(subset=["hs0", "hs1"])
    held, fitted = _held_out(pairs, test_fraction, seed)
    w = held["ws_weight"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        k2 = 1.0 + (held["hs1"] ** 2 - held["hs0"] ** 2) / (w * held["hs0"] ** 2)
    held["k_runs"] = np.sqrt(np.clip(k2, fitted.k_range[0] ** 2, fitted.k_range[1] ** 2))
    held = held[(held["u10"] >= fitted.u10_min) & (held["hs_ws"] >= fitted.hs_min) & np.isfinite(held["k_runs"])]
    held = held.dropna(subset=["fetch", "depth", "mean_depth"])
    # same bounds as the runs' and the fitted K (tiny wind seas would give huge K)
    bounds = fitted.k_range
    held["k_formula_deep"] = np.clip(formula_k(held["hs_ws"], held["u10"], held["fetch"]), *bounds)
    held["k_formula_depth"] = np.clip(
        formula_k(held["hs_ws"], held["u10"], held["fetch"], held["depth"]), *bounds
    )
    held["hs_ws_u2"] = 9.81 * held["hs_ws"] / held["u10"] ** 2
    keep = ["case", "sites", "hs_ws", "u10", "fetch", "depth", "angle", "hs_ws_u2",
            "k_runs", "k_fitted", "k_formula_deep", "k_formula_depth"]
    return held[[c for c in keep + ["site_label"] if c in held]].reset_index(drop=True)


def holdout_cases(model: str | None = None, test_fraction: float = 0.2, seed: int = 0) -> pd.DataFrame:
    """
    The training cases held out by :func:`fit_wind_correction` (same *test_fraction*, *seed*).

    Returns
    -------
    pandas.DataFrame
        Rows of ``wind_cases.csv`` (index = case), plus ``hs_ws_u2``
        (``g Hs_ws / U^2`` offshore: how developed the wind sea is, 0.24 =
        fully developed).
    """
    model = simulation_model(model)
    pairs = training_pairs(model).dropna(subset=["hs0", "hs1"])
    ids = np.sort(pairs.loc[_holdout(pairs, test_fraction, seed), "case"].unique())
    cases = pd.read_csv(wind_cases_dir(True, model).parent / "wind_cases.csv").loc[ids]
    cases["hs_ws_u2"] = 9.81 * cases["hs"] * np.sqrt(cases["ws_weight"]) / cases["u10"] ** 2
    return cases


def case_comparison(
    case_ids: list[int], model: str | None = None, test_fraction: float = 0.2, seed: int = 0
) -> xr.Dataset:
    """
    Held-out training cases at every site: the runs with and without wind, and the correction.

    The model is fitted without the held-out cases (as :func:`k_comparison`),
    so the corrected Hs is a fair test against the run with wind.

    Parameters
    ----------
    case_ids
        Cases of :func:`holdout_cases`.
    model
        Default ``simulations.model``.
    test_fraction, seed
        Held-out cases, as in :func:`fit_wind_correction`.

    Returns
    -------
    xarray.Dataset
        ``hs0`` (without wind), ``hs1`` (with wind), ``hs_corrected`` and
        ``k_fitted`` on ``(case, sites)``; site ``longitude`` / ``latitude``
        and the case parameters (``wind_cases.csv``, ``hs_ws_u2``) as
        coordinates.
    """
    model = simulation_model(model)
    cases = holdout_cases(model, test_fraction, seed)
    missing = sorted(set(case_ids) - set(cases.index))
    if missing:
        raise ValueError(f"cases {missing} are not held out: pick them from holdout_cases()")
    pairs = training_pairs(model).dropna(subset=["hs0", "hs1"])
    held, _ = _held_out(pairs, test_fraction, seed)
    held = held[held["case"].isin(case_ids)]
    ds = held.set_index(["case", "sites"])[["hs0", "hs1", "hs_corrected", "k_fitted"]].to_xarray()
    ds = ds.sel(sites=list(dict.fromkeys(held["sites"])))  # the runs' site order, not sorted
    table = fetch_table(model=model).sel(sites=ds["sites"].values)
    ds = ds.assign_coords(longitude=("sites", table["point_x"].values), latitude=("sites", table["point_y"].values))
    return ds.assign_coords({name: ("case", cases.loc[ds["case"].values, name].to_numpy()) for name in cases.columns})


def training_range(model: str | None = None, q: tuple[float, float] = (1.0, 99.0)) -> dict[str, tuple[float, float]]:
    """
    Range of the ``K`` inputs in the training pairs (percentiles *q*), as seen at the sites.

    Returns
    -------
    dict
        ``u10`` (m/s), ``angle`` (deg) and ``hs_ws_u2`` (``g Hs_ws / U^2``)
        -> (low, high).
    """
    pairs = training_pairs(model).dropna(subset=["hs0", "hs1"])
    pairs = pairs[pairs["hs0"] > 0]
    values = {
        "u10": pairs["u10"],
        "angle": pairs["angle"],
        "hs_ws_u2": 9.81 * np.sqrt(pairs["ws_weight"]) * pairs["hs0"] / pairs["u10"] ** 2,
    }
    return {name: tuple(float(v) for v in np.nanpercentile(s, q)) for name, s in values.items()}


def k_response(
    site_ids: list[str],
    wind_dir: float,
    hs_ws: float = 2.0,
    u10: float = 15.0,
    angle: float = 0.0,
    n: int = 60,
    model: str | None = None,
) -> xr.Dataset:
    """
    ``K`` of the fitted model at sites, one input varied at a time around a base state.

    The wind blows from *wind_dir*; fetch and mean depth along it come from
    the fetch table (nearest direction), depth from the site.

    Parameters
    ----------
    site_ids
        Sites of the fetch table.
    wind_dir
        Wind direction (deg, nautical, coming from).
    hs_ws, u10, angle
        Base state: wind-sea Hs at the site (m), wind speed (m/s), wind-wave
        angle (deg).
    n
        Points per curve.
    model
        Default ``simulations.model``.

    Returns
    -------
    xarray.Dataset
        ``k_u10`` on ``(sites, u10)`` (``hs_ws`` fixed), ``k_angle`` on
        ``(sites, angle)`` and ``k_hs_ws_u2`` on ``(sites, hs_ws_u2)``
        (``u10`` fixed, ``hs_ws`` varied); site ``depth`` and ``fetch``
        (m) coordinates; the base state in the attributes.
    """
    correction = load_wind_correction(model)
    table = fetch_table(model=model).sel(sites=list(site_ids))
    table = table.sel(dir=float(wind_dir) % 360.0, method="nearest")
    fetch = table["fetch_eff"].values[:, None]
    depth = table["site_depth"].values[:, None]
    mean_depth = table["mean_depth"].values[:, None]
    axes = {
        "u10": np.linspace(1.0, 40.0, n),
        "angle": np.linspace(-90.0, 90.0, n),
        "hs_ws_u2": np.geomspace(0.002, 0.6, n),
    }
    inputs = {
        "u10": (hs_ws, axes["u10"][None, :], angle),
        "angle": (hs_ws, u10, axes["angle"][None, :]),
        "hs_ws_u2": (axes["hs_ws_u2"][None, :] * u10**2 / 9.81, u10, angle),
    }
    data = {}
    for name, (h, u, a) in inputs.items():
        shape = (len(site_ids), n)
        k = correction.predict(
            *(np.broadcast_to(v, shape).ravel() for v in (h, u, fetch, depth, mean_depth, a))
        )
        data[f"k_{name}"] = (("sites", name), k.reshape(shape))
    return xr.Dataset(
        data,
        coords={"sites": list(site_ids), **axes, "depth": ("sites", depth[:, 0]), "fetch": ("sites", fetch[:, 0])},
        attrs={"wind_dir": float(wind_dir), "hs_ws": hs_ws, "u10": u10, "angle": angle},
    )


def load_wind_correction(model: str | None = None) -> WindCorrection:
    """The fitted correction of :func:`fit_wind_correction`."""
    path = correction_path(model)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run fit_wind_correction() first")
    return load_model(str(path))


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def wind_at_sites(times: Any, site_ids: list[str], model: str | None = None) -> xr.Dataset:
    """
    ``forcing.wind`` interpolated to the sites: ``u10`` and ``u10dir`` on ``(time, sites)``.

    Parameters
    ----------
    times
        Times to return.
    site_ids
        Sites (coordinates from the fetch table).
    model
        Default ``simulations.model``.

    Returns
    -------
    xarray.Dataset
        Wind at the sites.
    """
    sites = fetch_table(model=model).sel(sites=list(site_ids))
    index = pd.DatetimeIndex(times)
    wind = open_wind(slice(index.min() - pd.Timedelta(hours=1), index.max() + pd.Timedelta(hours=1)))
    at = wind.interp(
        time=index,
        latitude=xr.DataArray(sites["point_y"].values, dims="sites"),
        longitude=xr.DataArray(sites["point_x"].values, dims="sites"),
    ).load()
    speed, direction = wind_speed_direction(at["u10"].values, at["v10"].values)
    coords = {"time": index, "sites": list(site_ids)}
    return xr.Dataset(
        {"u10": (("time", "sites"), speed), "u10dir": (("time", "sites"), direction)},
        coords=coords,
    )


def reconstruction_inputs(
    times: Any, site_ids: list[str], model: str | None = None
) -> dict[str, Any]:
    """
    Keyword arguments of the wind-sea contribution for one reconstruction.

    Parameters
    ----------
    times
        Reconstructed times (for the wind data).
    site_ids
        Reconstructed sites.
    model
        Default ``simulations.model``.

    Returns
    -------
    dict
        ``correction``, ``fetch``, ``wind`` (None for ``"inferred"``),
        ``spr`` and ``tp`` (see
        :func:`bluemath_tk.waves.hywaves.wind_correction.wind_sea_contribution`).
    """
    cfg = config()
    if cfg["wind"] not in ("inferred", "forcing"):
        raise ValueError(f"wind_correction.wind must be 'inferred' or 'forcing', got {cfg['wind']!r}")
    return {
        "correction": load_wind_correction(model),
        "fetch": fetch_table(model=model).sel(sites=list(site_ids)),
        "wind": wind_at_sites(times, site_ids, model) if cfg["wind"] == "forcing" else None,
        "spr": float(cfg["added_spr"]),
        "tp": str(cfg["added_tp"]),
    }


__all__ = [
    "DEFAULTS",
    "config",
    "correction_path",
    "enabled",
    "fetch_table",
    "case_comparison",
    "fit_wind_correction",
    "formula_k",
    "generate_wind_cases",
    "holdout_cases",
    "k_response",
    "k_comparison",
    "load_wind_correction",
    "postprocessed_path",
    "reconstruction_inputs",
    "training_pairs",
    "training_range",
    "wind_at_sites",
    "wind_case_wrappers",
    "wind_cases_dir",
]
