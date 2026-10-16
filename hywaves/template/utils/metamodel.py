"""The HyWaves metamodel of a case: fit, predict and reconstruct.

The method is :mod:`bluemath_tk.waves.hywaves` (``metamodel``: MDA, PCA, GP
and prediction per goal; ``reconstruction``: the linear spectral sum). This
module is the case layer: it reads ``metamodel`` from the case JSON, keeps the
fitted artefacts in ``outputs/goal_XX`` (``mda.pkl``, ``split.json``, PCA/GP
pickles) and caches reconstructions in ``tmp/reconstruct`` (one zarr store
per setup, one chunk per site and month).

Metamodel sets (``metamodel.sets``) are PCA + GP models on a group of output
sites (``{"buoys": {"sites": "buoy"}, ...}``). They share each goal's model
cases and MDA, and every call names its set (``metamodel="buoys"``).

Resources: the user decides. :func:`fit_resources` and
:func:`reconstruction_resources` print what this machine has and propose
settings (at most 75% of it); the user edits them if wanted and passes them to
:func:`fit_all` / :func:`reconstruct` (``resources=``)::

    res = mm.reconstruction_resources("2013", 550, metamodel="buoys", spectral=True)
    res.workers = 20  # optional
    ds = mm.reconstruct("2013", 550, metamodel="buoys", spectral=True, resources=res)

Fits run as parallel single-thread processes; a reconstruction predicts the
goal contributions with threads and sums the spectra in worker processes
(see :mod:`bluemath_tk.waves.hywaves.resources`).

- Training: :func:`fit_resources`, :func:`fit_all` (parallel processes).
- Prediction: :func:`predict` (forcing rows -> nearshore fields of one goal),
  :func:`predict_goal` (one goal and offshore partition of a hindcast).
- Reconstruction: :func:`reconstruction_resources`,
  :func:`goal_contributions` (every goal x partition),
  :func:`sum_contributions` (spectral sum), :func:`reconstruct` (both, cached
  per site and calendar month in :func:`reconstruction_store`),
  :func:`load_reconstruction_spectra`.

Site selection and metadata: :mod:`utils.geometry`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import shutil
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import xarray as xr
from bluemath_tk.core.io import load_model
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.resources import GIB, USAGE, limit_threads, machine, run_processes
from bluemath_tk.waves.hywaves import metamodel as hw
from bluemath_tk.waves.hywaves import reconstruction as hwr
from bluemath_tk.waves.hywaves.resources import (
    FitResources,
    ReconstructionResources,
    call_memory,
    hours,
    plan_fit,
    plan_reconstruction,
    sum_budget,
)
from bluemath_tk.waves.hywaves.store import ReconstructionStore

from utils.config import (
    FORCING,
    METAMODEL,
    forcing_columns,
    get_offshore_wave,
    goal_water_level,
    load_goals,
    model_config,
    open_offshore_waves,
    resolve,
    simulation_model,
)
from utils.geometry import output_sites, output_sites_path, resolve_sites, site_meta

LOGGER_NAME = "hywaves.metamodel"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("logs/metamodel.log")),
    level="INFO",
    console=True,
    console_level="WARNING",
)
# bluemath_tk logs (summation and process-pool progress) go to the same file
for _name in ("bluemath_tk.waves.hywaves", "bluemath_tk.core.resources"):
    _toolkit_logger = logging.getLogger(_name)
    _toolkit_logger.setLevel(logging.INFO)
    for _handler in logger.handlers:
        if isinstance(_handler, logging.FileHandler) and _handler not in _toolkit_logger.handlers:
            _toolkit_logger.addHandler(_handler)

VARS = METAMODEL["vars_to_predict"]
MDA_INPUTS = METAMODEL["mda_inputs"]
MDA_DIRECTIONAL = METAMODEL["mda_directional_variables"]
MDA_CANDIDATES = [int(n) for n in METAMODEL["mda_candidates"]]
SITE_FILTER = METAMODEL["site_filter"]
SETS = METAMODEL["sets"]
RECONSTRUCTION = METAMODEL["reconstruction"]
PCA_PREPROCESSING = {
    key: {**hw.DEFAULT_PCA_PREPROCESSING[key], **METAMODEL.get("pca_preprocessing", {}).get(key, {})}
    for key in ("value_to_replace_nans", "nan_threshold_to_drop")
}
#: PCA targets of ``vars_to_predict``: ``(name, variance, stacked variables)``.
PCA_TARGETS = hw.pca_targets(VARS)

#: ``wl`` of the reconstruction: a constant (m), or this string for the
#: ``forcing.water_level`` series at each goal.
WL_FROM_FORCING = "forcing"
WaterLevel = float | str
BULK_PARTITION = hw.BULK_PARTITION

#: Bump to invalidate every reconstruction cache.
CACHE_VERSION = 10


class ModelsNotFittedError(FileNotFoundError):
    """PCA/GP pickles are missing for a goal, metamodel set and MDA size."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def goal_output_dir(goal: str | int) -> Path:
    """``outputs/goal_XX``: the fitted artefacts of one goal."""
    return resolve(f"outputs/goal_{int(goal):02d}")


def cases_path(goal: str | int) -> Path:
    """``output_sites_{goal}.nc`` in ``simulations.<model>.metamodel_cases``."""
    return resolve(model_config()["metamodel_cases"]) / f"output_sites_{int(goal)}.nc"


def mda_path(goal: str | int) -> Path:
    """``outputs/goal_XX/mda.pkl``."""
    return goal_output_dir(goal) / "mda.pkl"


def split_path(goal: str | int) -> Path:
    """``outputs/goal_XX/split.json``: model, train/test cases, removed sites."""
    return goal_output_dir(goal) / "split.json"


def _artefact_dir(key: str, goal: str | int, metamodel: str, n: int) -> Path:
    """``metamodel.paths.{key}`` for one goal, set and MDA size."""
    if metamodel not in SETS:
        raise ValueError(f"Unknown metamodel set {metamodel!r}; configured: {list(SETS)}")
    return resolve(
        METAMODEL["paths"][key].format(
            goal=f"goal_{int(goal):02d}", metamodel=metamodel, mda_candidate=int(n)
        )
    )


def gp_model_path(goal: str | int, metamodel: str, n: int) -> Path:
    """``outputs/goal_XX/{set}/gp_{n}/gp_model.pkl``."""
    return _artefact_dir("gp_dir", goal, metamodel, n) / "gp_model.pkl"


def pca_model_path(goal: str | int, metamodel: str, n: int, name: str, variance: float) -> Path:
    """``outputs/goal_XX/{set}/pca_{n}/pca_{name}_{variance}_model.pkl``."""
    return _artefact_dir("pca_dir", goal, metamodel, n) / f"pca_{name}_{variance:g}_model.pkl"


# ---------------------------------------------------------------------------
# Goals and sites
# ---------------------------------------------------------------------------


def _goals(goals: Iterable[str | int] | None = None) -> list[str]:
    """Goal ids as strings; default every goal of the goals JSON."""
    return [str(g) for g in (goals if goals is not None else load_goals())]


def set_sites(metamodel: str) -> list[str]:
    """``ok`` output sites of a metamodel set (its ``sites`` selector)."""
    return resolve_sites(site_meta(), SETS[metamodel]["sites"])


def reconstruction_sites(
    metamodel: str,
    sites: str | Iterable[str] | None = None,
    goals: Iterable[str | int] | None = None,
) -> list[str]:
    """
    Sites a set can reconstruct: in the set and kept by the site filter of every goal.

    Parameters
    ----------
    metamodel
        Metamodel set.
    sites
        Selector within the set (group, umbrella, site id or a list; see
        :func:`utils.geometry.resolve_sites`); default every site of the set.
    goals
        Goals whose ``removed_sites`` are excluded; default all.

    Returns
    -------
    list[str]
        Site ids, in table / selector order.

    Raises
    ------
    ValueError
        If no site is left.
    """
    removed = {s for goal in _goals(goals) for s in removed_sites(goal)}
    in_set = [s for s in set_sites(metamodel) if s not in removed]
    if sites is None:
        return in_set
    allowed = set(in_set)
    selected = resolve_sites(site_meta(), sites)
    dropped = [s for s in selected if s not in allowed]
    if dropped:
        logger.warning(
            "Dropped %d site(s) not in set %r or removed by the site filter: %s",
            len(dropped), metamodel, ", ".join(dropped[:8]) + ("…" if len(dropped) > 8 else ""),
        )
    kept = [s for s in selected if s in allowed]
    if not kept:
        raise ValueError(f"No site of {sites!r} can be reconstructed with set {metamodel!r}")
    return kept


# ---------------------------------------------------------------------------
# Model cases and train/test split
# ---------------------------------------------------------------------------


def _read_split(goal: str | int) -> dict[str, Any]:
    with split_path(goal).open(encoding="utf-8") as f:
        return json.load(f)


def removed_sites(goal: str | int) -> list[str]:
    """``split.json`` ``removed_sites`` of one goal; empty before it is fitted."""
    return list(_read_split(goal)["removed_sites"]) if split_path(goal).exists() else []


def load_split(goal: str | int) -> tuple[np.ndarray, np.ndarray]:
    """Train (MDA centroids, in MDA order) and test case indices of one goal."""
    split = _read_split(goal)
    return np.asarray(split["train_idx"], dtype=int), np.asarray(split["test_idx"], dtype=int)


def check_artefacts_model(goal: str | int) -> None:
    """Refuse a goal's artefacts when another model produced its cases (``split.json`` ``model``)."""
    fitted = _read_split(goal)["model"]
    if fitted != simulation_model():
        raise ValueError(
            f"Goal {goal}: outputs in {goal_output_dir(goal)} were fitted on {fitted} cases "
            f"but simulations.model is {simulation_model()!r}. Move them aside (snapshot the "
            "portal first) or switch the model back."
        )


def load_cases_goal(
    goal: str | int, *, apply_site_filter: bool = True, fill_zero: bool = True
) -> xr.Dataset:
    """
    Model cases of one goal at the ``ok`` output sites, with ``dir_u``/``dir_v``.

    Where no energy reaches a site (``hs = 0``) the other fields are NaN
    (:func:`bluemath_tk.waves.hywaves.metamodel.mask_zero_energy`).

    Parameters
    ----------
    goal
        Goal id.
    apply_site_filter
        Drop ``split.json`` ``removed_sites``.
    fill_zero
        With *apply_site_filter*, fill the zero-energy fields with the case
        forcing (training). ``False`` for a holdout reference.

    Returns
    -------
    xarray.Dataset
        ``(case_num, sites)``: the fields and the ``*_forcing`` variables.

    Raises
    ------
    ValueError
        If the cases were run with other output sites (re-run them).
    """
    with xr.open_dataset(cases_path(goal)) as ds:
        out = ds.load()
    run = {str(s) for s in out.sites.values}
    saved = set(output_sites()["source_id"].astype(str))
    if run != saved:
        raise ValueError(
            f"Goal {goal}: the cases hold {len(run)} sites, the output-sites table "
            f"{len(saved)} ({len(saved - run)} missing, {len(run - saved)} extra); "
            "re-run the metamodel cases (notebook 05)."
        )
    out = hw.mask_zero_energy(out)
    if apply_site_filter:
        out = hw.drop_sites(out, removed_sites(goal))
        if fill_zero:
            out = hw.fill_zero_energy(out)
    return hw.add_direction_components(out)


def forcing_dataframe(cases: xr.Dataset) -> pd.DataFrame:
    """MDA inputs of each model case (``mda_inputs`` columns, indexed by ``case_num``)."""
    df = cases[list(forcing_columns().values())].to_dataframe()
    df.columns = MDA_INPUTS
    return df


# ---------------------------------------------------------------------------
# Fitted models
# ---------------------------------------------------------------------------


def missing_models(goal: str | int, metamodel: str, n: int) -> list[Path]:
    """GP and PCA pickles not on disk for one goal, set and MDA size."""
    paths = [gp_model_path(goal, metamodel, n)] + [
        pca_model_path(goal, metamodel, n, name, variance) for name, variance, _ in PCA_TARGETS
    ]
    return [p for p in paths if not p.exists()]


def require_models(metamodel: str, n: int, goals: Iterable[str | int] | None = None) -> None:
    """
    Raise :class:`ModelsNotFittedError` if any goal lacks its fitted models.

    Raises
    ------
    ModelsNotFittedError
        Listing every missing pickle.
    """
    missing = [p for goal in _goals(goals) for p in missing_models(goal, metamodel, n)]
    if missing:
        raise ModelsNotFittedError(
            f"Models not fitted (set {metamodel!r}, n={n}); missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + "\nRun 06_Validate_Metamodel.ipynb (mm.fit_all) first."
        )


def load_models(
    goal: str | int, metamodel: str, n: int, *, device: str | None = None
) -> dict[str, Any]:
    """
    Fitted GP and PCAs of one goal, set and MDA size.

    Parameters
    ----------
    goal, metamodel, n
        Goal id, metamodel set and MDA size.
    device
        PyTorch device for GP inference (e.g. ``"cpu"`` on a small GPU).

    Returns
    -------
    dict
        ``gp`` and ``pcas`` (target name -> PCA).
    """
    require_models(metamodel, n, [goal])
    check_artefacts_model(goal)
    gp = load_model(gp_model_path(goal, metamodel, n))
    if device is not None:
        gp.to(device)
    pcas = {
        name: load_model(pca_model_path(goal, metamodel, n, name, variance))
        for name, variance, _ in PCA_TARGETS
    }
    return {"gp": gp, "pcas": pcas}


def _release_gp_memory() -> None:
    """Free the GPU memory of unloaded GP models."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _fit_mda(goal: str, force_retrain: bool = False) -> None:
    """
    MDA of one goal and its ``split.json`` (kept if fitted, unless *force_retrain*).

    The site filter runs here: its ``removed_sites`` are saved with the split.
    The MDA selects ``max(mda_candidates)`` cases (train, in MDA order); the
    rest are the holdout.
    """
    if not force_retrain and mda_path(goal).exists() and split_path(goal).exists():
        check_artefacts_model(goal)
        return

    raw = load_cases_goal(goal, apply_site_filter=False)
    dropped = hw.detect_removed_sites(raw, SITE_FILTER)
    if dropped:
        logger.warning(
            "Goal %s: site filter %s removed %d site(s): %s",
            goal, SITE_FILTER, len(dropped), ", ".join(dropped),
        )
    cases = hw.drop_sites(raw, dropped)
    num_centers = max(MDA_CANDIDATES)
    if num_centers >= cases.sizes["case_num"]:
        raise ValueError(
            f"Goal {goal}: the largest mda_candidates ({num_centers}) must be below the "
            f"number of model cases ({cases.sizes['case_num']}) to leave holdout cases."
        )
    mda, train_idx, test_idx = hw.fit_mda(forcing_dataframe(cases), num_centers, MDA_DIRECTIONAL)

    mda_path(goal).parent.mkdir(parents=True, exist_ok=True)
    mda.save_model(mda_path(goal))
    with split_path(goal).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": simulation_model(),
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "removed_sites": dropped,
                "site_filter": SITE_FILTER,
            },
            f,
            indent=2,
        )
    logger.info("Goal %s: MDA fitted (%d train, %d test)", goal, len(train_idx), len(test_idx))


def _fit_set(goal: str, metamodel: str, n: int, gp_epochs: int, device: str) -> None:
    """PCAs and GP of one goal, set and MDA size on the first *n* MDA cases (one process)."""
    mda, train_idx = load_model(mda_path(goal)), load_split(goal)[0]
    cases = load_cases_goal(goal).isel(case_num=train_idx[:n])
    kept = set(cases.sites.values)
    cases = cases.sel(sites=[s for s in set_sites(metamodel) if s in kept])
    forcing = forcing_dataframe(cases)
    centroids = mda.centroids[MDA_INPUTS].iloc[:n]
    if not np.allclose(forcing.values, centroids.values, rtol=1e-5, atol=1e-8):
        raise ValueError(f"Goal {goal}: the train cases are not the MDA centroids")

    pcas, summary = hw.fit_pcas(
        cases,
        PCA_TARGETS,
        PCA_PREPROCESSING["value_to_replace_nans"],
        PCA_PREPROCESSING["nan_threshold_to_drop"],
    )
    for name, variance, _ in PCA_TARGETS:
        path = pca_model_path(goal, metamodel, n, name, variance)
        path.parent.mkdir(parents=True, exist_ok=True)
        pcas[name].save_model(path)
    gp = hw.fit_gp(centroids, pcas, MDA_DIRECTIONAL, epochs=gp_epochs, device=device)
    gp_model_path(goal, metamodel, n).parent.mkdir(parents=True, exist_ok=True)
    gp.save_model(gp_model_path(goal, metamodel, n))
    logger.info(
        "Goal %s / %s / n=%d: %d cases x %d sites, PCA sites dropped %s, %d PCs",
        goal, metamodel, n, cases.sizes["case_num"], cases.sizes["sites"],
        summary["pca_sites_dropped"], hw.stack_pcs(pcas).shape[1],
    )


def _fit_jobs(
    goals: Iterable[str | int] | None,
    metamodels: Iterable[str] | None,
    candidates: Iterable[int] | None,
    force_retrain: bool,
) -> tuple[list[str], list[tuple[str, str, int]]]:
    """Goals whose MDA to fit, and (goal, set, n) to fit, largest sets first."""
    goals = _goals(goals)
    mdas = [g for g in goals if force_retrain or not (mda_path(g).exists() and split_path(g).exists())]
    fits = [
        (g, m, int(n))
        for g in goals
        for m in (metamodels or SETS)
        for n in (candidates or MDA_CANDIDATES)
        if g in mdas or missing_models(g, m, int(n))
    ]
    fits.sort(key=lambda job: (len(set_sites(job[1])), job[2]), reverse=True)
    return mdas, fits


def fit_resources(
    goals: Iterable[str | int] | None = None,
    *,
    metamodels: Iterable[str] | None = None,
    candidates: Iterable[int] | None = None,
    force_retrain: bool = False,
) -> FitResources:
    """
    Propose resources for :func:`fit_all` with the same arguments, and print why.

    Lists the MDAs and the (goal, set, MDA size) fits still to do. On CPU
    every fit is one single-thread process, as many at a time as 75% of the
    cores and free memory allow (a fit takes ~1 GiB + 8x its goal's model
    cases). Nothing runs::

        res = mm.fit_resources()      # prints machine, fits to do, proposal
        res.workers = 12              # optional
        mm.fit_all(resources=res, gp_epochs=1000)

    Parameters
    ----------
    goals, metamodels, candidates, force_retrain
        As for :func:`fit_all`.

    Returns
    -------
    FitResources
        ``device``, ``workers`` (fits at the same time), ``threads`` (each).
    """
    mdas, fits = _fit_jobs(goals, metamodels, candidates, force_retrain)
    host = machine()
    cores, memory = host.usable()
    print(f"Machine   {host} -> at most {cores} cores, {memory / GIB:.0f} GiB ({USAGE:.0%})")
    if not mdas and not fits:
        print("Nothing to fit (force_retrain=True refits).")
        return FitResources()
    cases = []
    for goal in {g for g, _, _ in fits} | set(mdas):
        with xr.open_dataset(cases_path(goal)) as ds:
            cases.append(ds.nbytes)
    job_memory = GIB + 8 * max(cases)
    res = plan_fit(max(len(mdas), len(fits)), job_memory, host=host)
    per_set = pd.Series([m for _, m, _ in fits], dtype=object).value_counts().to_dict()
    print(f"To fit    {len(mdas)} MDA(s), then {len(fits)} PCA + GP fit(s) {per_set}, ~{job_memory / GIB:.1f} GiB each")
    print(f"Proposed  {res}")
    print("Edit the fields if wanted (res.workers = ...), then fit_all(..., resources=res).")
    return res


def fit_all(
    goals: Iterable[str | int] | None = None,
    *,
    resources: FitResources,
    metamodels: Iterable[str] | None = None,
    candidates: Iterable[int] | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
) -> None:
    """
    MDA of every goal, then PCA + GP per goal, metamodel set and MDA size.

    Fits run in parallel processes as *resources* say (see
    :func:`fit_resources`); existing artefacts are kept unless *force_retrain*
    (a goal whose MDA is fitted again gets all its models fitted again).
    Progress goes to ``logs/metamodel.log``.

    Parameters
    ----------
    goals
        Goals to fit; default all.
    resources
        From :func:`fit_resources`.
    metamodels
        Sets to fit; default every set.
    candidates
        MDA sizes to fit; default ``mda_candidates``.
    force_retrain
        Re-fit the MDA (new split) and every model.
    gp_epochs
        GP optimisation iterations.
    """
    mdas, fits = _fit_jobs(goals, metamodels, candidates, force_retrain)
    kw = dict(workers=resources.workers, threads=resources.threads)
    t0 = _time.monotonic()
    run_processes(_fit_mda, [(g, True) for g in mdas], **kw)
    run_processes(_fit_set, [(g, m, n, gp_epochs, resources.device) for g, m, n in fits], **kw)
    logger.info("fit_all: %d MDA(s), %d fit(s) in %.0f s (%s)", len(mdas), len(fits),
                _time.monotonic() - t0, resources)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    goal: str | int,
    n: int,
    forcing: pd.DataFrame,
    *,
    metamodel: str,
    sites: Iterable[str] | None = None,
    models: dict[str, Any] | None = None,
    device: str | None = None,
) -> xr.Dataset:
    """
    Nearshore fields of one goal for each forcing row (GP -> PCA inverse).

    Parameters
    ----------
    goal, n
        Goal id and MDA size.
    forcing
        One row per case or time step with the ``mda_inputs`` columns; its
        index (``case_num``, ``time``, ...) is the row dimension of the output.
    metamodel
        Metamodel set.
    sites
        Sites to return; default every site the goal's models kept.
    models
        :func:`load_models` output; loaded here when omitted.
    device
        PyTorch device when the models are loaded here.

    Returns
    -------
    xarray.Dataset
        ``hs``, ``tp``, ``dir``, ``spr`` on ``(row, sites)`` with the site
        metadata coordinates.
    """
    models = models or load_models(goal, metamodel, n, device=device)
    if sites is None:
        removed = set(removed_sites(goal))
        sites = [s for s in set_sites(metamodel) if s not in removed]
    site_ids = list(sites)
    fields = hw.predict_fields(models["gp"], models["pcas"], forcing[MDA_INPUTS], VARS, site_ids)
    return fields.assign_coords(site_meta(site_ids).coords)


def _goal_forcing(offshore: xr.Dataset, goal: str, partition: int, wl: WaterLevel) -> pd.DataFrame:
    """MDA forcing (``hs``, ``tp``, ``dir``, ``spr``, ``wl``) of one goal and partition."""
    if not isinstance(wl, str):
        return hw.goal_forcing(offshore.sel(goal=int(goal)), partition, float(wl))
    if wl != WL_FROM_FORCING:
        raise ValueError(f"wl must be a number or {WL_FROM_FORCING!r}, got {wl!r}")
    df = hw.goal_forcing(offshore.sel(goal=int(goal)), partition)
    df["wl"] = goal_water_level(goal, df.index).to_numpy()
    return df


def predict_goal(
    offshore: xr.Dataset,
    goal: str | int,
    n: int,
    *,
    metamodel: str,
    sites: Iterable[str],
    partition: int = BULK_PARTITION,
    wl: WaterLevel = 0.0,
    models: dict[str, Any] | None = None,
    device: str | None = None,
    wind_sea: bool = False,
) -> xr.Dataset | None:
    """
    Nearshore contribution of one goal and offshore partition over a hindcast.

    Only the time steps with energy and a direction inside the goal's sector
    are predicted (:func:`bluemath_tk.waves.hywaves.metamodel.valid_timesteps`).

    Parameters
    ----------
    offshore
        Offshore hindcast on ``(time, goal)``, already selected in time.
    goal, n
        Goal id and MDA size.
    metamodel
        Metamodel set.
    sites
        Sites to predict.
    partition
        :data:`BULK_PARTITION` or a spectral partition (``0``-``3``).
    wl
        Water level (m), or ``"forcing"`` for the ``forcing.water_level``
        series at the goal.
    models, device
        As for :func:`predict`.
    wind_sea
        Also return the offshore wind-sea weight ``ws_weight`` and inferred
        wind ``u10`` / ``u10dir`` on ``time`` (for :mod:`utils.wind`).

    Returns
    -------
    xarray.Dataset or None
        Fields on ``(time, sites)``; ``None`` when no time step is valid.
    """
    goal = str(goal)
    forcing = _goal_forcing(offshore, goal, partition, wl)
    forcing = forcing[hw.valid_timesteps(forcing, sector=tuple(load_goals()[goal]["angles"]))]
    if forcing.empty:
        return None
    pred = predict(goal, n, forcing, metamodel=metamodel, sites=sites, models=models, device=device)
    if not wind_sea:
        return pred

    from bluemath_tk.waves.hywaves.wind_correction import offshore_wind_sea

    from utils.wind import config as wind_config

    cfg = wind_config()
    ws = offshore_wind_sea(
        forcing,
        steepness_range=tuple(cfg["steepness_range"]),
        spr_range=tuple(cfg["spr_range"]),
        speed_factor=float(cfg["speed_factor"]),
    )
    return pred.assign({name: ("time", ws[name].to_numpy()) for name in ws})


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def goal_contributions(
    offshore: xr.Dataset,
    n: int,
    *,
    metamodel: str,
    sites: Iterable[str],
    spectral: bool = True,
    resources: ReconstructionResources,
    goals: Iterable[str | int] | None = None,
    wl: WaterLevel = 0.0,
    wind_sea: bool = False,
) -> list[xr.Dataset]:
    """
    Contribution of every goal and offshore partition (:func:`predict_goal`).

    The models of one goal are loaded, used for all its partitions and
    released before the next goal (GPU memory).

    Parameters
    ----------
    offshore
        Offshore hindcast on ``(time, goal)``, already selected in time.
    n, metamodel, sites
        MDA size, metamodel set and sites.
    spectral
        Offshore partitions ``0``-``3`` (``True``) or the bulk sea state.
    resources
        ``device`` and ``threads`` of the GP prediction
        (:func:`reconstruction_resources`).
    goals
        Goals to include; default all.
    wl, wind_sea
        As for :func:`predict_goal`.

    Returns
    -------
    list[xarray.Dataset]
        One dataset per goal x partition with valid time steps, with
        ``goal`` and ``partition`` dimensions of size 1.
    """
    sites = list(sites)
    chunks: list[xr.Dataset] = []
    with limit_threads(resources.threads):
        for goal in _goals(goals):
            models = load_models(goal, metamodel, n, device=resources.device)
            for partition in hwr.partition_ids(offshore, spectral):
                pred = predict_goal(
                    offshore, goal, n, metamodel=metamodel, sites=sites, partition=partition,
                    wl=wl, models=models, wind_sea=wind_sea,
                )
                if pred is not None:
                    chunks.append(pred.expand_dims(goal=[int(goal)], partition=[partition]).load())
            del models
            _release_gp_memory()
    return chunks


def sum_contributions(
    chunks: list[xr.Dataset],
    sites: Iterable[str],
    *,
    resources: ReconstructionResources,
    variables: Mapping[str, str] | None = None,
    save_spectra: bool = False,
    wind_correction: bool = False,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """
    Linear spectral sum of goal contributions into bulk parameters.

    Parameters
    ----------
    chunks
        :func:`goal_contributions` output (or datasets on ``goal``,
        ``partition``, ``time``, ``sites``).
    sites
        Sites to sum.
    resources
        ``workers`` processes (up to 2 GiB of site x time blocks each, within
        what ``memory`` the contributions and spectra leave;
        :func:`reconstruction_resources`).
    variables
        Output name -> wavespectra stat; default
        ``metamodel.reconstruction.variables``.
    save_spectra
        Also return the summed spectra.
    wind_correction
        Add the local wind sea before the sum (see :mod:`utils.wind`); the
        chunks must carry the wind-sea variables (``wind_sea=True``).

    Returns
    -------
    bulk, spectra
        Bulk parameters on ``(time, sites)``; summed ``efth`` or ``None``.
    """
    sites = list(sites)
    times = np.unique(np.concatenate([c["time"].values for c in chunks]))
    wind_inputs = None
    if wind_correction:
        from utils.wind import reconstruction_inputs

        wind_inputs = reconstruction_inputs(times, sites)
    held = 2 * sum(c.nbytes for c in chunks)
    if save_spectra:
        held += 3 * hwr.spectra_bytes(len(times), len(sites))
    return hwr.sum_partition_spectra_batched(
        chunks,
        sites,
        dict(variables or RECONSTRUCTION["variables"]),
        budget_bytes=sum_budget(resources.memory, held, resources.workers),
        workers=resources.workers,
        save_spectra=save_spectra,
        log_file=str(resolve("logs/metamodel.log")),
        wind_correction=wind_inputs,
    )


def _wind_correction_on(wind_correction: bool | None) -> bool:
    """*wind_correction*, or ``metamodel.wind_correction.enabled`` when None."""
    if wind_correction is not None:
        return bool(wind_correction)
    from utils.wind import enabled

    return enabled()


def _input_files(
    metamodel: str, n: int, goals: list[str], offshore_name: str, wl: WaterLevel, wind_correction: bool
) -> list[Path]:
    """Files a reconstruction depends on: fitted models, splits, sites, offshore and forcing stores."""
    files = [output_sites_path(), resolve(get_offshore_wave(offshore_name)["data_path"])]
    for goal in goals:
        files += [split_path(goal), gp_model_path(goal, metamodel, n)]
        files += [pca_model_path(goal, metamodel, n, name, var) for name, var, _ in PCA_TARGETS]
    if wl == WL_FROM_FORCING:
        files.append(resolve(FORCING["water_level"]["data_path"]))
    if wind_correction:
        from utils import wind

        files += [wind.correction_path(), wind.fetch_table_path()]
        if wind.config()["wind"] == "forcing":
            files.append(resolve(FORCING["wind"]["data_path"]))
    return files


def _fingerprint(files: Iterable[Path]) -> str:
    """Hash of the size and modification time of *files* (a zarr store by its metadata)."""
    stats = []
    for path in files:
        meta = next((path / m for m in ("zarr.json", ".zmetadata") if (path / m).is_file()), path)
        st = meta.stat() if meta.exists() else None
        stats.append((str(path), st.st_size if st else -1, st.st_mtime_ns if st else -1))
    return hashlib.sha1(json.dumps(stats).encode()).hexdigest()


def _store_name(
    n: int, metamodel: str, spectral: bool, wl: WaterLevel, wind: bool,
    goals: list[str] | None, spectra: bool,
) -> str:
    """``{set}_mda{n}_{model}_{bulk|spectral}_wl-{wl}_wind-{on|off}[_goals-..][_spectra]``."""
    return (
        f"{metamodel}_mda{n}_{simulation_model()}_{'spectral' if spectral else 'bulk'}"
        f"_wl-{wl if isinstance(wl, str) else f'{wl:g}'}_wind-{'on' if wind else 'off'}"
        + (f"_goals-{'-'.join(goals)}" if goals else "")
        + ("_spectra" if spectra else "")
    )


def reconstruction_store(
    n: int,
    *,
    metamodel: str,
    spectral: bool = True,
    offshore_name: str = "WHACS",
    goals: Iterable[str | int] | None = None,
    wl: WaterLevel = 0.0,
    variables: Mapping[str, str] | None = None,
    wind_correction: bool | None = None,
    spectra: bool = False,
) -> ReconstructionStore:
    """
    The cache of one reconstruction setup (bulk, or ``spectra=True`` for ``efth``).

    ``tmp/reconstruct/{offshore}/{set}_mda{n}_{model}_{bulk|spectral}_wl-{wl}_wind-{on|off}[_goals-..][_spectra].zarr``,
    holding every site of the set (kept in every goal) on the whole offshore
    time axis, one chunk per site and calendar month (see
    :class:`bluemath_tk.waves.hywaves.store.ReconstructionStore`). It is
    rebuilt when the setup or any input file (fitted models, ``split.json``,
    sites table, offshore and forcing stores, wind correction) changes. Its
    attributes record the setup and the PCs per goal.
    """
    goals = _goals(goals)
    variables = dict(variables or RECONSTRUCTION["variables"])
    wind = _wind_correction_on(wind_correction)
    name = _store_name(n, metamodel, spectral, wl, wind, goals if goals != _goals() else None, spectra)
    key = {
        "version": CACHE_VERSION,
        "model": simulation_model(),
        "offshore": offshore_name,
        "metamodel": metamodel,
        "n": int(n),
        "spectral": bool(spectral),
        "goals": goals,
        "wl": wl,
        "variables": variables,
        "wind": None if not wind else {k: v for k, v in METAMODEL.get("wind_correction", {}).items() if k != "enabled"},
        "config": {k: METAMODEL.get(k) for k in ("mda_inputs", "vars_to_predict", "site_filter", "pca_preprocessing")},
    }
    if spectra:
        store_vars = {"efth": ("freq", "dir")}
        coords = {"freq": hwr.SPECTRAL_FREQ, "dir": hwr.SPECTRAL_DIR}
    else:
        store_vars = {v: () for v in [*variables, *(["wind_k"] if wind else [])]}
        coords = {}

    def pcs() -> dict[str, Any]:
        return {"pcs": {
            goal: {name: int(load_model(pca_model_path(goal, metamodel, n, name, var)).pca.n_components_)
                   for name, var, _ in PCA_TARGETS}
            for goal in goals
        }}

    return ReconstructionStore(
        resolve("tmp/reconstruct") / offshore_name / f"{name}.zarr",
        open_offshore_waves(offshore_name)["time"].values,
        reconstruction_sites(metamodel, None, goals),
        store_vars,
        coords=coords,
        key=key,
        fingerprint=_fingerprint(_input_files(metamodel, n, goals, offshore_name, wl, wind)),
        attrs=pcs,
    )


@dataclass
class _Request:
    """A reconstruction request: its sites, stores and contributions per site."""

    n: int
    metamodel: str
    spectral: bool
    goals: list[str]
    wl: WaterLevel
    site_ids: list[str]
    variables: dict[str, str]
    save_spectra: bool
    wind: bool
    stores: list[ReconstructionStore]

    @property
    def parts(self) -> int:
        """Contributions summed per site: goals x partitions (+ 1 wind sea)."""
        partitions = len(hwr.SPECTRAL_PARTITIONS) if self.spectral else 1
        return len(self.goals) * partitions + int(self.wind)

    def todo(self, time: str | slice, recompute: bool = False) -> dict[str, list[str]]:
        """Sites to compute per month (every requested one with *recompute*), in store order."""
        months = self.stores[0].months_of(time)
        if recompute:
            return {m: list(self.site_ids) for m in months}
        todo: dict[str, set[str]] = {}
        for store in self.stores:
            for month, missing in store.missing(time, self.site_ids).items():
                todo.setdefault(month, set()).update(missing)
        return {m: [s for s in self.stores[0].sites if s in todo[m]] for m in months if m in todo}


def _request(
    n: int, *, metamodel: str, spectral: bool, offshore_name: str, goals: Iterable[str | int] | None,
    wl: WaterLevel, sites: str | Iterable[str] | None, variables: Mapping[str, str] | None,
    save_spectra: bool | None, wind_correction: bool | None,
) -> _Request:
    if not (isinstance(wl, (int, float)) or (isinstance(wl, str) and wl == WL_FROM_FORCING)):
        raise ValueError(
            f"wl must be a number (m) or {WL_FROM_FORCING!r} (forcing.water_level at each goal), "
            f"got {type(wl).__name__}"
        )
    goals = _goals(goals)
    require_models(metamodel, n, goals)
    variables = dict(variables or RECONSTRUCTION["variables"])
    save_spectra = RECONSTRUCTION["save_spectra"] if save_spectra is None else bool(save_spectra)
    wind = _wind_correction_on(wind_correction)
    setup = dict(metamodel=metamodel, spectral=spectral, offshore_name=offshore_name,
                 goals=goals, wl=wl, variables=variables, wind_correction=wind)
    stores = [reconstruction_store(n, **setup)]
    if save_spectra:
        stores.append(reconstruction_store(n, **setup, spectra=True))
    return _Request(n, metamodel, spectral, goals, wl, reconstruction_sites(metamodel, sites, goals),
                    variables, save_spectra, wind, stores)


def reconstruction_resources(
    time: str | slice,
    n: int,
    *,
    metamodel: str,
    spectral: bool = True,
    offshore_name: str = "WHACS",
    goals: Iterable[str | int] | None = None,
    wl: WaterLevel = 0.0,
    sites: str | Iterable[str] | None = None,
    variables: Mapping[str, str] | None = None,
    save_spectra: bool | None = None,
    wind_correction: bool | None = None,
    recompute: bool = False,
) -> ReconstructionResources:
    """
    Propose resources for :func:`reconstruct` with the same arguments, and print why.

    Nothing runs. It prints:

    - ``Machine``: cores, free memory and GPU, and the 75% it may use;
    - ``Request``: site-months still to compute (stored ones are read);
    - ``Proposed``: the settings (see
      :class:`bluemath_tk.waves.hywaves.resources.ReconstructionResources`);
    - ``Per call``: memory of one call (contributions, spectra, sum workers);
    - ``Time``: a rough estimate from rates measured on a 48-core node;
    - ``Spectra``: their disk size for this request (saved or not) and the
      free disk, to decide on ``save_spectra``.

    Edit the returned fields if wanted and pass them to :func:`reconstruct`
    (``resources=``). One proposal can serve several calls: a call that
    does not fit in ``resources.memory`` stops before computing.

    Parameters
    ----------
    time, n, metamodel, spectral, offshore_name, goals, wl, sites, variables,
    save_spectra, wind_correction, recompute
        As for :func:`reconstruct`.

    Returns
    -------
    ReconstructionResources
        ``device``, ``threads``, ``workers``, ``months`` and ``sites`` per
        call, ``memory``.
    """
    req = _request(n, metamodel=metamodel, spectral=spectral, offshore_name=offshore_name,
                   goals=goals, wl=wl, sites=sites, variables=variables,
                   save_spectra=save_spectra, wind_correction=wind_correction)
    todo = req.todo(time, recompute)
    n_months, n_sites = len(todo), len(set().union(*todo.values())) if todo else 0
    site_months = sum(len(v) for v in todo.values())
    requested = len(req.stores[0].months_of(time)) * len(req.site_ids)
    host = machine()
    cores, memory = host.usable()
    mode = "spectral" if spectral else "bulk"
    print(f"Machine   {host} -> at most {cores} cores, {memory / GIB:.0f} GiB ({USAGE:.0%})")
    print(f"Request   {metamodel} n={n} {mode} ({req.parts} contributions/site), {time!r}: "
          f"{site_months} of {requested} site-months to compute, in {n_months} month(s)")
    res = plan_reconstruction(max(1, n_sites), max(1, n_months), req.parts,
                              save_spectra=req.save_spectra, host=host)
    print(f"Proposed  {res}")
    if todo:
        need = call_memory(res.months, res.sites or n_sites, req.parts, workers=res.workers,
                           save_spectra=req.save_spectra)
        held = need["contributions"] + need["spectra"]
        need = call_memory(res.months, res.sites or n_sites, req.parts, workers=res.workers,
                           save_spectra=req.save_spectra,
                           sum_per_worker=sum_budget(res.memory, held, res.workers) // res.workers)
        print(f"Per call  ~{need['total'] / GIB:.0f} GiB of {res.memory / GIB:.0f} GiB: contributions "
              f"{need['contributions'] / GIB:.1f}, spectra {need['spectra'] / GIB:.1f}, "
              f"spectral sum {need['sum'] / GIB:.1f} ({res.workers} workers)")
        h = hours(site_months, n_months, req.parts, res.workers)
        span = f"{h:.1f} h" if h >= 1 else f"{max(1, round(h * 60))} min"
        print(f"Time      ~{span} (rate measured on a 48-core Xeon Gold 5418Y node; yours may differ)")
    disk = site_months * hwr.spectra_bytes(744, 1)
    free = shutil.disk_usage(req.stores[0].path.parent).free
    state = "saved" if req.save_spectra else "not saved (save_spectra=True to keep them)"
    print(f"Spectra   {state}: ~{disk / 1e9:.1f} GB on disk for these site-months, "
          f"{free / 1e9:.0f} GB free")
    if req.save_spectra and disk > free:
        print("          They do not fit on disk: fewer sites (sites=...) or a shorter period.")
    print("Edit the fields if wanted (res.workers = ...), then reconstruct(..., resources=res).")
    return res


def reconstruct(
    time: str | slice,
    n: int,
    *,
    metamodel: str,
    resources: ReconstructionResources,
    spectral: bool = True,
    offshore_name: str = "WHACS",
    goals: Iterable[str | int] | None = None,
    wl: WaterLevel = 0.0,
    sites: str | Iterable[str] | None = None,
    variables: Mapping[str, str] | None = None,
    save_spectra: bool | None = None,
    wind_correction: bool | None = None,
    use_cache: bool = True,
    recompute: bool = False,
) -> xr.Dataset:
    """
    Nearshore waves from the offshore hindcast: goal contributions, summed.

    Results are cached per site and calendar month (:func:`reconstruction_store`):
    a request reads what is stored and computes only the missing site-months,
    whole months at a time, ``resources.months`` months x ``resources.sites``
    sites per call.

    Parameters
    ----------
    time
        Time selection of the offshore hindcast (``"2013-12"``, ``"2013"``,
        ``"2013-12-05"``, a slice, ...).
    n
        MDA size; the models must be fitted for every goal.
    metamodel
        Metamodel set.
    resources
        From :func:`reconstruction_resources` (called with the same arguments).
    spectral
        Sum the offshore partitions ``0``-``3`` (``True``) or the bulk sea state.
    offshore_name
        ``offshore_waves`` entry.
    goals
        Goals to include; default all.
    wl
        Water level (m), or ``"forcing"`` for ``forcing.water_level`` per goal.
    sites
        Sites within the set (see :func:`reconstruction_sites`).
    variables
        Output name -> wavespectra stat; default
        ``metamodel.reconstruction.variables``.
    save_spectra
        Store the summed spectra too (:func:`load_reconstruction_spectra`);
        default ``metamodel.reconstruction.save_spectra``.
    wind_correction
        Add the local wind sea (see :mod:`utils.wind`); the output then has
        ``wind_k``. Default ``metamodel.wind_correction.enabled``.
    use_cache
        Read and write the cache (``False``: compute the selection in one call).
    recompute
        Recompute the requested site-months even if stored.

    Returns
    -------
    xarray.Dataset
        The requested variables on ``(time, sites)``; NaN at hours with no
        wave energy from any goal.

    Raises
    ------
    ModelsNotFittedError
        If a goal lacks its fitted models.
    MemoryError
        If one call of this request needs more than ``resources.memory``.
    """
    req = _request(n, metamodel=metamodel, spectral=spectral, offshore_name=offshore_name,
                   goals=goals, wl=wl, sites=sites, variables=variables,
                   save_spectra=save_spectra, wind_correction=wind_correction)
    offshore = open_offshore_waves(offshore_name)

    def check(months: int, n_sites: int) -> None:
        need = call_memory(months, n_sites, req.parts, workers=resources.workers,
                           save_spectra=req.save_spectra and use_cache)
        if need["total"] > resources.memory:
            raise MemoryError(
                f"A call of {months} month(s) x {n_sites} sites needs ~{need['total'] / GIB:.0f} GiB, "
                f"resources.memory is {resources.memory / GIB:.0f} GiB: lower resources.months / "
                "resources.sites, or call reconstruction_resources() with this request"
            )

    def compute(off: xr.Dataset, sites_: list[str]) -> tuple[xr.Dataset, xr.Dataset | None]:
        t0 = _time.monotonic()
        chunks = goal_contributions(
            off, n, metamodel=metamodel, sites=sites_, spectral=spectral, goals=req.goals,
            wl=wl, resources=resources, wind_sea=req.wind,
        )
        t1 = _time.monotonic()
        out = sum_contributions(
            chunks, sites_, resources=resources, variables=req.variables,
            save_spectra=req.save_spectra and use_cache, wind_correction=req.wind,
        )
        logger.info("%d steps x %d sites: contributions %.0f s, sum %.0f s",
                    off.sizes["time"], len(sites_), t1 - t0, _time.monotonic() - t1)
        return out

    if not use_cache:
        check(len(req.stores[0].months_of(time)), len(req.site_ids))
        bulk, _ = compute(offshore.sel(time=time), req.site_ids)
        return bulk.assign_coords(site_meta(req.site_ids).coords)

    if recompute:
        for store in req.stores:
            store.invalidate(time, req.site_ids)
    todo = req.todo(time)
    months = list(todo)
    if months:
        widest = max(len(v) for v in todo.values())
        check(min(resources.months, len(months)), min(resources.sites or widest, widest))
    logger.info(
        "Reconstruct %r: set %s n=%d, %d sites, %d site-month(s) to compute in %d month(s); %s",
        time, metamodel, n, len(req.site_ids), sum(len(v) for v in todo.values()), len(months), resources,
    )
    for i in range(0, len(months), resources.months):
        batch = months[i : i + resources.months]
        needed = set().union(*(todo[m] for m in batch))
        sites_ = [s for s in req.stores[0].sites if s in needed]
        off = xr.concat([offshore.sel(time=m) for m in batch], dim="time")
        step = resources.sites or len(sites_)
        for j in range(0, len(sites_), step):
            bulk, spectra = compute(off, sites_[j : j + step])
            req.stores[0].write(bulk, batch)
            if req.save_spectra:
                req.stores[1].write(spectra, batch)
            del bulk, spectra
    return req.stores[0].read(time, req.site_ids).assign_coords(site_meta(req.site_ids).coords)


def load_reconstruction_spectra(time: str | slice, n: int, *, sites: str | Iterable[str] | None = None,
                                **setup: Any) -> xr.Dataset:
    """
    Summed spectra stored by :func:`reconstruct` with ``save_spectra=True``.

    *setup*: the same ``metamodel``, ``spectral``, ``offshore_name``, ``goals``,
    ``wl``, ``variables``, ``wind_correction`` as that call.

    Returns
    -------
    xarray.Dataset
        ``efth`` on ``(time, sites, freq, dir)``.

    Raises
    ------
    KeyError
        If some requested site-months have no stored spectra.
    """
    site_ids = reconstruction_sites(setup["metamodel"], sites, setup.get("goals"))
    return reconstruction_store(n, **setup, spectra=True).read(time, site_ids)


def reconstruction_cache() -> pd.DataFrame:
    """
    The reconstruction stores of this case (``tmp/reconstruct``).

    Returns
    -------
    pandas.DataFrame
        One row per store: ``offshore``, ``store``, ``metamodel``, ``n``,
        ``months`` and ``site_months`` stored, ``sites``, ``size_mb``.
    """
    import zarr

    rows = []
    for path in sorted(resolve("tmp/reconstruct").glob("*/*.zarr")):
        group = zarr.open_group(path, mode="r")
        key = json.loads(group.attrs["key"])
        done = group["done"][:]
        rows.append({
            "offshore": path.parent.name,
            "store": path.stem,
            "metamodel": key["metamodel"],
            "n": key["n"],
            "months": int(done.any(axis=1).sum()),
            "site_months": int(done.sum()),
            "sites": done.shape[1],
            "size_mb": round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1),
        })
    return pd.DataFrame(rows)


def clear_reconstruction_cache(pattern: str = "*") -> list[Path]:
    """
    Delete the reconstruction stores matching *pattern* (e.g. ``"isobaths_*"``).

    Returns
    -------
    list[Path]
        The deleted stores.
    """
    import shutil

    paths = sorted(resolve("tmp/reconstruct").glob(f"*/{pattern}.zarr"))
    for path in paths:
        shutil.rmtree(path)
        path.with_suffix(".lock").unlink(missing_ok=True)
    return paths

