"""GPR metamodel on SnapWave LHS cases (MDA → per-variable PCA → stacked GP).

The HyWaves method itself is :mod:`bluemath_tk.waves.hywaves` (``metamodel``:
fit/predict per goal; ``reconstruction``: linear spectral summation). This
module is the case layer: it reads ``metamodel`` from the case JSON, keeps the
fitted artefacts under ``outputs/goal_XX`` (``mda.pkl``, ``split.json``, PCA/GP
pickles), loops goals x partitions x months and caches reconstructions.

The metamodel inputs are the offshore ``hs``, ``tp``, ``dir``, ``spr`` and
``wl`` of each goal, and it predicts absolute nearshore fields (``hs`` in m).

Metamodel sets (``metamodel.sets`` in the case JSON) are PCA + GP models fitted
on a subset of the output sites, e.g. ``{"buoys": {"sites": "buoy"},
"isobaths": {"sites": "isobath"}}``: they share the SnapWave cases and the MDA
selection of each goal, and every fit/predict/reconstruct call names its set
(``metamodel="buoys"``). Artefacts: ``outputs/goal_XX/{set}/pca_{n}``,
``gp_{n}``.

Training: :func:`fit`, :func:`fit_all`
Prediction: :func:`predict` (LHS ``case_num`` or hindcast ``time`` via ``row_dim``)
Hindcast: :func:`reconstruct` — ``spectral=False`` bulk (partition ``-1``),
``spectral=True`` WHACS partitions ``0``–``3``; both via wavespectra linear sum.
Bulk output variables and optional full-spectrum caching are configured in the
case JSON under ``metamodel.reconstruction`` (``variables`` maps HyGloW names to
wavespectra bulk stats; ``save_spectra`` caches summed spectra).
Long hindcast periods are processed month by month; results are cached under
``tmp/reconstruct``.

Site grouping and selection: :mod:`utils.geometry`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.io import load_model
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.waves.hywaves import metamodel as hw
from bluemath_tk.waves.hywaves import reconstruction as hwr
from wavespectra.construct import STATS as WAVESPECTRA_STATS

from utils.config import (
    METAMODEL,
    forcing_columns,
    load_goals,
    goal_water_level,
    mda_max_points,
    model_config,
    open_offshore_waves,
    resolve,
    simulation_model,
    validate_metamodel_config,
)
from utils.geometry import SITE_META, resolve_sites

LOGGER_NAME = "hywaves.metamodel"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("logs/metamodel.log")),
    level="INFO",
    console=True,
    console_level="WARNING",
)
# bluemath_tk.waves.hywaves logs (e.g. summation progress) go to the same file
_hywaves_logger = logging.getLogger("bluemath_tk.waves.hywaves")
_hywaves_logger.setLevel(logging.INFO)
for _handler in logger.handlers:
    if isinstance(_handler, logging.FileHandler) and _handler not in _hywaves_logger.handlers:
        _hywaves_logger.addHandler(_handler)

PcaVariance = hw.PcaVariance

#: ``wl`` of :func:`reconstruct` and friends: a constant (m), or this string for
#: the ``forcing.water_level`` series at each goal.
WL_FROM_FORCING = "forcing"
WaterLevel = float | str

# case config → metamodel (single source; do not duplicate in other modules)
VARS = METAMODEL["vars_to_predict"]
MDA_INPUTS = METAMODEL["mda_inputs"]
MDA_DIRECTIONAL = METAMODEL["mda_directional_variables"]
MDA_CANDIDATES = METAMODEL["mda_candidates"]
SITE_FILTER = METAMODEL["site_filter"]
SETS = METAMODEL["sets"]

# Offshore partitions: bulk = -1; wind sea + swells = 0-3 (WHACS ``phs{i}``...)
BULK_PARTITION = hw.BULK_PARTITION
SPECTRAL_PARTITIONS = hwr.SPECTRAL_PARTITIONS

# Spectral grid of the linear summation (see bluemath_tk.waves.hywaves.reconstruction)
SPECTRAL_FREQ = hwr.SPECTRAL_FREQ
SPECTRAL_DIR = hwr.SPECTRAL_DIR

# Physical floors applied in predict() (see bluemath_tk.waves.hywaves.metamodel)
MIN_DIRECTIONAL_SPREAD_DEG = hw.MIN_DIRECTIONAL_SPREAD_DEG
MIN_HS = hw.MIN_HS

# `reconstruct()` batches every *uncached* month into as few GP prediction
# calls as possible (see `_reconstruct_periods_batch`) since each call's cost
# is dominated by a large fixed per-call overhead, not by row count — one
# call covering N months costs far less than N separate one-month calls. But
# batching *too much* at once grows GPyTorch's per-call kernel evaluation past
# available GPU memory: on a 5.61 GiB GPU, a full 11-year (132-month) batch
# and a 12-month batch both hit CUDA OOM partway through the 26-goal loop.
# This is a GPU-VRAM constraint that scales with row count (time span) per
# call, independent of how many sites are requested — confirmed by direct
# instrumentation (see `_reconstruct_batch_months`'s docstring): the goal
# loop's own host RSS stayed flat (~1.5-1.7 GiB) across both a 3-site and an
# 18-site request, so this cap does not need a site-count term.
#
# Re-measured with all 26 goals active (the original 3-month figure above was
# only ever exercised with 1-2 goals) — spectral mode (4x the partition data
# of bulk per call) is the binding constraint, not bulk:
#   - bulk:     clean through at least 8 months/call (18.9 s/month at 8)
#   - spectral: clean at 4 (73.3 s/month), thrashes near the OOM edge at 6
#     (>300s for what should be a ~2 min call — GPU memory pressure, not a
#     clean failure), hits CUDA OOM outright at 8
# 4 is the largest value safe for both modes in one reconstruct() call, and a
# real ~1.2-1.5x speedup over the old default of 3 for both. Raise it only
# after confirming headroom on the target GPU for *spectral* mode specifically
# at the full 26-goal count (or pass device="cpu" to reconstruct() for
# unbounded-but-slower CPU inference).
RECONSTRUCT_GPU_BATCH_MONTHS_CAP = 4


def _reconstruct_batch_months(n_sites: int) -> int:
    """Months per `_reconstruct_periods_batch` call, capped for GPU memory.

    Currently just returns `RECONSTRUCT_GPU_BATCH_MONTHS_CAP` — *n_sites* is
    accepted (not `len(site_ids)` directly, so callers don't need to special-
    case this) in case a future host-RAM axis needs it again, but as of the
    `WAVESPECTRA_PEAK_MULTIPLIER` fix in `_site_batch_size`, the goal loop
    this caps does not itself scale with site count — see that function's
    docstring for why the real host-RAM constraint lives downstream, in
    `_wavespectra_sum`, not here.

    Parameters
    ----------
    n_sites
        ``len(site_ids)`` for this `reconstruct()` call. Unused for now.

    Returns
    -------
    int
        `RECONSTRUCT_GPU_BATCH_MONTHS_CAP`.
    """
    del n_sites
    return RECONSTRUCT_GPU_BATCH_MONTHS_CAP


# Memory budget of the site-batched spectral summation on this host (see
# bluemath_tk.waves.hywaves.reconstruction.site_batch_size for the sizing,
# including its peak-memory safety multiplier).
WAVESPECTRA_MEMORY_BUDGET_BYTES = (
    12 * 1024**3
)  # 12 GiB; leaves headroom on a 62 GiB host
WAVESPECTRA_PEAK_MULTIPLIER = hwr.PEAK_MEMORY_MULTIPLIER


def _site_batch_size(
    n_hours: int, n_parts: int, *, budget_bytes: int | None = None
) -> int:
    """Sites per summation call within ``WAVESPECTRA_MEMORY_BUDGET_BYTES``.

    See :func:`bluemath_tk.waves.hywaves.reconstruction.site_batch_size`.
    """
    budget = WAVESPECTRA_MEMORY_BUDGET_BYTES if budget_bytes is None else budget_bytes
    return hwr.site_batch_size(n_hours, n_parts, budget)


def _set_gp_device(gp: Any, device: str) -> None:
    """Move a fitted :class:`~bluemath_tk.interpolation.gps.ExactGPInterpolation` to *device*."""
    import torch

    dev = torch.device(device)
    gp.device = dev
    for model in gp._models.values():
        model.to(dev)
    for likelihood in gp._likelihoods.values():
        likelihood.to(dev)


def _release_gp_memory() -> None:
    """Best-effort CUDA cache flush after unloading GP models."""
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Default PCA NaN handling (overridden by metamodel.pca_preprocessing in case JSON)
_DEFAULT_PCA_PREPROCESSING = hw.DEFAULT_PCA_PREPROCESSING

def _pca_preprocessing() -> tuple[dict[str, float], dict[str, float]]:
    """Return PCA NaN settings from case config (with code defaults)."""
    cfg = METAMODEL.get("pca_preprocessing", {})
    replace = {
        **_DEFAULT_PCA_PREPROCESSING["value_to_replace_nans"],
        **cfg.get("value_to_replace_nans", {}),
    }
    drop = {
        **_DEFAULT_PCA_PREPROCESSING["nan_threshold_to_drop"],
        **cfg.get("nan_threshold_to_drop", {}),
    }
    return replace, drop


def _reconstruction_config() -> dict[str, Any]:
    """Return ``metamodel.reconstruction`` from the case JSON.

    Raises
    ------
    ValueError
        If the section is missing or not a dict.
    """
    recon = METAMODEL.get("reconstruction")
    if not isinstance(recon, dict):
        raise ValueError(
            "metamodel.reconstruction must be defined in the case JSON "
            "(see validate_metamodel_config)"
        )
    return recon


def _parse_reconstruction_variables(raw: Any) -> dict[str, str]:
    """Validate a HyGloW name → wavespectra stat mapping.

    Parameters
    ----------
    raw
        Mapping from case JSON ``metamodel.reconstruction.variables`` or a
        runtime override passed to :func:`reconstruct`.

    Returns
    -------
    dict[str, str]
        Validated mapping.

    Raises
    ------
    ValueError
        If *raw* is empty, uses unknown stats, or keys are not in
        ``vars_to_predict``.
    TypeError
        If *raw* is not a dict.
    """
    if not isinstance(raw, dict):
        raise TypeError(
            "metamodel.reconstruction.variables must be a dict "
            "(HyGloW name → wavespectra stat)"
        )
    if not raw:
        raise ValueError("metamodel.reconstruction.variables must be non-empty")
    unknown_stat = [s for s in raw.values() if s not in WAVESPECTRA_STATS]
    if unknown_stat:
        raise ValueError(
            "metamodel.reconstruction.variables uses unsupported wavespectra stats: "
            f"{unknown_stat}; allowed: {WAVESPECTRA_STATS}"
        )
    unknown_keys = [k for k in raw if k not in VARS]
    if unknown_keys:
        raise ValueError(
            "metamodel.reconstruction.variables keys not in vars_to_predict: "
            f"{unknown_keys}"
        )
    return dict(raw)


def _reconstruction_variables(
    *, override: Mapping[str, str] | None = None
) -> dict[str, str]:
    """HyGloW output names mapped to wavespectra bulk stats.

    Reads ``metamodel.reconstruction.variables`` from the case JSON unless
    *override* is provided (e.g. ``mm.reconstruct(..., variables={...})``).

    Raises
    ------
    ValueError
        If the case JSON section or mapping is missing or invalid.
    """
    if override is not None:
        return _parse_reconstruction_variables(dict(override))
    recon = _reconstruction_config()
    if "variables" not in recon:
        raise ValueError(
            "metamodel.reconstruction.variables is required in the case JSON"
        )
    return _parse_reconstruction_variables(recon["variables"])


def _reconstruction_save_spectra(*, override: bool | None = None) -> bool:
    """Whether to cache full summed spectra alongside bulk reconstruction.

    Reads ``metamodel.reconstruction.save_spectra`` from the case JSON unless
    *override* is passed to :func:`reconstruct`.

    Raises
    ------
    ValueError
        If the key is missing from the case JSON.
    TypeError
        If the configured value is not a bool.
    """
    if override is not None:
        return bool(override)
    recon = _reconstruction_config()
    if "save_spectra" not in recon:
        raise ValueError(
            "metamodel.reconstruction.save_spectra is required in the case JSON"
        )
    save = recon["save_spectra"]
    if not isinstance(save, bool):
        raise TypeError("metamodel.reconstruction.save_spectra must be a bool")
    return save


class ModelsNotFittedError(FileNotFoundError):
    """Raised when PCA/GP pickles are missing for a goal, MDA size and PCA variance."""

    _FIT_HINT = " Run ``06_Validate_Metamodel.ipynb`` (``mm.fit_all``) first."


def _pca_variables(override: PcaVariance = None) -> list[tuple[str, float, list[str]]]:
    """PCA targets of ``vars_to_predict`` (see :func:`bluemath_tk.waves.hywaves.metamodel.pca_targets`)."""
    return hw.pca_targets(VARS, override)


def resolved_pca_variances(override: PcaVariance = None) -> dict[str, float]:
    """PCA variance fraction per stacked variable.

    Parameters
    ----------
    override
        Optional override; default values come from the case config JSON.

    Returns
    -------
    dict[str, float]
        Mapping ``variable_name → variance``.
    """
    return {name: var for name, var, _ in _pca_variables(override)}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _variance_tag(variance: float) -> str:
    """Format a PCA variance value for artefact filenames.

    Parameters
    ----------
    variance
        PCA variance fraction.

    Returns
    -------
    str
        Compact string (e.g. ``"0.99"``).
    """
    return f"{variance:g}"


def goal_output_dir(goal: str | int) -> Path:
    """Return the output directory for one goal's fitted artefacts.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Returns
    -------
    Path
        ``outputs/goal_XX`` under the current case directory.
    """
    return resolve(f"outputs/goal_{int(goal):02d}")


def cases_path(goal: str | int) -> Path:
    """Return the metamodel cases NetCDF path for one goal (active model).

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Returns
    -------
    Path
        ``output_sites_{goal}.nc`` in ``simulations.<model>.metamodel_cases``.
    """
    return resolve(model_config()["metamodel_cases"]) / f"output_sites_{int(goal)}.nc"


def mda_path(goal: str | int) -> Path:
    """Return the fitted MDA pickle path for one goal.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Returns
    -------
    Path
        ``outputs/goal_XX/mda.pkl``.
    """
    return goal_output_dir(goal) / "mda.pkl"


def split_path(goal: str | int) -> Path:
    """Return the train/test index JSON path for one goal.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Returns
    -------
    Path
        ``outputs/goal_XX/split.json`` written by :func:`_fit_mda`.
    """
    return goal_output_dir(goal) / "split.json"


def _check_set(metamodel: str) -> str:
    """Return *metamodel* if it names a set in ``metamodel.sets``."""
    if metamodel not in SETS:
        raise ValueError(
            f"Unknown metamodel set {metamodel!r}; configured: {list(SETS)}"
        )
    return metamodel


def _artefact_dir(key: str, goal: str | int, metamodel: str, n: int) -> Path:
    """Resolve ``metamodel.paths.{key}`` for one goal, set and MDA size."""
    template = METAMODEL["paths"][key]
    return resolve(
        template.format(
            goal=f"goal_{int(goal):02d}",
            metamodel=_check_set(metamodel),
            mda_candidate=n,
        )
    )


def gp_model_path(goal: str | int, metamodel: str, n: int) -> Path:
    """Return the fitted GP pickle path for one goal, metamodel set and MDA size.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    metamodel
        Metamodel set (key of ``metamodel.sets``).
    n
        MDA train size (candidate from ``mda_candidates``).

    Returns
    -------
    Path
        ``outputs/goal_XX/{set}/gp_{n}/gp_model.pkl``.
    """
    return _artefact_dir("gp_dir", goal, metamodel, n) / "gp_model.pkl"


def pca_model_path(
    goal: str | int, metamodel: str, n: int, variable: str, variance: float
) -> Path:
    """Return the fitted PCA pickle path for one goal, set, MDA size and variable.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    metamodel
        Metamodel set (key of ``metamodel.sets``).
    n
        MDA train size (candidate from ``mda_candidates``).
    variable
        Stacked variable name (e.g. ``"hs"``, ``"dir"``).
    variance
        PCA variance fraction used when fitting.

    Returns
    -------
    Path
        ``outputs/goal_XX/{set}/pca_{n}/pca_{variable}_{variance}_model.pkl``.
    """
    base = _artefact_dir("pca_dir", goal, metamodel, n)
    return base / f"pca_{variable}_{_variance_tag(variance)}_model.pkl"


def set_sites(cases: xr.Dataset, metamodel: str) -> list[str]:
    """Sites of *cases* that belong to a metamodel set.

    Parameters
    ----------
    cases
        Dataset with a ``sites`` dimension (e.g. :func:`load_cases_goal`).
    metamodel
        Metamodel set; its ``sites`` selector (group, umbrella, site ids or
        ``null`` for all) is resolved with :func:`utils.geometry.resolve_sites`.

    Returns
    -------
    list[str]
        Site labels of the set present in *cases*.
    """
    return resolve_sites(cases, SETS[_check_set(metamodel)]["sites"])


# ---------------------------------------------------------------------------
# Cases & site filter
# ---------------------------------------------------------------------------


def detect_removed_sites(cases: xr.Dataset) -> list[str]:
    """Return site labels that fail ``METAMODEL.site_filter`` on SnapWave cases.

    See :func:`bluemath_tk.waves.hywaves.metamodel.detect_removed_sites`.
    """
    return hw.detect_removed_sites(cases, SITE_FILTER)


def drop_sites(cases: xr.Dataset, site_ids: Iterable[str]) -> xr.Dataset:
    """Return *cases* without the listed site labels."""
    return hw.drop_sites(cases, site_ids)


def removed_sites(goal: str | int) -> list[str]:
    """Return sites dropped at training time for one goal.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Returns
    -------
    list[str]
        Labels from ``split.json`` ``removed_sites``, or empty if missing.
    """
    if not split_path(goal).exists():
        return []
    with split_path(goal).open(encoding="utf-8") as f:
        return list(json.load(f).get("removed_sites", []))


def load_cases_goal(goal: str | int, *, apply_site_filter: bool = True) -> xr.Dataset:
    """SnapWave LHS output for one goal, with ``dir_u``/``dir_v`` and optional site filter.

    Parameters
    ----------
    goal
        Goal id (as in the case goals JSON).
    apply_site_filter
        If ``True``, drop sites listed in ``split.json`` ``removed_sites``.

    Returns
    -------
    xarray.Dataset
        Dimensions ``(case_num, sites)``; variables ``hs``, ``tp``, ``dir``, …
    """
    with xr.open_dataset(cases_path(goal)) as ds:
        out = hw.add_direction_components(ds).load()
    if apply_site_filter:
        out = drop_sites(out, removed_sites(goal))
    return out


def forcing_dataframe(cases: xr.Dataset) -> pd.DataFrame:
    """Build the MDA forcing table for each LHS case.

    Parameters
    ----------
    cases
        SnapWave LHS dataset with ``*_forcing`` variables.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``case_num``; columns match ``mda_inputs`` in config.
    """
    cols = forcing_columns()
    df = cases[list(cols.values())].to_dataframe()
    df.columns = MDA_INPUTS
    return df


def load_split(goal: str | int) -> tuple[np.ndarray, np.ndarray]:
    """Load train and test ``case_num`` indices from ``split.json``.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Returns
    -------
    train_idx, test_idx
        Integer arrays of case indices.
    """
    split = _read_split_dict(goal)
    return (
        np.asarray(split["train_idx"], dtype=int),
        np.asarray(split["test_idx"], dtype=int),
    )


def _read_split_dict(goal: str | int) -> dict[str, Any]:
    """Load the full ``split.json`` payload for one goal."""
    with split_path(goal).open(encoding="utf-8") as f:
        return json.load(f)


def _write_split_dict(goal: str | int, data: dict[str, Any]) -> None:
    """Write ``split.json`` for one goal."""
    path = split_path(goal)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def check_artefacts_model(goal: str | int) -> None:
    """Refuse a goal's fitted artefacts when another model produced its cases.

    ``split.json`` records ``simulations.model`` at MDA fit time (artefacts
    written before it did were all SnapWave).

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Raises
    ------
    ValueError
        If the artefacts were fitted on a different model's cases.
    """
    fitted = _read_split_dict(goal).get("model", "snapwave")
    if fitted != simulation_model():
        raise ValueError(
            f"Goal {goal}: outputs in {goal_output_dir(goal)} were fitted on "
            f"{fitted} cases but simulations.model is {simulation_model()!r}. "
            "Move them aside (snapshot the portal first) or switch the model back."
        )


def load_fit_summary(
    goal: str | int, metamodel: str, n: int | None = None
) -> dict[str, Any]:
    """Return persisted PCA/GP fit summary from ``split.json``.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    metamodel
        Metamodel set.
    n
        MDA train size. If ``None``, return every size of the set.

    Returns
    -------
    dict
        Fit summary for one MDA size, or all sizes when *n* is ``None``.
    """
    fit_by = _read_split_dict(goal).get("fit_by_metamodel", {}).get(metamodel, {})
    if n is None:
        return fit_by
    return fit_by.get(str(int(n)), {})


def _consolidated_pca_sites(
    kept_by_stack: dict[str, list[str]], stack: list[str]
) -> list[str]:
    """Intersection of kept sites across all stacked variables for one PCA target."""
    if not stack:
        return []
    sets = [set(kept_by_stack.get(v, [])) for v in stack]
    if not sets or not sets[0]:
        return kept_by_stack.get(stack[0], [])
    consolidated = sets[0]
    for other in sets[1:]:
        consolidated &= other
    return sorted(consolidated)


def _persist_fit_summary(
    goal: str | int, metamodel: str, n: int, summary: dict[str, Any]
) -> None:
    """Merge one set and MDA-size fit summary into ``split.json``."""
    data = _read_split_dict(goal)
    if "pca_nan_policy" not in data:
        value_replace, nan_drop = _pca_preprocessing()
        data["pca_nan_policy"] = {
            "value_to_replace_nans": value_replace,
            "nan_threshold_to_drop": nan_drop,
        }
    fit_by = data.setdefault("fit_by_metamodel", {}).setdefault(metamodel, {})
    fit_by[str(int(n))] = summary
    _write_split_dict(goal, data)


def report_training_data(
    goal: str | int,
    n: int | None = None,
) -> dict[str, Any]:
    """Summarise LHS training data quality (site filter, NaN fractions).

    Read-only helper for notebook diagnostics before or after fit. When *n* is
    given and ``split.json`` exists, reports stats on the first *n* MDA train
    cases only.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    n
        Optional MDA train size to subset train cases.

    Returns
    -------
    dict
        Keys include ``removed_sites``, ``nan_fraction``, ``pca_nan_policy``.
    """
    goal = str(goal)
    raw = load_cases_goal(goal, apply_site_filter=False)
    dropped = detect_removed_sites(raw)
    cases = drop_sites(raw, dropped)
    summary: dict[str, Any] = {
        "goal": goal,
        "n_cases_total": int(cases.sizes["case_num"]),
        "n_sites_after_site_filter": int(cases.sizes["sites"]),
        "removed_sites": dropped,
        "n_removed_sites": len(dropped),
    }
    if n is not None and split_path(goal).exists():
        train_idx, _ = load_split(goal)
        cases = cases.isel(case_num=train_idx[: int(n)])
        summary["n_cases_requested"] = int(n)
        summary["n_cases_used"] = int(cases.sizes["case_num"])

    nan_frac: dict[str, float] = {}
    for var in ("hs", "tp", "dir", "spr"):
        if var in cases:
            nan_frac[var] = float(np.mean(np.isnan(cases[var].values)))
    summary["nan_fraction"] = nan_frac

    value_replace, nan_drop = _pca_preprocessing()
    summary["pca_nan_policy"] = {
        "nan_threshold_to_drop": nan_drop,
        "value_to_replace_nans": value_replace,
    }
    logger.info(
        "Goal %s training data: %d cases, %d sites (%d removed by site_filter); "
        "NaN fractions %s",
        goal,
        summary["n_cases_total"],
        summary["n_sites_after_site_filter"],
        summary["n_removed_sites"],
        {k: f"{v:.3f}" for k, v in nan_frac.items()},
    )
    return summary


def load_context(goal: str | int) -> dict[str, Any]:
    """Saved MDA object and train/test case indices for holdout validation.

    Returns
    -------
    dict
        Keys ``mda``, ``train_idx``, ``test_idx``.

    Raises
    ------
    FileNotFoundError
        If MDA or split file is missing (run :func:`fit` first).
    """
    goal = str(goal)
    if not mda_path(goal).exists() or not split_path(goal).exists():
        raise FileNotFoundError(f"Missing MDA for goal {goal!r}. Run mm.fit() first.")
    train_idx, test_idx = load_split(goal)
    return {
        "mda": load_model(mda_path(goal)),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


def _missing_model_paths(
    goal: str | int, metamodel: str, n: int, *, pca_variance: PcaVariance = None
) -> list[Path]:
    """Return paths of missing GP/PCA model files for one goal, set and MDA size.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    metamodel
        Metamodel set.
    n
        MDA train size.
    pca_variance
        PCA variance override used when checking PCA artefact names.

    Returns
    -------
    list[Path]
        Missing pickle paths; empty when all models exist.
    """
    missing: list[Path] = []
    if not gp_model_path(goal, metamodel, n).exists():
        missing.append(gp_model_path(goal, metamodel, n))
    for name, variance, _ in _pca_variables(pca_variance):
        p = pca_model_path(goal, metamodel, n, name, variance)
        if not p.exists():
            missing.append(p)
    return missing


def models_exist(
    goal: str | int, metamodel: str, n: int, *, pca_variance: PcaVariance = None
) -> bool:
    """Check whether GP and all PCA pickles exist for one goal, set and MDA size.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    metamodel
        Metamodel set.
    n
        MDA train size.
    pca_variance
        PCA variance override used when checking PCA artefact names.

    Returns
    -------
    bool
        ``True`` when no model files are missing.
    """
    return not _missing_model_paths(goal, metamodel, n, pca_variance=pca_variance)


def require_models(
    goal: str | int, metamodel: str, n: int, *, pca_variance: PcaVariance = None
) -> None:
    """Raise :class:`ModelsNotFittedError` if any model file is missing.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    metamodel
        Metamodel set.
    n
        MDA train size.
    pca_variance
        PCA variance override used when checking PCA artefact names.

    Raises
    ------
    ModelsNotFittedError
        If the GP or any PCA pickle is absent.
    """
    missing = _missing_model_paths(goal, metamodel, n, pca_variance=pca_variance)
    if missing:
        raise ModelsNotFittedError(
            f"Models not fitted: goal={goal!r}, metamodel={metamodel!r}, n={n}, "
            f"pca_variance={resolved_pca_variances(pca_variance)}. Missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + ModelsNotFittedError._FIT_HINT
        )


def require_models_all(
    metamodel: str,
    n: int,
    goals: Iterable[str | int] | None = None,
    *,
    pca_variance: PcaVariance = None,
) -> None:
    """Raise :class:`ModelsNotFittedError` if any goal is missing fitted models.

    Parameters
    ----------
    metamodel
        Metamodel set.
    n
        MDA train size.
    goals
        Goals to check; default all configured goals.
    pca_variance
        PCA variance override used when checking PCA artefact names.

    Raises
    ------
    ModelsNotFittedError
        If any goal is missing GP or PCA pickles.
    """
    for goal in _goals(goals):
        require_models(goal, metamodel, n, pca_variance=pca_variance)


def load_models(
    goal: str | int,
    metamodel: str,
    n: int,
    *,
    pca_variance: PcaVariance = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Load fitted MDA, GP and per-variable PCA models from disk.

    Parameters
    ----------
    goal, metamodel, n
        Goal id, metamodel set and MDA train size (must match saved artefacts).
    pca_variance
        Override PCA variance fractions; default uses the case config JSON.
    device
        Optional PyTorch device for GP inference (e.g. ``"cpu"`` on small GPUs).

    Returns
    -------
    dict
        Keys ``mda``, ``gp``, ``pcas`` (name → PCA object), ``pca_variances``.

    Raises
    ------
    ModelsNotFittedError
        If any pickle is missing.
    """
    require_models(goal, metamodel, n, pca_variance=pca_variance)
    check_artefacts_model(goal)
    pcas = {
        name: load_model(pca_model_path(goal, metamodel, n, name, var))
        for name, var, _ in _pca_variables(pca_variance)
    }
    gp = load_model(gp_model_path(goal, metamodel, n))
    if device is not None:
        _set_gp_device(gp, device)
    return {
        "mda": load_model(mda_path(goal)),
        "gp": gp,
        "pcas": pcas,
        "pca_variances": resolved_pca_variances(pca_variance),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _goals(goals: Iterable[str | int] | None = None) -> list[str]:
    """Return active goal ids.

    Parameters
    ----------
    goals
        Explicit goal ids; ``None`` uses all keys in the case goals JSON.

    Returns
    -------
    list[str]
        Goal ids as strings.
    """
    if goals is None:
        return list(load_goals().keys())
    return [str(g) for g in goals]


@lru_cache(maxsize=1)
def _goal_angles_lookup() -> dict[str, tuple[float, float]]:
    """Goal sector bounds in nautical clockwise ``[left, right]``.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping ``goal_id -> (left, right)`` from the case goals JSON.

    Raises
    ------
    ValueError
        If any goal stores invalid bounds (``left >= right``).
    """
    out: dict[str, tuple[float, float]] = {}
    for key, item in load_goals().items():
        left, right = float(item["angles"][0]), float(item["angles"][1])
        if left >= right:
            raise ValueError(
                f"Goal {key!r} has invalid nautical sector bounds: {left} >= {right}"
            )
        out[str(key)] = (left, right)
    return out


def _goal_sector(goal: str | int) -> tuple[float, float]:
    """Return nautical ``(left, right)`` bounds for one goal."""
    return _goal_angles_lookup()[str(goal)]


def _fit_mda(
    goal: str | int, *, force_retrain: bool = False
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit or load MDA centroids and persist train/test split for one goal.

    Applies ``site_filter``, fits MDA on offshore forcing columns, and writes
    ``mda.pkl`` plus ``split.json`` (train indices = centroid cases, rest = test).

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    force_retrain
        If ``True``, re-fit even when artefacts already exist.

    Returns
    -------
    mda
        Fitted :class:`~bluemath_tk.datamining.mda.MDA` object.
    train_idx, test_idx
        Integer arrays of MDA centroid and holdout case indices.
    """
    goal = str(goal)
    path, split_file = mda_path(goal), split_path(goal)

    if not force_retrain and path.exists() and split_file.exists():
        mda = load_model(path)
        missing = set(MDA_INPUTS) - set(mda.centroids.columns)
        if missing:
            raise ValueError(
                f"Goal {goal}: saved MDA ({path}) lacks mda_inputs "
                f"{sorted(missing)} (fitted on an older configuration). "
                "Re-fit with force_retrain=True (or remove outputs/goal_*)."
            )
        check_artefacts_model(goal)
        train_idx, test_idx = load_split(goal)
        logger.info(
            "Goal %s: loaded MDA (%d train, %d test)",
            goal,
            len(train_idx),
            len(test_idx),
        )
        return mda, train_idx, test_idx

    raw = load_cases_goal(goal, apply_site_filter=False)
    dropped = detect_removed_sites(raw)
    cases = drop_sites(raw, dropped)
    if dropped:
        logger.warning(
            "Goal %s: removed %d site(s) (site_filter=%s): %s",
            goal,
            len(dropped),
            SITE_FILTER,
            ", ".join(dropped),
        )

    num_centers = mda_max_points()
    n_cases = int(cases.sizes["case_num"])
    if num_centers >= n_cases:
        raise ValueError(
            f"Goal {goal}: largest mda_candidates ({num_centers}) must be below the "
            f"number of SnapWave cases ({n_cases}) to leave holdout cases; raise "
            "simulations.num_samples_per_goal or lower mda_candidates."
        )
    mda, train_idx, test_idx = hw.fit_mda(
        forcing_dataframe(cases), num_centers, MDA_DIRECTIONAL
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    mda.save_model(path)
    value_replace, nan_drop = _pca_preprocessing()
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": simulation_model(),
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "num_centers": num_centers,
                "removed_sites": dropped,
                "site_filter": SITE_FILTER,
                "pca_nan_policy": {
                    "value_to_replace_nans": value_replace,
                    "nan_threshold_to_drop": nan_drop,
                },
                "fit_by_metamodel": {},
            },
            f,
            indent=2,
        )
    logger.info(
        "Goal %s: MDA fitted (%d train, %d test)", goal, len(train_idx), len(test_idx)
    )
    return mda, train_idx, test_idx


def _fit_models(
    goal: str | int,
    metamodel: str,
    n: int,
    *,
    mda: Any | None = None,
    train_idx: np.ndarray | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
    pca_variance: PcaVariance = None,
) -> dict[str, Any]:
    """Fit or load PCA + GP models for one goal, metamodel set and MDA size *n*.

    Uses the first *n* MDA train cases and the sites of the set. Each variable
    in ``vars_to_predict`` (except ``raw``) gets a PCA on stacked site fields;
    all PC blocks are concatenated and regressed with an exact GP on MDA
    centroids.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    metamodel
        Metamodel set.
    n
        Number of MDA centroids (must be in ``mda_candidates``).
    mda, train_idx
        Optional pre-fitted MDA context from :func:`_fit_mda`.
    force_retrain
        Re-fit even when pickles already exist.
    gp_epochs
        GP optimisation iterations.
    pca_variance
        Override per-variable PCA variance; default from the case config JSON.

    Returns
    -------
    dict
        Keys ``mda``, ``gp``, ``pcas``, ``pca_variances``, ``train_idx``.
    """
    goal, n = str(goal), int(n)
    if n not in MDA_CANDIDATES:
        raise ValueError(f"{n} not in mda_candidates")
    if n > mda_max_points():
        raise ValueError(f"n={n} exceeds mda_max_points={mda_max_points()}")

    if not force_retrain and models_exist(
        goal, metamodel, n, pca_variance=pca_variance
    ):
        logger.info("Goal %s / %s / n=%d: loaded models", goal, metamodel, n)
        out = load_models(goal, metamodel, n, pca_variance=pca_variance)
        out["train_idx"] = train_idx
        return out

    if mda is None or train_idx is None:
        mda, train_idx, _ = _fit_mda(goal, force_retrain=force_retrain)

    cases = load_cases_goal(goal).isel(case_num=train_idx[:n])
    cases = cases.sel(sites=set_sites(cases, metamodel))
    value_replace, nan_drop = _pca_preprocessing()
    n_requested = n
    n_cases = int(cases.sizes["case_num"])
    n_sites = int(cases.sizes["sites"])
    if n_cases < n_requested:
        logger.warning(
            "Goal %s / n=%d: only %d training cases available",
            goal,
            n_requested,
            n_cases,
        )
    targets = _pca_variables(pca_variance)
    logger.info(
        "Goal %s / %s / n=%d: fitting PCA %s (%d cases × %d sites)",
        goal,
        metamodel,
        n,
        [name for name, _, _ in targets],
        n_cases,
        n_sites,
    )
    pcas, pca_summary = hw.fit_pcas(cases, targets, value_replace, nan_drop)
    for name, variance, _ in targets:
        logger.info(
            "Goal %s / n=%d: PCA %s kept %d/%d sites (%d dropped by nan threshold)",
            goal,
            n,
            name,
            n_sites - pca_summary["pca_sites_dropped"][name],
            n_sites,
            pca_summary["pca_sites_dropped"][name],
        )
        path = pca_model_path(goal, metamodel, n, name, variance)
        path.parent.mkdir(parents=True, exist_ok=True)
        pcas[name].save_model(path)

    forcing = forcing_dataframe(cases)
    centroids = mda.centroids[MDA_INPUTS].iloc[:n]
    if forcing.shape != centroids.shape:
        raise ValueError(
            f"GP training: forcing shape {forcing.shape} != centroids {centroids.shape}"
        )
    if not np.allclose(forcing.values, centroids.values, rtol=1e-5, atol=1e-8):
        diff = float(np.nanmax(np.abs(forcing.values - centroids.values)))
        raise ValueError(f"GP training: forcing != MDA centroids (max |Δ|={diff:.4g})")

    gp = hw.fit_gp(centroids, pcas, MDA_DIRECTIONAL, epochs=gp_epochs)
    n_components = int(hw.stack_pcs(pcas).shape[1])
    gp_p = gp_model_path(goal, metamodel, n)
    gp_p.parent.mkdir(parents=True, exist_ok=True)
    gp.save_model(gp_p)
    logger.info(
        "Goal %s / %s / n=%d: GP saved (%d PCs)", goal, metamodel, n, n_components
    )

    _persist_fit_summary(
        goal,
        metamodel,
        n,
        {
            "n_cases_requested": n_requested,
            "n_cases_used": n_cases,
            "n_sites_before_pca": n_sites,
            **pca_summary,
            "pca_variances": resolved_pca_variances(pca_variance),
            "n_gp_components": n_components,
        },
    )

    return {
        "mda": mda,
        "gp": gp,
        "pcas": pcas,
        "pca_variances": resolved_pca_variances(pca_variance),
        "train_idx": train_idx,
    }


def fit(
    goal: str | int,
    *,
    metamodels: Iterable[str] | None = None,
    candidates: Iterable[int] | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
    pca_variance: PcaVariance = None,
) -> None:
    """Train the MDA of one goal, then PCA and GP models per set and MDA size.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    metamodels
        Metamodel sets to fit; default every set in ``metamodel.sets``.
    candidates
        MDA train sizes to fit; default ``mda_candidates`` from config.
    force_retrain
        Re-fit even when artefacts already exist.
    gp_epochs
        GP optimisation iterations.
    pca_variance
        Override per-variable PCA variance; default from the case config JSON.
    """
    goal = str(goal)
    validate_metamodel_config()
    mda, train_idx, _ = _fit_mda(goal, force_retrain=force_retrain)
    for metamodel in metamodels or SETS:
        for n in candidates or MDA_CANDIDATES:
            _fit_models(
                goal,
                metamodel,
                int(n),
                mda=mda,
                train_idx=train_idx,
                force_retrain=force_retrain,
                gp_epochs=gp_epochs,
                pca_variance=pca_variance,
            )


def fit_all(
    goals: Iterable[str | int] | None = None,
    *,
    metamodels: Iterable[str] | None = None,
    candidates: Iterable[int] | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
    pca_variance: PcaVariance = None,
) -> None:
    """Train models for every goal via :func:`fit`.

    Parameters
    ----------
    goals
        Goals to fit; default all configured goals.
    metamodels
        Metamodel sets to fit; default every set in ``metamodel.sets``.
    candidates
        MDA train sizes to fit; default ``mda_candidates`` from config.
    force_retrain
        Re-fit even when artefacts already exist.
    gp_epochs
        GP optimisation iterations.
    pca_variance
        Override per-variable PCA variance; default from the case config JSON.
    """
    for goal in _goals(goals):
        fit(
            goal,
            metamodels=metamodels,
            candidates=candidates,
            force_retrain=force_retrain,
            gp_epochs=gp_epochs,
            pca_variance=pca_variance,
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict(
    goal: str | int,
    n: int,
    forcing: pd.DataFrame,
    *,
    metamodel: str,
    models: dict[str, Any] | None = None,
    pca_variance: PcaVariance = None,
    sites: Iterable[str] | None = None,
    row_dim: str = "case_num",
    device: str | None = None,
) -> xr.Dataset:
    """
    Map offshore forcing rows to nearshore wave fields via GP → PCA inverse.

    Pipeline: GP predicts stacked PCs from MDA inputs → each PCA inverts to
    spatial fields → ``dir`` is recovered from ``dir_u``/``dir_v``. Variables
    marked ``raw`` in config (e.g. ``tp``) are passed through unchanged.

    Parameters
    ----------
    goal, n
        Goal id and MDA train size used at fitting time.
    forcing
        DataFrame indexed by case/time row, with the ``mda_inputs`` columns
        from config (extra columns are ignored). Index is renamed internally
        to ``case_num`` for PCA compatibility.
    metamodel
        Metamodel set.
    models
        Pre-loaded dict from :func:`load_models`; loaded from disk if omitted.
    pca_variance
        Must match the variances used when models were fitted.
    sites
        Subset of site labels; default all sites of the set.
    row_dim
        Name of the output row dimension (``case_num`` for LHS holdout,
        ``time`` for hindcast).
    device
        Optional PyTorch device for GP inference when models are loaded here.

    Returns
    -------
    xarray.Dataset
        Variables ``hs``, ``tp``, ``dir``, ``spr``, … with dims
        ``(row_dim, sites)`` plus ``SITE_META`` coordinates.

    Raises
    ------
    ModelsNotFittedError
        If models must be loaded and pickles are missing.
    ValueError
        If no requested *sites* exist for the goal.
    """

    goal = str(goal)
    if models is None:
        models = load_models(
            goal, metamodel, n, pca_variance=pca_variance, device=device
        )

    cases = load_cases_goal(goal)
    all_sites = set_sites(cases, metamodel)
    site_ids = all_sites if sites is None else [s for s in sites if s in all_sites]
    if not site_ids:
        raise ValueError(
            f"No requested sites found for goal {goal!r} in set {metamodel!r}"
        )

    fields = hw.predict_fields(
        models["gp"], models["pcas"], forcing[MDA_INPUTS], VARS, site_ids
    )
    meta = cases.sel(sites=site_ids)
    ds = fields.assign_coords(
        {name: meta[name] for name in SITE_META if name in meta.coords}
    )
    if row_dim != "case_num":
        ds = ds.rename(case_num=row_dim)  # case_num → time for hindcast output

    return ds


# ---------------------------------------------------------------------------
# Hindcast: sites & offshore forcing
# ---------------------------------------------------------------------------


def common_model_sites(goals: Iterable[str | int] | None = None) -> list[str]:
    """Return site labels present in every goal after ``site_filter``.

    Parameters
    ----------
    goals
        Goals to intersect; default all configured goals.

    Returns
    -------
    list[str]
        Sorted intersection of site labels across goals.
    """
    active = _goals(goals)
    common = set(str(s) for s in load_cases_goal(active[0]).sites.values)
    for goal in active[1:]:
        common &= {str(s) for s in load_cases_goal(goal).sites.values}
    return sorted(common)


def resolve_reconstruction_sites(
    ds: xr.Dataset,
    sites: str | Iterable[str] | None = None,
    *,
    metamodel: str,
    goals: Iterable[str | int] | None = None,
) -> list[str]:
    """Site labels valid for multi-goal hindcast reconstruction with one set.

    Intersects user *sites* selection (via :func:`resolve_sites`) with the
    sites of the metamodel set and :func:`common_model_sites`, so every
    retained site exists in all goals.

    Parameters
    ----------
    ds
        Reference dataset for site metadata (typically one goal's cases).
    sites
        Group name, umbrella (``buoy``/``isobath``/``mesh``), site id, or list;
        ``None`` returns all common sites of the set.
    metamodel
        Metamodel set.
    goals
        Goals to intersect; default all configured goals.

    Returns
    -------
    list[str]
        Sorted site labels.

    Raises
    ------
    ValueError
        If the intersection is empty.
    """
    common = set(common_model_sites(goals))
    common &= set(set_sites(load_cases_goal(_goals(goals)[0]), metamodel))
    if sites is None:
        return sorted(common)

    selected = resolve_sites(ds, sites)
    kept = [s for s in selected if s in common]
    dropped = [s for s in selected if s not in common]
    if dropped:
        logger.warning(
            "Dropped %d site(s) not in all goals or not in set %r: %s",
            len(dropped),
            metamodel,
            ", ".join(dropped[:8]) + ("…" if len(dropped) > 8 else ""),
        )
    if not kept:
        raise ValueError(
            "No sites left after intersecting with common model sites. "
            f"common count={len(common)}."
        )
    return kept


def _reconstruction_partitions(spectral: bool, offshore: xr.Dataset) -> list[int]:
    """Partition ids for bulk (``[-1]``) or spectral (``[0, 1, 2, 3]``) mode.

    See :func:`bluemath_tk.waves.hywaves.reconstruction.partition_ids`.
    """
    return hwr.partition_ids(offshore, spectral)


def _valid_timesteps(
    forcing_df: pd.DataFrame, *, goal: str | int | None = None
) -> pd.Series:
    """Usable hindcast time steps; restricted to the goal's sector when *goal* is given.

    See :func:`bluemath_tk.waves.hywaves.metamodel.valid_timesteps`.
    """
    return hw.valid_timesteps(
        forcing_df, sector=_goal_sector(goal) if goal is not None else None
    )


def _offshore_forcing(
    offshore: xr.Dataset,
    goal: str | int,
    *,
    partition: int,
    wl: WaterLevel = 0.0,
) -> pd.DataFrame:
    """MDA forcing (``hs``, ``tp``, ``dir``, ``spr``, ``wl``) of one goal and partition.

    ``wl`` is a constant (m) or ``"forcing"``: the ``forcing.water_level``
    series at the goal (see :func:`utils.config.goal_water_level`). See
    :func:`bluemath_tk.waves.hywaves.metamodel.goal_forcing`.
    """
    if not isinstance(wl, str):
        return hw.goal_forcing(offshore.sel(goal=int(goal)), partition, wl)
    if wl != WL_FROM_FORCING:
        raise ValueError(f"wl must be a number or {WL_FROM_FORCING!r}, got {wl!r}")
    df = hw.goal_forcing(offshore.sel(goal=int(goal)), partition)
    df["wl"] = goal_water_level(goal, df.index).to_numpy()
    return df


def _predict_offshore(
    offshore: xr.Dataset,
    goal: str | int,
    n: int,
    site_ids: list[str],
    *,
    metamodel: str,
    partition: int,
    wl: WaterLevel = 0.0,
    pca_variance: PcaVariance = None,
    models: dict[str, Any] | None = None,
    device: str | None = None,
    wind_sea: bool = False,
) -> xr.Dataset | None:
    """Predict nearshore bulk parameters for one goal, partition and time slice.

    Parameters
    ----------
    offshore
        Offshore dataset already selected on ``time``.
    goal
        Goal id from the case goals JSON.
    n
        MDA train size used at fitting time.
    site_ids
        Nearshore site labels to predict.
    metamodel
        Metamodel set.
    partition
        WHACS partition id (bulk ``-1`` or spectral ``0``–``3``).
    wl
        Water level (m) for the MDA forcing, or ``"forcing"`` for the
        ``forcing.water_level`` series at each goal.
    pca_variance
        PCA variance override; must match fitted models.
    models
        Pre-loaded model dict from :func:`load_models`; loaded if omitted.
    device
        Optional PyTorch device for GP inference when models are loaded here.
    wind_sea
        Also return, from the offshore forcing, the wind-sea weight
        ``ws_weight`` and the inferred wind ``u10`` / ``u10dir`` on ``time``
        (for the wind correction, see :mod:`utils.wind`).

    Returns
    -------
    xarray.Dataset or None
        Predicted ``hs``, ``tp``, ``dir``, ``spr`` on ``(time, sites)``, or
        ``None`` when no valid timesteps remain.
    """
    forcing_df = _offshore_forcing(offshore, goal, partition=partition, wl=wl)
    valid = _valid_timesteps(forcing_df, goal=goal)
    if not valid.any():
        return None

    pred = predict(
        goal,
        n,
        forcing_df.loc[valid],
        metamodel=metamodel,
        models=models,
        pca_variance=pca_variance,
        sites=site_ids,
        row_dim="time",
        device=device,
    )
    if not wind_sea:
        return pred

    from bluemath_tk.waves.hywaves.wind_correction import offshore_wind_sea

    from utils.wind import config as wind_config

    cfg = wind_config()
    ws = offshore_wind_sea(
        forcing_df.loc[valid],
        steepness_range=tuple(cfg["steepness_range"]),
        spr_range=tuple(cfg["spr_range"]),
        speed_factor=float(cfg["speed_factor"]),
    )
    return pred.assign({name: ("time", ws[name].to_numpy()) for name in ws})


def _wavespectra_sum(
    cube: xr.Dataset,
    site_ids: list[str],
    *,
    variables: Mapping[str, str] | None = None,
    save_spectra: bool = False,
    load: bool = True,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Linearly sum goal × partition spectra into bulk parameters.

    *variables* defaults to ``metamodel.reconstruction.variables`` from the case
    JSON; see :func:`bluemath_tk.waves.hywaves.reconstruction.sum_partition_spectra`.
    """
    return hwr.sum_partition_spectra(
        cube,
        site_ids,
        _reconstruction_variables(override=variables),
        save_spectra=save_spectra,
        load=load,
    )


def _sum_chunks(
    chunks: list[xr.Dataset],
    site_ids: list[str],
    *,
    n_hours: int,
    n_parts: int,
    variable_map: Mapping[str, str],
    save_spectra: bool,
    workers: int,
    load: bool = True,
    wind_correction: Mapping[str, Any] | None = None,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Site-batched summation within this host's memory budget, optionally parallel.

    See :func:`bluemath_tk.waves.hywaves.reconstruction.sum_partition_spectra_batched`;
    *wind_correction* is :func:`utils.wind.reconstruction_inputs`.
    """
    return hwr.sum_partition_spectra_batched(
        chunks,
        site_ids,
        n_hours,
        n_parts,
        variable_map,
        save_spectra=save_spectra,
        budget_bytes=WAVESPECTRA_MEMORY_BUDGET_BYTES,
        workers=workers,
        load=load,
        log_file=str(resolve("logs/metamodel.log")),
        wind_correction=wind_correction,
    )


# ---------------------------------------------------------------------------
# Hindcast cache
# ---------------------------------------------------------------------------


def reconstruction_cache_dir() -> Path:
    """Return the directory used for cached hindcast reconstructions.

    Creates ``tmp/reconstruct`` in the current case if it does not exist.

    Returns
    -------
    Path
        Absolute path to the reconstruction cache folder.
    """
    path = resolve("tmp/reconstruct")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _time_label_for_cache(time: str | slice) -> str:
    """Encode a time selector as a filesystem-safe label.

    Parameters
    ----------
    time
        xarray time selector (month string, year, or slice).

    Returns
    -------
    str
        Sanitized label used in cache filenames.
    """
    if isinstance(time, str):
        return time.replace(":", "-")
    if isinstance(time, slice):

        def _fmt(value: Any) -> str:
            """Format one slice bound as a compact timestamp, or ``""`` if unset."""
            if value is None:
                return ""
            return pd.Timestamp(value).strftime("%Y%m%dT%H%M")

        return f"{_fmt(time.start)}_to_{_fmt(time.stop)}"
    return str(time).replace(":", "-")


def _metamodel_cache_label() -> str:
    """Short hash of site-filter and PCA preprocessing settings for cache filenames."""
    payload = {
        "site_filter": METAMODEL.get("site_filter", {}),
        "pca_preprocessing": METAMODEL.get("pca_preprocessing", {}),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]
    return digest


def _pca_variance_label(pca_variance: PcaVariance) -> str:
    """Encode PCA variance settings for cache filenames.

    Uses :func:`resolved_pca_variances` so every stacked variable from the case
    config appears in the label, not only keys present in a partial override dict.

    Parameters
    ----------
    pca_variance
        PCA variance override passed to reconstruction.

    Returns
    -------
    str
        Compact, order-stable label.
    """
    if pca_variance is None:
        return "default"
    resolved = resolved_pca_variances(pca_variance)
    return "_".join(f"{key}{value:g}" for key, value in sorted(resolved.items()))


def _goals_label(goals: Iterable[str | int] | None) -> str:
    """Encode goal selection for cache filenames.

    Parameters
    ----------
    goals
        Goal ids passed to reconstruction; ``None`` means all goals.

    Returns
    -------
    str
        ``"all"`` or a hyphen-separated goal list.
    """
    if goals is None:
        return "all"
    return "-".join(str(goal) for goal in _goals(goals))


def _wind_label() -> str:
    """Cache-label token of the wind correction (model and wind source)."""
    from utils.wind import config as wind_config

    cfg = wind_config()
    return f"wind-{simulation_model()}-{cfg['wind']}-{cfg['added_tp']}__"


def _wl_label(wl: WaterLevel) -> str:
    """Cache-label token for a constant water level or the forcing series."""
    return wl if isinstance(wl, str) else f"{wl:g}"


def _sanitize_cache_stem(stem: str, *, max_len: int = 220) -> str:
    """Sanitize and optionally truncate a cache filename stem.

    Parameters
    ----------
    stem
        Raw filename stem built from reconstruction parameters.
    max_len
        Maximum allowed stem length before appending a hash suffix.

    Returns
    -------
    str
        Filesystem-safe stem.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(stem.encode()).hexdigest()[:10]
    return f"{safe[: max_len - 11]}_{digest}"


def _variables_label(variables: Mapping[str, str]) -> str:
    """Encode HyGloW → wavespectra mapping for cache filenames."""
    return "-".join(f"{k}-{v}" for k, v in sorted(variables.items()))


def reconstruction_spectra_cache_path(bulk_cache_path: Path) -> Path:
    """Return the spectra cache path paired with one bulk reconstruction file.

    Parameters
    ----------
    bulk_cache_path
        Path from :func:`reconstruction_cache_path` for the bulk variables.

    Returns
    -------
    Path
        ``{bulk_stem}__spectra.nc`` alongside the bulk cache file.
    """
    return bulk_cache_path.with_name(f"{bulk_cache_path.stem}__spectra.nc")


def reconstruction_cache_path(
    time_label: str | slice,
    n: int,
    *,
    metamodel: str,
    spectral: bool,
    offshore_name: str,
    goals: Iterable[str | int] | None,
    wl: WaterLevel,
    site_ids: list[str],
    pca_variance: PcaVariance,
    variables: Mapping[str, str] | None = None,
    wind_correction: bool = False,
) -> Path:
    """Build the NetCDF cache path for one reconstruction request.

    The filename encodes the numerical model, offshore source, metamodel set, MDA size, spectral
    mode, goals, water level, resolved site list, output variable mapping, PCA
    settings, metamodel config fingerprint (``site_filter`` +
    ``pca_preprocessing``), and time span.

    Parameters
    ----------
    time_label
        Time selector for this cache entry (month or full period).
    n
        MDA train size.
    metamodel
        Metamodel set.
    spectral
        ``True`` for spectral partition sum; ``False`` for bulk only.
    offshore_name
        Processed hindcast key in the case config JSON.
    goals
        Goals included in reconstruction.
    wl
        Water level (m) for the MDA forcing, or ``"forcing"`` for the
        ``forcing.water_level`` series at each goal.
    site_ids
        Resolved nearshore site labels.
    pca_variance
        PCA variance override used for prediction.
    variables
        HyGloW name → wavespectra stat mapping. When omitted, uses the case JSON
        ``metamodel.reconstruction.variables``.
    wind_correction
        Whether the local wind sea is added (labelled with the model and the
        wind source, see :mod:`utils.wind`).

    Returns
    -------
    Path
        Target ``.nc`` path under ``tmp/reconstruct`` (not created on disk).
    """
    var_map = _reconstruction_variables(override=variables)
    stem = (
        "reconstruct-v8__"  # v8: model in the key (v7: hs input, metamodel sets)
        f"reconstruct__model-{simulation_model()}__"
        f"offshore-{offshore_name}__set-{metamodel}__n-{n}__"
        f"{'spectral' if spectral else 'bulk'}__"
        f"goals-{_goals_label(goals)}__wl-{_wl_label(wl)}__"
        f"{_wind_label() if wind_correction else ''}"
        f"sites-{'-'.join(sorted(site_ids))}__"
        f"vars-{_variables_label(var_map)}__"
        f"pca-{_pca_variance_label(pca_variance)}__"
        f"cfg-{_metamodel_cache_label()}__"
        f"time-{_time_label_for_cache(time_label)}"
    )
    return resolve("tmp/reconstruct") / f"{_sanitize_cache_stem(stem)}.nc"


def load_reconstruction_spectra(
    time_label: str | slice,
    n: int,
    *,
    metamodel: str,
    spectral: bool,
    offshore_name: str,
    goals: Iterable[str | int] | None = None,
    wl: WaterLevel = 0.0,
    site_ids: list[str],
    pca_variance: PcaVariance,
    variables: Mapping[str, str] | None = None,
    load: bool = True,
    wind_correction: bool | None = None,
) -> xr.Dataset | None:
    """Load cached summed spectra written when ``save_spectra`` was enabled.

    Parameters
    ----------
    time_label
        Same time selector used for the paired bulk reconstruction.
    n
        MDA train size.
    metamodel
        Metamodel set used for reconstruction.
    spectral
        Spectral or bulk partition mode used for reconstruction.
    offshore_name
        Processed hindcast key in the case config JSON.
    goals
        Goals included in reconstruction.
    wl
        Water level (m) for the MDA forcing, or ``"forcing"`` for the
        ``forcing.water_level`` series at each goal.
    site_ids
        Resolved nearshore site labels.
    pca_variance
        PCA variance override used at fit time.
    variables
        HyGloW → wavespectra mapping; must match the bulk cache entry. When
        omitted, uses ``metamodel.reconstruction.variables``.
    load
        Eager-load arrays into memory.
    wind_correction
        Same as :func:`reconstruct` (default ``metamodel.wind_correction.enabled``).

    Returns
    -------
    xarray.Dataset or None
        ``efth`` on ``(time, sites, freq, dir)``, or ``None`` if no spectra
        cache exists for the request.
    """
    bulk_path = reconstruction_cache_path(
        time_label,
        n,
        metamodel=metamodel,
        spectral=spectral,
        offshore_name=offshore_name,
        goals=goals,
        wl=wl,
        site_ids=site_ids,
        pca_variance=pca_variance,
        variables=variables,
        wind_correction=_wind_correction_on(wind_correction),
    )
    return _load_reconstruction_cache(
        reconstruction_spectra_cache_path(bulk_path), load=load
    )


def _wind_correction_on(wind_correction: bool | None) -> bool:
    """*wind_correction*, or ``metamodel.wind_correction.enabled`` when None."""
    if wind_correction is not None:
        return bool(wind_correction)
    from utils.wind import enabled

    return enabled()


def _load_reconstruction_cache(path: Path, *, load: bool) -> xr.Dataset | None:
    """Load one reconstruction cache file if present.

    Parameters
    ----------
    path
        NetCDF cache path from :func:`reconstruction_cache_path`.
    load
        Eager-load arrays into memory.

    Returns
    -------
    xarray.Dataset or None
        Cached dataset, or ``None`` if *path* does not exist.
    """
    if not path.is_file():
        return None
    logger.info("Loading reconstruction cache: %s", path.name)
    with xr.open_dataset(path) as ds:
        data = ds.load()
    return data


def _save_reconstruction_cache(ds: xr.Dataset, path: Path) -> None:
    """Write one reconstruction result to the cache.

    Parameters
    ----------
    ds
        Reconstructed dataset on ``(time, sites)``.
    path
        Target NetCDF path from :func:`reconstruction_cache_path`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    logger.info(
        "Saved reconstruction cache: %s (%.2f MB)",
        path.name,
        path.stat().st_size * 1e-6,
    )


def _month_periods(time: str | slice, offshore: xr.Dataset) -> list[str | slice]:
    """Split a time selector into calendar months when needed.

    Parameters
    ----------
    time
        User time selector passed to :func:`reconstruct`.
    offshore
        Offshore dataset used only to inspect available timesteps.

    Returns
    -------
    list
        Single-element list for one month; otherwise month strings such as
        ``"2020-05"``.
    """
    subset = offshore.sel(time=time)
    if subset.sizes.get("time", 0) <= 1:
        return [time]
    months = pd.DatetimeIndex(subset.time.values).to_period("M").unique()
    if len(months) <= 1:
        return [time]
    return [str(period) for period in months]


def _reconstruct_period(
    offshore: xr.Dataset,
    n: int,
    *,
    metamodel: str,
    time_label: str | slice,
    active: list[str],
    site_ids: list[str],
    spectral: bool = True,
    offshore_name: str = "WHACS",
    pca_variance: PcaVariance = None,
    wl: WaterLevel = 0.0,
    load: bool = True,
    models_by_goal: dict[str, dict[str, Any]] | None = None,
    cache_path: Path | None = None,
    recompute: bool = False,
    use_cache: bool = True,
    device: str | None = None,
    save_spectra: bool | None = None,
    variables: Mapping[str, str] | None = None,
    workers: int = 1,
    wind_correction: bool = False,
) -> xr.Dataset:
    """Reconstruct one offshore time slice already selected with ``.sel(time=...)``.

    Parameters
    ----------
    offshore
        Offshore forcing for the period to reconstruct.
    n
        MDA train size.
    metamodel
        Metamodel set.
    time_label
        Time label used for logging and cache lookup.
    active
        Goal ids included in reconstruction.
    site_ids
        Resolved nearshore site labels (shared across goals).
    spectral
        ``True`` for WHACS partitions ``0``–``3``; ``False`` for bulk only.
    offshore_name
        Processed hindcast key in the case config JSON.
    pca_variance
        PCA variance override used for prediction.
    wl
        Water level (m) for the MDA forcing, or ``"forcing"`` for the
        ``forcing.water_level`` series at each goal.
    load
        Eager-load the output dataset.
    models_by_goal
        Pre-loaded models per goal. When omitted, each goal is loaded only for
        its own predictions and released before the next (saves GPU memory).
    cache_path
        Optional explicit cache path; built from parameters if omitted.
    recompute
        If ``True``, ignore any existing cache file.
    use_cache
        If ``False``, skip cache read and write.
    device
        Optional PyTorch device for GP inference (``"cpu"`` on small GPUs).
    save_spectra
        Cache full summed spectra alongside bulk stats. When ``None``, reads
        ``metamodel.reconstruction.save_spectra`` from the case JSON.
    variables
        HyGloW name → wavespectra stat mapping. When ``None``, reads
        ``metamodel.reconstruction.variables`` from the case JSON.
    workers
        Process-pool workers for the site-batched merge + _wavespectra_sum
        step (see :func:`_sum_chunks`). ``1`` (default) runs in this
        process. Only that pure-CPU step is parallelized; the goal loop above
        it (GP inference, often on GPU) is not.
    wind_correction
        Add the local wind sea (see :mod:`utils.wind`).

    Returns
    -------
    xarray.Dataset
        Bulk parameters on ``(time, sites)`` for the mapped HyGloW variables.

    Raises
    ------
    ValueError
        If no goal × partition produced valid predictions.
    """
    if cache_path is None:
        cache_path = reconstruction_cache_path(
            time_label,
            n,
            metamodel=metamodel,
            spectral=spectral,
            offshore_name=offshore_name,
            goals=active,
            wl=wl,
            site_ids=site_ids,
            pca_variance=pca_variance,
            variables=variables,
            wind_correction=wind_correction,
        )
    variable_map = _reconstruction_variables(override=variables)
    save_spectra = _reconstruction_save_spectra(override=save_spectra)
    spectra_cache_path = reconstruction_spectra_cache_path(cache_path)
    part_ids = _reconstruction_partitions(spectral, offshore)

    if use_cache and not recompute:
        cached = _load_reconstruction_cache(cache_path, load=load)
        if cached is not None:
            if save_spectra and not spectra_cache_path.is_file():
                logger.info(
                    "Bulk cache hit but spectra cache missing — recomputing %s",
                    spectra_cache_path.name,
                )
            else:
                return cached

    logger.info(
        "Reconstruct time=%r n=%d goals=%s partitions=%s sites=%d",
        time_label,
        n,
        active,
        part_ids,
        len(site_ids),
    )

    chunks: list[xr.Dataset] = []
    for goal in active:
        lazy_load = models_by_goal is None
        goal_models = (
            models_by_goal[str(goal)]
            if models_by_goal is not None
            else load_models(
                goal, metamodel, n, pca_variance=pca_variance, device=device
            )
        )
        try:
            for part in part_ids:
                pred = _predict_offshore(
                    offshore,
                    goal,
                    n,
                    site_ids,
                    metamodel=metamodel,
                    partition=part,
                    wl=wl,
                    pca_variance=pca_variance,
                    models=goal_models,
                    device=device,
                    wind_sea=wind_correction,
                )
                if pred is None:
                    logger.debug(
                        "Goal %s partition %s: no valid timesteps in %r",
                        goal,
                        part,
                        time_label,
                    )
                    continue
                chunks.append(
                    pred.expand_dims(goal=[int(goal)], partition=[part]).load()
                )
        finally:
            if lazy_load:
                del goal_models
                _release_gp_memory()

    if not chunks:
        raise ValueError(f"Reconstruction produced no data for {time_label!r}")

    # See _site_batch_size's docstring: xr.merge and, above all, wavespectra's
    # own broadcast multiply both scale with site count even though
    # prediction above doesn't, so the merge + wavespectra construction runs
    # per site-batch rather than on all of site_ids at once.
    n_hours = offshore.sizes.get("time", 0)
    n_parts = len(active) * len(part_ids)
    result, spectra = _sum_chunks(
        chunks,
        site_ids,
        n_hours=n_hours,
        n_parts=n_parts,
        variable_map=variable_map,
        save_spectra=save_spectra,
        workers=workers,
        load=load,
        wind_correction=_wind_inputs(wind_correction, offshore, site_ids),
    )

    if use_cache:
        _save_reconstruction_cache(result, cache_path)
        if save_spectra and spectra is not None:
            _save_reconstruction_cache(spectra, spectra_cache_path)
            logger.info(
                "Saved spectra cache: %s (%.2f MB)",
                spectra_cache_path.name,
                spectra_cache_path.stat().st_size * 1e-6,
            )
    logger.info(
        "Reconstruction done: %d site(s), %.2f MB",
        result.sizes["sites"],
        result.nbytes * 1e-6,
    )
    return result


def _wind_inputs(
    wind_correction: bool, offshore: xr.Dataset, site_ids: list[str]
) -> dict[str, Any] | None:
    """:func:`utils.wind.reconstruction_inputs` when the correction is on."""
    if not wind_correction:
        return None
    from utils.wind import reconstruction_inputs

    return reconstruction_inputs(offshore["time"].values, site_ids)


def _reconstruct_periods_batch(
    offshore_store: xr.Dataset,
    periods: list[str],
    n: int,
    *,
    metamodel: str,
    active: list[str],
    site_ids: list[str],
    spectral: bool,
    pca_variance: PcaVariance,
    wl: WaterLevel,
    cache_kw: Mapping[str, Any],
    use_cache: bool = True,
    device: str | None = None,
    save_spectra: bool | None = None,
    variables: Mapping[str, str] | None = None,
    workers: int = 1,
    wind_correction: bool = False,
) -> dict[str, xr.Dataset]:
    """Reconstruct several uncached months in one GP prediction pass per goal.

    GP inference cost is dominated by a large *fixed* per-call overhead (each
    call loops over every PCA component and dispatches one GPyTorch forward
    pass per component), not by how many rows it predicts — reconstructing a
    single month's ~50-700 valid hours measured about the same wall time as a
    whole year's ~1000+ hours in one call. ``reconstruct()``'s original month
    -by-month loop called :func:`_predict_offshore` once per (goal, partition,
    month), so a multi-year request re-paid that fixed cost once per month —
    e.g. an 11-year request paid it 132 times instead of once.

    This concatenates every *missing* month's offshore forcing into one
    combined selection, runs the goal x partition prediction loop and
    :func:`_wavespectra_sum` exactly once over the whole span, then splits the
    combined result back into each month's own cache file — so per-month
    cache granularity (and thus cheap partial/overlapping future requests)
    behaves exactly as before; only the redundant re-prediction is removed.

    Parameters
    ----------
    offshore_store
        Full offshore dataset (not yet time-sliced).
    periods
        Calendar-month labels to compute; caller has already confirmed none
        of these are cached.
    n, metamodel, active, site_ids, spectral, pca_variance, wl, use_cache,
    device, save_spectra, variables
        Same as :func:`reconstruct`.
    cache_kw
        Shared kwargs for :func:`reconstruction_cache_path` (built once by
        the caller, so every month's cache path matches what a normal
        single-month :func:`_reconstruct_period` call would have produced).
    workers, wind_correction
        Same as :func:`_reconstruct_period`.

    Returns
    -------
    dict[str, xarray.Dataset]
        Month label -> bulk-parameter dataset, one entry per period that had
        offshore data (periods with zero timesteps are omitted, same as the
        original per-month loop).

    Raises
    ------
    ValueError
        If no goal x partition produced valid predictions for any requested
        month.
    """
    variable_map = _reconstruction_variables(override=variables)
    save_spec = _reconstruction_save_spectra(override=save_spectra)

    period_offshore = {p: offshore_store.sel(time=p) for p in periods}
    period_offshore = {
        p: off for p, off in period_offshore.items() if off.sizes.get("time", 0) > 0
    }
    if not period_offshore:
        return {}

    combined_offshore = xr.concat(list(period_offshore.values()), dim="time")
    part_ids = _reconstruction_partitions(spectral, combined_offshore)

    logger.info(
        "Reconstruct batch: %d month(s) n=%d goals=%s partitions=%s sites=%d",
        len(period_offshore),
        n,
        active,
        part_ids,
        len(site_ids),
    )

    chunks: list[xr.Dataset] = []
    for goal in active:
        goal_models = load_models(
            goal, metamodel, n, pca_variance=pca_variance, device=device
        )
        try:
            for part in part_ids:
                pred = _predict_offshore(
                    combined_offshore,
                    goal,
                    n,
                    site_ids,
                    metamodel=metamodel,
                    partition=part,
                    wl=wl,
                    pca_variance=pca_variance,
                    models=goal_models,
                    device=device,
                    wind_sea=wind_correction,
                )
                if pred is None:
                    continue
                chunks.append(
                    pred.expand_dims(goal=[int(goal)], partition=[part]).load()
                )
        finally:
            del goal_models
            _release_gp_memory()

    if not chunks:
        raise ValueError(f"Reconstruction produced no data for {periods!r}")

    # Prediction above is unbatched across sites (cheap, see
    # _site_batch_size's docstring), but xr.merge and wavespectra's own
    # broadcast multiply both scale with site count — merge + wavespectra_sum
    # runs per site-batch to keep memory bounded regardless of how many sites
    # were requested.
    n_hours = combined_offshore.sizes.get("time", 0)
    n_parts = len(active) * len(part_ids)
    combined_result, combined_spectra = _sum_chunks(
        chunks,
        site_ids,
        n_hours=n_hours,
        n_parts=n_parts,
        variable_map=variable_map,
        save_spectra=save_spec,
        workers=workers,
        load=True,
        wind_correction=_wind_inputs(wind_correction, combined_offshore, site_ids),
    )

    out: dict[str, xr.Dataset] = {}
    for period in period_offshore:
        month_result = combined_result.sel(time=period)
        if use_cache:
            month_path = reconstruction_cache_path(period, **cache_kw)
            _save_reconstruction_cache(month_result, month_path)
            if save_spec and combined_spectra is not None:
                month_spec = combined_spectra.sel(time=period)
                spectra_path = reconstruction_spectra_cache_path(month_path)
                _save_reconstruction_cache(month_spec, spectra_path)
                logger.info(
                    "Saved spectra cache: %s (%.2f MB)",
                    spectra_path.name,
                    spectra_path.stat().st_size * 1e-6,
                )
        out[period] = month_result
    logger.info(
        "Reconstruction batch done: %d month(s), %d site(s)",
        len(out),
        month_result.sizes["sites"] if out else 0,
    )
    return out


def reconstruct(
    time: str | slice,
    n: int,
    *,
    metamodel: str,
    spectral: bool = True,
    offshore_name: str = "WHACS",
    goals: Iterable[str | int] | None = None,
    wl: WaterLevel = 0.0,
    sites: str | Iterable[str] | None = None,
    pca_variance: PcaVariance = None,
    load: bool = True,
    recompute: bool = False,
    use_cache: bool = True,
    device: str | None = None,
    save_spectra: bool | None = None,
    variables: Mapping[str, str] | None = None,
    workers: int = 1,
    wind_correction: bool | None = None,
) -> xr.Dataset:
    """Reconstruct nearshore waves from offshore hindcast forcing.

    Runs the metamodel for each goal and WHACS partition, then linearly sums
    partition spectra with wavespectra. Each goal is only evaluated for
    timesteps whose offshore direction lies inside that goal's nautical sector
    (from the case goals JSON). Long periods are split month by month for
    offshore I/O and caching; GP models are loaded one goal at a time to limit
    GPU memory. Results are cached under ``tmp/reconstruct``.

    Parameters
    ----------
    time
        Period passed to the offshore zarr (e.g. ``"2020-05"`` or ``"2020"``).
    n
        MDA train size; fitted models must exist for every active goal.
    metamodel
        Metamodel set (key of ``metamodel.sets``); *sites* must belong to it.
    spectral
        ``False`` — bulk offshore partition ``-1`` per goal.
        ``True`` — WHACS partitions ``phs0``–``phs3`` (``0``–``3``).
    offshore_name
        Processed hindcast key in the case config JSON.
    goals
        Goals to include; default all configured goals.
    wl
        Water level (m) for the MDA forcing, or ``"forcing"`` for the
        ``forcing.water_level`` series at each goal.
    sites
        Site selector within the set; see :func:`resolve_sites`.
    pca_variance
        PCA fractions used at fit time.
    load
        Eager-load the output dataset.
    recompute
        If ``True``, ignore and overwrite any matching cache file. Use after
        retraining models or changing ``site_filter`` / ``pca_preprocessing``.
    use_cache
        If ``False``, skip reading and writing cache files.
    device
        Optional PyTorch device for GP inference. Use ``"cpu"`` when CUDA runs
        out of memory (each goal holds ~100+ exact GPs at ``n=450``).
    save_spectra
        Write full summed spectra to a companion cache file
        (``…__spectra.nc``). When ``None``, reads
        ``metamodel.reconstruction.save_spectra``. Expensive for many sites /
        long periods — prefer a small ``sites`` subset.
    variables
        HyGloW name → wavespectra stat (e.g. ``dir`` → ``dm``, ``spr`` →
        ``dpspr``). When ``None``, reads ``metamodel.reconstruction.variables``.
        All wavespectra bulk stats are computed per site; only the mapped names
        are returned.
    workers
        Process-pool workers for the site-batched merge + wavespectra
        summation step (the dominant cost at large site counts — see
        :func:`_sum_chunks`). ``1`` (default) is the original
        sequential behavior; every other reconstruct() call in this codebase
        is unaffected by this parameter's existence. GPU inference (the goal
        loop) is not parallelized — only pure-CPU wavespectra work is, so
        this is safe to raise regardless of ``device``. Pick a value that
        leaves headroom on the host: each worker gets its own share of
        ``WAVESPECTRA_MEMORY_BUDGET_BYTES``, so total memory stays roughly
        constant as ``workers`` grows, but CPU/core usage scales with it —
        e.g. ``workers=os.cpu_count() // 2`` to leave half the machine free.
    wind_correction
        Add the local wind sea before the spectral sum (see :mod:`utils.wind`;
        needs the fetch table and the fitted correction). Default
        ``metamodel.wind_correction.enabled``. The output then also has
        ``wind_k`` (``K``, 1 where nothing was added).

    Returns
    -------
    xarray.Dataset
        ``(time, sites)`` with the requested HyGloW variables. Use
        :func:`load_reconstruction_spectra` to read cached spectra when
        ``save_spectra=True``.

    Raises
    ------
    ModelsNotFittedError
        If fitted models are missing for any active goal.
    ValueError
        If reconstruction produces no valid data for the requested period.
    """
    from utils.config import OFFSHORE_WAVES

    if not OFFSHORE_WAVES:
        logger.warning(
            "offshore_waves is empty in case config; reconstruct requires a "
            "processed hindcast zarr (see notebook 07)"
        )

    active = _goals(goals)
    wind_correction = _wind_correction_on(wind_correction)
    validate_metamodel_config()
    require_models_all(metamodel, n, active, pca_variance=pca_variance)

    template = load_cases_goal(active[0])
    site_ids = resolve_reconstruction_sites(
        template, sites, metamodel=metamodel, goals=active
    )
    variable_map = _reconstruction_variables(override=variables)

    cache_kw = dict(
        n=n,
        metamodel=metamodel,
        spectral=spectral,
        offshore_name=offshore_name,
        goals=goals,
        wl=wl,
        site_ids=site_ids,
        pca_variance=pca_variance,
        variables=variable_map,
        wind_correction=wind_correction,
    )
    full_cache_path = reconstruction_cache_path(time, **cache_kw)
    save_spec = _reconstruction_save_spectra(override=save_spectra)
    full_spectra_cache_path = reconstruction_spectra_cache_path(full_cache_path)
    if use_cache and not recompute:
        cached = _load_reconstruction_cache(full_cache_path, load=load)
        if cached is not None:
            if save_spec and not full_spectra_cache_path.is_file():
                logger.info(
                    "Bulk cache hit but full spectra cache missing — recomputing"
                )
            else:
                return cached

    offshore_store = open_offshore_waves(offshore_name)
    periods = _month_periods(time, offshore_store)

    period_kw = dict(
        metamodel=metamodel,
        active=active,
        site_ids=site_ids,
        spectral=spectral,
        offshore_name=offshore_name,
        pca_variance=pca_variance,
        wl=wl,
        recompute=recompute,
        use_cache=use_cache,
        device=device,
        save_spectra=save_spectra,
        variables=variable_map,
        workers=workers,
        wind_correction=wind_correction,
    )

    if len(periods) == 1:
        period = periods[0]
        result = _reconstruct_period(
            offshore_store.sel(time=period),
            n,
            time_label=period,
            load=load,
            cache_path=reconstruction_cache_path(period, **cache_kw),
            **period_kw,
        )
        if use_cache and _time_label_for_cache(period) != _time_label_for_cache(time):
            _save_reconstruction_cache(result, full_cache_path)
            if save_spec:
                period_spec = _load_reconstruction_cache(
                    reconstruction_spectra_cache_path(
                        reconstruction_cache_path(period, **cache_kw)
                    ),
                    load=False,
                )
                if period_spec is not None:
                    _save_reconstruction_cache(period_spec, full_spectra_cache_path)
        return result

    logger.info(
        "Reconstruct time=%r spans %d month(s)",
        time,
        len(periods),
    )
    # Two passes: first split into already-cached vs missing months without
    # computing anything, then reconstruct every missing month in one batch
    # (see _reconstruct_periods_batch) instead of one _predict_offshore call
    # per month — GP inference's fixed per-call overhead otherwise gets
    # re-paid for every single month of a long request.
    monthly_by_period: dict[str, xr.Dataset] = {}
    missing_periods: list[str] = []
    for period in periods:
        if offshore_store.sel(time=period).sizes.get("time", 0) == 0:
            logger.debug("Month %r: no timesteps — skipped", period)
            continue
        month_path = reconstruction_cache_path(period, **cache_kw)
        if use_cache and not recompute:
            cached_month = _load_reconstruction_cache(month_path, load=False)
            if cached_month is not None:
                monthly_by_period[period] = cached_month
                continue
        missing_periods.append(period)

    if missing_periods:
        batch_months = _reconstruct_batch_months(len(site_ids))
        logger.info(
            "Reconstructing %d missing month(s) of %d (%d already cached), in "
            "batches of up to %d month(s) (%d site(s))",
            len(missing_periods),
            len(periods),
            len(monthly_by_period),
            batch_months,
            len(site_ids),
        )
        for i in range(0, len(missing_periods), batch_months):
            batch = missing_periods[i : i + batch_months]
            monthly_by_period.update(
                _reconstruct_periods_batch(
                    offshore_store,
                    batch,
                    n,
                    metamodel=metamodel,
                    active=active,
                    site_ids=site_ids,
                    spectral=spectral,
                    pca_variance=pca_variance,
                    wl=wl,
                    cache_kw=cache_kw,
                    use_cache=use_cache,
                    device=device,
                    save_spectra=save_spectra,
                    variables=variable_map,
                    workers=workers,
                    wind_correction=wind_correction,
                )
            )

    monthly = [monthly_by_period[p] for p in periods if p in monthly_by_period]
    if not monthly:
        raise ValueError("Reconstruction produced no data (check time range)")

    result = xr.concat(monthly, dim="time")
    if load:
        result = result.load()
    if use_cache:
        _save_reconstruction_cache(result, full_cache_path)
        if save_spec:
            monthly_specs: list[xr.Dataset] = []
            for period in periods:
                spec_path = reconstruction_spectra_cache_path(
                    reconstruction_cache_path(period, **cache_kw)
                )
                spec_month = _load_reconstruction_cache(spec_path, load=False)
                if spec_month is not None:
                    monthly_specs.append(spec_month)
            if monthly_specs:
                full_spec = xr.concat(monthly_specs, dim="time")
                if load:
                    full_spec = full_spec.load()
                _save_reconstruction_cache(full_spec, full_spectra_cache_path)
                logger.info(
                    "Saved full spectra cache: %s (%.2f MB)",
                    full_spectra_cache_path.name,
                    full_spectra_cache_path.stat().st_size * 1e-6,
                )
    logger.info(
        "Reconstruction done: %d site(s), %.2f MB (%d months)",
        result.sizes["sites"],
        result.nbytes * 1e-6,
        len(monthly),
    )
    return result


__all__ = [
    "BULK_PARTITION",
    "LOGGER_NAME",
    "MDA_CANDIDATES",
    "ModelsNotFittedError",
    "PcaVariance",
    "SETS",
    "SITE_META",
    "SPECTRAL_PARTITIONS",
    "VARS",
    "WL_FROM_FORCING",
    "check_artefacts_model",
    "common_model_sites",
    "fit",
    "fit_all",
    "forcing_dataframe",
    "goal_output_dir",
    "load_cases_goal",
    "load_context",
    "load_models",
    "logger",
    "predict",
    "reconstruct",
    "reconstruction_cache_dir",
    "reconstruction_cache_path",
    "reconstruction_spectra_cache_path",
    "load_reconstruction_spectra",
    "require_models",
    "require_models_all",
    "resolve_reconstruction_sites",
    "resolve_sites",
    "resolved_pca_variances",
    "set_sites",
]
