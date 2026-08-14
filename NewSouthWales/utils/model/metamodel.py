"""GPR metamodel on SnapWave LHS cases (MDA → per-variable PCA → stacked GP).

Training: :func:`fit`, :func:`fit_all`
Prediction: :func:`predict` (LHS ``case_num`` or hindcast ``time`` via ``row_dim``)
Hindcast: :func:`reconstruct` — ``spectral=False`` bulk (partition ``-1``),
``spectral=True`` WHACS partitions ``0``–``3``; both via wavespectra linear sum.
Bulk output variables and optional full-spectrum caching are configured in the
case JSON under ``metamodel.reconstruction`` (``variables`` maps HyGloW names to
wavespectra bulk stats; ``save_spectra`` caches summed spectra).
Long hindcast periods are processed month by month; results are cached under
``tmp/reconstruct``.

Site grouping and selection: :mod:`utils.geometry.sites`.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.io import load_model
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.operations import get_degrees_from_uv, get_uv_components
from bluemath_tk.datamining.mda import MDA
from bluemath_tk.datamining.pca import PCA
from bluemath_tk.interpolation.gps import ExactGPInterpolation
from wavespectra.construct import STATS as WAVESPECTRA_STATS
from wavespectra.construct import construct_partition

from utils.config import (
    METAMODEL,
    SIMULATIONS,
    forcing_columns,
    load_goals,
    mda_max_points,
    open_offshore_waves,
    resolve,
    validate_metamodel_config,
)
from utils.directions import in_nautical_sector
from utils.geometry.sites import SITE_META, resolve_sites

LOGGER_NAME = "hyglow.metamodel"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("logs/metamodel.log")),
    level="INFO",
    console=True,
    console_level="WARNING",
)

PcaVariance = float | dict[str, float] | None
VarSpec = Literal["raw"] | dict[str, Any]

# case config → metamodel (single source; do not duplicate in other modules)
VARS = METAMODEL["vars_to_predict"]
MDA_INPUTS = METAMODEL["mda_inputs"]
MDA_DIRECTIONAL = METAMODEL["mda_directional_variables"]
MDA_CANDIDATES = METAMODEL["mda_candidates"]
SITE_FILTER = METAMODEL["site_filter"]
FORCING_COLS = tuple(
    c for c in MDA_INPUTS if c != "wl"
)  # validity mask for hindcast rows (wl optional)

# WHACS: bulk = partition -1; wind sea + swells = 0–3 (see utils/simulation/offshore_forcing.py)
BULK_PARTITION = -1
SPECTRAL_PARTITIONS = (0, 1, 2, 3)

# Fixed frequency/direction grids for wavespectra linear summation (reconstruct)
SPECTRAL_FREQ = np.logspace(np.log10(0.035), np.log10(0.5), 29)
SPECTRAL_DIR = np.arange(0.0, 360.0, 5)
# float32 copies for _wavespectra_sum (see its docstring) — precomputed once
# rather than cast on every call; these two arrays are tiny either way, this
# is purely for tidiness.
SPECTRAL_FREQ32 = SPECTRAL_FREQ.astype("float32")
SPECTRAL_DIR32 = SPECTRAL_DIR.astype("float32")

# Directional spreading (`spr`) is GP-predicted, not raw-passthrough, so it can
# occasionally extrapolate to a non-physical near-zero or negative value for
# rare storm-forcing combinations. wavespectra's Cartwright spreading function
# (wavespectra.construct.direction.cartwright) computes
# ``s = 2 / deg2rad(dspr)**2 - 1``: as dspr -> 0 that exponent explodes, the
# discretised cos(...)**(2*s) spreading curve underflows to exactly zero at
# every point on SPECTRAL_DIR's 5-degree grid, and the normalisation step
# (``1 / gth.sum()``) then divides by that zero -> `inf`, silently poisoning
# the whole reconstructed spectrum (and anything averaged over it — a single
# bad hour turns an entire monthly/seasonal mean spectrum blank). Predicted
# `spr` is floored to this value in `predict()` before it ever reaches
# wavespectra, rather than each caller having to guard against it downstream.
MIN_DIRECTIONAL_SPREAD_DEG = 1.0

# `reconstruct()` batches every *uncached* month into as few GP prediction
# calls as possible (see `_reconstruct_periods_batch`) since each call's cost
# is dominated by a large fixed per-call overhead, not by row count — one
# call covering 3 months costs far less than 3 separate one-month calls. But
# batching *too much* at once grows GPyTorch's per-call kernel evaluation past
# available GPU memory: on a 5.61 GiB GPU, a full 11-year (132-month) batch
# and a 12-month batch both hit CUDA OOM partway through the 26-goal loop,
# while a 3-month batch measured fine. This is a GPU-VRAM constraint that
# scales with row count (time span) per call, independent of how many sites
# are requested — confirmed by direct instrumentation (see
# `_reconstruct_batch_months`'s docstring): the goal loop's own host RSS
# stayed flat (~1.5-1.7 GiB) across both a 3-site and an 18-site request, in
# both bulk and spectral mode, so this cap does not need a site-count term.
# Raise it only after confirming headroom on the target GPU (or pass
# device="cpu" to reconstruct() for unbounded-but-slower CPU inference).
RECONSTRUCT_GPU_BATCH_MONTHS_CAP = 3


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


# `_reconstruct_periods_batch`/`_reconstruct_period` always predict *all*
# requested sites in one pass rather than site-batching prediction itself —
# confirmed safe by direct instrumentation: the goal loop's host RSS stayed
# flat (~1.5-1.7 GiB) regardless of site count (3 vs 18) or mode (bulk vs
# spectral), so `predict()`'s per-goal PCA inverse_transform (which always
# reconstructs the *full* ~3,890-site catalog before subsetting to
# `site_ids` — see `predict()`) is released promptly each iteration and
# never became the bottleneck an earlier version of this comment assumed.
#
# Downstream of that, the xr.merge right after (combining the goal x
# partition chunks into one cube) and, far more, wavespectra's own
# `efth1d * spread` broadcast inside construct_partition, both scale with
# site count — and the wavespectra multiply is the dominant one: its output
# is the *full* (time, sites, goal x partition, freq, dir) array, not just
# the small (time, sites, goal x partition) bulk-variable cube xr.merge
# builds. A fixed site-batch size that was safe for the merge step OOM'd on
# the wavespectra step alone (80 sites x 1 month x 104 goal x partition
# combos tried to allocate 93.2 GiB against a single-array estimate). So the
# batch size is computed per call from how many hours/goal x partition combos
# are actually in play (both vary — hours with `_reconstruct_batch_months`,
# combos with bulk vs spectral mode and len(goals)) rather than guessed as
# one constant.
#
# That single-array estimate turned out to itself be an underestimate of the
# true peak: `construct_partition`'s jonswap/cartwright/sum pipeline holds
# several comparably-sized temporaries alive at once (the frequency
# spectrum, the directional spread, their broadcast product, the
# partition-sum reduction), not just the one final combined array the
# formula below sizes against. This went unnoticed while every exercised
# site count landed well under the computed batch threshold (bulk mode's
# smaller n_parts gives a lot of headroom) — until an 18-site spectral-mode
# request (n_parts=104, just under the *unmultiplied* formula's ~20-site
# threshold, so it ran unsplit) hit a real 62 GiB host, reaching at least
# 55 GiB RSS (the kill point, so a lower bound on the true peak — total-vm
# in the OOM record was still climbing past 92 GiB) against a 12 GiB target.
# That implies a true-to-estimated ratio of at least ~5x, plausibly higher
# since the process was killed before finishing its allocation.
# WAVESPECTRA_PEAK_MULTIPLIER=8 bakes in that safety margin; recalibrate (see
# tmp/mem_probe.py's pattern in the conversation this was diagnosed in) if
# construct_partition's internals change enough to move the true ratio.
WAVESPECTRA_MEMORY_BUDGET_BYTES = (
    12 * 1024**3
)  # 12 GiB; leaves headroom on a 62 GiB host
WAVESPECTRA_PEAK_MULTIPLIER = 8


def _site_batch_size(n_hours: int, n_parts: int) -> int:
    """Sites per _wavespectra_sum call keeping its peak array under budget.

    Parameters
    ----------
    n_hours
        Timesteps in the current reconstruct call (grows with
        ``_reconstruct_batch_months``).
    n_parts
        ``len(active goals) * len(partitions)`` — up to 4x more for spectral
        mode than bulk.

    Returns
    -------
    int
        At least 1, however small the budget — never returns 0 (which would
        silently drop sites).
    """
    # 4 bytes: _wavespectra_sum computes construct_partition in float32 (see
    # its docstring) — this must track that dtype, or batches would be sized
    # for an array twice as big as the one actually allocated.
    # WAVESPECTRA_PEAK_MULTIPLIER: construct_partition holds several
    # comparably-sized temporaries at once, not just this one final array —
    # see WAVESPECTRA_MEMORY_BUDGET_BYTES's docstring above for how that was
    # calibrated.
    bytes_per_site = (
        max(1, n_hours)
        * max(1, n_parts)
        * len(SPECTRAL_FREQ)
        * len(SPECTRAL_DIR)
        * 4
        * WAVESPECTRA_PEAK_MULTIPLIER
    )
    return max(1, WAVESPECTRA_MEMORY_BUDGET_BYTES // bytes_per_site)


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
_DEFAULT_PCA_PREPROCESSING = {
    "nan_threshold_to_drop": {
        "hs": 0.99,
        "dir_u": 0.99,
        "dir_v": 0.99,
        "spr": 0.99,
    },
    "value_to_replace_nans": {
        "hs": 0.0,
        "dir_u": 0.0,
        "dir_v": 0.0,
        "spr": 0.0,
    },
}


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


def _resolve_pca_variance(variable: str, override: PcaVariance = None) -> float:
    """Return the PCA variance fraction for one stacked variable.

    Parameters
    ----------
    variable
        Key in ``metamodel.vars_to_predict`` (must not be ``"raw"``).
    override
        Single float for all variables, or per-variable dict override.

    Returns
    -------
    float
        PCA variance fraction in ``(0, 1]``.

    Raises
    ------
    ValueError
        If *variable* is marked ``"raw"`` in the config.
    """
    spec = VARS[variable]
    if spec == "raw":
        raise ValueError(f"{variable!r} is raw")
    if isinstance(override, dict) and variable in override:
        return float(override[variable])
    if isinstance(override, (int, float)):
        return float(override)
    return float(spec["pca_variance"])


def _pca_variables(override: PcaVariance = None) -> list[tuple[str, float, list[str]]]:
    """List PCA targets configured in the case config JSON.

    Parameters
    ----------
    override
        Optional PCA variance override passed to :func:`_resolve_pca_variance`.

    Returns
    -------
    list of tuple
        ``(variable_name, variance, vars_to_stack)`` for each non-raw entry.
    """
    return [
        (name, _resolve_pca_variance(name, override), spec["vars_to_stack"])
        for name, spec in VARS.items()
        if spec != "raw"
    ]


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
    """Return the SnapWave LHS NetCDF path for one goal.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.

    Returns
    -------
    Path
        Path to ``output_sites_{goal}.nc``.
    """
    return resolve(SIMULATIONS["metamodel_cases"]) / f"output_sites_{int(goal)}.nc"


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


def gp_model_path(goal: str | int, n: int) -> Path:
    """Return the fitted GP pickle path for one goal and MDA size.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    n
        MDA train size (candidate from ``mda_candidates``).

    Returns
    -------
    Path
        ``outputs/goal_XX/gp_{n}/gp_model.pkl``.
    """
    template = METAMODEL["paths"]["gp_dir"]
    return (
        resolve(template.format(goal=f"goal_{int(goal):02d}", mda_candidate=n))
        / "gp_model.pkl"
    )


def pca_model_path(goal: str | int, n: int, variable: str, variance: float) -> Path:
    """Return the fitted PCA pickle path for one goal, MDA size and variable.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    n
        MDA train size (candidate from ``mda_candidates``).
    variable
        Stacked variable name (e.g. ``"hs"``, ``"dir"``).
    variance
        PCA variance fraction used when fitting.

    Returns
    -------
    Path
        ``outputs/goal_XX/pca_{n}/pca_{variable}_{variance}_model.pkl``.
    """
    template = METAMODEL["paths"]["pca_dir"]
    base = resolve(template.format(goal=f"goal_{int(goal):02d}", mda_candidate=n))
    return base / f"pca_{variable}_{_variance_tag(variance)}_model.pkl"


# ---------------------------------------------------------------------------
# Cases & site filter
# ---------------------------------------------------------------------------


def detect_removed_sites(cases: xr.Dataset) -> list[str]:
    """Return site labels that fail ``METAMODEL.site_filter`` on SnapWave cases.

    Each ``site_filter`` key must end with ``_max`` or ``_min``:

    - ``{var}_max`` — drop sites whose maximum over ``case_num`` exceeds the threshold.
    - ``{var}_min`` — drop sites whose minimum over ``case_num`` falls below the threshold.

    Parameters
    ----------
    cases
        SnapWave LHS dataset with variables on ``(case_num, sites)``.

    Returns
    -------
    list[str]
        Sorted site labels failing any configured rule.
    """
    bad: set[str] = set()
    for key, threshold in SITE_FILTER.items():
        if key.endswith("_max"):
            var = key[:-4]
            op = "max"
        elif key.endswith("_min"):
            var = key[:-4]
            op = "min"
        else:
            logger.warning(
                "Ignoring site_filter key %r (expected {var}_max or {var}_min)",
                key,
            )
            continue
        if var not in cases.data_vars:
            logger.warning("site_filter %r: variable %r not in cases", key, var)
            continue
        if op == "max":
            agg = cases[var].max("case_num")
            offenders = agg.where(agg > float(threshold), drop=True)
        else:
            agg = cases[var].min("case_num")
            offenders = agg.where(agg < float(threshold), drop=True)
        bad.update(str(s) for s in offenders.sites.values)
    return sorted(bad)


def drop_sites(cases: xr.Dataset, site_ids: Iterable[str]) -> xr.Dataset:
    """Return *cases* without the listed site coordinate labels.

    Parameters
    ----------
    cases
        Dataset with a ``sites`` coordinate.
    site_ids
        Site labels to remove.

    Returns
    -------
    xarray.Dataset
        Subset with the requested sites dropped.
    """
    remove = {str(s) for s in site_ids}
    if not remove:
        return cases
    kept = [s for s in cases.sites.values if str(s) not in remove]
    return cases.sel(sites=kept)


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
        u, v = get_uv_components(ds["dir"])
        out = ds.assign(dir_u=u, dir_v=v).load()
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


def load_fit_summary(goal: str | int, n: int | None = None) -> dict[str, Any]:
    """Return persisted PCA/GP fit summary from ``split.json``.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    n
        MDA train size. If ``None``, return the full ``fit_by_mda`` mapping.

    Returns
    -------
    dict
        Fit summary for one MDA size, or all sizes when *n* is ``None``.
    """
    fit_by = _read_split_dict(goal).get("fit_by_mda", {})
    if n is None:
        return fit_by
    return fit_by.get(str(int(n)), {})


def _pca_kept_site_labels(pca: Any) -> dict[str, list[str]]:
    """Map stacked variable name → site labels kept by PCA NaN threshold."""
    sites = np.asarray(pca.coords_values.get("sites", []))
    if sites.size == 0:
        return {}
    out: dict[str, list[str]] = {}
    for var, positions in getattr(pca, "not_nan_positions", {}).items():
        idx = np.asarray(positions, dtype=int)
        out[var] = [str(sites[i]) for i in idx]
    return out


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


def _persist_fit_summary(goal: str | int, n: int, summary: dict[str, Any]) -> None:
    """Merge one MDA-size fit summary into ``split.json``."""
    data = _read_split_dict(goal)
    if "pca_nan_policy" not in data:
        value_replace, nan_drop = _pca_preprocessing()
        data["pca_nan_policy"] = {
            "value_to_replace_nans": value_replace,
            "nan_threshold_to_drop": nan_drop,
        }
    fit_by = data.setdefault("fit_by_mda", {})
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
    goal: str | int, n: int, *, pca_variance: PcaVariance = None
) -> list[Path]:
    """Return paths of missing GP/PCA model files for one goal and MDA size.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
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
    if not gp_model_path(goal, n).exists():
        missing.append(gp_model_path(goal, n))
    for name, variance, _ in _pca_variables(pca_variance):
        p = pca_model_path(goal, n, name, variance)
        if not p.exists():
            missing.append(p)
    return missing


def models_exist(goal: str | int, n: int, *, pca_variance: PcaVariance = None) -> bool:
    """Check whether GP and all PCA pickles exist for one goal and MDA size.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    n
        MDA train size.
    pca_variance
        PCA variance override used when checking PCA artefact names.

    Returns
    -------
    bool
        ``True`` when no model files are missing.
    """
    return not _missing_model_paths(goal, n, pca_variance=pca_variance)


def require_models(
    goal: str | int, n: int, *, pca_variance: PcaVariance = None
) -> None:
    """Raise :class:`ModelsNotFittedError` if any model file is missing.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
    n
        MDA train size.
    pca_variance
        PCA variance override used when checking PCA artefact names.

    Raises
    ------
    ModelsNotFittedError
        If the GP or any PCA pickle is absent.
    """
    missing = _missing_model_paths(goal, n, pca_variance=pca_variance)
    if missing:
        raise ModelsNotFittedError(
            f"Models not fitted: goal={goal!r}, n={n}, "
            f"pca_variance={resolved_pca_variances(pca_variance)}. Missing:\n"
            + "\n".join(f"  {p}" for p in missing)
            + ModelsNotFittedError._FIT_HINT
        )


def require_models_all(
    n: int,
    goals: Iterable[str | int] | None = None,
    *,
    pca_variance: PcaVariance = None,
) -> None:
    """Raise :class:`ModelsNotFittedError` if any goal is missing fitted models.

    Parameters
    ----------
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
        require_models(goal, n, pca_variance=pca_variance)


def load_models(
    goal: str | int,
    n: int,
    *,
    pca_variance: PcaVariance = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Load fitted MDA, GP and per-variable PCA models from disk.

    Parameters
    ----------
    goal, n
        Goal id and MDA train size (must match saved artefacts).
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
    require_models(goal, n, pca_variance=pca_variance)
    pcas = {
        name: load_model(pca_model_path(goal, n, name, var))
        for name, var, _ in _pca_variables(pca_variance)
    }
    gp = load_model(gp_model_path(goal, n))
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
    mda = MDA(num_centers=num_centers)
    mda.fit(forcing_dataframe(cases), directional_variables=MDA_DIRECTIONAL)

    train_idx = np.asarray(mda.centroid_real_indices, dtype=int)
    all_idx = set(range(cases.sizes["case_num"]))
    test_idx = np.asarray(sorted(all_idx - set(train_idx.tolist())), dtype=int)

    path.parent.mkdir(parents=True, exist_ok=True)
    mda.save_model(path)
    value_replace, nan_drop = _pca_preprocessing()
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "num_centers": num_centers,
                "removed_sites": dropped,
                "site_filter": SITE_FILTER,
                "pca_nan_policy": {
                    "value_to_replace_nans": value_replace,
                    "nan_threshold_to_drop": nan_drop,
                },
                "fit_by_mda": {},
            },
            f,
            indent=2,
        )
    logger.info(
        "Goal %s: MDA fitted (%d train, %d test)", goal, len(train_idx), len(test_idx)
    )
    return mda, train_idx, test_idx


def _stack_pcs(pcas: dict[str, Any]) -> pd.DataFrame:
    """Concatenate per-variable PC columns into one GP target matrix.

    Parameters
    ----------
    pcas
        Mapping ``variable_name → fitted PCA`` object.

    Returns
    -------
    pandas.DataFrame
        Columns named ``{variable}__{pc}``; one row per training case.
    """
    frames = []
    for name in pcas:
        df = pcas[name].pcs_df.copy()
        df.columns = [f"{name}__{c}" for c in df.columns]
        frames.append(df)
    return pd.concat(frames, axis=1)


def _split_pcs(pcs: np.ndarray, pcas: dict[str, Any]) -> dict[str, np.ndarray]:
    """Split stacked GP output back into per-variable PC blocks.

    Parameters
    ----------
    pcs
        GP prediction array with columns in :func:`_stack_pcs` order.
    pcas
        Mapping ``variable_name → fitted PCA`` object.

    Returns
    -------
    dict[str, numpy.ndarray]
        PC blocks keyed by variable name.
    """
    blocks, offset = {}, 0
    for name in pcas:
        k = pcas[name].pcs_df.shape[1]
        blocks[name] = pcs[:, offset : offset + k]
        offset += k
    return blocks


def _fit_models(
    goal: str | int,
    n: int,
    *,
    mda: Any | None = None,
    train_idx: np.ndarray | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
    pca_variance: PcaVariance = None,
) -> dict[str, Any]:
    """Fit or load PCA + GP models for one goal and MDA size *n*.

    Uses the first *n* MDA train cases. Each variable in ``vars_to_predict``
    (except ``raw``) gets a PCA on stacked site fields; all PC blocks are
    concatenated and regressed with an exact GP on MDA centroids.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
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

    if not force_retrain and models_exist(goal, n, pca_variance=pca_variance):
        logger.info("Goal %s / n=%d: loaded models", goal, n)
        out = load_models(goal, n, pca_variance=pca_variance)
        out["train_idx"] = train_idx
        return out

    if mda is None or train_idx is None:
        mda, train_idx, _ = _fit_mda(goal, force_retrain=force_retrain)

    cases = load_cases_goal(goal).isel(case_num=train_idx[:n])
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
    pcas: dict[str, Any] = {}
    pca_sites_kept: dict[str, list[str]] = {}
    pca_sites_kept_by_stack: dict[str, list[str]] = {}
    pca_sites_dropped: dict[str, int] = {}
    for name, variance, stack in _pca_variables(pca_variance):
        logger.info(
            "Goal %s / n=%d: fitting PCA %s (%d cases × %d sites; nan_threshold=%s)",
            goal,
            n,
            name,
            n_cases,
            n_sites,
            {k: nan_drop.get(k) for k in stack},
        )
        pca = PCA(n_components=variance)
        pca.fit_transform(
            data=cases,
            vars_to_stack=stack,
            coords_to_stack=["sites"],
            pca_dim_for_rows="case_num",
            value_to_replace_nans=value_replace,
            nan_threshold_to_drop=nan_drop,
        )
        kept_by_stack = _pca_kept_site_labels(pca)
        kept = _consolidated_pca_sites(kept_by_stack, stack)
        pca_sites_kept[name] = kept
        for var in stack:
            if var in kept_by_stack:
                pca_sites_kept_by_stack[var] = kept_by_stack[var]
        n_dropped = n_sites - len(kept)
        pca_sites_dropped[name] = n_dropped
        logger.info(
            "Goal %s / n=%d: PCA %s kept %d/%d sites (%d dropped by nan threshold)",
            goal,
            n,
            name,
            len(kept),
            n_sites,
            n_dropped,
        )
        path = pca_model_path(goal, n, name, variance)
        path.parent.mkdir(parents=True, exist_ok=True)
        pca.save_model(path)
        pcas[name] = pca
        logger.info("Goal %s / n=%d: saved %s", goal, n, path.name)

    forcing = forcing_dataframe(cases)
    centroids = mda.centroids[MDA_INPUTS].iloc[:n]
    if forcing.shape != centroids.shape:
        raise ValueError(
            f"GP training: forcing shape {forcing.shape} != centroids {centroids.shape}"
        )
    if not np.allclose(forcing.values, centroids.values, rtol=1e-5, atol=1e-8):
        diff = float(np.nanmax(np.abs(forcing.values - centroids.values)))
        raise ValueError(f"GP training: forcing != MDA centroids (max |Δ|={diff:.4g})")

    target = _stack_pcs(pcas)
    gp = ExactGPInterpolation(epochs=gp_epochs)
    gp.fit(
        subset_data=mda.centroids[MDA_INPUTS].iloc[:n],
        subset_directional_variables=MDA_DIRECTIONAL,
        target_data=target,
    )
    gp_p = gp_model_path(goal, n)
    gp_p.parent.mkdir(parents=True, exist_ok=True)
    gp.save_model(gp_p)
    logger.info("Goal %s / n=%d: GP saved (%d PCs)", goal, n, target.shape[1])

    _persist_fit_summary(
        goal,
        n,
        {
            "n_cases_requested": n_requested,
            "n_cases_used": n_cases,
            "n_sites_before_pca": n_sites,
            "pca_sites_kept": pca_sites_kept,
            "pca_sites_kept_by_stack": pca_sites_kept_by_stack,
            "pca_sites_dropped": pca_sites_dropped,
            "pca_variances": resolved_pca_variances(pca_variance),
            "n_gp_components": int(target.shape[1]),
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
    candidates: Iterable[int] | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
    pca_variance: PcaVariance = None,
) -> None:
    """Train MDA, PCA and GP models for one goal and all MDA candidate sizes.

    Parameters
    ----------
    goal
        Goal id from the case goals JSON.
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
    for n in candidates or MDA_CANDIDATES:
        _fit_models(
            goal,
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
        DataFrame indexed by case/time row, columns ``mda_inputs`` from config.
        Index is renamed internally to ``case_num`` for PCA compatibility.
    models
        Pre-loaded dict from :func:`load_models`; loaded from disk if omitted.
    pca_variance
        Must match the variances used when models were fitted.
    sites
        Subset of site labels; default all sites for the goal.
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
        models = load_models(goal, n, pca_variance=pca_variance, device=device)

    # PCA inverse_transform expects the row dimension to be named case_num
    if forcing.index.name != "case_num":
        forcing = forcing.rename_axis("case_num")
    rows = forcing.index.values

    cases = load_cases_goal(goal)
    all_sites = [str(s) for s in cases.sites.values]
    site_ids = all_sites if sites is None else [s for s in sites if s in all_sites]
    if not site_ids:
        raise ValueError(f"No requested sites found for goal {goal!r}")

    blocks = _split_pcs(models["gp"].predict(dataset=forcing).values, models["pcas"])
    out: dict[str, Any] = {}
    for name, spec in VARS.items():
        if spec == "raw":
            v = forcing[name].to_numpy()
            out[name] = xr.DataArray(
                np.broadcast_to(v[:, None], (len(v), len(site_ids))),
                dims=("case_num", "sites"),
                coords={"case_num": rows, "sites": site_ids},
            )
            continue
        pts = models["pcas"][name].inverse_transform(
            xr.Dataset(
                {"PCs": (("case_num", "n_component"), blocks[name])},
                coords={"case_num": rows},
            )
        )
        if name == "dir":
            pca_sites = pts["dir_u"].coords["sites"].values
            arr = xr.DataArray(
                get_degrees_from_uv(pts["dir_u"].values, pts["dir_v"].values),
                dims=("case_num", "sites"),
                coords={"case_num": rows, "sites": pca_sites},
            )
        else:
            arr = pts[name]
            if "time" in arr.dims:
                arr = arr.rename(time="case_num")
        if name == "spr":
            # Floor GP-extrapolated non-physical spread; see
            # MIN_DIRECTIONAL_SPREAD_DEG's module-level docstring for why.
            arr = arr.clip(min=MIN_DIRECTIONAL_SPREAD_DEG)
        out[name] = arr.sel(sites=site_ids)

    meta = cases.sel(sites=site_ids)
    ds = xr.Dataset(
        out,
        coords={name: meta[name] for name in SITE_META if name in meta.coords},
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
    goals: Iterable[str | int] | None = None,
) -> list[str]:
    """Site labels valid for multi-goal hindcast reconstruction.

    Intersects user *sites* selection (via :func:`resolve_sites`) with
    :func:`common_model_sites` so every retained site exists in all goals.

    Parameters
    ----------
    ds
        Reference dataset for site metadata (typically one goal's cases).
    sites
        Group name, umbrella (``buoy``/``isobath``/``mesh``), site id, or list;
        ``None`` returns all common sites.
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
    if sites is None:
        return sorted(common)

    selected = resolve_sites(ds, sites)
    kept = [s for s in selected if s in common]
    dropped = [s for s in selected if s not in common]
    if dropped:
        logger.warning(
            "Dropped %d site(s) not in all goals: %s",
            len(dropped),
            ", ".join(dropped[:8]) + ("…" if len(dropped) > 8 else ""),
        )
    if not kept:
        raise ValueError(
            "No sites left after intersecting with common model sites. "
            f"common count={len(common)}."
        )
    return kept


def _reconstruction_partitions(spectral: bool, offshore: xr.Dataset) -> list[int]:
    """Return WHACS partition ids used for one reconstruction mode.

    Parameters
    ----------
    spectral
        If ``False``, return bulk partition ``-1`` only.
    offshore
        Offshore dataset used to verify spectral partition variables exist.

    Returns
    -------
    list[int]
        ``[-1]`` for bulk-only, else ``[0, 1, 2, 3]``.

    Raises
    ------
    ValueError
        If spectral mode is requested but partition variables are missing.
    """
    if not spectral:
        return [BULK_PARTITION]
    missing = [i for i in SPECTRAL_PARTITIONS if f"phs{i}" not in offshore.data_vars]
    if missing:
        raise ValueError(
            "Offshore dataset missing partition variables: "
            + ", ".join(f"phs{i}" for i in missing)
        )
    return list(SPECTRAL_PARTITIONS)


def _valid_timesteps(
    forcing_df: pd.DataFrame,
    hs_offshore: xr.DataArray,
    partition: int,
    *,
    goal: str | int | None = None,
) -> pd.Series:
    """Return a boolean mask of usable hindcast timesteps.

    Parameters
    ----------
    forcing_df
        Offshore forcing indexed by time (or case id).
    hs_offshore
        Offshore significant height used for transfer-function scaling.
    partition
        WHACS partition id; bulk ``-1`` skips the ``phs > 0`` filter.
    goal
        Goal id. When provided, timesteps are additionally filtered to that
        goal's nautical propagation sector from the case goals JSON.

    Returns
    -------
    pandas.Series
        ``True`` where forcing is finite (and partition height is positive).
    """
    valid = forcing_df[list(FORCING_COLS)].notna().all(axis=1)
    if goal is not None:
        left, right = _goal_sector(goal)
        in_sector = pd.Series(
            in_nautical_sector(forcing_df["dir"].to_numpy(), left, right),
            index=forcing_df.index,
        )
        valid = valid & in_sector
    if partition == BULK_PARTITION:
        return valid
    row_dim = forcing_df.index.name or "time"
    hs = hs_offshore
    if hs.dims[0] != row_dim:
        hs = hs.rename({hs.dims[0]: row_dim})
    hs_pos = hs.reindex({row_dim: forcing_df.index}).fillna(0).values > 0
    return valid & hs_pos


def _offshore_forcing(
    offshore: xr.Dataset,
    goal: str | int,
    *,
    partition: int,
    wl: float = 0.0,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Build MDA forcing and offshore ``hs`` scale for one goal and partition.

    Parameters
    ----------
    offshore
        Offshore dataset already selected on ``time``.
    goal
        Goal id from the case goals JSON.
    partition
        Bulk ``-1`` uses ``hs/tp/dir/spr``; ``0``–``3`` use ``phs*`` variables.
    wl
        Constant water level appended to forcing (m).

    Returns
    -------
    forcing_df
        DataFrame with MDA input columns indexed by time.
    hs_offshore
        Offshore partition/bulk height aligned with forcing rows.
    """
    g = offshore.sel(goal=int(goal))
    if "seapoint" in g.coords:
        g = g.drop_vars("seapoint")

    if partition == BULK_PARTITION:
        part = g[["hs", "tp", "dir", "spr"]]
    else:
        i = int(partition)
        part = g[[f"phs{i}", f"ptp{i}", f"pdir{i}", f"pspr{i}"]].rename(
            {f"phs{i}": "hs", f"ptp{i}": "tp", f"pdir{i}": "dir", f"pspr{i}": "spr"}
        )

    df = part[["tp", "dir", "spr"]].to_dataframe()
    df["wl"] = wl
    return df, part["hs"]


def _predict_offshore(
    offshore: xr.Dataset,
    goal: str | int,
    n: int,
    site_ids: list[str],
    *,
    partition: int,
    wl: float = 0.0,
    pca_variance: PcaVariance = None,
    models: dict[str, Any] | None = None,
    device: str | None = None,
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
    partition
        WHACS partition id (bulk ``-1`` or spectral ``0``–``3``).
    wl
        Constant water level passed to MDA forcing (m).
    pca_variance
        PCA variance override; must match fitted models.
    models
        Pre-loaded model dict from :func:`load_models`; loaded if omitted.
    device
        Optional PyTorch device for GP inference when models are loaded here.

    Returns
    -------
    xarray.Dataset or None
        Predicted ``hs``, ``tp``, ``dir``, ``spr`` on ``(time, sites)``, or
        ``None`` when no valid timesteps remain.
    """
    forcing_df, hs_offshore = _offshore_forcing(
        offshore, goal, partition=partition, wl=wl
    )
    valid = _valid_timesteps(forcing_df, hs_offshore, partition, goal=goal)
    if not valid.any():
        return None

    pred = predict(
        goal,
        n,
        forcing_df.loc[valid],
        models=models,
        pca_variance=pca_variance,
        sites=site_ids,
        row_dim="time",
        device=device,
    )
    # Scale hs by offshore partition/bulk height (transfer function)
    pred["hs"] = pred["hs"] * hs_offshore.sel(time=pred.coords["time"])
    return pred


def _wavespectra_sum(
    cube: xr.Dataset,
    site_ids: list[str],
    *,
    variables: Mapping[str, str] | None = None,
    save_spectra: bool = False,
    load: bool = True,
) -> tuple[xr.Dataset, xr.Dataset | None]:
    """Linearly sum goal × partition spectra into bulk parameters.

    All wavespectra bulk stats are computed per site; the case JSON mapping
    (``metamodel.reconstruction.variables``) selects which stat becomes each
    HyGloW name (``hs``, ``tp``, ``dir``, ``spr``, …).

    Parameters
    ----------
    cube
        Dataset with ``goal``, ``partition``, ``time`` and ``sites`` dimensions.
    site_ids
        Nearshore site labels to process.
    variables
        HyGloW name → wavespectra stat. When omitted, reads
        ``metamodel.reconstruction.variables`` from the case JSON.
    save_spectra
        When ``True``, also return the summed spectrum on ``(time, sites, freq, dir)``.
        When omitted at the :func:`reconstruct` level, reads
        ``metamodel.reconstruction.save_spectra`` from the case JSON.
    load
        Eager-load the output dataset(s).

    Returns
    -------
    tuple[xarray.Dataset, xarray.Dataset | None]
        Bulk parameters on ``(time, sites)`` and optional ``efth`` spectra.

    Note
    ----
    ``jonswap``/``cartwright``/``construct_partition`` are plain xarray
    arithmetic (no explicit site loop inside wavespectra itself), so every
    site in *site_ids* is constructed and summed in one vectorised call
    instead of looping per site in Python. Callers with many sites should
    still chunk *cube*/*site_ids* themselves before calling this — see
    :func:`_site_batch_size` — since the ``efth1d * spread`` broadcast
    multiply inside ``construct_partition`` allocates the *full*
    ``(time, sites, goal x partition, freq, dir)`` array, so this function's
    own peak memory still scales with ``len(site_ids)``, just without the
    redundant per-site Python overhead.
    """
    variable_map = _reconstruction_variables(override=variables)
    output_names = tuple(variable_map)

    parts = cube.sel(sites=site_ids).stack(part=("goal", "partition"))
    # float32 here: measured only a ~3% direct speedup (jonswap/cartwright are
    # dominated by exp/cos evaluation, not memory bandwidth, so halving
    # precision doesn't roughly halve time the way a bandwidth-bound op
    # would) — but it does halve the peak (time, sites, goal x partition,
    # freq, dir) array _site_batch_size sizes against, so the same memory
    # budget fits ~2x more sites per call, meaningfully cutting how many
    # merge + construct_partition calls a given site count needs. Verified
    # correctness on a real reconstruction: max relative difference from
    # float64 was 1.2e-6, well below anything that matters physically here.
    spec = construct_partition(
        freq_name="jonswap",
        freq_kwargs={
            "freq": SPECTRAL_FREQ32,
            "fp": (1.0 / parts.tp).astype("float32"),
            "hs": parts.hs.astype("float32"),
        },
        dir_name="cartwright",
        dir_kwargs={
            "dir": SPECTRAL_DIR32,
            "dm": parts.dir.astype("float32"),
            "dspr": parts.spr.astype("float32"),
        },
    )
    combined = spec.sum(dim="part")
    if "sites" not in combined.dims:
        combined = combined.expand_dims(sites=list(site_ids))

    spectra: xr.Dataset | None = None
    if save_spectra:
        spectra = (
            combined
            if isinstance(combined, xr.Dataset)
            else combined.to_dataset(name="efth")
        )
        if load:
            spectra = spectra.load()

    all_stats = combined.spec.stats(list(WAVESPECTRA_STATS))
    result = xr.Dataset(
        {hyglow: all_stats[ws_stat] for hyglow, ws_stat in variable_map.items()}
    )[list(output_names)]
    if load:
        result = result.load()
    return result, spectra


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
    spectral: bool,
    offshore_name: str,
    goals: Iterable[str | int] | None,
    wl: float,
    site_ids: list[str],
    pca_variance: PcaVariance,
    variables: Mapping[str, str] | None = None,
) -> Path:
    """Build the NetCDF cache path for one reconstruction request.

    The filename encodes offshore source, MDA size, spectral mode, goals,
    water level, resolved site list, output variable mapping, PCA settings,
    metamodel config fingerprint (``site_filter`` + ``pca_preprocessing``),
    and time span.

    Parameters
    ----------
    time_label
        Time selector for this cache entry (month or full period).
    n
        MDA train size.
    spectral
        ``True`` for spectral partition sum; ``False`` for bulk only.
    offshore_name
        Processed hindcast key in the case config JSON.
    goals
        Goals included in reconstruction.
    wl
        Constant water level (m).
    site_ids
        Resolved nearshore site labels.
    pca_variance
        PCA variance override used for prediction.
    variables
        HyGloW name → wavespectra stat mapping. When omitted, uses the case JSON
        ``metamodel.reconstruction.variables``.

    Returns
    -------
    Path
        Target ``.nc`` path under ``tmp/reconstruct`` (not created on disk).
    """
    var_map = _reconstruction_variables(override=variables)
    stem = (
        "reconstruct-v5__"
        f"reconstruct__offshore-{offshore_name}__n-{n}__"
        f"{'spectral' if spectral else 'bulk'}__"
        f"goals-{_goals_label(goals)}__wl-{wl:g}__"
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
    spectral: bool,
    offshore_name: str,
    goals: Iterable[str | int] | None = None,
    wl: float = 0.0,
    site_ids: list[str],
    pca_variance: PcaVariance,
    variables: Mapping[str, str] | None = None,
    load: bool = True,
) -> xr.Dataset | None:
    """Load cached summed spectra written when ``save_spectra`` was enabled.

    Parameters
    ----------
    time_label
        Same time selector used for the paired bulk reconstruction.
    n
        MDA train size.
    spectral
        Spectral or bulk partition mode used for reconstruction.
    offshore_name
        Processed hindcast key in the case config JSON.
    goals
        Goals included in reconstruction.
    wl
        Constant water level (m).
    site_ids
        Resolved nearshore site labels.
    pca_variance
        PCA variance override used at fit time.
    variables
        HyGloW → wavespectra mapping; must match the bulk cache entry. When
        omitted, uses ``metamodel.reconstruction.variables``.
    load
        Eager-load arrays into memory.

    Returns
    -------
    xarray.Dataset or None
        ``efth`` on ``(time, sites, freq, dir)``, or ``None`` if no spectra
        cache exists for the request.
    """
    bulk_path = reconstruction_cache_path(
        time_label,
        n,
        spectral=spectral,
        offshore_name=offshore_name,
        goals=goals,
        wl=wl,
        site_ids=site_ids,
        pca_variance=pca_variance,
        variables=variables,
    )
    return _load_reconstruction_cache(
        reconstruction_spectra_cache_path(bulk_path), load=load
    )


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


def _load_models_by_goal(
    active: list[str],
    n: int,
    *,
    pca_variance: PcaVariance = None,
) -> dict[str, dict[str, Any]]:
    """Load fitted models once per goal for hindcast reconstruction.

    Parameters
    ----------
    active
        Goal ids to load.
    n
        MDA train size.
    pca_variance
        PCA variance override; must match fitted models.

    Returns
    -------
    dict[str, dict]
        Mapping ``goal_id → load_models(...)`` dict.
    """
    logger.info("Loading metamodels for goals %s", active)
    return {
        str(goal): load_models(goal, n, pca_variance=pca_variance) for goal in active
    }


def _reconstruct_period(
    offshore: xr.Dataset,
    n: int,
    *,
    time_label: str | slice,
    active: list[str],
    site_ids: list[str],
    spectral: bool = True,
    offshore_name: str = "WHACS",
    pca_variance: PcaVariance = None,
    wl: float = 0.0,
    load: bool = True,
    models_by_goal: dict[str, dict[str, Any]] | None = None,
    cache_path: Path | None = None,
    recompute: bool = False,
    use_cache: bool = True,
    device: str | None = None,
    save_spectra: bool | None = None,
    variables: Mapping[str, str] | None = None,
) -> xr.Dataset:
    """Reconstruct one offshore time slice already selected with ``.sel(time=...)``.

    Parameters
    ----------
    offshore
        Offshore forcing for the period to reconstruct.
    n
        MDA train size.
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
        Constant water level passed to MDA forcing (m).
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
            spectral=spectral,
            offshore_name=offshore_name,
            goals=active,
            wl=wl,
            site_ids=site_ids,
            pca_variance=pca_variance,
            variables=variables,
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
            else load_models(goal, n, pca_variance=pca_variance, device=device)
        )
        try:
            for part in part_ids:
                pred = _predict_offshore(
                    offshore,
                    goal,
                    n,
                    site_ids,
                    partition=part,
                    wl=wl,
                    pca_variance=pca_variance,
                    models=goal_models,
                    device=device,
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
    site_batch_size = _site_batch_size(
        offshore.sizes.get("time", 0), len(active) * len(part_ids)
    )
    result_parts: list[xr.Dataset] = []
    spectra_parts: list[xr.Dataset] = []
    for i in range(0, len(site_ids), site_batch_size):
        site_batch = site_ids[i : i + site_batch_size]
        merged_batch = xr.merge([c.sel(sites=site_batch) for c in chunks])
        result_batch, spectra_batch = _wavespectra_sum(
            merged_batch,
            site_batch,
            variables=variable_map,
            save_spectra=save_spectra,
            load=load,
        )
        result_parts.append(result_batch)
        if save_spectra and spectra_batch is not None:
            spectra_parts.append(spectra_batch)
        del merged_batch

    result = xr.concat(result_parts, dim="sites")
    spectra = xr.concat(spectra_parts, dim="sites") if spectra_parts else None

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


def _reconstruct_periods_batch(
    offshore_store: xr.Dataset,
    periods: list[str],
    n: int,
    *,
    active: list[str],
    site_ids: list[str],
    spectral: bool,
    pca_variance: PcaVariance,
    wl: float,
    cache_kw: Mapping[str, Any],
    use_cache: bool = True,
    device: str | None = None,
    save_spectra: bool | None = None,
    variables: Mapping[str, str] | None = None,
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
    n, active, site_ids, spectral, pca_variance, wl, use_cache, device,
    save_spectra, variables
        Same as :func:`reconstruct`.
    cache_kw
        Shared kwargs for :func:`reconstruction_cache_path` (built once by
        the caller, so every month's cache path matches what a normal
        single-month :func:`_reconstruct_period` call would have produced).

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
        goal_models = load_models(goal, n, pca_variance=pca_variance, device=device)
        try:
            for part in part_ids:
                pred = _predict_offshore(
                    combined_offshore,
                    goal,
                    n,
                    site_ids,
                    partition=part,
                    wl=wl,
                    pca_variance=pca_variance,
                    models=goal_models,
                    device=device,
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
    site_batch_size = _site_batch_size(
        combined_offshore.sizes.get("time", 0), len(active) * len(part_ids)
    )
    result_parts: list[xr.Dataset] = []
    spectra_parts: list[xr.Dataset] = []
    for i in range(0, len(site_ids), site_batch_size):
        site_batch = site_ids[i : i + site_batch_size]
        merged_batch = xr.merge([c.sel(sites=site_batch) for c in chunks])
        result_batch, spectra_batch = _wavespectra_sum(
            merged_batch,
            site_batch,
            variables=variable_map,
            save_spectra=save_spec,
            load=True,
        )
        result_parts.append(result_batch)
        if save_spec and spectra_batch is not None:
            spectra_parts.append(spectra_batch)
        del merged_batch

    combined_result = xr.concat(result_parts, dim="sites")
    combined_spectra = xr.concat(spectra_parts, dim="sites") if spectra_parts else None

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
    spectral: bool = True,
    offshore_name: str = "WHACS",
    goals: Iterable[str | int] | None = None,
    wl: float = 0.0,
    sites: str | Iterable[str] | None = None,
    pca_variance: PcaVariance = None,
    load: bool = True,
    recompute: bool = False,
    use_cache: bool = True,
    device: str | None = None,
    save_spectra: bool | None = None,
    variables: Mapping[str, str] | None = None,
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
    spectral
        ``False`` — bulk offshore partition ``-1`` per goal.
        ``True`` — WHACS partitions ``phs0``–``phs3`` (``0``–``3``).
    offshore_name
        Processed hindcast key in the case config JSON.
    goals
        Goals to include; default all configured goals.
    wl
        Constant water level for MDA forcing (m).
    sites
        Site selector; see :func:`resolve_sites`.
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
    validate_metamodel_config()
    require_models_all(n, active, pca_variance=pca_variance)

    template = load_cases_goal(active[0])
    site_ids = resolve_reconstruction_sites(template, sites, goals=active)
    variable_map = _reconstruction_variables(override=variables)

    cache_kw = dict(
        n=n,
        spectral=spectral,
        offshore_name=offshore_name,
        goals=goals,
        wl=wl,
        site_ids=site_ids,
        pca_variance=pca_variance,
        variables=variable_map,
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
    "SITE_META",
    "SPECTRAL_PARTITIONS",
    "VARS",
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
]
