"""GPR metamodel on SnapWave LHS cases (MDA → per-variable PCA → stacked GP).

Training: :func:`fit`, :func:`fit_all`
Prediction: :func:`predict` (LHS ``case_num`` or hindcast ``time`` via ``row_dim``)
Hindcast: :func:`reconstruct` — ``spectral=False`` bulk (partition ``-1``),
``spectral=True`` WHACS partitions ``0``–``3``; both via wavespectra linear sum.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.io import load_model
from bluemath_tk.core.logging import get_file_logger
from bluemath_tk.core.operations import get_degrees_from_uv, get_uv_components
from bluemath_tk.datamining.mda import MDA
from bluemath_tk.datamining.pca import PCA
from bluemath_tk.interpolation.gps import ExactGPInterpolation
from utils.config import (
    METAMODEL,
    SIMULATIONS,
    forcing_columns,
    load_goals,
    mda_max_points,
    open_offshore_waves,
    resolve,
)
from wavespectra.construct import construct_partition

LOGGER_NAME = "Samoa.metamodel"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("logs/metamodel.log")),
    level="INFO",
    console=True,
    console_level="INFO",
)

PcaVariance = float | dict[str, float] | None
VarSpec = Literal["raw"] | dict[str, Any]

# Samoa.json → metamodel (single source; do not duplicate in other modules)
VARS = METAMODEL["vars_to_predict"]
MDA_INPUTS = METAMODEL["mda_inputs"]
MDA_DIRECTIONAL = METAMODEL["mda_directional_variables"]
MDA_CANDIDATES = METAMODEL["mda_candidates"]
SITE_FILTER = METAMODEL["site_filter"]
FORCING_COLS = tuple(c for c in MDA_INPUTS if c != "wl")  # MDA inputs passed through GP

# Wavespectra grid and WHACS partition ids (see offshore_waves.py)
SPECTRAL_FREQ = np.logspace(np.log10(0.035), np.log10(0.5), 29)
SPECTRAL_DIR = np.arange(0.0, 360.0, 5)
# WHACS: bulk = partition -1; wind sea + swells = 0–3 (see offshore_waves.py)
BULK_PARTITION = -1
SPECTRAL_PARTITIONS = (0, 1, 2, 3)
SITE_META = (
    "site_source",
    "site_label",
    "site_depth",
    "site_longitude",
    "site_latitude",
)


class ModelsNotFittedError(FileNotFoundError):
    """Raised when PCA/GP pickles are missing for a goal, *n* and PCA variance."""

    _FIT_HINT = " Run ``06_Validate_Metamodel.ipynb`` (``mm.fit_all``) first."


def _resolve_pca_variance(variable: str, override: PcaVariance = None) -> float:
    """PCA variance for one stacked variable (JSON default or *override*)."""
    spec = VARS[variable]
    if spec == "raw":
        raise ValueError(f"{variable!r} is raw")
    if isinstance(override, dict) and variable in override:
        return float(override[variable])
    if isinstance(override, (int, float)):
        return float(override)
    return float(spec["pca_variance"])


def _pca_variables(override: PcaVariance = None) -> list[tuple[str, float, list[str]]]:
    """``(name, variance, vars_to_stack)`` for every non-raw entry in ``VARS``."""
    return [
        (name, _resolve_pca_variance(name, override), spec["vars_to_stack"])
        for name, spec in VARS.items()
        if spec != "raw"
    ]


def resolved_pca_variances(override: PcaVariance = None) -> dict[str, float]:
    """PCA variance fraction per stacked variable (from ``Samoa.json`` or *override*)."""
    return {name: var for name, var, _ in _pca_variables(override)}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _variance_tag(variance: float) -> str:
    """Filename-safe PCA variance label (e.g. ``0.95`` → ``0.95``)."""
    return f"{variance:g}"


def goal_output_dir(goal: str | int) -> Path:
    """Directory for one goal's MDA, split, PCA and GP artefacts."""
    return resolve(f"outputs/goal_{int(goal):02d}")


def cases_path(goal: str | int) -> Path:
    """SnapWave LHS NetCDF for one goal."""
    return resolve(SIMULATIONS["metamodel_cases"]) / f"output_sites_{int(goal)}.nc"


def mda_path(goal: str | int) -> Path:
    """Fitted MDA pickle for one goal."""
    return goal_output_dir(goal) / "mda.pkl"


def split_path(goal: str | int) -> Path:
    """Train/test index JSON written by :func:`_fit_mda`."""
    return goal_output_dir(goal) / "split.json"


def gp_model_path(goal: str | int, n: int) -> Path:
    """Fitted GP pickle for one goal and MDA train size."""
    template = METAMODEL["paths"]["gp_dir"]
    return (
        resolve(template.format(goal=f"goal_{int(goal):02d}", mda_candidate=n))
        / "gp_model.pkl"
    )


def pca_model_path(goal: str | int, n: int, variable: str, variance: float) -> Path:
    """Fitted PCA pickle for one goal, MDA size and stacked variable."""
    template = METAMODEL["paths"]["pca_dir"]
    base = resolve(template.format(goal=f"goal_{int(goal):02d}", mda_candidate=n))
    return base / f"pca_{variable}_{_variance_tag(variance)}_model.pkl"


# ---------------------------------------------------------------------------
# Cases & site filter
# ---------------------------------------------------------------------------


def detect_removed_sites(cases: xr.Dataset) -> list[str]:
    """Site labels failing ``METAMODEL.site_filter`` thresholds on SnapWave cases."""
    bad: set[str] = set()
    if "hs_max" in SITE_FILTER:
        peak = cases["hs"].max("case_num")
        bad.update(
            str(s)
            for s in peak.where(peak > SITE_FILTER["hs_max"], drop=True).sites.values
        )
    if "tp_min" in SITE_FILTER:
        low = cases["tp"].min("case_num")
        bad.update(
            str(s)
            for s in low.where(low < SITE_FILTER["tp_min"], drop=True).sites.values
        )
    return sorted(bad)


def drop_sites(cases: xr.Dataset, site_ids: Iterable[str]) -> xr.Dataset:
    """Return *cases* without the listed ``sites`` coordinate labels."""
    remove = {str(s) for s in site_ids}
    if not remove:
        return cases
    kept = [s for s in cases.sites.values if str(s) not in remove]
    return cases.sel(sites=kept)


def removed_sites(goal: str | int) -> list[str]:
    """Sites dropped at training time for *goal* (from ``split.json``)."""
    if not split_path(goal).exists():
        return []
    with split_path(goal).open(encoding="utf-8") as f:
        return list(json.load(f).get("removed_sites", []))


def load_cases_goal(goal: str | int, *, apply_site_filter: bool = True) -> xr.Dataset:
    """SnapWave LHS output for one goal, with ``dir_u``/``dir_v`` and optional site filter.

    Parameters
    ----------
    goal
        Goal id (as in ``Samoa_goals.json``).
    apply_site_filter
        If ``True``, drop sites listed in ``split.json`` ``removed_sites``.

    Returns
    -------
    xarray.Dataset
        Dimensions ``(case_num, sites)``; variables ``hs``, ``tp``, ``dir``, …
    """
    ds = xr.open_dataset(cases_path(goal))
    u, v = get_uv_components(ds["dir"])
    ds = ds.assign(dir_u=u, dir_v=v)
    if apply_site_filter:
        ds = drop_sites(ds, removed_sites(goal))
    return ds


def forcing_dataframe(cases: xr.Dataset) -> pd.DataFrame:
    """MDA input columns for each LHS case (index ``case_num``)."""
    cols = forcing_columns()
    df = cases[list(cols.values())].to_dataframe()
    df.columns = MDA_INPUTS
    return df


def load_split(goal: str | int) -> tuple[np.ndarray, np.ndarray]:
    """Train and test ``case_num`` indices from ``split.json``."""
    with split_path(goal).open(encoding="utf-8") as f:
        split = json.load(f)
    return (
        np.asarray(split["train_idx"], dtype=int),
        np.asarray(split["test_idx"], dtype=int),
    )


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
    missing: list[Path] = []
    if not gp_model_path(goal, n).exists():
        missing.append(gp_model_path(goal, n))
    for name, variance, _ in _pca_variables(pca_variance):
        p = pca_model_path(goal, n, name, variance)
        if not p.exists():
            missing.append(p)
    return missing


def models_exist(goal: str | int, n: int, *, pca_variance: PcaVariance = None) -> bool:
    """Whether GP and all PCA pickles exist for *goal* and MDA size *n*."""
    return not _missing_model_paths(goal, n, pca_variance=pca_variance)


def require_models(
    goal: str | int, n: int, *, pca_variance: PcaVariance = None
) -> None:
    """Raise :class:`ModelsNotFittedError` if any model file is missing."""
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
    """Call :func:`require_models` for every active goal."""
    for goal in _goals(goals):
        require_models(goal, n, pca_variance=pca_variance)


def load_models(
    goal: str | int, n: int, *, pca_variance: PcaVariance = None
) -> dict[str, Any]:
    """Load fitted MDA, GP and per-variable PCA models from disk.

    Parameters
    ----------
    goal, n
        Goal id and MDA train size (must match saved artefacts).
    pca_variance
        Override PCA variance fractions; default uses ``Samoa.json``.

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
    return {
        "mda": load_model(mda_path(goal)),
        "gp": load_model(gp_model_path(goal, n)),
        "pcas": pcas,
        "pca_variances": resolved_pca_variances(pca_variance),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _goals(goals: Iterable[str | int] | None = None) -> list[str]:
    """Active goal ids (default: all keys in ``Samoa_goals.json``)."""
    if goals is None:
        return list(load_goals().keys())
    return [str(g) for g in goals]


def _fit_mda(
    goal: str | int, *, force_retrain: bool = False
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit or load MDA centroids and persist train/test split for one goal.

    Applies ``site_filter``, fits MDA on offshore forcing columns, and writes
    ``mda.pkl`` plus ``split.json`` (train indices = centroid cases, rest = test).

    Returns
    -------
    mda, train_idx, test_idx
        Fitted MDA object and integer case indices.
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
    with split_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train_idx": train_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "num_centers": num_centers,
                "removed_sites": dropped,
                "site_filter": SITE_FILTER,
            },
            f,
            indent=2,
        )
    logger.info(
        "Goal %s: MDA fitted (%d train, %d test)", goal, len(train_idx), len(test_idx)
    )
    return mda, train_idx, test_idx


def _stack_pcs(pcas: dict[str, Any]) -> pd.DataFrame:
    """Concatenate per-variable PC columns for GP target matrix."""
    frames = []
    for name in pcas:
        df = pcas[name].pcs_df.copy()
        df.columns = [f"{name}__{c}" for c in df.columns]
        frames.append(df)
    return pd.concat(frames, axis=1)


def _split_pcs(pcs: np.ndarray, pcas: dict[str, Any]) -> dict[str, np.ndarray]:
    """Split stacked GP output back into per-variable PC blocks."""
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
    n
        Number of MDA centroids (must be in ``mda_candidates``).
    mda, train_idx
        Optional pre-fitted MDA context from :func:`_fit_mda`.
    force_retrain
        Re-fit even when pickles already exist.
    gp_epochs
        GP optimisation iterations.
    pca_variance
        Override per-variable PCA variance; default from ``Samoa.json``.

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
    pcas: dict[str, Any] = {}
    for name, variance, stack in _pca_variables(pca_variance):
        pca = PCA(n_components=variance)
        pca.fit_transform(
            data=cases,
            vars_to_stack=stack,
            coords_to_stack=["sites"],
            pca_dim_for_rows="case_num",
        )
        path = pca_model_path(goal, n, name, variance)
        path.parent.mkdir(parents=True, exist_ok=True)
        pca.save_model(path)
        pcas[name] = pca
        logger.info("Goal %s / n=%d: saved %s", goal, n, path.name)

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
    """Train MDA + PCA + GP for one goal and all MDA candidate sizes."""
    goal = str(goal)
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
    """Run :func:`fit` for every goal (default: all in ``Samoa_goals.json``)."""
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
        models = load_models(goal, n, pca_variance=pca_variance)

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
            # tp, spr, wl: identical at every site for a given forcing row
            v = forcing[name].to_numpy()
            out[name] = xr.DataArray(
                np.broadcast_to(v[:, None], (len(v), len(all_sites))),
                dims=("case_num", "sites"),
                coords={"case_num": rows, "sites": all_sites},
            ).sel(sites=site_ids)
            continue
        pts = models["pcas"][name].inverse_transform(
            xr.Dataset(
                {"PCs": (("case_num", "n_component"), blocks[name])},
                coords={"case_num": rows},
            )
        )
        if name == "dir":
            arr = xr.DataArray(
                get_degrees_from_uv(pts["dir_u"].values, pts["dir_v"].values),
                dims=("case_num", "sites"),
                coords={"case_num": rows, "sites": all_sites},
            )
        else:
            arr = pts[name]
            if "time" in arr.dims:
                arr = arr.rename(time="case_num")  # PCA may keep hindcast dim name
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
    """Intersection of site labels kept after ``site_filter`` in every goal.

    Use before multi-goal hindcast reconstruction so all goals predict the
    same nearshore points.
    """
    active = _goals(goals)
    common = set(str(s) for s in load_cases_goal(active[0]).sites.values)
    for goal in active[1:]:
        common &= {str(s) for s in load_cases_goal(goal).sites.values}
    return sorted(common)


def resolve_sites(
    ds: xr.Dataset,
    sites: str | Iterable[str] | None = None,
) -> list[str]:
    """Resolve a site selector to concrete ``sites`` coordinate labels.

    Accepts metric group names (from validation plots), umbrella sources
    (``buoy``, ``isobath``, ``mesh``), individual site ids, or a list of
    those. ``None`` returns every site in *ds*.
    """
    from utils.validation import metric_groups

    labels = [str(s) for s in ds.sites.values]
    if sites is None:
        return labels

    def from_selector(selector: str) -> list[str]:
        if selector in ("buoy", "isobath", "mesh"):
            return [
                sid
                for _name, source, ids in metric_groups(ds)
                if source == selector
                for sid in ids
            ]
        for name, _source, ids in metric_groups(ds):
            if name == selector:
                return list(ids)
        if selector in labels:
            return [selector]
        raise ValueError(
            f"Unknown site selector {selector!r}. "
            f"Use a group name ({[g[0] for g in metric_groups(ds)]}), "
            f"umbrella ('buoy', 'isobath', 'mesh'), or a site id."
        )

    if isinstance(sites, str):
        return from_selector(sites)

    out: list[str] = []
    seen: set[str] = set()
    for item in sites:
        for sid in resolve_sites(ds, item):
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


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
    """``[-1]`` for bulk-only; ``[0, 1, 2, 3]`` for WHACS spectral partitions."""
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
) -> pd.Series:
    """Boolean mask of timesteps with finite forcing (and ``phs > 0`` if not bulk)."""
    valid = forcing_df[list(FORCING_COLS)].notna().all(axis=1)
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
    """MDA forcing DataFrame and offshore ``hs`` scale for one goal × partition.

    ``partition == -1``: bulk ``hs/tp/dir/spr``. ``0``–``3``: ``phs0``…``phs3`` renamed.
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
) -> xr.Dataset | None:
    """Metamodel prediction for one goal × partition on valid timesteps only."""
    forcing_df, hs_offshore = _offshore_forcing(
        offshore, goal, partition=partition, wl=wl
    )
    valid = _valid_timesteps(forcing_df, hs_offshore, partition)
    if not valid.any():
        return None

    pred = predict(
        goal,
        n,
        forcing_df.loc[valid],
        pca_variance=pca_variance,
        sites=site_ids,
        row_dim="time",
    )
    # Scale hs by offshore partition/bulk height (transfer function)
    pred["hs"] = pred["hs"] * hs_offshore.sel(time=pred.coords["time"])
    return pred


def _wavespectra_sum(
    cube: xr.Dataset, site_ids: list[str], *, load: bool = True
) -> xr.Dataset:
    """Jonswap × Cartwright linear sum over goal × partition for each site."""
    out: list[xr.Dataset] = []
    for i, site in enumerate(site_ids, start=1):
        logger.info("Wavespectra site %3d / %d: %s", i, len(site_ids), site)
        parts = cube.sel(sites=[site]).stack(part=("goal", "partition"))
        spec = construct_partition(
            freq_name="jonswap",
            freq_kwargs={"freq": SPECTRAL_FREQ, "fp": 1.0 / parts.tp, "hs": parts.hs},
            dir_name="cartwright",
            dir_kwargs={"dir": SPECTRAL_DIR, "dm": parts.dir, "dspr": parts.spr},
        )
        bulk = spec.sum(dim="part").spec.stats(["hs", "tp", "dm"])
        if "sites" not in bulk.dims:
            bulk = bulk.expand_dims(sites=[site])
        out.append(bulk)

    result = xr.concat(out, dim="sites")
    if "dm" in result:
        result = result.rename(dm="dir")  # wavespectra stats name
    return result.load() if load else result


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
) -> xr.Dataset:
    """Reconstruct nearshore waves: metamodel per goal × partition, wavespectra sum.

    Parameters
    ----------
    time
        Period passed to the offshore zarr (e.g. ``"2020-05"``).
    n
        MDA train size; fitted models must exist for every active goal.
    spectral
        ``False`` — bulk ``hs/tp/dir/spr`` only (partition ``-1``).
        ``True`` — WHACS partitions ``phs0``–``phs3`` (``0``–``3``).
    offshore_name
        Processed hindcast key in ``Samoa.json``.
    goals
        Goals to include; default all.
    wl
        Constant water level for MDA forcing (m).
    sites
        Site selector; see :func:`resolve_sites`.
    pca_variance
        PCA fractions used at fit time.
    load
        Eager-load the output dataset.

    Returns
    -------
    xarray.Dataset
        ``(time, sites)`` with ``hs``, ``tp``, ``dir``.
    """
    active = _goals(goals)
    require_models_all(n, active, pca_variance=pca_variance)

    offshore = open_offshore_waves(offshore_name).sel(time=time)
    template = load_cases_goal(active[0])
    site_ids = resolve_reconstruction_sites(template, sites, goals=active)
    part_ids = _reconstruction_partitions(spectral, offshore)

    logger.info(
        "Reconstruct time=%r n=%d goals=%s partitions=%s sites=%d",
        time,
        n,
        active,
        part_ids,
        len(site_ids),
    )

    chunks: list[xr.Dataset] = []
    for goal in active:
        for part in part_ids:
            pred = _predict_offshore(
                offshore,
                goal,
                n,
                site_ids,
                partition=part,
                wl=wl,
                pca_variance=pca_variance,
            )
            if pred is None:
                logger.debug(
                    "Goal %s partition %s: no valid timesteps in %r",
                    goal,
                    part,
                    time,
                )
                continue
            chunks.append(pred.expand_dims(goal=[int(goal)], partition=[part]).load())

    if not chunks:
        raise ValueError("Reconstruction produced no data (check time range)")

    result = _wavespectra_sum(xr.merge(chunks), site_ids, load=load)
    logger.info(
        "Reconstruction done: %d site(s), %.2f MB",
        result.sizes["sites"],
        result.nbytes * 1e-6,
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
    "require_models",
    "require_models_all",
    "resolve_reconstruction_sites",
    "resolve_sites",
    "resolved_pca_variances",
]
