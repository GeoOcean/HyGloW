"""Samoa metamodel training (MDA → PCA → GPR per goal).

Typical notebook flow::

    from utils import metamodel as mm

    mm.fit("1")          # one goal, all ``mda_candidates`` from config
    mm.fit_all()         # every goal

Models are written under ``outputs/goal_{NN}/``. Validation (:mod:`utils.validation`)
loads these pickles only — it never retrains.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.logging import get_file_logger
from utils.config import (
    METAMODEL,
    SIMULATIONS,
    forcing_columns,
    load_goals,
    mda_max_points,
    resolve,
)

_log_dir = resolve("outputs/logs")
_log_dir.mkdir(parents=True, exist_ok=True)
_makedirs = os.makedirs
os.makedirs = lambda path, *args, **kwargs: (
    _makedirs(path, exist_ok=True) if path else None
)

LOGGER_NAME = "Samoa.metamodel"
logger = get_file_logger(
    LOGGER_NAME,
    logs_path=str(resolve("outputs/logs/metamodel.log")),
    level="INFO",
    console=True,
    console_level="INFO",
)

os.makedirs = _makedirs


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def goal_output_dir(goal: str | int) -> Path:
    """``outputs/goal_{NN}/`` for one goal."""
    return resolve(f"outputs/goal_{int(goal):02d}")


def cases_path(goal: str | int) -> Path:
    """SnapWave metamodel cases NetCDF for one goal."""
    cases_dir = SIMULATIONS["metamodel_cases"]
    return resolve(cases_dir) / f"output_sites_{int(goal)}.nc"


def mda_path(goal: str | int) -> Path:
    return goal_output_dir(goal) / "mda.pkl"


def split_path(goal: str | int) -> Path:
    return goal_output_dir(goal) / "split.json"


def model_path(kind: str, goal: str | int, mda_candidate: int) -> Path:
    """PCA or GP pickle for one goal and MDA size ``n``."""
    template = METAMODEL["paths"][f"{kind}_dir"]
    return (
        resolve(
            template.format(goal=f"goal_{int(goal):02d}", mda_candidate=mda_candidate)
        )
        / f"{kind}_model.pkl"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _hs_max_threshold() -> float:
    return float(METAMODEL.get("site_filter", {}).get("hs_max", 10.0))


def detect_removed_sites(
    cases: xr.Dataset,
    *,
    hs_max: float | None = None,
) -> list[str]:
    """Sites whose peak Hs over all cases exceeds ``hs_max`` (unphysical SnapWave)."""
    limit = _hs_max_threshold() if hs_max is None else float(hs_max)
    peak_hs = cases["hs"].max("case_num")
    bad = peak_hs.where(peak_hs > limit, drop=True)
    return [str(s) for s in bad.sites.values]


def drop_sites(cases: xr.Dataset, site_ids: Iterable[str]) -> xr.Dataset:
    """Drop sites by id; no-op when ``site_ids`` is empty."""
    remove = {str(s) for s in site_ids}
    if not remove:
        return cases
    kept = [s for s in cases.sites.values if str(s) not in remove]
    return cases.sel(sites=kept)


def _read_split(goal: str | int) -> dict[str, Any]:
    with split_path(goal).open(encoding="utf-8") as f:
        return json.load(f)


def removed_sites(goal: str | int) -> list[str]:
    """Sites excluded from PCA/validation for this goal (from ``split.json``)."""
    if not split_path(goal).exists():
        return []
    return list(_read_split(goal).get("removed_sites", []))


def load_cases_goal(goal: str | int, *, apply_site_filter: bool = True) -> xr.Dataset:
    """Load propagated cases, add direction UV, and drop flagged sites."""
    from bluemath_tk.core.operations import get_uv_components

    ds = xr.open_dataset(cases_path(goal))
    u, v = get_uv_components(ds["dir"])
    ds = ds.assign(dir_u=u, dir_v=v)
    if apply_site_filter:
        ds = drop_sites(ds, removed_sites(goal))
    return ds


def forcing_dataframe(cases: xr.Dataset) -> pd.DataFrame:
    """Forcing table with MDA input column names (``tp``, ``dir``, …)."""
    cols = forcing_columns()
    inputs = METAMODEL["mda_inputs"]
    df = cases[list(cols.values())].to_dataframe()
    df.columns = inputs
    return df


def load_split(goal: str | int) -> tuple[np.ndarray, np.ndarray]:
    """Train and test case indices from a previous MDA fit."""
    split = _read_split(goal)
    train_idx = np.asarray(split["train_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    return train_idx, test_idx


def load_context(goal: str | int) -> dict[str, Any]:
    """Load saved MDA and train/test split (no fitting)."""
    from bluemath_tk.core.io import load_model

    goal = str(goal)
    if not mda_path(goal).exists() or not split_path(goal).exists():
        raise FileNotFoundError(
            f"Missing MDA for goal {goal!r}. Run mm.fit() or mm.fit_mda() first."
        )
    train_idx, test_idx = load_split(goal)
    return {
        "mda": load_model(mda_path(goal)),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


# ---------------------------------------------------------------------------
# Model I/O
# ---------------------------------------------------------------------------


def models_exist(goal: str | int, mda_candidate: int) -> bool:
    """True when both PCA and GP pickles exist for this goal / ``n``."""
    return (
        model_path("pca", goal, mda_candidate).exists()
        and model_path("gp", goal, mda_candidate).exists()
    )


def require_models(goal: str | int, mda_candidate: int) -> None:
    if not models_exist(goal, mda_candidate):
        raise FileNotFoundError(
            f"Missing PCA/GP for goal {goal!r}, n={mda_candidate}. Run mm.fit() first."
        )


def load_models(goal: str | int, mda_candidate: int) -> dict[str, Any]:
    """Load saved MDA, PCA and GP for one goal and candidate size."""
    from bluemath_tk.core.io import load_model

    require_models(goal, mda_candidate)
    return {
        "mda": load_model(mda_path(goal)),
        "pca": load_model(model_path("pca", goal, mda_candidate)),
        "gp": load_model(model_path("gp", goal, mda_candidate)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _goal_ids(goals: Iterable[str | int] | None = None) -> list[str]:
    if goals is None:
        return list(load_goals().keys())
    return [str(g) for g in goals]


def fit_mda(
    goal: str | int,
    *,
    force_retrain: bool = False,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Fit (or load) MDA with ``mda_max_points()`` centers; return train/test indices."""

    from bluemath_tk.core.io import load_model
    from bluemath_tk.datamining.mda import MDA

    goal = str(goal)
    path = mda_path(goal)
    split_file = split_path(goal)

    if not force_retrain and path.exists() and split_file.exists():
        mda = load_model(path)
        train_idx, test_idx = load_split(goal)
        n_removed = len(removed_sites(goal))
        logger.info(
            "Goal %s: loaded MDA (%d train, %d test, %d sites removed)",
            goal,
            len(train_idx),
            len(test_idx),
            n_removed,
        )
        return mda, train_idx, test_idx

    raw_cases = load_cases_goal(goal, apply_site_filter=False)
    dropped = detect_removed_sites(raw_cases)
    cases = drop_sites(raw_cases, dropped)
    if dropped:
        logger.warning(
            "Goal %s: removing %d site(s) with Hs > %.1f m: %s",
            goal,
            len(dropped),
            _hs_max_threshold(),
            ", ".join(dropped),
        )

    forcing = forcing_dataframe(cases)
    num_centers = mda_max_points()

    mda = MDA(num_centers=num_centers)
    mda.fit(forcing, directional_variables=METAMODEL["directional_variables"])

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
                "site_filter": {"hs_max": _hs_max_threshold()},
            },
            f,
            indent=2,
        )

    logger.info(
        "Goal %s: MDA fitted with %d centers (%d train, %d test, %d sites removed)",
        goal,
        num_centers,
        len(train_idx),
        len(test_idx),
        len(dropped),
    )
    return mda, train_idx, test_idx


def fit_models(
    goal: str | int,
    mda_candidate: int,
    *,
    mda: Any | None = None,
    train_idx: np.ndarray | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
) -> dict[str, Any]:
    """Train (or load) PCA + GPR using the first ``mda_candidate`` MDA train cases."""

    from bluemath_tk.core.io import load_model
    from bluemath_tk.datamining.pca import PCA
    from bluemath_tk.interpolation.gps import ExactGPInterpolation

    goal = str(goal)
    mda_candidate = int(mda_candidate)
    candidates = [int(n) for n in METAMODEL["mda_candidates"]]
    if mda_candidate not in candidates:
        raise ValueError(f"{mda_candidate} not in mda_candidates {candidates}")
    if mda_candidate > mda_max_points():
        raise ValueError(
            f"mda_candidate={mda_candidate} exceeds mda_max_points={mda_max_points()}"
        )

    pca_p = model_path("pca", goal, mda_candidate)
    gp_p = model_path("gp", goal, mda_candidate)

    if not force_retrain and pca_p.exists() and gp_p.exists():
        logger.info("Goal %s / n=%d: loaded existing PCA + GP", goal, mda_candidate)
        return {
            "mda": mda or load_model(mda_path(goal)),
            "pca": load_model(pca_p),
            "gp": load_model(gp_p),
            "train_idx": train_idx,
        }

    if mda is None or train_idx is None:
        mda, train_idx, _ = fit_mda(goal, force_retrain=force_retrain)

    sel = train_idx[:mda_candidate]
    cases = load_cases_goal(goal)
    inputs = METAMODEL["mda_inputs"]
    pca_var = float(METAMODEL["pca_variance"])

    pca = PCA(n_components=pca_var)
    pca.fit_transform(
        data=cases.isel(case_num=sel),
        vars_to_stack=METAMODEL["vars_to_stack"],
        coords_to_stack=["sites"],
        pca_dim_for_rows="case_num",
    )
    pca_p.parent.mkdir(parents=True, exist_ok=True)
    pca.save_model(pca_p)

    gp = ExactGPInterpolation(epochs=gp_epochs)
    gp.fit(
        subset_data=mda.centroids[inputs].iloc[:mda_candidate],
        subset_directional_variables=METAMODEL["directional_variables"],
        target_data=pca.pcs_df,
    )
    gp_p.parent.mkdir(parents=True, exist_ok=True)
    gp.save_model(gp_p)

    logger.info("Goal %s / n=%d: PCA + GP saved", goal, mda_candidate)
    return {"mda": mda, "pca": pca, "gp": gp, "train_idx": train_idx}


def fit(
    goal: str | int,
    *,
    candidates: Iterable[int] | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
) -> None:
    """Fit MDA once, then PCA/GP for each MDA candidate size (one goal)."""
    goal = str(goal)
    cand = [int(n) for n in (candidates or METAMODEL["mda_candidates"])]
    mda, train_idx, _ = fit_mda(goal, force_retrain=force_retrain)
    for n in cand:
        fit_models(
            goal,
            n,
            mda=mda,
            train_idx=train_idx,
            force_retrain=force_retrain,
            gp_epochs=gp_epochs,
        )


def fit_all(
    goals: Iterable[str | int] | None = None,
    *,
    candidates: Iterable[int] | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
) -> None:
    """Fit every goal (or a subset) for each MDA candidate size."""
    for goal in _goal_ids(goals):
        fit(
            goal,
            candidates=candidates,
            force_retrain=force_retrain,
            gp_epochs=gp_epochs,
        )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_cases(
    goal: str | int,
    mda_candidate: int,
    case_idx: Iterable[int],
    *,
    models: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Reconstruct ``hs``, ``tp``, ``dir`` for selected propagated cases."""
    from bluemath_tk.core.operations import get_degrees_from_uv

    goal = str(goal)
    case_idx = np.asarray(list(case_idx), dtype=int)
    if models is None:
        models = load_models(goal, mda_candidate)

    cases = load_cases_goal(goal)
    forcing = forcing_dataframe(cases).iloc[case_idx]

    pcs = models["gp"].predict(dataset=forcing).values
    points = models["pca"].inverse_transform(
        xr.Dataset(
            {"PCs": (("case_num", "n_component"), pcs)},
            coords={"case_num": forcing.index},
        )
    )

    return xr.Dataset(
        {
            "hs": points["hs"],
            "tp": points["tp"],
            "dir": (
                ("case_num", "sites"),
                get_degrees_from_uv(points["dir_u"].values, points["dir_v"].values),
            ),
        },
        coords={
            "case_num": forcing.index,
            "sites": cases["sites"].values,
            "site_source": cases["site_source"],
            "site_label": cases["site_label"],
            "site_depth": cases["site_depth"],
            "site_longitude": cases["site_longitude"],
            "site_latitude": cases["site_latitude"],
        },
    )


def truth_cases(goal: str | int, case_idx: Iterable[int]) -> xr.Dataset:
    """Observed SnapWave outputs for selected cases."""
    case_idx = np.asarray(list(case_idx), dtype=int)
    cases = load_cases_goal(goal)
    return cases.isel(case_num=case_idx)[["hs", "tp", "dir"]].assign(
        site_source=cases["site_source"],
        site_label=cases["site_label"],
        site_depth=cases["site_depth"],
        site_longitude=cases["site_longitude"],
        site_latitude=cases["site_latitude"],
    )


__all__ = [
    "LOGGER_NAME",
    "cases_path",
    "detect_removed_sites",
    "drop_sites",
    "fit",
    "fit_all",
    "fit_mda",
    "fit_models",
    "forcing_dataframe",
    "goal_output_dir",
    "load_cases_goal",
    "load_context",
    "load_models",
    "load_split",
    "logger",
    "mda_path",
    "model_path",
    "models_exist",
    "predict_cases",
    "removed_sites",
    "require_models",
    "split_path",
    "truth_cases",
]
