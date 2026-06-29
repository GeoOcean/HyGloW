"""Samoa metamodel: MDA → per-variable PCA → stacked GPR.

mm.fit("1")
mm.fit_all()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Literal

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


class ModelsNotFittedError(FileNotFoundError):
    """PCA/GP pickles missing for the requested goal, n and variance."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def vars_to_predict() -> dict[str, VarSpec]:
    return METAMODEL["vars_to_predict"]


def _variance_tag(variance: float) -> str:
    return f"{variance:g}"


def resolve_pca_variance(variable: str, override: PcaVariance = None) -> float:
    spec = vars_to_predict()[variable]
    if spec == "raw":
        raise ValueError(f"{variable!r} is raw")
    if isinstance(override, dict) and variable in override:
        return float(override[variable])
    if isinstance(override, (int, float)):
        return float(override)
    return float(spec["pca_variance"])


def pca_variables(override: PcaVariance = None) -> list[tuple[str, float, list[str]]]:
    """(name, variance, vars_to_stack) in config order."""
    return [
        (name, resolve_pca_variance(name, override), spec["vars_to_stack"])
        for name, spec in vars_to_predict().items()
        if spec != "raw"
    ]


def resolved_pca_variances(override: PcaVariance = None) -> dict[str, float]:
    return {name: var for name, var, _ in pca_variables(override)}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def goal_output_dir(goal: str | int) -> Path:
    return resolve(f"outputs/goal_{int(goal):02d}")


def cases_path(goal: str | int) -> Path:
    return resolve(SIMULATIONS["metamodel_cases"]) / f"output_sites_{int(goal)}.nc"


def mda_path(goal: str | int) -> Path:
    return goal_output_dir(goal) / "mda.pkl"


def split_path(goal: str | int) -> Path:
    return goal_output_dir(goal) / "split.json"


def _model_dir(kind: str, goal: str | int, n: int) -> Path:
    template = METAMODEL["paths"][f"{kind}_dir"]
    return resolve(template.format(goal=f"goal_{int(goal):02d}", mda_candidate=n))


def gp_model_path(goal: str | int, n: int) -> Path:
    return _model_dir("gp", goal, n) / "gp_model.pkl"


def pca_model_path(goal: str | int, n: int, variable: str, variance: float) -> Path:
    return (
        _model_dir("pca", goal, n)
        / f"pca_{variable}_{_variance_tag(variance)}_model.pkl"
    )


def model_path(kind: str, goal: str | int, n: int) -> Path:
    if kind != "gp":
        raise ValueError("Use pca_model_path() for PCA")
    return gp_model_path(goal, n)


def missing_model_paths(
    goal: str | int, n: int, *, pca_variance: PcaVariance = None
) -> list[Path]:
    missing = []
    if not gp_model_path(goal, n).exists():
        missing.append(gp_model_path(goal, n))
    for name, variance, _ in pca_variables(pca_variance):
        p = pca_model_path(goal, n, name, variance)
        if not p.exists():
            missing.append(p)
    return missing


# ---------------------------------------------------------------------------
# Cases & split
# ---------------------------------------------------------------------------


def _site_filter() -> dict[str, float]:
    return {k: float(v) for k, v in METAMODEL.get("site_filter", {}).items()}


def detect_removed_sites(cases: xr.Dataset) -> list[str]:
    """Sites failing ``site_filter`` rules in ``Samoa.json`` (e.g. hs_max, tp_min)."""
    bad: set[str] = set()
    filt = _site_filter()

    if "hs_max" in filt:
        peak_hs = cases["hs"].max("case_num")
        bad.update(
            str(s)
            for s in peak_hs.where(peak_hs > filt["hs_max"], drop=True).sites.values
        )

    if "tp_min" in filt:
        min_tp = cases["tp"].min("case_num")
        bad.update(
            str(s)
            for s in min_tp.where(min_tp < filt["tp_min"], drop=True).sites.values
        )

    return sorted(bad)


def drop_sites(cases: xr.Dataset, site_ids: Iterable[str]) -> xr.Dataset:
    remove = {str(s) for s in site_ids}
    if not remove:
        return cases
    kept = [s for s in cases.sites.values if str(s) not in remove]
    return cases.sel(sites=kept)


def _read_split(goal: str | int) -> dict[str, Any]:
    with split_path(goal).open(encoding="utf-8") as f:
        return json.load(f)


def removed_sites(goal: str | int) -> list[str]:
    if not split_path(goal).exists():
        return []
    return list(_read_split(goal).get("removed_sites", []))


def load_cases_goal(goal: str | int, *, apply_site_filter: bool = True) -> xr.Dataset:
    from bluemath_tk.core.operations import get_uv_components

    ds = xr.open_dataset(cases_path(goal))
    u, v = get_uv_components(ds["dir"])
    ds = ds.assign(dir_u=u, dir_v=v)
    if apply_site_filter:
        ds = drop_sites(ds, removed_sites(goal))
    return ds


def forcing_dataframe(cases: xr.Dataset) -> pd.DataFrame:
    cols = forcing_columns()
    df = cases[list(cols.values())].to_dataframe()
    df.columns = METAMODEL["mda_inputs"]
    return df


def load_split(goal: str | int) -> tuple[np.ndarray, np.ndarray]:
    split = _read_split(goal)
    return (
        np.asarray(split["train_idx"], dtype=int),
        np.asarray(split["test_idx"], dtype=int),
    )


def load_context(goal: str | int) -> dict[str, Any]:
    from bluemath_tk.core.io import load_model

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


def models_exist(goal: str | int, n: int, *, pca_variance: PcaVariance = None) -> bool:
    return not missing_model_paths(goal, n, pca_variance=pca_variance)


def require_models(
    goal: str | int, n: int, *, pca_variance: PcaVariance = None
) -> None:
    missing = missing_model_paths(goal, n, pca_variance=pca_variance)
    if missing:
        raise ModelsNotFittedError(
            f"Models not fitted: goal={goal!r}, n={n}, "
            f"pca_variance={resolved_pca_variances(pca_variance)}. Missing:\n"
            + "\n".join(f"  {p}" for p in missing)
        )


def load_models(
    goal: str | int, n: int, *, pca_variance: PcaVariance = None
) -> dict[str, Any]:
    from bluemath_tk.core.io import load_model

    require_models(goal, n, pca_variance=pca_variance)
    pcas = {
        name: load_model(pca_model_path(goal, n, name, var))
        for name, var, _ in pca_variables(pca_variance)
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


def _goal_ids(goals: Iterable[str | int] | None = None) -> list[str]:
    if goals is None:
        return list(load_goals().keys())
    return [str(g) for g in goals]


def fit_mda(
    goal: str | int, *, force_retrain: bool = False
) -> tuple[Any, np.ndarray, np.ndarray]:
    from bluemath_tk.core.io import load_model
    from bluemath_tk.datamining.mda import MDA

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
            _site_filter(),
            ", ".join(dropped),
        )

    num_centers = mda_max_points()
    mda = MDA(num_centers=num_centers)
    mda.fit(
        forcing_dataframe(cases),
        directional_variables=METAMODEL["mda_directional_variables"],
    )

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
                "site_filter": _site_filter(),
            },
            f,
            indent=2,
        )
    logger.info(
        "Goal %s: MDA fitted (%d train, %d test)", goal, len(train_idx), len(test_idx)
    )
    return mda, train_idx, test_idx


def _stack_pcs(pcas: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for name, _ in vars_to_predict().items():
        if name not in pcas:
            continue
        df = pcas[name].pcs_df.copy()
        df.columns = [f"{name}__{c}" for c in df.columns]
        frames.append(df)
    return pd.concat(frames, axis=1)


def _split_pcs(pcs: np.ndarray, pcas: dict[str, Any]) -> dict[str, np.ndarray]:
    blocks, offset = {}, 0
    for name, _ in vars_to_predict().items():
        if name not in pcas:
            continue
        k = pcas[name].pcs_df.shape[1]
        blocks[name] = pcs[:, offset : offset + k]
        offset += k
    return blocks


def fit_models(
    goal: str | int,
    n: int,
    *,
    mda: Any | None = None,
    train_idx: np.ndarray | None = None,
    force_retrain: bool = False,
    gp_epochs: int = 1000,
    pca_variance: PcaVariance = None,
) -> dict[str, Any]:
    from bluemath_tk.datamining.pca import PCA
    from bluemath_tk.interpolation.gps import ExactGPInterpolation

    goal, n = str(goal), int(n)
    if n not in METAMODEL["mda_candidates"]:
        raise ValueError(f"{n} not in mda_candidates")
    if n > mda_max_points():
        raise ValueError(f"n={n} exceeds mda_max_points={mda_max_points()}")

    if not force_retrain and models_exist(goal, n, pca_variance=pca_variance):
        logger.info("Goal %s / n=%d: loaded models", goal, n)
        out = load_models(goal, n, pca_variance=pca_variance)
        out["train_idx"] = train_idx
        return out

    if mda is None or train_idx is None:
        mda, train_idx, _ = fit_mda(goal, force_retrain=force_retrain)

    cases = load_cases_goal(goal).isel(case_num=train_idx[:n])
    pcas: dict[str, Any] = {}

    for name, variance, stack in pca_variables(pca_variance):
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
        subset_data=mda.centroids[METAMODEL["mda_inputs"]].iloc[:n],
        subset_directional_variables=METAMODEL["mda_directional_variables"],
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
    goal = str(goal)
    mda, train_idx, _ = fit_mda(goal, force_retrain=force_retrain)
    for n in candidates or METAMODEL["mda_candidates"]:
        fit_models(
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
    for goal in _goal_ids(goals):
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


def _broadcast_forcing(
    forcing: pd.DataFrame, var: str, sites: np.ndarray
) -> xr.DataArray:
    v = forcing[var].to_numpy()
    return xr.DataArray(
        np.broadcast_to(v[:, None], (len(v), len(sites))),
        dims=("case_num", "sites"),
        coords={"case_num": forcing.index, "sites": sites},
    )


def predict_cases(
    goal: str | int,
    n: int,
    case_idx: Iterable[int],
    *,
    models: dict[str, Any] | None = None,
    pca_variance: PcaVariance = None,
) -> xr.Dataset:
    from bluemath_tk.core.operations import get_degrees_from_uv

    goal = str(goal)
    case_idx = np.asarray(list(case_idx), dtype=int)
    if models is None:
        models = load_models(goal, n, pca_variance=pca_variance)

    cases = load_cases_goal(goal)
    sites = cases.sites.values
    forcing = forcing_dataframe(cases).iloc[case_idx]
    blocks = _split_pcs(models["gp"].predict(dataset=forcing).values, models["pcas"])

    out: dict[str, Any] = {}
    for name, spec in vars_to_predict().items():
        if spec == "raw":
            out[name] = _broadcast_forcing(forcing, name, sites)
            continue
        pts = models["pcas"][name].inverse_transform(
            xr.Dataset(
                {"PCs": (("case_num", "n_component"), blocks[name])},
                coords={"case_num": forcing.index},
            )
        )
        if name == "dir":
            out["dir"] = (
                ("case_num", "sites"),
                get_degrees_from_uv(pts["dir_u"].values, pts["dir_v"].values),
            )
        else:
            out[name] = pts[name]

    return xr.Dataset(
        out,
        coords={
            "case_num": forcing.index,
            "sites": sites,
            "site_source": cases["site_source"],
            "site_label": cases["site_label"],
            "site_depth": cases["site_depth"],
            "site_longitude": cases["site_longitude"],
            "site_latitude": cases["site_latitude"],
        },
    )


def truth_cases(goal: str | int, case_idx: Iterable[int]) -> xr.Dataset:
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
    "ModelsNotFittedError",
    "PcaVariance",
    "cases_path",
    "detect_removed_sites",
    "drop_sites",
    "fit",
    "fit_all",
    "fit_mda",
    "fit_models",
    "forcing_dataframe",
    "goal_output_dir",
    "gp_model_path",
    "load_cases_goal",
    "load_context",
    "load_models",
    "load_split",
    "logger",
    "mda_path",
    "missing_model_paths",
    "model_path",
    "models_exist",
    "pca_model_path",
    "pca_variables",
    "predict_cases",
    "removed_sites",
    "require_models",
    "resolve_pca_variance",
    "resolved_pca_variances",
    "split_path",
    "truth_cases",
    "vars_to_predict",
]
