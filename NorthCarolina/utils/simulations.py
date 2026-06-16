"""LHS case generation for SnapWave runs.

All numeric defaults can be overridden by passing the ``simulations``
section of ``NorthCarolina.json`` (a dict with ``num_samples_per_goal``,
``bounds`` and ``physical_filter`` keys) via the ``sims_config`` argument.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.waves.series import waves_dispersion

DEFAULT_BOUNDS: dict[str, list[float]] = {
    "tp": [1.0, 15.0],
    "ds": [1.0, 45.0],
    "zs": [-3.5, 3.5],
}
DEFAULT_FILTER: dict[str, float] = {"kh_max": 1.0, "tp_min": 7.0}
DEFAULT_NUM_SAMPLES: int = 100


def get_real_scenarios_from_dataset(
    lhs_dataset: pd.DataFrame,
    h0: float,
    *,
    kh_max: float = DEFAULT_FILTER["kh_max"],
    tp_min: float = DEFAULT_FILTER["tp_min"],
) -> pd.DataFrame:
    """Filter the LHS dataset to keep only physically realistic sea states.

    Parameters
    ----------
    lhs_dataset
        DataFrame with at least ``hs``, ``tp``, ``wd``, ``ds`` columns.
    h0
        Water depth in meters used in the dispersion relation.
    kh_max
        Maximum admissible ``k * h`` (default 1.0).
    tp_min
        Minimum admissible peak period in seconds (default 7.0).
    """

    df = lhs_dataset.copy()
    df["L"] = [waves_dispersion(t, h0)[0] for t in df["tp"]]
    df["h/L"] = h0 / df["L"]
    df["kh"] = (2.0 * np.pi / df["L"]) * h0
    df = df.loc[(df["kh"] < kh_max) & (df["tp"] > tp_min) & (df["h/L"] < 0.5)]
    return lhs_dataset.loc[df.index]


def generate_cases(
    goals_config: Mapping[str, Mapping[str, Any]],
    num_samples_per_goal: int | None = None,
    *,
    sims_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate LHS cases for each goal node.

    Parameters
    ----------
    goals_config
        Mapping ``goal_id -> {"angles": [low, high], ...}``.
    num_samples_per_goal
        Number of LHS samples per goal. If ``None``, falls back to
        ``sims_config["num_samples_per_goal"]`` and finally
        :data:`DEFAULT_NUM_SAMPLES`.
    sims_config
        Optional ``simulations`` section from the case JSON. May provide
        ``num_samples_per_goal`` and a ``bounds`` mapping for ``tp``,
        ``ds`` and ``zs``.
    """

    cfg = dict(sims_config or {})
    bounds = {**DEFAULT_BOUNDS, **(cfg.get("bounds") or {})}
    n = (
        num_samples_per_goal
        if num_samples_per_goal is not None
        else int(cfg.get("num_samples_per_goal", DEFAULT_NUM_SAMPLES))
    )

    tp_lo, tp_hi = bounds["tp"]
    ds_lo, ds_hi = bounds["ds"]
    zs_lo, zs_hi = bounds["zs"]

    list_of_lhs_dfs: list[pd.DataFrame] = []
    for _goal_id, goal_values in goals_config.items():
        ang_lo, ang_hi = goal_values["angles"]
        lhs = LHS(num_dimensions=4)
        lhs_df = lhs.generate(
            dimensions_names=["tp", "wd", "ds", "zs"],
            lower_bounds=[tp_lo, ang_lo, ds_lo, zs_lo],
            upper_bounds=[tp_hi, ang_hi, ds_hi, zs_hi],
            num_samples=n,
        )
        lhs_df["wd"] = (90.0 - lhs_df["wd"]) % 360.0
        list_of_lhs_dfs.append(lhs_df)

    return (
        pd.concat(list_of_lhs_dfs, keys=list(goals_config.keys()))
        .reset_index()
        .rename(columns={"level_0": "goal"})
        .drop(columns=["level_1"])
    )


__all__ = [
    "DEFAULT_BOUNDS",
    "DEFAULT_FILTER",
    "DEFAULT_NUM_SAMPLES",
    "get_real_scenarios_from_dataset",
    "generate_cases",
]
