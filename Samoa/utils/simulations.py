"""LHS case generation for SnapWave metamodel runs."""

from __future__ import annotations

from typing import Mapping

import pandas as pd
from bluemath_tk.datamining.lhs import LHS
from utils.config import SIMULATIONS


def generate_cases(goals: Mapping[str, Mapping]) -> pd.DataFrame:
    """Generate LHS cases for each goal using ``SIMULATIONS`` from ``Samoa.json``."""

    bounds = SIMULATIONS["bounds"]
    n = SIMULATIONS["num_samples_per_goal"]
    tp_lo, tp_hi = bounds["tp"]
    spr_lo, spr_hi = bounds["spr"]
    wl_lo, wl_hi = bounds["wl"]

    frames: list[pd.DataFrame] = []
    for _goal_id, goal in goals.items():
        ang_lo, ang_hi = goal["angles"]
        lhs = LHS(num_dimensions=4)
        lhs_df = lhs.generate(
            dimensions_names=["tp", "dir", "spr", "wl"],
            lower_bounds=[tp_lo, ang_lo, spr_lo, wl_lo],
            upper_bounds=[tp_hi, ang_hi, spr_hi, wl_hi],
            num_samples=n,
        )
        lhs_df["dir"] = (90.0 - lhs_df["dir"]) % 360.0
        frames.append(lhs_df)

    return (
        pd.concat(frames, keys=list(goals.keys()))
        .reset_index()
        .rename(columns={"level_0": "goal"})
        .drop(columns=["level_1"])
    )
