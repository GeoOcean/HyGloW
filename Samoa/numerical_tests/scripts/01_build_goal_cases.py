"""Linearity test, part 1: per hour, one single-goal SnapWave case per goal with the real offshore forcing (no GP).

Hours of the 2018 dynamic run (WHACS_cal, CODEC), spread over the offshore directional spread.
The cases use the metamodel templates and wrapper, as the training cases. Summed in energy they must
equal the dynamic run if the sum is linear (02_compare_exact_gp.py).

Run from the case folder: PYTHONPATH=. python numerical_tests/scripts/01_build_goal_cases.py
Runs go to tmp/linearity_test/ (git-ignored); the case table to numerical_tests/results/goal_cases.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils import simulation as sim
from utils.config import get_offshore_wave, load_goals, open_offshore_waves, open_water_level

OUT = str(Path("tmp/linearity_test/cases").resolve())
N_PER_BIN = 8


def pick_hours(off) -> list[pd.Timestamp]:
    spr = off["spr"].mean("goal").to_series()
    bins = [(0, 45), (50, 58), (62, 90)]
    rng = np.random.default_rng(0)
    hours = [pd.Timestamp("2018-02-09 21:00")]  # Gita peak
    for lo, hi in bins:
        pool = spr[(spr >= lo) & (spr < hi)].index
        hours += list(rng.choice(pool, size=min(N_PER_BIN, len(pool)), replace=False))
    return sorted(pd.Timestamp(h) for h in hours)


def main() -> None:
    off = open_offshore_waves(entry=get_offshore_wave("WHACS_cal")).sel(time="2018").load()
    wl = open_water_level().sel(time="2018").load()
    hours = pick_hours(off)
    rows = []
    for h in hours:
        for g in off.goal.values:
            o = off.sel(time=h, goal=g)
            rows.append(dict(goal=str(int(g)), active_node=int(g), hs=float(o.hs), tp=float(o.tp),
                             dir=float(o.dir), spr=float(o.spr),
                             wl=float(wl.sel(time=h, goal=g).fillna(0.0)), time=h, save_map=False))
    cases = pd.DataFrame(rows)
    cases.to_csv("tmp/linearity_test/cases.csv", index=False)
    cases.to_csv("numerical_tests/results/goal_cases.csv", index=False)
    print(cases.groupby("time").spr.agg(["mean", "max"]).round(1))
    setup = sim.HyWavesWrapperSetup.from_case(load_goals())
    wrapper = setup.metamodel_wrapper(cases.drop(columns="time"), output_dir=OUT)
    wrapper.build_cases(cases_name_format=lambda ctx: f"{ctx.get('case_num'):04}")
    wrapper.run_cases_slurm(**sim.slurm_kwargs(max_parallel=200))
    print(f"{len(cases)} cases submitted ({len(hours)} hours x {off.sizes['goal']} goals)")


if __name__ == "__main__":
    main()
