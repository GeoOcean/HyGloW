"""Linearity test, part 3: the same hours with parts of SnapWave switched off, one at a time.

Variants (lines replaced in copies of the metamodel and dynamic templates):
  A  fw = 0              (no bottom friction)
  B  fw = 0, gamma = 10  (no friction, practically no depth-induced breaking)
  C  sector = 360        (full directional circle; SnapWave centres a 180 deg sector on the mean
                          boundary direction of each run)
  D  C + B, and the goal cases forced with the hour's mean Tp and water level (SnapWave runs one
     domain Tp = the boundary mean, and the depth of the mean level): every known nonlinearity removed
For each: the 16 single-goal cases per hour (cases.csv of part 1) and the all-goals dynamic case of the
same hour (04_compare_variants.py).

Run from the case folder: PYTHONPATH=. python numerical_tests/scripts/03_physics_variants.py <A|B|C|D>
(after 01). E and F are not built here: see 04 and the README (test build of SnapWave).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from utils import simulation as sim
from utils.config import FORCING, get_offshore_wave, load_goals, open_offshore_waves

ROOT = Path("tmp/linearity_test").resolve()
VARIANTS = {
    "A": {"fw": "0.0"},
    "B": {"fw": "0.0", "gamma": "10.0"},
    "C": {"sector": "360"},
    "D": {"sector": "360", "fw": "0.0", "gamma": "10.0"},
}


def patched_templates(kind: str, variant: str) -> Path:
    src = Path(sim.templates_dir(kind))
    dst = ROOT / f"templates_{kind}_{variant}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    inp = dst / "snapwave.inp"
    keys = VARIANTS[variant]
    lines = []
    for line in inp.read_text().splitlines():
        prefix, _, rest = line.rpartition("%}")  # the dynamic template sets sector inside a Jinja statement
        key = (rest if prefix else line).split("=")[0].strip()
        if key in keys:
            if prefix:
                lines.append(prefix + "%}")
            continue
        lines.append(line)
    lines += [f"{k:<15}= {v}" for k, v in VARIANTS[variant].items()]
    inp.write_text("\n".join(lines) + "\n")
    return dst


def main(variant: str) -> None:
    cases = pd.read_csv(ROOT / "cases.csv", parse_dates=["time"])
    if variant == "D":
        cases[["tp", "wl"]] = cases.groupby("time")[["tp", "wl"]].transform("mean")
        cases.to_csv(ROOT / "cases_D.csv", index=False)
    hours = sorted(cases.time.unique())
    setup = sim.HyWavesWrapperSetup.from_case(load_goals())

    goal_cases = setup.metamodel_wrapper(
        cases.drop(columns="time"), output_dir=str(ROOT / f"cases_{variant}"),
        templates=patched_templates("metamodel", variant),
    )
    goal_cases.build_cases(cases_name_format=lambda ctx: f"{ctx.get('case_num'):04}")
    goal_cases.run_cases_slurm(**sim.slurm_kwargs(max_parallel=200))

    off = open_offshore_waves(entry=get_offshore_wave("WHACS_cal")).sel(time=hours).load()
    params = sim.dynamic_parameters(off, water_level="water_level" in FORCING)
    params["save_map"] = [False] * len(hours)
    dyn = setup.dynamic_wrapper(
        params, output_dir=str(ROOT / f"dynamic_{variant}"), templates=patched_templates("dynamic", variant),
    )
    dyn.build_cases(cases_name_format=lambda ctx: f"{ctx.get('tref')}")
    dyn.run_cases_slurm(**sim.slurm_kwargs(max_parallel=200))
    print(f"variant {variant}: {len(cases)} goal cases + {len(hours)} dynamic hours submitted")


if __name__ == "__main__":
    main(sys.argv[1])
