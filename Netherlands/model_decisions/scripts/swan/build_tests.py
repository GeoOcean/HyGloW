"""Build the SWAN physics tests (cases/{variant}/{case}) and their SLURM array.

Run from Netherlands/: python SWAN_test/build_tests.py
Dynamic hours: strongest finished hours, calm controls and Xaver, from the
built SWAN_CASES_Dynamic. Metamodel: finished cases across steepness.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from utils import model_comparison as mc  # noqa: E402
from utils import simulation as sim  # noqa: E402
from utils.config import load_goals  # noqa: E402

ROOT = Path("SWAN_test").resolve()
CASES = ROOT / "cases"
HOURS_TABLE = sys.argv[1]  # t, obs per finished hour
SEED = 0

GEN3 = {
    "komen": "GEN3 KOMEN",
    "st6": "GEN3 ST6 5.7E-7 8.0E-6 4.0 4.0 UP HWANG VECTAU TRUE10\nSSWELL ZIEGER",
}
XAVER = pd.date_range("2013-12-05T06", "2013-12-06T12", freq="3h")


def link(source: Path, target: Path) -> None:
    os.symlink(source.resolve(), target)


def build(source: Path, target: Path, patch: tuple[str, str] | None) -> Path:
    target.mkdir(parents=True)
    text = (source / "INPUT").read_text()
    if patch:
        assert text.count(patch[0]) == 1, source
        text = text.replace(*patch)
    (target / "INPUT").write_text(text)
    link(source / "fort.14", target / "fort.14")
    link(source / "output_sites.txt", target / "output_sites.txt")
    if (source / "wind.dat").exists():
        link(source / "wind.dat", target / "wind.dat")
    return target


def storm_hours(table: pd.DataFrame, n: int, gap_hours: int = 2) -> list[pd.Timestamp]:
    chosen: list[pd.Timestamp] = []
    for time in table.sort_values("obs", ascending=False)["t"]:
        if all(abs((time - c).total_seconds()) >= gap_hours * 3600 for c in chosen):
            chosen.append(time)
        if len(chosen) == n:
            return chosen
    raise ValueError("not enough hours")


def dynamic_hours() -> pd.DataFrame:
    table = pd.read_csv(HOURS_TABLE, parse_dates=["t"])
    storms = storm_hours(table, 20)
    calm = table[table["obs"].between(0.6, 1.2)].sample(8, random_state=SEED)["t"]
    groups = [("storm", storms), ("calm", list(calm)), ("xaver", list(XAVER))]
    return pd.DataFrame(
        [(group, time) for group, times in groups for time in times], columns=["group", "time"]
    )


def metamodel_cases() -> pd.DataFrame:
    snapwave = mc.load_snapwave_metamodel(load_goals())
    table = pd.DataFrame(
        {
            "case": snapwave.case_num.values,
            "goal": snapwave.goal.values,
            "hs": snapwave.hs_forcing.values,
            "tp": snapwave.tp_forcing.values,
            "dir": snapwave.dir_forcing.values,
        }
    )
    table["steepness"] = table["hs"] / (9.81 * table["tp"] ** 2 / (2 * np.pi))
    table["folder"] = [mc.case_folders(snapwave)[case] for case in table["case"]]
    table = table[table["dir"].between(250, 340) & (table["hs"] >= 1) & table["goal"].between(2, 8)]
    table = table[[(f / "norm_end").exists() for f in table["folder"]]]
    table["bin"] = pd.cut(table["steepness"], [0, 0.01, 0.02, 0.03, 0.04, 0.07])
    return (
        table.dropna(subset="bin")
        .groupby("bin", observed=True)
        .sample(8, random_state=SEED)
        .drop(columns="bin")
    )


def main() -> None:
    hours = dynamic_hours()
    dynamic = sim.cases_dir("dynamic", model="swan")
    folders = []
    for time in hours["time"]:
        source = dynamic / time.strftime("%Y%m%d %H%M%S")
        name = time.strftime("%Y%m%dT%H")
        folders += [
            build(source, CASES / variant / name, ("\nGEN3\n", f"\n{line}\n"))
            for variant, line in GEN3.items()
        ]
        if not (source / "norm_end").exists():
            folders.append(build(source, CASES / "base" / name, None))

    cases = metamodel_cases()
    for case, folder in zip(cases["case"], cases["folder"]):
        patch = ("OFF QUAD\n", "OFF QUAD\nOFF WCAP\n")
        folders.append(build(folder, CASES / "no_wcap" / f"{case:04}", patch))

    hours.to_csv(ROOT / "dynamic_hours.csv", index=False)
    cases.drop(columns="folder").to_csv(ROOT / "metamodel_cases.csv", index=False)
    (ROOT / "case_dirs.txt").write_text("\n".join(str(f) for f in folders) + "\n")
    (ROOT / "slurm_logs").mkdir()
    (ROOT / "swan_array.sh").write_text(f"""#!/bin/bash
#SBATCH --job-name=swan_test
#SBATCH --partition=geocean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --output={ROOT}/slurm_logs/%A_%a.out
#SBATCH --error={ROOT}/slurm_logs/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=1-{len(folders)}%48
module use /nfs/software/geocean/modulefiles
module load swan/4151

case_dir=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {ROOT}/case_dirs.txt)
cd "$case_dir"
swan.exe > wrapper_out.log 2> wrapper_error.log
""")
    print(len(folders), "cases")
    print(hours.groupby("group")["time"].agg(["count", "min", "max"]))
    print(cases.groupby(pd.cut(cases["steepness"], [0, 0.01, 0.02, 0.03, 0.04, 0.07]), observed=True)["hs"].describe()[["count", "min", "max"]])


if __name__ == "__main__":
    main()
