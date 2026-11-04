"""Second round of SWAN tests (cases_2/{variant}/{case}), on top of round 1.

Run from Netherlands/: python SWAN_test/build_tests_2.py
Dynamic variants start from cases/komen; metamodel ones from cases/no_wcap.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from utils import simulation as sim  # noqa: E402
from utils.config import (  # noqa: E402
    get_offshore_wave,
    load_goals,
    open_offshore_waves,
    open_wind,
)

ROOT = Path("SWAN_test").resolve()
CASES = ROOT / "cases_2"
OFFSHORE = "WHACS_cal"
FORCING = "ERA5_cal"
#: Regular water-level grid: the ERA5 wind grid, which covers the mesh.
LON = np.arange(2.0, 7.51, 0.25)
LAT = np.arange(51.0, 54.51, 0.25)
WL_POWER = 2.0


def copy_case(source: Path, target: Path, patches: list[tuple[str, str]]) -> Path:
    target.mkdir(parents=True)
    text = (source / "INPUT").read_text()
    for old, new in patches:
        assert text.count(old) == 1, (source, old)
        text = text.replace(old, new)
    (target / "INPUT").write_text(text)
    for name in ("fort.14", "output_sites.txt", "wind.dat"):
        if (source / name).exists():
            os.symlink((source / name).resolve(), target / name)
    return target


def wl_field(wl: np.ndarray, goals: dict) -> np.ndarray:
    """Goal water levels on the regular grid, inverse-distance weighted."""
    lon = np.array([g["target"]["longitude"] for g in goals.values()])
    lat = np.array([g["target"]["latitude"] for g in goals.values()])
    grid_lon, grid_lat = np.meshgrid(LON, LAT)
    distance = np.hypot(
        (grid_lon[..., None] - lon) * np.cos(np.radians(52.5)), grid_lat[..., None] - lat
    )
    weights = 1.0 / np.maximum(distance, 1e-3) ** WL_POWER
    return (weights * wl).sum(-1) / weights.sum(-1)


def build_wl_case(source: Path, target: Path, wl: np.ndarray, goals: dict) -> Path:
    text = (source / "INPUT").read_text()
    level = re.search(r"SET LEVEL=\S+ ", text).group(0)
    grid = f"{LON[0]:.2f} {LAT[0]:.2f} 0 {LON.size - 1} {LAT.size - 1} 0.25 0.25"
    read = f"INPGRID WLEV REGULAR {grid}\nREADINP WLEV 1 'wl.dat' 3 0 FREE\nBOUND SHAPE"
    copy_case(source, target, [(level, "SET LEVEL=0 "), ("BOUND SHAPE", read)])
    np.savetxt(target / "wl.dat", wl_field(wl, goals), fmt="%.3f")
    return target


def build_forcing_cases(hours: pd.DatetimeIndex, goals: dict, forcing: str, variant: str) -> list[Path]:
    """Dynamic cases with GEN3 KOMEN forced by *forcing*, through the production wrapper."""
    offshore = open_offshore_waves(entry=get_offshore_wave(forcing)).sel(time=hours).load()
    parameters = sim.dynamic_parameters(offshore, water_level=True)
    setup = sim.HyWavesWrapperSetup.from_case(goals)
    wind = open_wind(slice(hours.min(), hours.max()))
    wrapper = setup.dynamic_wrapper(parameters, wind=wind, output_dir=CASES / variant)
    wrapper.build_cases(
        cases_name_format=lambda ctx: f"{ctx.get('tref')[:8]}T{ctx.get('tref')[9:11]}"
    )
    folders = [Path(f) for f in wrapper.cases_dirs]
    for folder in folders:
        input_file = folder / "INPUT"
        text = input_file.read_text()
        assert text.count("\nGEN3\n") == 1
        input_file.write_text(text.replace("\nGEN3\n", "\nGEN3 KOMEN\n"))
    return folders


def main() -> None:
    hours = pd.read_csv(ROOT / "dynamic_hours.csv", parse_dates=["time"])
    times = pd.DatetimeIndex(hours["time"]).sort_values()
    goals = load_goals()
    komen = ROOT / "cases" / "komen"
    snapwave = sim.cases_dir("dynamic", model="snapwave")

    folders = build_forcing_cases(times, goals, FORCING, "era5cal")
    for time in times:
        name = time.strftime("%Y%m%dT%H")
        source = komen / name
        wl = (snapwave / time.strftime("%Y%m%d %H%M%S") / "wl.txt").read_text().splitlines()[0]
        folders.append(
            build_wl_case(source, CASES / "wl_field" / name, np.array(wl.split()[1:], float), goals)
        )
        folders.append(
            copy_case(source, CASES / "bkd" / name, [("BREAKING CONSTANT 1.0 0.73", "BREAKING BKD")])
        )
    for time in hours.query("group == 'calm'")["time"]:
        name = time.strftime("%Y%m%dT%H")
        folders.append(
            copy_case(komen / name, CASES / "iter100" / name, [("STATIONARY 50", "STATIONARY 100")])
        )
    for source in sorted((ROOT / "cases" / "no_wcap").iterdir()):
        folders.append(
            copy_case(
                source, CASES / "friction" / source.name,
                [("FRICTION JONSWAP CONSTANT 0.038", "FRICTION JONSWAP CONSTANT 0.019")],
            )
        )
        folders.append(copy_case(source, CASES / "no_triad" / source.name, [("\nTRIAD\n", "\n")]))

    (ROOT / "case_dirs_2.txt").write_text("\n".join(str(f) for f in folders) + "\n")
    (ROOT / "slurm_logs_2").mkdir()
    (ROOT / "swan_array_2.sh").write_text(f"""#!/bin/bash
#SBATCH --job-name=swan_test2
#SBATCH --partition=geocean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6gb
#SBATCH --output={ROOT}/slurm_logs_2/%A_%a.out
#SBATCH --error={ROOT}/slurm_logs_2/%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --array=1-{len(folders)}%96
module use /nfs/software/geocean/modulefiles
module load swan/4151

case_dir=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {ROOT}/case_dirs_2.txt)
cd "$case_dir"
swan.exe > wrapper_out.log 2> wrapper_error.log
""")
    print(len(folders), "cases")


if __name__ == "__main__":
    main()
