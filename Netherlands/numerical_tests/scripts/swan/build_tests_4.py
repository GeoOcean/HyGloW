"""No-wind dynamic reference: 500 hours with the goal-case physics (OFF WCAP).

The 39 test hours plus a 2013 sample by offshore Hs, forced by the new WHACS_cal:
tests the linear sum without K, and wind vs no wind at the buoys. Run from
Netherlands/ with GDAL_DATA and PROJ_DATA set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from utils import simulation as sim  # noqa: E402
from utils.config import get_offshore_wave, load_goals, open_offshore_waves  # noqa: E402

ROOT = Path("SWAN_test").resolve()
FORCING = "WHACS_cal"
SAMPLE = {(0, 1): 100, (1, 2): 120, (2, 3): 120, (3, 5): 100, (5, 15): 21}
SEED = 0


def sample_hours(offshore_hs: pd.Series, chosen: pd.DatetimeIndex) -> pd.DatetimeIndex:
    rest = offshore_hs.drop(chosen)
    picks = [
        rest[rest.between(low, high, inclusive="left")]
        .sample(frac=1, random_state=SEED)
        .iloc[:n]
        .index
        for (low, high), n in SAMPLE.items()
    ]
    return chosen.append(picks).sort_values()


def main() -> None:
    goals = load_goals()
    tested = pd.DatetimeIndex(
        pd.read_csv(ROOT / "dynamic_hours.csv", parse_dates=["time"])["time"]
    )
    year = open_offshore_waves(entry=get_offshore_wave(FORCING)).sel(time="2013").load()
    hours = sample_hours(year["hs"].mean("goal").to_series(), tested)
    wrapper = sim.HyWavesWrapperSetup.from_case(goals).dynamic_wrapper(
        sim.dynamic_parameters(year.sel(time=hours), water_level=True),
        output_dir=ROOT / "cases_4" / "no_wind",
    )
    wrapper.build_cases(
        cases_name_format=lambda ctx: f"{ctx.get('tref')[:8]}T{ctx.get('tref')[9:11]}"
    )
    folders = [Path(f) for f in wrapper.cases_dirs]
    for folder in folders:
        text = (folder / "INPUT").read_text()
        assert text.count("\nOFF QUAD\n") == 1 and "GEN3\n" not in text
        (folder / "INPUT").write_text(text.replace("\nOFF QUAD\n", "\nOFF QUAD\nOFF WCAP\n"))
    pd.DataFrame({"time": hours}).to_csv(ROOT / "no_wind_hours.csv", index=False)
    (ROOT / "case_dirs_4.txt").write_text("\n".join(str(f) for f in folders) + "\n")
    script = (ROOT / "swan_array_2.sh").read_text()
    for old, new in (
        ("swan_test2", "swan_test4"),
        ("slurm_logs_2", "slurm_logs_4"),
        ("case_dirs_2", "case_dirs_4"),
        ("1-205%96", f"1-{len(folders)}%96"),
    ):
        script = script.replace(old, new)
    (ROOT / "swan_array_4.sh").write_text(script)
    (ROOT / "slurm_logs_4").mkdir()
    print(len(folders), "cases", int(np.isin(hours, tested).sum()), "tested")


if __name__ == "__main__":
    main()
