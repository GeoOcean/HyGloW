"""Friction and forcing on a wider 2013 sample: 120 hours outside the 39 test hours,
Komen with wind (current template). Run from Netherlands/ with GDAL_DATA and PROJ_DATA set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from utils import simulation as sim  # noqa: E402
from utils.config import get_offshore_wave, load_goals, open_offshore_waves, open_wind  # noqa: E402

ROOT = Path("SWAN_test").resolve()
N_HOURS = 120
SEED = 0
FRICTION = ("FRICTION JONSWAP CONSTANT 0.019", "FRICTION JONSWAP CONSTANT 0.038")


def build(hours: pd.DatetimeIndex, forcing: str, variant: str, goals: dict) -> list[Path]:
    offshore = open_offshore_waves(entry=get_offshore_wave(forcing)).sel(time=hours).load()
    wrapper = sim.HyWavesWrapperSetup.from_case(goals).dynamic_wrapper(
        sim.dynamic_parameters(offshore, water_level=True),
        wind=open_wind(slice(hours.min(), hours.max())),
        output_dir=ROOT / "cases_6" / variant,
    )
    wrapper.build_cases(
        cases_name_format=lambda ctx: f"{ctx.get('tref')[:8]}T{ctx.get('tref')[9:11]}"
    )
    return [Path(f) for f in wrapper.cases_dirs]


def main() -> None:
    goals = load_goals()
    tested = pd.read_csv(ROOT / "dynamic_hours.csv", parse_dates=["time"])["time"]
    sample = pd.read_csv(ROOT / "no_wind_hours.csv", parse_dates=["time"])["time"]
    hours = pd.DatetimeIndex(
        sample[~sample.isin(tested)].sample(N_HOURS, random_state=SEED).sort_values()
    )
    pd.DataFrame({"time": hours}).to_csv(ROOT / "hours_6.csv", index=False)
    folders = []
    for forcing, variant in (("WHACS_cal", "whacs"), ("ERA5_cal", "era5")):
        folders += build(hours, forcing, variant, goals)
    (ROOT / "case_dirs_6.txt").write_text("\n".join(str(f) for f in folders) + "\n")
    # friction 0.038 on WHACS_cal, same forcing
    extra = []
    for folder in [f for f in folders if f.parent.name == "whacs"]:
        target = ROOT / "cases_6" / "whacs_038" / folder.name
        target.mkdir(parents=True)
        for item in folder.iterdir():
            (target / item.name).symlink_to(item.resolve()) if item.name in ("fort.14", "wind.dat", "output_sites.txt", "output_spectra.txt") else (target / item.name).write_bytes(item.read_bytes())
        text = (target / "INPUT").read_text()
        assert text.count(FRICTION[0]) == 1
        (target / "INPUT").write_text(text.replace(*FRICTION))
        extra.append(target)
    with open(ROOT / "case_dirs_6.txt", "a") as f:
        f.write("\n".join(str(p) for p in extra) + "\n")
    print(len(folders) + len(extra), "cases")


if __name__ == "__main__":
    main()
