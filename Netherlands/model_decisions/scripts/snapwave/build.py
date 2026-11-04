"""SnapWave tests: boundary enclosure, unstable hours, convergence. Run from Netherlands/
with GDAL_DATA and PROJ_DATA set: python SWAN_test/snapwave/build.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "SWAN_test/snapwave")
from enclosure import open_boundary_enclosure, write_enclosure  # noqa: E402
from utils import simulation as sim  # noqa: E402
from utils.config import get_offshore_wave, load_goals, open_offshore_waves, open_wind  # noqa: E402
from utils.geometry import output_sites  # noqa: E402

ROOT = Path("SWAN_test/snapwave").resolve()
FORCING = "WHACS_cal"
N_UNSTABLE = 24
SEED = 0

#: variant -> (thin enclosure, snapwave.inp keys)
DYNAMIC = {
    "base": (False, {}),
    "enc": (True, {}),
    "enc_upw": (True, {"upwindref": 1}),
}
STABILITY = {
    "base": (False, {}),
    "enc": (True, {}),
    "enc_upw": (True, {"upwindref": 1}),
    "enc_nodcg": (True, {"jadcgdx": 0}),
    "enc_dt": (True, {"dt": 3600.0}),
    "enc_iter": (True, {"niter": 800}),
}
METAMODEL = {
    "enc": (True, {}),
    "enc_n200": (True, {"niter": 200}),
    "enc_upw": (True, {"upwindref": 1, "niter": 200}),
    "enc_360": (True, {"sector": 360, "niter": 200}),
}


def apply_variant(folder: Path, enclosure, keys: dict) -> None:
    if enclosure is not None:
        write_enclosure(folder / "enclosure.txt", enclosure)
    inp = folder / "snapwave.inp"
    text = inp.read_text()
    for key, value in keys.items():
        pattern = rf"^{key}\s*=.*$"
        line = f"{key:<15}= {value}"
        text, n = re.subn(pattern, line, text, flags=re.M)
        if n == 0:
            text = f"{line}\n{text}"
    inp.write_text(text)


def variants(source: Path, target: Path, table: dict, enclosure) -> list[Path]:
    out = []
    for name, (thin, keys) in table.items():
        folder = target / name / source.name
        shutil.copytree(source, folder, ignore=shutil.ignore_patterns("*.nc", "*.log", "*.upw"))
        apply_variant(folder, enclosure if thin else None, keys)
        out.append(folder)
    return out


def build_hours(hours: pd.DatetimeIndex, goals: dict, setup) -> list[Path]:
    offshore = open_offshore_waves(entry=get_offshore_wave(FORCING)).sel(time=hours).load()
    wrapper = setup.dynamic_wrapper(
        sim.dynamic_parameters(offshore, water_level=True),
        wind=open_wind(slice(hours.min(), hours.max())),
        output_dir=ROOT / "source",
    )
    wrapper.build_cases(
        cases_name_format=lambda ctx: f"{ctx.get('tref')[:8]}T{ctx.get('tref')[9:11]}"
    )
    return [Path(f) for f in wrapper.cases_dirs]


def main() -> None:
    goals = load_goals()
    enclosure = open_boundary_enclosure()
    setup = sim.HyWavesWrapperSetup.from_case(goals, model="snapwave", output_sites_df=output_sites("swan"))
    tested = pd.DatetimeIndex(pd.read_csv("SWAN_test/dynamic_hours.csv", parse_dates=["time"])["time"]).sort_values()
    table = pd.read_csv(ROOT / "unstable_hours_all.csv", parse_dates=["time"])
    unstable = table.sample(N_UNSTABLE, random_state=SEED)["time"].sort_values()
    pd.DataFrame({"time": unstable}).to_csv(ROOT / "unstable_hours.csv", index=False)

    folders = build_hours(tested.union(pd.DatetimeIndex(unstable)), goals, setup)
    by_name = {f.name: f for f in folders}
    cases = []
    for t in tested:
        cases += variants(by_name[t.strftime("%Y%m%dT%H")], ROOT / "dyn", DYNAMIC, enclosure)
    for t in unstable:
        cases += variants(by_name[t.strftime("%Y%m%dT%H")], ROOT / "stab", STABILITY, enclosure)
    root = sim.cases_dir("metamodel", model="snapwave")
    for case in pd.read_csv("SWAN_test/metamodel_cases.csv").itertuples():
        source = root / f"{case.goal:02}" / f"{case.case:04}"
        cases += variants(source, ROOT / "mm", METAMODEL, enclosure)
    (ROOT / "case_dirs.txt").write_text("\n".join(str(f) for f in cases) + "\n")
    print(len(cases), "cases")


if __name__ == "__main__":
    main()
