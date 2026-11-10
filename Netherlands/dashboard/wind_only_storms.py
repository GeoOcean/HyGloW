"""
SnapWave wind-only runs on the 2013 storms of the dashboard (run from this folder).

Usage: ``python wind_only_storms.py build | slurm | postprocess``

Every hour of every event of ``store/storms.csv``: the dynamic template, ERA5 wind field and
CODEC water level as the wind-on run, but no waves at the boundary
(:func:`utils.simulation.wind_only_parameters`). Cases in
:func:`utils.simulation.wind_only_cases_dir` (``SnapWave_CASES_Dynamic_wind-only``), postprocessed
to ``Dynamic_Cases_storms.nc`` there; then ``python build_store.py --only wind_only``. The Wind and
Storms pages show it (``hywaves/WIND.md``, wind-only storm test). 520 hours, ~2-5 min each.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "share", "gdal"))
os.environ.setdefault("PROJ_DATA", os.path.join(sys.prefix, "share", "proj"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from utils import simulation as sim  # noqa: E402
from utils.config import load_goals, open_offshore_waves, open_wind  # noqa: E402

MODEL = "snapwave"
#: Fills Tp, Dir, Spr of the boundary files (Hs is 0).
OFFSHORE = "WHACS_cal"
STORMS = HERE / "store" / "storms.csv"
OUTPUT = "Dynamic_Cases_storms.nc"


def _name(ctx: dict) -> str:
    return f"{ctx.get('tref')}"


def storm_hours() -> pd.DatetimeIndex:
    """Every hour of every event of the dashboard's ``storms.csv``."""
    storms = pd.read_csv(STORMS, parse_dates=["start", "end"])
    hours = [pd.date_range(r.start, r.end, freq="h") for r in storms.itertuples()]
    return pd.DatetimeIndex(sorted(set().union(*hours)))


def wrapper():
    hours = storm_hours()
    parameters = sim.wind_only_parameters(open_offshore_waves(OFFSHORE).sel(time=hours).load())
    parameters["save_map"] = [False] * len(hours)
    setup = sim.HyWavesWrapperSetup.from_case(load_goals(), model=MODEL)
    return setup.dynamic_wrapper(parameters, wind=open_wind(slice(hours[0], hours[-1])),
                                 output_dir=sim.wind_only_cases_dir(MODEL))


def main(step: str) -> None:
    if step not in ("build", "slurm", "postprocess"):
        raise SystemExit(__doc__)
    w = wrapper()
    if step == "build":
        w.build_cases(cases_name_format=_name)
        return
    w.load_cases(cases_name_format=_name)
    if step == "slurm":
        w.run_cases_slurm(**sim.slurm_kwargs(MODEL, max_parallel=192))
    else:
        w.postprocess_cases().to_netcdf(sim.wind_only_cases_dir(MODEL) / OUTPUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
