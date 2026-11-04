"""The 39 dynamic hours with GEN3 KOMEN and the WHACS_cal of the new calibration
(directional + QuantileMapping). Run from Netherlands/ with GDAL_DATA and PROJ_DATA set.
"""

from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "SWAN_test")
from build_tests_2 import ROOT, build_forcing_cases, load_goals  # noqa: E402

hours = pd.read_csv(ROOT / "dynamic_hours.csv", parse_dates=["time"])
times = pd.DatetimeIndex(hours["time"]).sort_values()
folders = build_forcing_cases(times, load_goals(), "WHACS_cal", "whacs_map")
(ROOT / "case_dirs_3.txt").write_text("\n".join(str(f) for f in folders) + "\n")
script = (ROOT / "swan_array_2.sh").read_text()
for old, new in (("swan_test2", "swan_test3"), ("slurm_logs_2", "slurm_logs_3"), ("case_dirs_2", "case_dirs_3"), ("1-205%96", f"1-{len(folders)}%48")):
    script = script.replace(old, new)
(ROOT / "swan_array_3.sh").write_text(script)
(ROOT / "slurm_logs_3").mkdir()
print(len(folders), "cases")
