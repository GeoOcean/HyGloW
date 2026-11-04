"""Wind × 1.1, AGROW (39 hours) and 5 deg directions (40 metamodel cases).
Run from Netherlands/: python SWAN_test/compare_tests_7.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "SWAN_test")
import compare_tests_2 as ct  # noqa: E402

ct.VARIANTS = {
    v: ct.ROOT / "cases_2" / v
    for v in ("whacs_map_fric", "whacs_fric_agrow", "whacs_fric_wind110", "era5cal_fric", "era5cal_fric_wind110")
}
ct.report_dynamic()

cases = pd.read_csv(ct.ROOT / "metamodel_cases.csv").sort_values("case")
base = np.stack([ct.read_hs(ct.ROOT / "cases" / "no_wcap" / f"{c:04}") for c in cases["case"]])
fine = np.stack([ct.read_hs(ct.ROOT / "cases_2" / "dir72" / f"{c:04}") for c in cases["case"]])
keep = base > 0.05
print(f"\n5 vs 10 deg directions, Hs ratio {np.sqrt(np.nansum(fine[keep] ** 2) / np.nansum(base[keep] ** 2)):.3f}, "
      f"mean diff {np.nanmean((fine - base)[keep]):+.4f} m, p1/p99 {np.nanpercentile((fine - base)[keep], [1, 99]).round(3)}")
for i, c in enumerate(cases["case"]):
    print(c, round(float(np.sqrt(np.nansum(fine[i][keep[i]] ** 2) / np.nansum(base[i][keep[i]] ** 2))), 3))
