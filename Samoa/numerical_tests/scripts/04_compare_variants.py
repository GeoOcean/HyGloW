"""Linearity test, part 4: exact sum vs dynamic, same physics, for the baseline and the variants A-F.

E and F rerun the dynamic hours of D and C with a test SnapWave build whose boundary spectra are centred on
the direction of each boundary point (snapwave_perpoint_direction.patch, built in tmp/snapwave_perpoint); the goal cases are the same
(one direction per case, so the build makes no difference there: checked, identical output)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path("tmp/linearity_test")


def hs_of(case: Path) -> np.ndarray:
    with xr.open_dataset(case / "output_sites.nc") as ds:
        return ds.point_hm0.values[0]


def main() -> None:
    cases = pd.read_csv(ROOT / "cases.csv", parse_dates=["time"])
    hours = sorted(cases.time.unique())
    sites = xr.open_dataset(
        "SnapWave_CASES_Dynamic_WHACS_cal_wind-off/Dynamic_Cases_2018.nc"
    )[["site_label"]].load()
    label = sites.site_label.values
    spr = cases.groupby("time").spr.mean().reindex(hours).values
    runs = {
        "baseline": (ROOT / "cases", Path("SnapWave_CASES_Dynamic_WHACS_cal_wind-off")),
        "A: no friction": (ROOT / "cases_A", ROOT / "dynamic_A"),
        "B: no friction, no breaking": (ROOT / "cases_B", ROOT / "dynamic_B"),
        "C: sector 360": (ROOT / "cases_C", ROOT / "dynamic_C"),
        "D: sector 360, no dissipation, mean Tp/wl": (ROOT / "cases_D", ROOT / "dynamic_D"),
        "E: D + per-point directions": (ROOT / "cases_D", ROOT / "dynamic_E"),
        "F: C + per-point directions": (ROOT / "cases_C", ROOT / "dynamic_F"),
    }
    rows = []
    for name, (goal_dir, dyn_dir) in runs.items():
        e = np.stack([hs_of(goal_dir / f"{i:04}") for i in range(len(cases))]) ** 2
        exact = np.stack([np.nansum(e[(cases.time == h).values], axis=0) for h in hours])
        dyn = np.stack([hs_of(dyn_dir / pd.Timestamp(h).strftime("%Y%m%d %H%M%S")) for h in hours]) ** 2
        for g in np.unique(label):
            m = label == g
            r = np.nansum(exact[:, m], axis=1) / np.nansum(dyn[:, m], axis=1)
            rows.append(dict(physics=name, group=g, ratio_median=np.median(r),
                             spr_lt45=np.median(r[spr < 45]), spr_gt60=np.median(r[spr > 60])))
    out = pd.DataFrame(rows)
    out.to_csv("numerical_tests/results/variants.csv", index=False)
    print(out.pivot_table(index="group", columns="physics", values="ratio_median").round(3).to_string())
    print()
    print(out.pivot_table(index="group", columns="physics", values=["spr_lt45", "spr_gt60"]).round(3).to_string())


if __name__ == "__main__":
    main()
