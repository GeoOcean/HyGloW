"""Linearity test, part 2: dynamic vs exact sum of single-goal runs vs metamodel (GP) sum, same hours and sites."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from utils import metamodel as mm
from utils import simulation as sim

CASES = Path("tmp/linearity_test/cases")


def last_error(case: Path) -> float:
    lines = [l for l in (case / "wrapper_out.log").read_text().splitlines() if "error =" in l]
    return float(lines[-1].split("error =")[1].split()[0]) if lines else np.nan


def main() -> None:
    logging.disable(logging.WARNING)
    cases = pd.read_csv("tmp/linearity_test/cases.csv", parse_dates=["time"])
    hs, err = [], []
    for i in range(len(cases)):
        case = CASES / f"{i:04}"
        with xr.open_dataset(case / "output_sites.nc") as ds:
            hs.append(ds.point_hm0.values[0])  # stations in output_sites.txt order, as the dynamic run
        err.append(last_error(case))
    cases["hs_sites"] = hs
    cases["error"] = err
    print(f"cases: {len(cases)}, not converged (> 0.01): {int((cases.error > 0.01).sum())}")
    exact = (
        cases.groupby("time")["hs_sites"].apply(lambda s: np.sqrt(np.nansum(np.stack(s.values) ** 2, axis=0)))
    )
    dyn = xr.open_dataset(sim.cases_dir("dynamic", offshore="WHACS_cal", wind=False) / "Dynamic_Cases_2018.nc")
    dyn = dyn.sel(time=exact.index.values).load()
    exact = xr.DataArray(np.stack(exact.values), dims=("time", "sites"),
                         coords={"time": exact.index.values, "sites": dyn.sites.values}, name="hs")
    res = mm.reconstruction_resources("2018", 800, metamodel="isobaths", spectral=False, offshore_name="WHACS_cal",
                                      wl="forcing", save_spectra=False)
    gp = xr.concat([
        mm.reconstruct("2018", 800, metamodel=st, spectral=False, offshore_name="WHACS_cal", wl="forcing",
                       save_spectra=False, resources=res, depth_cap=False).hs.sel(time=exact.time.values)
        for st in mm.SETS], dim="sites").sel(sites=dyn.sites.values)

    label = dyn.site_label.values
    spr = cases.groupby("time").spr.mean().reindex(exact.time.values).values
    rows = []
    for g in np.unique(label):
        m = label == g
        d2 = np.nansum(dyn.hs.values[:, m] ** 2, axis=1)
        for name, est in (("exact sum", exact.values[:, m]), ("GP sum", gp.values[:, m])):
            r = np.nansum(est ** 2, axis=1) / d2
            rows.append(dict(group=g, estimate=name, ratio_median=np.median(r),
                             ratio_spr_lt45=np.median(r[spr < 45]), ratio_spr_gt60=np.median(r[spr > 60]),
                             hs_bias=float(np.nanmean(est - dyn.hs.values[:, m]))))
    out = pd.DataFrame(rows)
    out.to_csv("numerical_tests/results/exact_vs_gp_sum.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
