"""Which reference is closer to the sea: the dynamic run or the linear sum, against the Aunu'u buoys in 2018.

Run from the case folder: PYTHONPATH=. python numerical_tests/scripts/05_dynamic_vs_buoys.py
Needs the 2018 dynamic run (08) and the 2018 buoys reconstruction cache (09).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xarray as xr

from utils import metamodel as mm
from utils import simulation as sim
from utils import validation as val

SITES = ["buoy_Aunuu_1", "buoy_Aunuu_2"]


def main() -> None:
    logging.disable(logging.WARNING)
    buoy = val.load_buoy_observations("2018", SITES, variables=["hs"])
    res = mm.reconstruction_resources("2018", 800, metamodel="buoys", spectral=False, offshore_name="WHACS_cal",
                                      wl="forcing", save_spectra=False)
    estimates = {
        f"linear sum, {mode}": mm.reconstruct("2018", 800, metamodel="buoys", spectral=mode == "spectral",
                                              offshore_name="WHACS_cal", wl="forcing", save_spectra=False,
                                              resources=res)
        for mode in ("bulk", "spectral")
    }
    dyn = xr.open_dataset(sim.cases_dir("dynamic", offshore="WHACS_cal", wind=False) / "Dynamic_Cases_2018.nc")
    estimates["dynamic run"] = dyn.sel(sites=SITES).load()
    rows = []
    for site in buoy.sites.values:
        obs = buoy.hs.sel(sites=site).dropna("time")
        for name, est in estimates.items():
            x = est.hs.sel(sites=site, time=obs.time)
            ok = np.isfinite(x.values)
            d = (x - obs)[ok]
            rows.append(dict(site=site, estimate=name, n=int(ok.sum()), bias=float(d.mean()),
                             rmse=float(np.sqrt((d ** 2).mean())),
                             energy_ratio=float((x[ok] ** 2).sum() / (obs[ok] ** 2).sum())))
    out = pd.DataFrame(rows)
    out.to_csv("numerical_tests/results/dynamic_vs_buoys.csv", index=False)
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
