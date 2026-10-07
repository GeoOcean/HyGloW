"""Per-site Hs / Dir / Spr metrics: bulk metamodel reconstruction vs dynamic SnapWave (2010 + 2015).

Same comparison as Netherlands notebook 09, but kept per site instead of pooled
per isobath. Hand-run snapshot (needs the case's models + dynamic runs):

    cd Netherlands && PYTHONPATH=. python3 ../presentation/data/persite/compute_persite.py

Writes Netherlands_linear_summation.csv next to this script; build_data.py
summarises it into data.js.
"""
from pathlib import Path
OUT = Path(__file__).resolve().parent / "Netherlands_linear_summation.csv"
import os, time
os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np, pandas as pd, xarray as xr
from utils.config import SIMULATIONS, resolve
from utils import metamodel as mm
t0=time.perf_counter()
meta = mm.reconstruct(slice("2010","2015"), 450, spectral=False, offshore_name="WHACS",
                      sites=None, pca_variance=0.99, workers=8, save_spectra=False)
print("reconstruct s", time.perf_counter()-t0, dict(meta.sizes))
dyn = xr.open_dataset(resolve(SIMULATIONS["dynamic_cases"])/"Dynamic_Cases_WHACS_2010-2015.nc")
meta = xr.concat([meta.sel(time="2010"), meta.sel(time="2015")], dim="time")
common_t = np.intersect1d(meta.time.values, dyn.time.values)
common_s = np.intersect1d(meta.sites.values, dyn.sites.values)
m = meta.sel(time=common_t, sites=common_s); d = dyn.sel(time=common_t, sites=common_s).load()
print(list(meta.data_vars))
from utils import validation as val
rows=[]
for var in ("hs", "dir", "spr"):
    spec = val.PARAMETERS[var]
    mv, dv = m[var].values, d[var].values
    for i, s in enumerate(common_s):
        # reference = dynamic SnapWave, estimate = metamodel
        r = val.compute_metrics(dv[:, i], mv[:, i], circular=bool(spec["circular"]), variable=var)
        if r["n"] < 100:
            continue
        rows.append(dict(site=str(s), group=str(s).split("_")[0], variable=var, n=r["n"], bias=r["bias"],
                         rmse=r["rmse"], rrmse=r["rrmse"], r2=r["r2"], ref_mean=float(np.nanmean(dv[:, i])),
                         lon=float(d.site_longitude.sel(sites=s)), lat=float(d.site_latitude.sel(sites=s))))
df=pd.DataFrame(rows); df.to_csv(OUT, index=False)
for (v,g),x in df.groupby(["variable","group"]):
    print(v,g, len(x), "R2 quantiles 10/25/50/75/90:", np.round(x.r2.quantile([.1,.25,.5,.75,.9]).values,3),
          "frac R2>0.9:", round((x.r2>0.9).mean(),2), "rmse med", round(x.rmse.median(),3))
