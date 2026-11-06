# Analysis of the first iteration (Netherlands, SnapWave, 2013)

Every estimate of the case on the same sites and hours of 2013, to draw the conclusions of
the SnapWave iteration before SWAN and before any wind in the reconstruction
(`hywaves/WIND.md`). What proves useful may move to the pipeline notebooks (09-12).

| Notebook | Compares |
| --- | --- |
| `01_Error_Budget` | All estimates against the 27 buoys: skill, storm tail (quantiles), and the split of the error into partitions + method + missing wind + model and forcing |
| `02_Linear_Sum_vs_Dynamic` | The method error at every output site: bulk and spectral metamodel vs the dynamic run without wind, per group, map and sea state |
| `03_Wind_Effect` | Dynamic with wind − without, by site, wind speed and direction, stretch of coast, sea state; an analog explorer (closest hours of 2013 to a chosen wind and sea, interactive); the buoys by wind direction; the hours that do not converge with wind |
| `04_Calibration_Effect` | WHACS_cal vs WHACS from the goals to the coast, and against the buoys |
| `05_Storms` | Xaver and the January-February storms: peaks, time series, maps at the peak |

**Estimates** (`results.py`, for one offshore source):

- `offshore`: the hindcast at each buoy's closest goal;
- `metamodel bulk` / `metamodel spectral`: the linear sum (MDA 600, CODEC water level), read
  from the reconstruction cache;
- `dynamic no wind` / `dynamic wind`: notebook 08 (`Dynamic_Cases_2013.nc`).

The dynamic runs force every goal with its bulk sea state, so `metamodel bulk` vs
`dynamic no wind` is the method error; `metamodel spectral` is the product. Metrics are
reference first (bias = estimate − reference); comparisons use only pairs valid in every
estimate they involve (the run with wind has non-converged hours).

**Inputs:** the cache must hold 2013 for every set, both sources, bulk and spectral
(`tmp/reconstruct_2013.py` from the case folder; ~3 h on the 48-core node), and the four
dynamic runs postprocessed. The notebooks run from this folder (they put the case folder on
`sys.path`).

## What it shows (first iteration, 2026-11-05)

The numbers are in the notebooks and on the dashboard (`../dashboard/`); here only the
conclusions.

- **The linear sum works.** The bulk metamodel matches the dynamic run without wind (like
  for like) at every isobath and at the buoys; the method error is a negligible part of
  the error at the buoys (01, 02).
- **But it is high in the biggest seas near the coast**, most at 5 m and more with
  WHACS_cal (02, each site's top 1 % of hours). Likely depth-limited breaking, which does
  not add linearly: each goal alone breaks less than all together. To check.
- **The missing wind is the largest error** at the buoys, and it grows with the sea; with
  wind the dynamic run is the best estimate (01, 03; numbers and options in
  `hywaves/WIND.md`).
- **What SnapWave's wind adds depends on the sea already there** and on the fetch inside
  the domain: little where WHACS already brings a grown sea from offshore (NW), most with
  wind from land or along the coast (E-S); SW wind adds most in Holland and the Wadden, E
  wind at the outer Belgium-Zeeland sites. At the buoys the run with wind is too high with
  S-SW wind and about right with NW wind: SnapWave grows too much sea along the coast (03).
- **Spectral Tp, direction and spreading are poor**, while spectral Hs is fine: the spectral
  Tp is the worst Tp of all estimates at the buoys, and far off the dynamic run at 5 m (01,
  02). To check how the summed spectrum gives Tp, Dir, Spr. Spreading is too narrow without
  wind, in the metamodel and the dynamic run.
- **The satellite calibration helps the storm tail** at the buoys and in Xaver (01, 04,
  05); the offshore change shrinks towards the coast (04).
