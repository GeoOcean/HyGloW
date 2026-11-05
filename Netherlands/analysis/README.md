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

## What it shows (run of 2026-11-05)

- **The linear sum works.** Bulk metamodel vs dynamic run without wind (like for like),
  Hs bias / RMSE / R²: 5 m +0.03 / 0.16 / 0.89, 10 m +0.01 / 0.14 / 0.93, 20 m
  −0.02 / 0.15 / 0.94, buoys −0.00 / 0.14 / 0.94 (WHACS; WHACS_cal the same). At the
  buoys the method error is −0.003 m of a −0.15 m total.
- **But it is high in the biggest seas near the coast**: each site's top 1 % of hours
  +0.44 m at 5 m, +0.17 at 10 m, +0.08 at 20 m (WHACS_cal +0.61 / +0.29 / +0.11). Likely
  depth-limited breaking, which does not add linearly: each goal alone breaks less than all
  together. To check.
- **The missing wind is the largest error** (−0.31 m of −0.15 m at the buoys, −0.59 m in
  each buoy's top 1 %); with wind the dynamic run is the best estimate (R² 0.81). Numbers
  and options: `hywaves/WIND.md`.
- **What SnapWave's wind adds depends on the sea already there.** At 8-14 m/s the added Hs is
  largest with wind from land or along the coast (E-S, ~0.7 m at 20 m) and smallest from the sea
  (NW 0.27 m), where WHACS already brings a grown sea; the added energy is flatter (1.1-1.9 m²).
  It follows the fetch inside the domain: SW wind adds most in Holland and the Wadden, E wind
  most at the outer Belgium-Zeeland sites. At the buoys (U10 ≥ 8 m/s) the run with wind
  overestimates with wind from S-SW (+0.29 / +0.33 m) and is close from NW (+0.02): SnapWave
  grows too much sea along the coast.
- **Spectral Tp, direction and spreading are poor.** Against the dynamic run at 5 m the
  spectral metamodel has Tp bias +1.7 s (R² −3.2; bulk −0.04 s, R² 0.46), direction RMSE
  32° (bulk 18°). At the buoys spectral Tp is the worst Tp of all estimates (RMSE 3.1 s).
  Spectral Hs is fine (+0.05 m vs bulk). To check how the summed spectrum gives Tp, Dir,
  Spr. Spreading is ~17° too narrow without wind, in the metamodel and the dynamic run.
- **The satellite calibration helps the storm tail.** Spectral metamodel at the buoys, p99.9
  bias −0.69 m (WHACS) → −0.22 m (WHACS_cal); Xaver peak, median over 19 buoys −1.14 → −0.48
  m. The offshore change shrinks towards the coast (p99.9: +1.69 m at the goals, +0.96 at
  20 m, +0.40 at 5 m).
