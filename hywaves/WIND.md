# Wind: what was tried, what we know, what to do next

Final state on 2026-11-10 (Netherlands, SnapWave; the last wind test of this iteration). The
reconstruction has **no wind**: the metamodel cases run without it and the linear sum adds the
goals only. Wind stays as forcing of the dynamic runs (`forcing.wind`, folders
`<dynamic_cases>_<offshore>_wind-on|off`, both wrappers) and of the wind-only run
(`<dynamic_cases>_wind-only`, `utils.simulation.wind_only_parameters` / `wind_only_cases_dir`).

**Conclusion.** The missing wind is the largest error of the reconstruction at the buoys and
grows with the sea. The growth factor K (removed) failed with the open-boundary enclosure. The
way to add it, if it is added: **a wind-only term with a damping**,
`Hs = sqrt(E + f(share) E_wo)`, E the energy of the linear sum, E_wo the sea the local wind alone
grows (no boundary waves), f a curve of the share of the sea already there, E / (E + E_wo).
Tested on the six 2013 storms (last section): learned on the other storms, it reproduces SnapWave
with wind from SnapWave without wind, and matches it on the metamodel at 20 m; nearer the coast
it adds to the linear sum's storm overshoot, which has to be fixed first. To build it: E_wo from
analogs of the ERA5 wind field or its own MDA -> PCA -> GP on `u10, u10dir, wl` (option 2), f
learned from wind-on / wind-off / wind-only runs over more than storms (calm and offshore winds
untested). It can only reproduce SnapWave's wind, which is high with S-SW wind.

## Why the wind is separate

Each metamodel case is one goal's share of the boundary. With wind, every
goal's run would grow its own copy of the local sea and the sum would count it
once per goal. Growth is also non-linear: it slows as the sea develops, which
depends on the total energy, not on one goal's share. So any wind term has to
enter once per hour and site, after the goals are predicted.

## Tried: the growth factor K (removed 2026-11-04)

Restore with HyGloW `b36596e` and bluemath_tk `e280155` (branch
`feature/more-hywaves`): `bluemath_tk.waves.hywaves.wind_correction`, the
`wind_correction=` option of the reconstruction, case `utils/wind.py`, notebooks
`05b_Wind_Correction` and `10b_Storm_Event`, config `metamodel.wind_correction`.
Kept: `bluemath_tk.waves.wind` (wind fields, wind-sea weight, fully developed
speed) and `bluemath_tk.waves.fetch` (fetch and mean depth per site and
direction over a mesh).

How it worked:

1. **Training cases**: full-domain seas (every goal forced the same), run twice
   with the dynamic templates, with and without a uniform wind of `speed_ratio`
   x the fully developed speed (clipped to `u10_range`), blowing
   `wind_wave_angle` off the waves.
2. **K** = Hs(with wind) / Hs(without) of the wind-sea part,
   `K² = 1 + (hs1² − hs0²) / (w hs0²)` (w = offshore wind-sea weight, steepness
   0.02-0.03 and spreading 20-25° ramps, cases with w ≥ 0.5): gradient boosting
   of log K on g·Hs_ws/U², g·F/U², g·d/U², g·d̄/U², cos(wind − wave), K in
   [0.5, 4].
3. **Reconstruction**: `E_ws = Σ w·hs²` over the goal x partition
   contributions; one extra contribution `(K² − 1)·E_ws` from the wind direction
   (ERA5 or inferred wind; wind-sea Tp, spreading 30°) joined the sum.

## What we know

**SnapWave wind mode** (39 hours of 2013, 27 buoys, `Netherlands/numerical_tests/`):
adds ~0.4 m in calm hours (SWAN ~0.2), so SnapWave stays ~+0.2 m high there and
any correction learned from it inherits that; `fw` 0.02 / 0.04 trade it for worse
storms. It does not converge in some hours (swell, light wind; 241 of 8760 in the
2013 tests, 413 in the WHACS wind-on run below) and no setting fixes it:
postprocessing drops them (last error > `MAX_ERROR` 0.01). One period per node:
wind energy over swell takes the swell period.

**K, old setup** (old AoI rectangle forcing
the coast nodes, no-wind twins with `niter` 40; 250 cases, `speed_ratio`
[0.8, 2.0], u10 up to 35 m/s). Held out (reference = run with wind), `waves`
sites Hs bias / RMSE −0.70 / 1.00 -> −0.06 / 0.39 m. But 131 of the 250 no-wind
twins did not converge, and part of K came from the coast forcing.

- More cases do not help: 5-fold by case, 50 / 100 / 150 / 200 training cases
  give RMSE 0.41 / 0.42 / 0.38 / 0.38 m (no correction 0.86).
- The error grows with the wind: per case, Spearman with u10 0.78, with
  `speed_ratio` 0.64, with the angle 0.00; RMSE 0.10 m below 10 m/s, 0.45 m at
  25-35 m/s.
- Real hours (WHACS_cal wind-sea partitions at the 10 goals x ERA5 at the goal,
  1979-2023, 2.6 M partition-hours): u10 p50 / p90 / p99 9.2 / 13.4 / 17.0 m/s,
  `speed_ratio` 1.24 / 1.53 / 1.89; 95 % of the hours (98.8 % of the wind-sea
  energy) inside that design: it was too windy, not too narrow.

**K, current setup** (2026-11-03: open-boundary enclosure, `niter` 200,
`upwindref` 1; `speed_ratio` [0.7, 1.7], `u10_range` [3, 25]): every twin
converges, but held out the correction is much worse, `waves` −1.03 / 1.41 ->
−0.51 / 1.09 m. Waves enter only through the open boundary, so with waves from
the land side the no-wind run has almost nothing at the sites, the wind grows the
whole sea there and a factor on the sea already present cannot learn it.

| wave direction (from) | 0-45 | 45-90 | 90-135 | 135-180 | 180-225 | 225-270 | 270-315 | 315-360 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no-wind Hs / offshore Hs, old enclosure | 0.62 | 0.39 | 0.30 | 0.27 | 0.39 | 0.59 | 0.74 | 0.70 |
| same, open-boundary enclosure | 0.43 | 0.22 | 0.00 | 0.01 | 0.18 | 0.52 | 0.65 | 0.58 |
| sites below 0.3 m without wind | 9 % | 28 % | 76 % | 73 % | 24 % | 4 % | 8 % | 6 % |

Pairs needing K > 4: 2.7 % before, 11.3 % now (17.6 % of the energy the wind adds).
Real wind-sea energy at the goals from 70-200°: ~9 % (WHACS_cal, 1979-2023).

Variants on the current runs (20 % of cases held out; all sites, Hs bias / RMSE
in m; no correction −0.84 / 1.20):

| variant | all sites | no-wind Hs < 0.3 m | no-wind Hs ≥ 0.3 m |
| --- | --- | --- | --- |
| K ≤ 4 (as fitted) | −0.43 / 0.95 | −1.18 / 1.54 | −0.07 / 0.45 |
| K ≤ 20 | −0.35 / 0.89 | −1.03 / 1.43 | −0.02 / 0.43 |
| K only where Hs_ws ≥ 0.3 m | −0.48 / 0.99 | not corrected | −0.07 / 0.44 |
| added Hs (g·ΔHs/U² learned directly, ΔHs² = hs1² − hs0²) | −0.11 / 0.43 | −0.19 / 0.48 | −0.07 / 0.41 |
| K on the 163 cases with waves from 200° -> N -> 70° (5-fold) | −0.11 / 0.41 | −0.31 / 0.52 (8.9 % of pairs) | |

## Options

1. **Added Hs instead of K**: the same kind of training runs; the target is the
   energy the wind adds, so sheltered sites and offshore winds are covered. Still
   uniform training winds.
2. **Wind as one more goal**: cases with no boundary waves and a uniform wind
   (`u10`, `u10dir`, `wl`), MDA -> PCA -> GP like a goal, added to the sum. Misses
   the interaction (wind grows an existing sea less than calm water). Tested on the
   2013 storms with the real wind field (last section): the plain sum overshoots, a
   damping on the share of the sea already there fixes it. The chosen option.
3. **Correction learned from the dynamic runs**: the 2013 wind-on / wind-off pairs
   are the same hours with and without the real ERA5 field. Per site and hour,
   target ΔE = Hs_on² − Hs_off²; inputs the ERA5 wind at the site, fetch and mean
   depth along it, site depth, the no-wind Hs (and wind-sea part) and the
   wind-wave angle; one gradient boosting over all sites. In use, the no-wind sea
   comes from the metamodel sum and ΔE joins it as the wind contribution. Real wind
   fields, offshore winds and Xaver included; one year only (check its wind
   coverage against 1979-2023); inherits the SnapWave calm-hour bias.
4. **Wind as an input of every goal**: rejected (double counting, 7 inputs, no
   longer linear).

## The 2013 dynamic pairs (2026-11-05)

Netherlands 2013, SnapWave every hour with and without the ERA5 wind field, for WHACS and
WHACS_cal (`Netherlands/SnapWave_CASES_Dynamic_<offshore>_wind-<on|off>/Dynamic_Cases_2013.nc`;
open-boundary enclosure, `niter` 200, `upwindref` 1, sector 360 with wind; ~136 s per hour
with wind, 5.6 s without). See them on the dashboard (`Netherlands/dashboard/`: Error budget, Wind).

**What the reconstruction misses** (27 buoys, MDA 600, CODEC; pairs valid in every
estimate). Mean Hs error of the spectral metamodel at the buoys, split into parts:

| part (m) | WHACS | WHACS_cal |
| --- | --- | --- |
| partitions (spectral − bulk metamodel) | +0.05 | +0.05 |
| method (bulk metamodel − dynamic without wind) | −0.003 | −0.003 |
| **missing wind** (dynamic without − with wind) | **−0.31** | **−0.29** |
| model + forcing (dynamic with wind − buoy) | +0.11 | +0.13 |
| total (spectral metamodel − buoy) | −0.15 | −0.12 |
| total, each buoy's top 1 % of hours | −0.92 | −0.56 |

The wind is the largest part, and grows with the sea (top 1 %: −0.59 / −0.52 m). With
wind the dynamic run is the best estimate at the buoys (median over buoys, Hs bias / RMSE /
R²: WHACS +0.11 / 0.28 / 0.81, WHACS_cal +0.14 / 0.29 / 0.80, against −0.17 / 0.35 / 0.69 and
−0.13 / 0.34 / 0.69 for the spectral metamodel); it also fixes the spreading (−17° → −7°).

**Where and when** (4977 isobath sites). The wind adds a mean 0.20 / 0.24 / 0.30 m at
5 / 10 / 20 m, growing with the wind (20 m, WHACS: +0.05 m below 5 m/s, +0.65 at 10-15,
+1.0 at 15-20 m/s) and, at 10 and 20 m, with the sea (each site's top 1 % of hours:
+0.36 / +0.57 m). At 5 m it is depth-limited (+0.02 m in the top 1 %); in the few hours with
Hs ≥ 3 m there (151 / 381 site-hours) it lowers Hs by 0.20 / 0.28 m as the wind-sea period
takes over (Tp −3.2 / −3.7 s: SnapWave keeps one period per node).

**By direction** (`03_Wind_Effect` sections 3-6, U10 8-14 m/s). The added Hs follows the sea
already there and the fetch inside the domain: 20 m, WHACS, 0.27 m with wind from NW (no-wind
Hs 1.85 m) to 0.71 m from SE (0.45 m); added energy 1.1-1.9 m². At the buoys with U10 ≥ 8 m/s
the run with wind is high with wind from S-SW (+0.33 / +0.29 m, WHACS) and right from NW
(+0.02); without wind it is low from E-SE (−0.61 / −0.62 m). Any correction learned from these
runs (option 3) would inherit the S-SW overestimate.

**Non-converged hours** with wind: 413 (WHACS) / 331 (WHACS_cal), NaN in the files; calm
(median site-mean U10 3.5 m/s, all hours 6.5; Hs 0.32-0.35 m, all 0.68).

The K training pairs of both setups were deleted on 2026-11-04; the K numbers above are all
that is left of them.


## Wind-only storm test (2026-11-09/10)

Option 2 checked on real fields before building it: SnapWave on every hour of the six 2013
storms of the dashboard (520 hours) with **no boundary waves**, the ERA5 wind field and the
CODEC water level, otherwise the wind-on dynamic setup (Tp, Dir, Spr of WHACS_cal only fill the
boundary files). Run: `cd Netherlands/dashboard && python wind_only_storms.py build | slurm |
postprocess`, then `python build_store.py --only wind_only` (cases
`Netherlands/SnapWave_CASES_Dynamic_wind-only/`, ~3-5 min per hour, about 30 min on the cluster).
The Wind page (wind-only section: damping curve, maps, scores) and the Storms page ("wind-only
test" estimates) show it.

506 of 520 hours converged. Hs bias / RMSE (m) over all storm hours, WHACS_cal, against SnapWave
with wind (isobaths, buoy sites) and against the buoy records (9544 pairs):

| estimate | 5 m | 10 m | 20 m | buoy sites | vs buoys |
| --- | --- | --- | --- | --- | --- |
| SnapWave no wind | −0.20 / 0.31 | −0.35 / 0.45 | −0.47 / 0.59 | −0.39 / 0.50 | −0.27 / 0.50 |
| SnapWave no wind ⊕ wind only | +0.24 / 0.38 | +0.11 / 0.21 | +0.12 / 0.17 | +0.10 / 0.17 | +0.28 / 0.49 |
| metamodel spectral | +0.03 / 0.49 | −0.24 / 0.46 | −0.42 / 0.58 | −0.32 / 0.50 | −0.19 / 0.46 |
| metamodel spectral ⊕ wind only | +0.43 / 0.67 | +0.20 / 0.39 | +0.16 / 0.23 | +0.16 / 0.28 | +0.34 / 0.51 |
| SnapWave with wind | | | | | +0.17 / 0.43 |

The energy sum overshoots the run with wind (the wind grows an existing sea less than calm
water), most at 5 m where it also adds to the depth-limited overshoot of the linear sum. At 10-20 m
and the buoy sites it turns a −0.35 to −0.47 m bias into +0.10 to +0.16 m with a third to half the
RMSE against the run with wind; against the buoys the RMSE does not improve (the bias changes
sign). A wind-only term is worth keeping only with a damping that depends on the sea already
there (e.g. a factor on the wind-only energy learned from these pairs, option 3) and a depth
limit at 5 m.

**Damping** (Wind page, wind-only section; `dashboard/data.py` `damped_wind_only`). Ratio added energy / wind-only energy,
(Hs_on² − Hs_off²) / Hs_wo², against the share of the sea already there, E_off / (E_off + E_wo):
about 1-1.4 when the wind-only sea dominates, falling smoothly to ~0.3-0.6 at shares of 0.6-0.8,
the same shape at 10 m, 20 m and the buoy sites; at 5 m it falls faster (depth limit). A damped
term sqrt(E + f(E / (E + E_wo)) E_wo), f the median curve per site group fitted on the SnapWave
pairs with each storm left out (20 bins of the share, median ratio per bin with >= 50 pairs,
linear between bins, flat beyond; pairs with E_wo < 1e-4 m² left out), E the energy without wind of
the estimate (for the metamodel its own, as a reconstruction would) (WHACS_cal, all storm
hours, bias / RMSE vs SnapWave with wind; last column vs the buoy records):

| group | estimate | no wind | ⊕ wind only | ⊕ damped wind only | damped vs buoys |
| --- | --- | --- | --- | --- | --- |
| 5 m | SnapWave | −0.20 / 0.31 | +0.24 / 0.38 | +0.03 / 0.20 | |
| 5 m | metamodel spectral | +0.03 / 0.48 | +0.43 / 0.67 | +0.21 / 0.48 | |
| 10 m | SnapWave | −0.34 / 0.44 | +0.11 / 0.21 | +0.02 / 0.16 | |
| 10 m | metamodel spectral | −0.23 / 0.45 | +0.20 / 0.39 | +0.10 / 0.30 | |
| 20 m | SnapWave | −0.46 / 0.59 | +0.12 / 0.17 | −0.02 / 0.14 | |
| 20 m | metamodel spectral | −0.41 / 0.57 | +0.16 / 0.23 | +0.02 / 0.15 | |
| buoys | SnapWave | −0.42 / 0.54 | +0.12 / 0.19 | +0.02 / 0.15 | +0.19 / 0.44 |
| buoys | metamodel spectral | −0.35 / 0.54 | +0.18 / 0.31 | +0.08 / 0.25 | +0.25 / 0.44 |

SnapWave with wind vs the buoys: +0.16 / 0.43. The damped term on SnapWave without wind
reproduces the run with wind; on the metamodel it works at 20 m, is +0.10 m at 10 m and +0.21 m
at 5 m, where it adds to the linear sum's storm overshoot (the metamodel's own error, not the
wind term's). Against the buoys the damped metamodel has the RMSE of SnapWave with wind and a
slightly larger bias. So a wind term = a wind-only field (from analogs or its own MDA -> PCA -> GP
on u10, u10dir, wl) times a damping on the share of the sea already there looks feasible; the
5 m overshoot has to be fixed in the sum first.

Open: six storms of one year only (onshore winds); the wind-only field came from a run every
hour, not from analogs or a metamodel; calm and offshore-wind hours untested; the ratio above 1
when the wind-only sea dominates (the wind adds more to a small sea than to calm water) is not
explained (growth from a seed, or SnapWave's one period per node).

