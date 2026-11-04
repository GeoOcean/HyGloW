# Wind: how the local wind enters, what we know, what is open

State on 2026-11-04 (Netherlands, SnapWave, hywaves 0.15.0). The method summary
is in [`README.md`](README.md); this file holds the wind details, the numbers
and the options under review.

## Why the wind is separate

The metamodel cases run **without wind**. Each case is one goal's share of the
boundary; with wind, every goal's run would grow its own copy of the local
sea and the sum would count it once per goal. Growth is also non-linear: it
slows as the sea develops, which depends on the total energy, not on one
goal's share. So the wind enters once per hour and site, after the goals are
predicted and before their spectra are summed.

## Current method: the growth factor K

Code: `bluemath_tk.waves.hywaves.wind_correction` (model, reconstruction
step), case layer `utils/wind.py`, notebook `05b_Wind_Correction`, config
`metamodel.wind_correction`.

1. **Fetch table**: fetch and mean depth along it per output site and wind
   direction (5°), over the mesh (`outputs/wind/fetch_table_{model}.nc`).
2. **Training cases**: `num_cases` full-domain seas (every goal forced the same),
   each run twice with the dynamic templates, with and without a uniform wind of
   `speed_ratio` x the fully developed speed (clipped to `u10_range`), blowing
   `wind_wave_angle` off the waves; waves from the goal sectors and, if set, the
   clockwise sector `directions`.
3. **K** = Hs(with wind) / Hs(without) of the wind-sea part
   (`K² = 1 + (hs1² − hs0²) / (w hs0²)`, w = offshore wind-sea weight, cases with
   w ≥ 0.5), a gradient boosting of log K on g·Hs_ws/U², g·F/U², g·d/U², g·d̄/U²,
   cos(wind − wave), K in [0.5, 4].
4. **Reconstruction** (`wind_correction=True`): `E_ws = Σ w·hs²` over the goal x
   partition contributions (w from their offshore steepness 0.02-0.03 and spreading
   20-25° ramps), wind from ERA5 (`wind: forcing`) or inferred from the sea state;
   one extra contribution `(K² − 1)·E_ws` from the wind direction (wind-sea Tp,
   spreading 30°) joins the sum.

Netherlands settings: 250 cases, `speed_ratio` [0.7, 1.7], `u10_range` [3, 25],
`wind_wave_angle` ±60°, `directions` [-160, 70] (waves from 200° through N to 70°).

## What we know

**SnapWave wind mode** (39 hours of 2013, 27 buoys, `Netherlands/model_decisions/`):
adds ~0.4 m in calm hours (SWAN ~0.2), so SnapWave stays ~+0.2 m high there and
any correction learned from it inherits that; `fw` 0.02 / 0.04 trade it for worse
storms. It does not converge in 241 of 8760 hours of 2013 (swell, light wind; Hs
up to ~10 m), and no setting fixes it: postprocessing drops non-converged runs
(last error > `MAX_ERROR` 0.01), metamodel cases are left out, dynamic hours NaN.
One period per node: wind energy over swell takes the swell period.

**Archived 05b run** (`portal_snapwave/SnapWave_CASES_Wind`, before the
`model_decisions/` setup: old AoI rectangle forcing the coast nodes, no-wind
twins with `niter` 40): 250 cases, `speed_ratio` [0.8, 2.0], u10 up to 35 m/s.
Held out (reference = run with wind), `waves` sites Hs bias / RMSE −0.70 / 1.00 ->
−0.06 / 0.39 m. But 131 of the 250 no-wind twins did not converge (mostly the
smaller seas), and part of K came from the coast forcing.

- *More cases do not help*: 5-fold by case, 50 / 100 / 150 / 200 training cases
  give RMSE 0.41 / 0.42 / 0.38 / 0.38 m (no correction 0.86).
- *Error grows with the wind*: per case, Spearman with u10 0.78, with
  `speed_ratio` 0.64, with the angle 0.00; RMSE 0.10 m below 10 m/s, 0.45 m at
  25-35 m/s.
- *Real hours* (WHACS_cal wind-sea partitions at the 10 goals x ERA5 at the goal,
  1979-2023, 2.6 M partition-hours): u10 p50 / p90 / p99 9.2 / 13.4 / 17.0 m/s,
  `speed_ratio` 1.24 / 1.53 / 1.89; 95 % of the hours (98.8 % of the wind-sea
  energy) lie inside the old design, so the design was too windy, not too narrow.

**New 05b run** (2026-11-03, current templates: open-boundary enclosure, `niter`
200, `upwindref` 1; `speed_ratio` [0.7, 1.7], `u10_range` [3, 25], no `directions`):
every twin converges, but held out the correction is much worse, `waves`
−1.03 / 1.41 -> −0.51 / 1.09 m. The cause is the enclosure: waves enter only
through the open boundary, so with waves from the land side the no-wind run has
almost nothing at the sites, the wind grows the whole sea there and K (a factor on
the sea already present) cannot learn it (capped at 4).

| wave direction (from) | 0-45 | 45-90 | 90-135 | 135-180 | 180-225 | 225-270 | 270-315 | 315-360 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no-wind Hs / offshore Hs, old enclosure | 0.62 | 0.39 | 0.30 | 0.27 | 0.39 | 0.59 | 0.74 | 0.70 |
| same, open-boundary enclosure | 0.43 | 0.22 | 0.00 | 0.01 | 0.18 | 0.52 | 0.65 | 0.58 |
| sites below 0.3 m without wind | 9 % | 28 % | 76 % | 73 % | 24 % | 4 % | 8 % | 6 % |

Pairs needing K > 4: 2.7 % before, 11.3 % now (17.6 % of the energy the wind adds).
Real wind-sea energy at the goals from 70-200°: ~9 % (WHACS_cal, 1979-2023).

Variants on the new runs (notebook split, 20 % of cases held out; all sites, Hs
bias / RMSE in m; no correction −0.84 / 1.20):

| variant | all sites | no-wind Hs < 0.3 m | no-wind Hs ≥ 0.3 m |
| --- | --- | --- | --- |
| K ≤ 4 (as fitted) | −0.43 / 0.95 | −1.18 / 1.54 | −0.07 / 0.45 |
| K ≤ 20 | −0.35 / 0.89 | −1.03 / 1.43 | −0.02 / 0.43 |
| K, fitted and applied only where Hs_ws ≥ 0.3 m | −0.48 / 0.99 | not corrected | −0.07 / 0.44 |
| added Hs (g·ΔHs/U² learned directly, ΔHs² = hs1² − hs0²) | −0.11 / 0.43 | −0.19 / 0.48 | −0.07 / 0.41 |
| K on the 163 cases from 200° -> N -> 70° (5-fold) | −0.11 / 0.41 | (8.9 % of pairs) −0.31 / 0.52 | |

So the Netherlands config now sets `directions` [-160, 70] (re-run of 05b pending).
The waves from 70-200° then keep almost no correction: their no-wind sea at the
sites is ~0, so `(K² − 1)·E_ws` adds ~nothing.

## Options under review

1. **Added Hs instead of K**: same runs and features, the target is the energy
   the wind adds; covers sheltered sites and offshore winds. Still uniform
   training winds.
2. **Wind as one more goal**: cases with no boundary waves and a uniform wind
   (`u10`, `u10dir`, `wl`), MDA -> PCA -> GP like a goal, added to the sum. Misses
   the interaction (wind grows an existing sea less than calm water); measurable
   with 250 cheap runs (the training cases with zero boundary waves).
3. **Correction learned from the dynamic runs**: the 2013 wind-on / wind-off pairs
   are the same hours with and without the real ERA5 field. Per site and hour,
   target ΔE = Hs_on² − Hs_off², inputs the ERA5 wind at the site, fetch and mean
   depth along it, site depth, the no-wind Hs (and wind-sea part) and the
   wind-wave angle; one gradient boosting over all sites (subsampled, every storm
   kept). In use, the no-wind sea comes from the metamodel sum and ΔE joins it as
   the wind contribution. Real wind fields, offshore winds and Xaver included; one
   year only (check its wind coverage against 1979-2023), inherits the SnapWave
   calm-hour bias.
4. **Wind as an input of every goal**: rejected (double counting, 7 inputs,
   no longer linear).

## Data for the analysis

Netherlands 2013 dynamic runs, all with the current setup (open-boundary
enclosure, `niter` 200, `upwindref` 1; wind-on: ERA5 field, `sector` 360):

| folder (`Netherlands/`) | postprocessed |
| --- | --- |
| `SnapWave_CASES_Dynamic_WHACS_wind-off` | `Dynamic_Cases_2013.nc` |
| `SnapWave_CASES_Dynamic_WHACS_wind-on` | `Dynamic_Cases_2013.nc` |
| `SnapWave_CASES_Dynamic_WHACS_cal_wind-off` | `Dynamic_Cases_2013.nc` |
| `SnapWave_CASES_Dynamic_WHACS_cal_wind-on` | running (2026-11-04) |

~136 s per hour with wind, 5.6 s without. WHACS wind-on: 413 non-converged hours
(366 with offshore Hs ≤ 1 m), NaN in the nc. First check against 27 buoys (median
over buoys, 8347 converged hours, Hs bias / RMSE / R²): WHACS wind-off
−0.21 / 0.36 / 0.63, WHACS_cal wind-off −0.17 / 0.34 / 0.64, WHACS wind-on
+0.11 / 0.28 / 0.81. Quick check per run: `PYTHONPATH=. python
tmp/check_dynamic_run.py <offshore> wind-on wind-off` from `Netherlands/`.

Plus the 05b training pairs (`SnapWave_CASES_Wind/`, the archived ones in
`portal_snapwave/`) and the buoys (notebook 10).

Next: what the wind adds in the dynamic pairs (per site group, by wind speed and
direction, storms vs calm, non-converged hours); how the current K and the options
above reproduce it on held-out months of 2013; then the buoys. Decide the method
from that.

## Open

- K is a step function (gradient boosting): mm-level input changes flip it at ~1 %
  of site-hours (up to 2.6). Smooth it before trusting wind-corrected extremes.
- Training winds are uniform; real ones are fields.
- Wind-wave angles beyond the trained range keep the edge value.
