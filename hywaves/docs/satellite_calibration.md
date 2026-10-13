# Offshore hindcast vs satellite altimetry: validation and calibration

Since hywaves 0.13.0; optional (`satellite` block of the case JSON, notebook
`07b_Satellite_Calibration.ipynb`). Code: bluemath_tk `waves/altimetry.py`
and `waves/calibration.py` (`CalVal`), case layer `utils/calibration.py`.

## Summary

HyWaves is only as good as its offshore forcing. Before the dynamic runs and
the reconstruction, notebook 07b compares the offshore hindcast at every goal
with satellite altimeter Hs and, if wanted, calibrates it per goal:

1. altimeter observations within `radius_km` of each goal's hindcast node, in
   water deeper than `min_depth`, quality flags `flags`;
2. one median Hs per satellite pass, paired with the hindcast at the pass
   time;
3. a directional calibration per goal (wind sea and swell, per direction
   sector), fitted on `train_fraction` of the passes (75 % by default) and
   validated on the rest;
4. the calibrated hindcast written as a new zarr and added to
   `offshore_waves` (e.g. `WHACS_cal`), used from 08 onwards like any other
   offshore source.

Nothing is specific to one hindcast: any `offshore_waves` entry on
`(time, goal)` with `hs`, `dir` (and partitions `phs{i}`, `pdir{i}`, partition
0 the wind sea) works, so ERA5, WHACS or another product go through the same
steps.

## The altimeter store

A zarr of along-track observations on one `obs` dimension, sorted by
1-degree cell and then time, with a cell index (`cell_start`, `cell_count` on
`lat_cell`, `lon_cell`): the observations near a point are a few contiguous
slices, so reading the passes around ten goals takes seconds, locally or from
a server (the zarr chunks are read over HTTP the same way). The GeoOcean copy
of the IMOS multi-mission product:

`/lustre/geocean/DATA/IMOS/processed/imos_altimeter_1985_2020.zarr`
(1.6 · 10⁹ observations, 15 missions, 1985–2020): `hs` (calibrated Ku band),
`time`, `latitude`, `longitude` (0–360), `flag` (1 good, 2 probably good,
3 correctable; bad data removed), `depth` (negative in the ocean), `dist2coast`
(km), `satellite`, winds.

A few tracks carry depth fill values (below −11 000 m); they are read as
missing, so they fail the depth filter.

## Matching

| Step | Choice | Why |
| --- | --- | --- |
| Where | Observations within `radius_km` (default 50 km) of the goal's **hindcast node** (goals JSON `hindcast.<name>`), not of the goal | The hindcast series is the node's |
| Depth | Only water deeper than `min_depth` (default 15 m) | The altimeter footprint degrades in shallow water and near the coast |
| Quality | `flags` (default 1 and 2) | Flag 3 is CryoSat-2 only, correctable |
| Per pass | Consecutive observations of a satellite less than 5 min apart are one pass; its **median** Hs, if it has at least `min_points` (3) | 1 Hz observations along a track are not independent; one value per pass keeps the statistics honest and resists outliers |
| In time | Hindcast at the nearest step, within `max_time_diff` (30 min) | Nearest, not interpolated, so directions and partitions stay consistent |

## Calibration

Directional calibration (Mínguez et al., 2011): the satellite energy is
fitted as a sum of the hindcast energy of each family in each direction
sector (16 sectors of 22.5°, centred on North),

```
Hs_sat² = Σ_k a_sea,k · E_sea,k + Σ_k a_swell,k · E_swell,k        (no intercept)
E_family,k = Σ Hs_i²  over the partitions of the family whose direction is in sector k
```

- Partition 0 is the wind sea, the others swell. Without partitions, the
  bulk `hs`/`dir` is one family.
- No intercept, so calm seas stay calm: the correction is multiplicative.
- A coefficient is kept only with at least `min_samples` (20) training passes
  with energy in its sector, a p-value below 0.05 and a positive value; the
  others are fixed at 1 (no change) and the rest refitted.
- The correction multiplies each partition Hs by `√a` of its family and
  sector, and scales the bulk Hs by the change in total partition energy.
  Periods, directions and spreads are unchanged.

The passes of each goal are split at random (seed 0) into `train_fraction`
for the fit and the rest for validation; `split: "time"` fits on the first
part of the record and validates on the last, to check that the correction
holds in time. Metrics are reference first (bias = hindcast − satellite).

## Outputs

`outputs/calibration/{offshore}/`:

- `pairs.csv`: one row per goal and pass (`goal`, `time`, `sat_*` for the pass,
  hindcast variables);
- `calval_goal_XX.pkl`: the fitted calibration of each goal;
- `coefficients.csv`: `goal, family, sector, center, n, a, factor, fitted, p_value`;
- `metrics.csv`: `offshore, goal, set (train/test/all), stage (raw/calibrated), n, bias, rmse, si, r2, slope`.

`write_calibrated_hindcast` writes `<source>_cal.zarr` next to the source and
returns the entry to add to `offshore_waves`:

```json
{"name": "WHACS_cal", "data_path": "inputs/WHACS/processed/Netherlands_WHACS_goal_cal.zarr", "hindcast": "WHACS"}
```

`hindcast` names the goals JSON hindcast the series sit on, so the calibrated
entry reuses the WHACS nodes (e.g. in the offshore buoy validation).

## After calibrating

- The dynamic runs (08) must be forced with the calibrated hindcast too, so
  that 09 compares the reconstruction and the dynamic runs with the same
  forcing.
- The offshore buoys (10a) are an independent check: run it with the raw and
  the calibrated names.
- The metamodel does not change (its cases are sampled within bounds), but
  check that the calibrated Hs stays inside `simulations.bounds.hs`.

## Netherlands (2026-10-13)

WHACS against IMOS, 1985–2020, radius 50 km, depth > 15 m, flags 1–2, ≥ 3
points per pass: **2145–4549 passes per goal** (goal 1, at the Belgian corner,
has the fewest). Metrics on the 25 % test passes:

| goal | n | bias raw → cal (m) | RMSE raw → cal (m) | slope raw → cal |
| --- | --- | --- | --- | --- |
| 1 | 535 | −0.069 → −0.033 | 0.286 → 0.284 | 0.88 → 0.90 |
| 2 | 1071 | −0.065 → −0.003 | 0.247 → 0.237 | 0.92 → 0.96 |
| 3 | 1115 | −0.091 → +0.005 | 0.280 → 0.267 | 0.87 → 0.94 |
| 4 | 874 | −0.085 → +0.044 | 0.304 → 0.278 | 0.81 → 0.93 |
| 5 | 1092 | −0.115 → +0.062 | 0.304 → 0.270 | 0.81 → 0.95 |
| 6 | 935 | −0.105 → +0.060 | 0.298 → 0.261 | 0.83 → 0.95 |
| 7 | 1135 | −0.085 → +0.077 | 0.277 → 0.270 | 0.86 → 0.98 |
| 8 | 904 | −0.032 → +0.074 | 0.244 → 0.256 | 0.89 → 0.98 |
| 9 | 905 | −0.065 → +0.069 | 0.279 → 0.270 | 0.85 → 0.96 |
| 10 | 1031 | −0.046 → +0.062 | 0.284 → 0.281 | 0.87 → 0.98 |

WHACS is low, more so at high Hs (slope 0.81–0.92), consistent with the
−0.6 m Hs bias at the buoys during storm Xaver (Dec 2013). The calibration
brings the slope to 0.90–0.98 and lowers the RMSE at 9 of 10 goals (not goal 8), but it
**overshoots the mean**: the test bias goes from −0.03…−0.12 m to
−0.03…+0.08 m. Fitting energy (Hs²) weights the large waves, which WHACS
underestimates most, so moderate seas are pushed slightly high (e.g. goal 5:
wind-sea factors 1.06–1.23 in all 16 sectors). Whether this is acceptable, or a fit that
weights the moderate seas more is needed, is still open; the offshore buoys
(10a) should decide.

## ERA5

`07a_Download_ERA5.ipynb` reads ERA5 ocean waves from the Earth Data Hub zarr
at the goal nodes (bluemath_tk `waves/hywaves/offshore.py`: `open_era5_waves`,
`era5_at_goals`; key in the `EDH_API_KEY` environment variable) into the same
HyWaves names, partitions included (0 = wind sea, 1–3 = swells), so 07b runs on
it unchanged (`offshore_name = "ERA5"`). ERA5 has no partition peak periods:
`ptp{i}` are mean periods (Tm-1,0), and the swells share one spread. Netherlands,
1979–2023, 10 goals: about 5 minutes.

With `radius_km = 10` (the circle covers only a few tracks: 58–1422 passes per
goal, 12–355 test passes), ERA5 has a lower test RMSE than WHACS at 8 of 10
goals (equal at goal 4, higher at goal 2, 14 test passes) and slopes closer to
1 at most goals; with so few passes per sector most coefficients
stay at 1, so 10 km is a good validation radius but a thin one for calibrating.

## Limitations

- Altimetry gives Hs only: periods and directions are not calibrated.
- The coefficients are fitted around each node (50 km) and applied to its
  series; between goals the forcing is interpolated as before.
- Altimeter Hs itself has errors (calibrated against buoys by IMOS), larger in
  shallow water and near the coast, hence the depth filter.
