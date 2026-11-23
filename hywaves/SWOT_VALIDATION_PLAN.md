# SWOT validation plan

Plan (2026-11-23, not started) to validate the reconstruction at the output sites against
SWOT KaRIn, automatically, for every case. Like `WIND.md`, it keeps the numbers that
motivated it; when it is built, the method moves to docstrings, the notebook and
`CLAUDE.md`, and this file goes.

## Data

- SWOT KaRIn wind-wave store: `https://geoocean.unican.es/zarr/SWOT/swot_karin_windwave.zarr`
  (2 km pixels, 2023-03 to now, per-pixel `swh_karin_qual`, `dist2coast`, `depth`, `cycle`,
  `pass`). Read with `bluemath_tk.waves.altimetry.altimeter_near_points` (one read for many
  sites). Public URLs in the case JSONs (users off the cluster).
- SWOT nadir altimeter: in the IMOS store `https://geoocean.unican.es/zarr/IMOS/imos_altimeter_dm00.zarr`,
  `satellite` 17 (same pass as KaRIn, at the same time, ~60 km away).
- Period: 2023-07 (science orbit) to 2026-01 (end of WHACS).
- KaRIn reads 0.2-0.3 m high below 1 m Hs and is unbiased above 2 m (against buoys).

## KaRIn artefacts: a quality check is required first

Found at the Cantabrian node (-3.8, 43.7), 2023-07 to 2026-01, WHACS against KaRIn
(5 km median, qual 0, dist2coast >= 2 km): RMSE 1.06 m, r 0.79 on 155 passes, against
0.36 m, r 0.975 for WHACS vs the IMOS altimeters (50 km). About 6 passes read 4.5-8 m
where the SWOT nadir on the same pass read 1.5-2.9 m (2024-02-09, 2024-04-30, 2024-09-05,
2024-09-23, 2024-10-17, 2025-03-22); real storms (2024-03-01, Darragh 2024-12-08, Herminia
2025-01-27) agree with the nadir. `swh_karin_qual` and the pixel spread do not flag them,
and an artefact can cover the whole 30 km around the site.

Screen used in the BlueMath notebooks (`toolkit/data/WHACS_Validation_Zarr.ipynb`,
`WHACS_Node_Selection_Zarr.ipynb`, function `screen_with_nadir`, 2026-11-23): drop a KaRIn pass
(5 km median) that is **higher** than the SWOT nadir of the same pass (IMOS passes within 100 km,
satellite `SWOT`, +-15 min) by more than max(0.75 m, 25 %); one-sided, because near the coast
KaRIn is rightly lower than the nadir offshore. At that node it drops 11 of 155 passes and WHACS
vs KaRIn goes to RMSE 0.41 m, r 0.971 (altimeters: 0.36 m, 0.975). A two-sided variant on the
30 km KaRIn median plus a 5 km / 30 km spike ratio <= 1.5 gave a similar result.

To do: move `screen_with_nadir` to `bluemath_tk.waves.altimetry` with tests (the notebooks then
import it), and check the tolerance on nodes of Samoa, NSW and the Netherlands.

## Step A: SWOT at the goal nodes (07b section 6)

Independent check of the satellite calibration (fitted on IMOS): KaRIn (with the check)
against `WHACS` and `WHACS_cal` at each goal node, the same metrics and tail quantiles as
section 5. Cheap (minutes). Questions it answers: the per-goal tail (Samoa goal 8 overshoots
at p99.9 on the Aunu'u buoys) and NSW, where calibration makes the Eden buoy worse.

## Step B: validation at the output sites

### Configuration (`validation.swot` in the case JSON, checked in `validate.py`)

```json
"swot": {"data_path": "https://geoocean.unican.es/zarr/SWOT/swot_karin_windwave.zarr",
         "nadir_path": "https://geoocean.unican.es/zarr/IMOS/imos_altimeter_dm00.zarr",
         "period": ["2023-07-01", "2026-01-31"], "sets": ["isobaths", "buoys"],
         "offshore": ["WHACS_cal", "WHACS"], "spacing_km": 2.5, "radius_km": 5,
         "depth_ratio": [0.5, 2.0], "min_pixels": 3, "min_dist2coast_km": 2,
         "min_passes": 20, "hs_min": 1.0}
```

### Per site (any set)

1. Footprint: KaRIn pixels with centre within `radius_km`, qual 0, `dist2coast` >=
   `min_dist2coast_km`, Hs > 0, and pixel depth within `depth_ratio` x site depth (a 5 km
   circle around a 10 m site otherwise measures mostly deeper water).
2. One KaRIn value per pass: median of those pixels, kept with >= `min_pixels` and if the
   pass passes the artefact check; pixel count and spread saved.
3. Estimate at the pass time: hourly reconstruction at the site interpolated linearly, per
   offshore source; baseline = offshore hindcast at the nearest goal (no propagation).
4. Metrics (reference = SWOT, `bluemath_tk.core.metrics`): n, bias, rmse, si, r, slope, on
   all passes and on SWOT Hs >= `hs_min`; only sites with >= `min_passes`.
5. Groups: per isobath depth, per nearest goal, per set.

### Per kind of site

- Isobaths: ok sites thinned to one every `spacing_km` along each isobath (neighbours do
  not share pixels). Maps of bias and RMSE along 10/20/50 m. The 5 m isobath will be almost
  empty (coast mask + depth match; worst on the steep NSW coast): shown as missing coverage.
- Buoys: three series on the same passes: SWOT vs buoy says how far SWOT can be trusted at
  that spot; reconstruction vs SWOT and vs buoy. Where SWOT agrees with a buoy, the SWOT map
  around it is credible. Useful mostly for buoys >= 10-15 m and a few km offshore.
- Polygon mesh nodes (e.g. Netherlands `deenhaag`): thinned to the same spacing; coverage
  map (passes per node) first, few nodes will score.

### Reconstruction

`reconstruct` over the SWOT months on the thinned sites, bulk, `wl=0` (CODEC ends in 2018:
the tide is missing, it matters at 5 m in the Netherlands; flag it), best MDA, depth cap as
configured, one `reconstruction_resources` cell first. Cost from the measured rate (~1 s per
site-year in bulk on the 48-core node): ~30-60 min per offshore source per case. The user
launches it. The same 2023-2026 store also serves notebook 10 for buoys with recent data only
(Samoa `King_Poloa`, the Belgian buoys).

### Outputs, notebook, portal

- `outputs/swot_validation/<set>/`: `pairs.csv` (site, time, swot_hs, n_pixels, spread,
  nadir, flags, each estimate, baseline), `metrics_sites.csv`, `metrics_groups.csv`,
  `coverage.csv` (passes per site or why dropped).
- Template notebook `10b_Validate_SWOT`: config, resources, pairs, reconstruction, metrics,
  coverage and metric maps, depth summaries, scatter; save cell.
- `publish.py` -> `portal/swot_validation/`; GUI map layer. `validate.py`, `publish.py`,
  `hywaves/tests/` and the `CLAUDE.md` reference change together (rule 3).

## Order

1. Artefact check in bluemath_tk, tested on several nodes; BlueMath notebooks re-run.
2. Step A (07b section 6) in the three cases.
3. Step B: template code, JSON block, 10b, validate/publish/GUI/tests; then the runs.
