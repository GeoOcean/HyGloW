# CLAUDE.md — HyGloW

Guide for maintainers and coding agents. User docs: `README.md`,
`hywaves/README.md` (the method, resources and timings), the case README. Case rules:
`hywaves/validate.py`.

## What it is

**HyGloW** (project) / **HyWaves** (method): offshore hindcast -> nearshore
waves without a spectral model every hour. Per offshore *goal*, a few hundred
stationary SnapWave or SWAN runs train an **MDA -> PCA -> GP** metamodel; the
nearshore series (bulk or spectra) is the **linear sum of the goal spectra**.
Any offshore hindcast can be reconstructed (WHACS, ERA5, raw or satellite-
calibrated), with water level from another hindcast (CODEC). The wrappers also
run **dynamic** reference cases (stationary, every hour, all goals, optional
ERA5 wind and water level) to validate the sum. No wind in the reconstruction
yet (`hywaves/WIND.md`).

| Repo / path | Role |
| --- | --- |
| `hywaves/` | One module per command (`start`, `validate` = the case rules, `publish`, `snapshot`, `update`), `common.py` (shared helpers, case manifest), `cli.py`, the case `template/` |
| `Netherlands/` | Reference case, latest code. `Samoa/`: migrated to 0.17 (2026-11-13), re-run in progress. `NorthCarolina/`: started 2026-11-15 on a delivered mesh, run stopped at 08 by a full disk (2026-11-16). `NewSouthWales/`: migrated to 0.17 (2026-11-16), run not started |
| `GUI/` | Dash results explorer, reads only `{Case}/portal/` |
| `deck/` | Slide deck on GitHub Pages: `python deck/build.py` -> `deck/_site/` from the `portal/` bundles (`.github/workflows/deck.yml`) |
| `~/BlueMath_tk` | **bluemath_tk**: all generic code (HyWaves in `waves/hywaves/`, wrappers in `wrappers/{snapwave,swan}/` holding only `_wrapper/_utils/_plots.py`) |
| `~/BlueMath` | Examples (`climate_services/hindcast/HyWaves_Netherlands`, branch `hywaves/netherlands`) |

## Environment and commands

- Python: `~/miniforge3/envs/bluemath-312/bin/python` (bluemath_tk editable
  from `~/BlueMath_tk`). No GPU on the login node (GP inference runs on CPU).
- SnapWave / SWAN are external binaries; on the GeoOcean cluster cases run as
  SLURM arrays (`wrapper.run_cases_slurm(**sim.slurm_kwargs())`). SWAN: one
  serial `swan.exe` per task (module `swan/4151`; MPI misplaces
  `BOUNDSPEC ... VARIABLE PAR` on unstructured grids). SnapWave: `snapwave/b663c81-wind2`
  (wind1 + boundary spectra centred on each boundary point's direction; installed
  2026-11-15 in `/nfs/software/geocean/models/snapwave`, owned by `tausiaj`; the module
  default since 2026-11-16, wind1 kept for old runs). Case JSONs load it by name. Draft
  issue for upstream (github.com/danoroelvink/SnapWave):
  `/lustre/geocean/WORK/users/javiert/software/snapwave/upstream_issue.md`.
- Heavy data (`*.nc`, `*.zarr`, `*.tiff`, `*_CASES_*`, `outputs/`, `logs/`,
  `tmp/`) is git-ignored and huge: don't scan it recklessly.

```bash
python -m hywaves validate <Case>/ [--publish]
python -m hywaves publish  <Case>/                 # -> portal/, commit it (book/ is git-ignored)
python -m hywaves snapshot <Case>/ <tag> --note "…"  # portal/ -> portal_<tag>/ before a re-publish
python -m hywaves update   <Case>/ [--dry-run]     # case utils/ <- template
python -m unittest discover -s hywaves/tests -t .  # all hywaves tests (case_utils/ run inside a started case)
cd Netherlands && PYTHONPATH=. python -m unittest discover -s ../hywaves/tests/case_utils -t ../hywaves/tests/case_utils
cd ~/BlueMath_tk && python -m unittest tests.waves.test_hywaves tests.waves.test_hywaves_plots tests.waves.test_hywaves_store tests.waves.test_hywaves_depth_cap tests.waves.test_hywaves_resources tests.waves.test_fetch tests.core.test_resources tests.core.test_core_metrics
cd GUI && python app.py                            # http://localhost:8050
```

## Rules

1. **Latest working code only, zero legacy.** No backward compatibility, old
   keys, fallbacks or dead code in hywaves or `template/utils/`; replace and
   migrate the template and the case being worked on. Old cases stay
   reproducible through their vendored `utils/` and the bluemath_tk commit in
   `.hywaves/manifest.json`. Small, simple functions; generic code goes to
   bluemath_tk, HyGloW keeps the JSON/case layer.
2. **`hywaves/template/` is canonical.** General fixes go there (tests in
   `hywaves/tests/case_utils/`), then `python -m hywaves update <Case>/`. Case-only code stays in
   the case (`Netherlands/utils/buoy_archive.py`, `rijkswaterstaat.py`). Say
   which copy you edit. Netherlands has no drift in `utils/` (machine
   settings are passed as resources, never hard-coded).
3. **Case rules live in `hywaves/validate.py`** (no spec document). Changing what a case provides: `validate.py`, `publish.py`,
   `hywaves/tests/` and the reference below together.
4. **The GUI reads only `portal/`**: extend `publish.py`, never make the GUI
   read `outputs/` or inputs.
5. **Notebooks are thin** (logic in `utils/` or bluemath_tk; import
   bluemath_tk directly where it owns the code). Edit sources only (a JSON
   round-trip that keeps outputs byte-for-byte); don't strip outputs unless
   asked. The user launches long runs (models, fits, reconstructions);
   prepare them.
6. **Refactors are checked numerically**: snapshot outputs on Netherlands
   data before, compare after (rebuild the script in the scratchpad).
7. **Docs stay minimal**: root README, `hywaves/README.md` (brief physics, resources + timings table: update it when timings are re-measured),
   the case README (one line per notebook + JSON keys), notebook markdown,
   short GUI/deck/dashboard READMEs, `hywaves/RECONSTRUCTION.md` (how the
   reconstruction uses the machine, buoy timings, open points),
   `hywaves/WIND.md` (wind: what was tried, the numbers, the options;
   wind facts go there, not here). Method
   details go in bluemath_tk docstrings, maintainer facts here. No other
   contract or method-note files. **No metrics in docs** (they go stale when a
   run changes): give the conclusion in words and where to see it (notebook,
   the dashboard); timings and settings are fine. `WIND.md` is the
   exception.
8. **Style**: Python 3.12, type hints, `from __future__ import annotations`,
   NumPy docstrings, `pathlib`, loggers `hywaves.*`; sparse comments.
9. **Honest numbers**: in answers, quote what was computed, with period, MDA
   size and set; flag doubts about physics or data.

**Git**: PRs to `main` of `GeoOcean/HyGloW` from branches; 
messages `[JTH] short lowercase description`; commit only when asked;
after publishing, commit `portal/`.

## How a case works

- Notebooks run from the case folder; `utils.config` loads `<Case>/<Case>.json`
  on import (`PATHS`, `SIMULATIONS`, `METAMODEL`, `FORCING`, `SATELLITE`,
  `resolve()`, `load_goals()`, `open_offshore_waves()`); restart the kernel
  after editing the JSON. No hard-coded paths in notebooks.
- Pipeline: 01 GEBCO (optionally merged with local sources,
  `bluemath_tk.topo_bathy.merge`) -> 02 mesh -> 03 observations + output sites -> 04 goals
  -> 05 training cases -> 06 fit + holdout -> 07a
  offshore (`07a_Offshore_WHACS` or `_ERA5`; 07b satellite calibration; 07c
  CODEC water level + ERA5 wind, `offshore.codec_at_goals` / `era5_wind`) ->
  08 dynamic runs (08b storms: metamodel and dynamic runs against the buoys, maps, runs
  against a `reference_run` such as SWAN) -> 09 linear sum vs dynamic -> 10 buoys
  (`10_Validate_Buoys`) -> 11 spectra -> 12 GUI figures. Save cells of
  06/09/10/12 write the `outputs/` that `publish` bundles.
- `Samoa/numerical_tests/`: why the linear sum is high against the dynamic run
  in Samoa (exact sums of goal runs, SnapWave variants, a per-point-direction
  test build, SWAN and wind1 runs of Cyclone Gita checking wind2); README +
  `scripts/` + frozen `results/`.
- `Netherlands/numerical_tests/` is the reference for the models: how the SWAN
  and SnapWave templates were chosen (01-02, from `results/*.csv`) and SnapWave
  vs SWAN on the same cases (03-04, `model_comparison.py`; not a clean
  comparison yet). Its notebooks put the case folder on `sys.path`; `runs/` is
  git-ignored. `Netherlands/dashboard/` (case-only Dash app, not `GUI/`): every estimate on
  the same sites and hours of 2013 (error budget, linear sum vs dynamic, wind, storms,
  metamodel, calibration, numerical tests; SnapWave and SWAN dynamic runs, their grid maps
  compared on analog hours); `build_store.py` packs the results (read by
  `sources.py`, every dynamic read names its model) into `store/` (git-ignored), `app.py` reads only it; rebuild the
  store after re-running 06-08 or the 2013 reconstruction. Its README holds the
  first-iteration conclusions (the Findings page shows them).
  `dashboard/wind_only_storms.py` runs SnapWave wind-only on the storm hours
  (`build_store.py --only wind_only`; Wind and Storms pages).
  `Netherlands/tmp/` holds only caches (`reconstruct/`) and scripts
  (`check_dynamic_run.py`, `reconstruct_2013.py`: fills the 2013 cache).
- Conventions: directions nautical, coming-from, clockwise; goal `angles` =
  `[left, right]` (left may be negative across North); goal ids are string
  ints. **Metrics reference first** (`bluemath_tk.core.metrics`): bias =
  estimate − reference, R² normalised by the reference, reference on the x-axis.
- Model: `simulations.model` (`snapwave`|`swan`) + `simulations.<model>`
  (`grid_file`, template and cases folders, `slurm`). Each model
  has its own cases folders; `split.json` records the model and the
  artefacts refuse to mix. SWAN: predict `tp` (PCA target), not `"raw"`.
- **Output sites** (notebook 03, `geometry.output_sites`): isobaths traced on
  the topobathy raster blanked outside the mesh, mesh nodes in polygons
  (optional `fraction`), buoys; each gets `mesh_depth` and `status` (`ok`
  only if inside the mesh and `mesh_depth + bounds.wl[0] >= min_depth`).
  Saved once in `outputs/output_sites_{model}.csv` (`# grid:` header);
  everything reads its `ok` sites, and cases run with another table are
  refused. `hs = 0` is a valid zero contribution, not dry. `site_filter`
  (`hs_ratio_max`, `tp_min`, `hs_nan_max`) removes failing sites per goal
  (`split.json` `removed_sites`).
- **Metamodel** (`utils/metamodel.py`): inputs `hs, tp, dir, spr, wl` (LHS,
  steep seas dropped by `max_steepness`); predicts absolute nearshore fields.
  **Sets** (`metamodel.sets`: `buoys`, `isobaths`, ...) = PCA + GP per site
  group on the same cases and MDA; every call names its set. **Resources:
  the user decides**: `fit_resources()` / `reconstruction_resources(<same
  args as reconstruct>)` print the machine and a proposal (<= 75% of it,
  never SLURM) and return an editable `FitResources` /
  `ReconstructionResources`, passed as `resources=` (required) to
  `fit_all` / `reconstruct` / `goal_contributions` / `sum_contributions`.
  Notebooks: one resources cell before the first heavy call. API:
  `fit_all(resources=)`; `predict(goal, n, forcing, metamodel=)` (row dim = forcing
  index name); `predict_goal(offshore, goal, n, metamodel=, sites=,
  partition=, wl=)`; `goal_contributions` + `sum_contributions` =
  `reconstruct(time, n, metamodel=, spectral=, wl=0|"forcing",
  save_spectra=, ...)`: reads `reconstruction_store(...)`
  (one zarr per setup in `tmp/reconstruct/{offshore}/{set}_mda{n}_{model}_
  {bulk|spectral}_wl-{wl}[_spectra].zarr`, chunk = 1 site x
  1 month, rebuilt when the setup or an input file changes), computes only
  missing site-months (whole months), returns exactly the requested times;
  `reconstruction_cache()` lists stores, `clear_reconstruction_cache(pattern)`
  deletes them; `reconstruction_sites(set, selector)`; `require_models(set,
  n)`; `load_reconstruction_spectra`. PCA variance only from `vars_to_predict`.
  `reconstruction.variables` = wavespectra stats of the summed spectrum (both modes
  build it: bulk = one partition per goal), so `tm01` is a stat, never predicted;
  `validation.metrics_variables` may name predicted variables or these stats
  (`validate.py`), and every comparison keeps what both sides have (`tm01`: buoys
  only; the cases, the SnapWave dynamic runs and the hindcasts have none).
  Depth cap (`reconstruction.depth_cap`, needs `wl` in `mda_inputs`): the store
  keeps the plain sum, `reconstruct(depth_cap=)` / `load_reconstruction_spectra`
  return it capped (`apply_depth_cap`: Hs <= `depth_ceiling()` x `water_depth()`,
  spectra x the energy factor; generic part `bluemath_tk.waves.hywaves.depth_cap`).
- **Validation** (`utils/validation.py`): `validate` / `sweep_all` /
  `plot_sweep` (holdout); `compare(reference, estimate)` / `compare_all` ->
  `Comparison` (`metrics`, `groups`, `plot`, `plot_all`, `plot_scatter`,
  `plot_timeseries`); `load_offshore_hindcast_at_buoys`, `plot_buoy_summary`.
  Hs pairs outside 0-15 m are dropped (corrupted dynamic output).
- Site metadata: `geometry.site_meta` / `attach_site_meta` (from the table);
  buoys: `onshore_sites.list_snapwave_buoy_sites()` = ok buoys.

## Reference

**`validate.py` checks**: `{Case}/{Case}.json` (`name`; `description` at
publish; `paths` strings; `TODO`s are gaps); goals JSON `{"goals": {"1":
{"target": {"longitude", "latitude"}, "angles"}}}`; observations GeoJSON
(Points, unique `instrument_id`, `measured_depth`); `offshore_waves` list
(`name`, `data_path`, optional `hindcast` = goals-JSON hindcast it sits on);
`outputs/goal_*/<set>/validation_metrics.csv` (`goal, metamodel,
mda_candidate, group, variable, n, bias, rmse, r2`); the `metamodel`,
`simulations` (bounds, `max_steepness`, model block, `output_sites_config`,
`min_depth`), `forcing`, `satellite` blocks; the sites CSV
(present, grid matches, every set selector names ok sites); `portal/`
staleness.

**`portal/`** (`publish`): `case.json`, `goals.json`, `observations.json`,
`aoi.geojson`, `isobaths.json` (masked to the mesh), `output_sites.json` (ok
non-buoy sites), `validation_metrics.csv`, `metamodel_validation/`,
`buoy_validation/` (`*_summary_*.csv`: the prefix `onshore`/`offshore` is the
GUI's buoy type; `figures/<site>.png`), `gui_figures/` (fig01_domain,
fig02_map_comparison, fig03_metamodel, fig04_spectra_reconstruction), `book/`
(Jupyter Book, git-ignored, copied to the server by hand; links built for
`[/<HYGLOW_URL_BASE>]/case-book/<id>`), `manifest.json` (`hywaves_version`,
`bluemath_tk`, `case`, `model`, `published_at`, `skipped`, `files`,
`sources`). Missing sources skip a layer. Snapshots: `portal_<tag>/`, never
re-published. Case manifest `.hywaves/manifest.json`: `hywaves_version`,
`source`, `created`/`updated`, `bluemath_tk {version, commit, branch, dirty}`
(start, update, publish), `notebooks` (copied at start), `engine_files`
(hashes of `utils/`). Cases carry
no `tests/`.

**Method facts** (decisions and settings; the numbers behind them live where
they are recomputed: notebooks, `numerical_tests/`, the dashboard)
- *Boundary forcing*: the sum equals an all-goals run only if each case puts
  a fraction w of its goal's energy along the boundary and the fractions add
  to 1. SnapWave: two nearest goals, inverse distance in energy (ends decay);
  but **every boundary spectrum is centred on one direction**, the Hs-weighted
  mean of all boundary points (`build_boundary_support_points_spectra`), with
  one domain Tp (the boundary mean) and the solver sector on that mean: right
  for a goal case (one direction everywhere), wrong for dynamic runs whose goals
  look different ways (Samoa's closed ring: the exact sum of goal runs is well above
  it even at 50 m, and equal to it with per-point directions; `Samoa/numerical_tests/`). Its
  spreading `cos^ms`, `ms = 1/σ² − 1`, inverts above σ = 57.3°. SWAN: spectra interpolated linearly
  along the boundary (Hs ~ √w; one peak at the mean direction; ends take the
  nearest goal in full via `boundary_par_table`). Checked on Netherlands case
  01/0000: the boundary follows the energy-linear profile.
- *SWAN*: ADCIRC `.grd` with counter-clockwise triangles (else it stops);
  ~32 min/case serial (SnapWave ~2 s); on test case 01/0000 it agrees with SnapWave.
- *SWAN physics* (templates; chosen on 39 hours of 2013 against 27 buoys,
  `Netherlands/numerical_tests/` README + 01/02): dynamic `GEN3 KOMEN` (beat
  Westhuysen and ST6); no-wind cases `OFF QUAD` + `OFF WCAP` (else steep seas come
  out well below SnapWave); friction `JONSWAP CONSTANT 0.019` (better storms,
  confirmed on 120 other hours, slight overestimate in calm seas). No gain from
  `BREAKING BKD`, 100 iterations, no triads, a water-level field, `AGROW`, 5°
  directions or wind × 1.1. Dynamic runs add 2D spectra at the buoys (`output_spectra.spc`).
- *SnapWave setup* (same tests, `Netherlands/numerical_tests/`): the enclosure holds
  only the open boundary (`open_boundary_nodes`: outline through the goals, out to
  where the laterals meet the coast; 767 nodes = SWAN's `.grd`); the old AoI
  rectangle forced all 6,329 outline nodes, coast included, and was too high
  nearshore. `upwindref = 1`, `niter = 200` (sweeps): every goal case converges and
  gentle seas agree with SWAN. `sector = 360` in both templates (2026-11-15): a 180°
  sector centred on the boundary direction drops the energy refraction turns past 90°
  around islands (Samoa goal cases, `Samoa/numerical_tests/`). Boundary `spr` at most
  `MAX_SPREAD` 55° (`bluemath_tk.wrappers.snapwave`; `boundary_time_series` clips it,
  `generate_cases` refuses wider `bounds.spr`): SnapWave's spreading inverts above 57.3°.
  Only the SnapWave boundary is clipped: the dynamic run is forced narrower than the
  hindcast where the bulk spread is broad (a mix of seas), while the reconstruction keeps
  the real values (bulk mode extrapolates above the bounds, the partitions sit inside
  them); there the buoys are the referee, not the dynamic run (notebooks 08-09 say so).
  Open boundary on a mesh built elsewhere or with goals on the lateral sides: `simulations.snapwave.coast_depth`
  (m; `open_boundary_nodes(coast_depth=)` follows the outline from the goals to the coast instead of
  stopping at corners; NorthCarolina). The corner rule stays the default: in the Netherlands it stops
  the laterals before shallow sea that must not be forced.
  Partitions are scaled to the bulk energy in 07a (`offshore.scale_partitions_to_bulk`),
  before the calibration. Postprocessing drops non-converged runs (last error >
  `MAX_ERROR` 0.01 in `wrapper_out.log`): metamodel cases are left out, dynamic
  hours are NaN (wind mode: `hywaves/WIND.md`).
- *Depth cap* (2026-11-11): hard cap Hs <= ceiling x h, ceiling = highest Hs / h of
  any training case of any goal (h = mesh depth + case wl, h > 0.5 m), h of the hour =
  mesh depth + the mean goal level (the dynamic runs' single level). Applied everywhere:
  sheltered sites (ceiling < 0.1) are not hurt. Hard, not smooth: the smooth form
  (p chosen on the scored hours) also lowered 20 m storm hours where nothing breaks.
  The dynamic runs stay under the ceiling; dashboard Breaking page.
- *Wind*: the reconstruction has no wind (the wind-correction code was removed
  on 2026-11-04); wind stays as dynamic-run forcing (`forcing.wind`, runs
  `_wind-on`/`_wind-off`, and `_wind-only`: no boundary waves,
  `simulation.wind_only_parameters` / `wind_only_cases_dir`). Wind tests are
  closed (2026-11-10); the way chosen if it is added: a wind-only term damped by
  the share of the sea already there. What was tried, the numbers and the
  options: `hywaves/WIND.md` (the one doc that keeps numbers).
- *Satellite calibration* (07b, `bluemath_tk.waves.{altimetry,calibration}`):
  IMOS `/lustre/geocean/DATA/IMOS/processed/imos_altimeter_1985_2020.zarr`
  (also over HTTP); passes within `radius_km` of the hindcast node, depth >
  `min_depth`, flags 1-2, one median per pass. Step 1, per goal: Mínguez 2011
  directional energy fit, no intercept, sea/swell × 16 sectors (sectors with too
  few passes stay at 1); it removes the mean bias but overshoots moderate seas
  (open). Step 2, one `QuantileMapping` over the passes of all goals (per goal
  fails: too few storm passes), constant offset beyond p99.9; it is fitted on every
  pass, so judge it on the deep buoys (07b section 5, dashboard Calibration page),
  never on the passes. On those buoys WHACS_cal fixes the storm tail; ERA5 barely
  needs calibrating. ERA5 (`EDH_API_KEY`) is closer to the satellite than WHACS at
  most goals, on few passes. Binning errors by observed Hs overstates storm bias (regression to
  the mean): use quantiles.
- *ERA5 partition periods* (07a, `offshore.peak_period_samples` /
  `partition_peak_periods`): ERA5 gives Tm-1,0 per partition; Tp / Tm-1,0 is
  the median of bulk `pp1d / ptp{i}` where one partition holds >= 90 % of the
  energy, computed per case (Netherlands: sea close to Pierson-Moskowitz, swell
  above a narrow JONSWAP: broad swell partitions). One processed ERA5 zarr, with Tp.

## Where we are (2026-11-16)

hywaves **0.17.0** (depth cap in the reconstruction; no wind) on `main` (HyGloW);
bluemath_tk on `feature/hywaves-depth-cap` (into `develop`).

**Netherlands** (`Netherlands.json`): SnapWave, 05-12 re-run with the
`numerical_tests/` setup (open-boundary enclosure, `upwindref`, `niter` 200,
Hs up to 10 m). Mesh `inputs/GRID/Netherlands_mesh.nc` / `.grd`; CODEC water
level ends in 2018. Buoys: `utils/buoy_archive.py` (stations < 1 km merged,
hourly; Belgian ones Hs-only 2020-25). 2013 complete: the four dynamic runs (WHACS, WHACS_cal x
wind on/off) and the reconstruction cache (mda600, `wl="forcing"`, every set,
both sources, bulk and spectral). First-iteration conclusions:
`Netherlands/dashboard/README.md` (linear sum fine on average, high in the
biggest seas at 5 m; missing wind the largest error; spectral Tp/Dir/Spr poor;
calibration helps the storm tail). Notebooks 10 and 12 run on 2013 and
`portal/` is published (SnapWave).
Notebook 11 waits for buoy spectra (the processed buoys are bulk only).
`Netherlands.json` is on `model: swan` for the SWAN pass (notebook 08); switch it
back to `snapwave` to re-run 09-12 or publish the SnapWave results.

**Reconstruction internals** (bluemath_tk `waves/hywaves/store.py`,
`core/resources.py`, `waves/hywaves/resources.py`; details and open points in
`hywaves/RECONSTRUCTION.md`): bulk zarr sharded per month, and zarr 3.1.5
cannot write scattered sites into a shard, so writes read-modify-write the
month under a file lock; the setup `key` + input `fingerprint` in the attrs
rebuild the store on any change; hours with no energy are NaN. Rates are
measured on the 48-core node: re-measure there if the sum changes.

**Samoa** (`Samoa.json`, migrated 2026-11-13): SnapWave, 16 goals on a closed
ring, WHACS + WHACS_cal, CODEC + ERA5 wind, dynamic year 2018 (CODEC ends 2018;
Aunu'u at both sites; Cyclone Gita). Bathymetry `inputs/Bathymetry/Samoa_topobathy_1s.tiff`:
GEBCO 2026 + Samoa project bathymetry (Upolu/Savai'i, UTM 326 m) + NOAA Pago Pago 3"
and Tutuila 1/3" (MHW + 0.394 m -> MSL); mesh geometry of 2026-07 kept, node
depths re-interpolated. Buoys: CDIP 189 Aunu'u split by GPS into `Aunuu_1`
(to 2018-04, ~79 m) / `Aunuu_2` (2018-05 on, ~55 m), CDIP 273 `King_Poloa`
(2024-06 on, after WHACS), `bluemath_tk.waves.cdip`. The July goal zarr had
`hs` = partition sum, not WHACS bulk (fixed from the global archive
`/lustre/geocean/DATA/CSIRO/WHACS/global-smc`); WHACS partitions hold only ~91 %
of the bulk energy in Samoa (99.9 % in the Netherlands), so the spectral sum
starts ~4-5 % low in Hs. 0.9.0 runs kept as `*_v0.9.0`. 2026-11-14: 05-12 run
(mda 1500 LHS -> 1108 cases/goal, fits to mda800, dynamic WHACS_cal wind-off 2018,
reconstruction cache 2018 every set bulk + spectral, `wl="forcing"`). The linear sum
is high against the dynamic run everywhere, growing with the offshore spread: mostly
the dynamic run (one boundary direction, see *Boundary forcing*; it is also the low
one against the Aunu'u buoys), then breaking at 5-10 m, then the GP extrapolating in
bulk mode (WHACS bulk `spr` median 53°, above `bounds.spr` [1, 45]; 38 % above
57.3°, where SnapWave inverts the spreading; the partitions sit inside the bounds).
Tests and conclusions: `Samoa/numerical_tests/README.md` (scripts, frozen CSVs, the
per-point SnapWave patch; runs in `tmp/linearity_test/`). 2026-11-15: re-run with the
fixes (wind2, sector 360, `bounds.spr` [1, 55], partitions scaled, 07a/07b re-run) by
`Samoa/tmp/run_samoa_chain.sh` (05 -> 06 || 08 -> reconstruction -> 09-12; finished
2026-11-16, every case and dynamic hour converged); the previous results are kept in
`*_sector180` / `outputs_sector180/` and the `*_unscaled.zarr` offshore files. With the
fixes the linear sum matches the dynamic run from 10 to 50 m and both sit close to the
Aunu'u buoys (the old dynamic run was the low one); left: breaking at 5 m (and some at
10 m), spectral Tp/Dir/Spr. A SWAN dynamic run of Cyclone Gita
(`numerical_tests/scripts/06_swan_storm.py`, 08b section 5) sides with wind2 over wind1 in
every site group. Numbers: notebook 09/10 and `numerical_tests/README.md`.
2026-11-16: `tm01` added (config, buoy files with `tm01`); cache refilled and 09-12, 08b
re-run (`tmp/run_tm01.sh`). Buoy files from CDIP
(`bluemath_tk.waves.cdip`): bulk `processed/<id>.nc`, MEM spectra
`processed/spectra/<id>.nc` (`val.buoy_spectra_path`). In this env,
`gpd.read_file` after importing bluemath_tk needs `GDAL_DATA` set to the env's
`share/gdal`.

**NorthCarolina** (`NorthCarolina.json`, started 2026-11-15): SnapWave on the mesh delivered with the
ShoreShop work (`inputs/delivery_2026-11-15/`, git-ignored: mesh scripts, enclosure, 8 m output points,
a daily ERA5-forced SnapWave run 1985-2023 at those points = a reference to compare against; the DEM was
not delivered, vertical datum unknown). 01-04 done: AoI = mesh open boundary closed over land, raster =
GEBCO 2026 + mesh depths (3"), 19 NDBC buoys (hourly tables of
`/lustre/geocean/WORK/users/montanoj/projects/shoreshop3_2026_pipeline/01_BinWaves/inputs/buoy_data`,
`utils/shoreshop_buoys.py`; no spreading; ten are CDIP stations, spectra not downloaded), 14 goals along
the three open sides (coast-end goals dropped by mesh depth, goal 14 sector set by hand), sets `buoys`,
`transects` (8 m isobath every 250 m, the delivered output line) and `isobaths`. 07a, 07b and the CODEC part of 07c done;
bounds from them (Hs 17 m, steepness 0.07, wl -1.25 to 2.25 m); on the deep buoys `WHACS_cal` does
not clearly beat WHACS in the storm tail. `tmp/whacs_seapoints.py` builds the
`_WHACS_seapoint.zarr` 07a reads from the global archive (no notebook does it); `tmp/run_inputs.sh`
(07a, 07b, CODEC part of 07c) then `tmp/run_chain.sh` (05 -> 06 || 08 -> reconstruction 2018 -> 09, 10,
12). The chains wait on the whole SLURM queue of the user: one at a time. ERA5 waves and wind need
`EDH_API_KEY` (not in the non-interactive shell). Template notebook 04 now renumbers the goals after
dropping any. 2026-11-16: synced to the template (`utils/`, `tm01` config keys, notebooks 09, 10, 08b)
and `tmp/run_chain.sh` launched (`WHACS_cal`): 05 done (14,769 of 14,770 cases), 06 fit left running, 08 failed
(our folders write to the SATA pool of `/lustre/geocean/WORK`, full; the free space is in the SAS-HA pool:
`lfs df`, `lfs getstripe -d`, case README); resume with `tmp/run_chain_from08.sh` (case README). Checked against Dano's run
(`tmp/dano_check/`, conclusion in the case README): his input with the wind1 build reproduces his output
exactly; our setup is lower in Hs because of wind2 and full convergence, not the enclosure or the goals.
`Netherlands/utils` is behind the template (`coast_depth`, `tm01`): `update` + the `tm01` config keys
when next touched.
The template 08b was rewritten on 2026-11-16 (`viz.plot_storm_fields` replaced `plot_storm_hour`,
`event_peaks` returns one row per site and source): NorthCarolina and NewSouthWales keep their old 08b and
`utils/` until `update`; copy the template 08b with it.

**NewSouthWales** (`NewSouthWales.json`, migrated 2026-11-16): 0.9.0 results kept (`portal_v0-9-0/`,
`*_v0.9.0`). Template `utils/` and notebooks 03, 05-12; 00/01/02/04 kept as records of the inputs.
`paths.topobathy_file` is the NSW 50 m raster (`NewSouthWales_MAO.tiff`: the mesh depths come from it,
not from the GEBCO tiff the 0.9.0 config named); isobaths every 1 km, smoothed 200 m at 5-10 m (1 km
smoothing erased them on this steep coast). 26 goals kept (default corner rule); 950 samples -> 701
cases per goal, 18,226 in all (two arrays under the 20,000-record limit), MDA up to 600. WHACS extract
of 0.9.0 checked against the global archive (bulk, not the July partition-sum bug). Dynamic year 2016
(June East Coast Low). Nothing submitted (the waiter was stopped when the disk filled): `tmp/run_chain.sh`
after North Carolina (case README).

**Next (user)**, also listed in `TODO.md` (keep both in step): decide from the dashboard what to fix or move into 09-12
(spectral Tp/Dir/Spr, the 5 m storm overshoot; wind tests closed, the damped
wind-only term in `hywaves/WIND.md` if it is built); buoy spectra for notebook 11; SWAN pass (05-12 with `model: swan`,
`tp` as PCA target, then snapshot `swan`); Samoa: check 08b-12, then
breaking near the coast (the depth cap barely acts); send the upstream issue (wind2 is
the default since 2026-11-16); then the Netherlands with the same setup (05-06 again:
sector 360 and the spread bounds, 08 with wind2; `tm01` needs a mean period in
`utils/buoy_archive.py`: check which one Waterinfo/CMEMS give); NewSouthWales: check its first 0.17 run, publish. Cluster: arrays <= 9999 tasks and 20000 job
records (bigger runs in `OFFSET` blocks).
