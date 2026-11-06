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
| `Netherlands/` | Reference case, latest code. `Samoa/`, `NewSouthWales/`: older code (0.9.0), migrate when next touched |
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
  `BOUNDSPEC ... VARIABLE PAR` on unstructured grids). SnapWave with wind
  needs `snapwave/b663c81-wind1`, the module default (cases load plain `snapwave`).
- Heavy data (`*.nc`, `*.zarr`, `*.tiff`, `*_CASES_*`, `outputs/`, `logs/`,
  `tmp/`) is git-ignored and huge: don't scan it recklessly.

```bash
python -m hywaves validate <Case>/ [--publish]
python -m hywaves publish  <Case>/                 # -> portal/, commit it (book/ is git-ignored)
python -m hywaves snapshot <Case>/ <tag> --note "…"  # portal/ -> portal_<tag>/ before a re-publish
python -m hywaves update   <Case>/ [--dry-run]     # case utils/ <- template
python -m unittest discover -s hywaves/tests -t .  # all hywaves tests (case_utils/ run inside a started case)
cd Netherlands && PYTHONPATH=. python -m unittest discover -s ../hywaves/tests/case_utils -t ../hywaves/tests/case_utils
cd ~/BlueMath_tk && python -m unittest tests.waves.test_hywaves tests.waves.test_hywaves_plots tests.waves.test_hywaves_store tests.waves.test_hywaves_resources tests.waves.test_fetch tests.core.test_resources tests.core.test_core_metrics
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
   `analysis/`, the dashboard); timings and settings are fine. `WIND.md` is the
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
- Pipeline: 01 GEBCO -> 02 mesh -> 03 observations + output sites -> 04 goals
  -> 05 training cases -> 06 fit + holdout -> 07a
  offshore (`07a_Offshore_WHACS` or `_ERA5`; 07b satellite calibration) ->
  08 dynamic runs -> 09 linear sum vs dynamic -> 10 buoys
  (`10_Validate_Buoys`) -> 11 spectra -> 12 GUI figures. Save cells of
  06/09/10/12 write the `outputs/` that `publish` bundles.
- `Netherlands/numerical_tests/` is the reference for the models: how the SWAN
  and SnapWave templates were chosen (01-02, from `results/*.csv`) and SnapWave
  vs SWAN on the same cases (03-04, `model_comparison.py`; not a clean
  comparison yet). Its notebooks put the case folder on `sys.path`; `runs/` is
  git-ignored. `Netherlands/analysis/` (case-only, like `numerical_tests/`): every estimate
  on the same sites and hours of 2013 (`results.py` + notebooks 01-05: error
  budget at the buoys, linear sum vs dynamic, wind effect, calibration, storms).
  `Netherlands/dashboard/` (case-only Dash app, not `GUI/`): `build_store.py` packs every 2013
  result into `store/` (git-ignored, 4.2 GB; reuses `analysis/results.py`), `app.py` reads only it;
  rebuild the store after re-running 06-08 or the 2013 reconstruction.
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
they are recomputed: notebooks, `numerical_tests/`, `analysis/`, the dashboard)
- *Boundary forcing*: the sum equals an all-goals run only if each case puts
  a fraction w of its goal's energy along the boundary and the fractions add
  to 1. SnapWave: two nearest goals, inverse distance (bimodal between
  different directions; ends decay). SWAN: spectra interpolated linearly
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
  gentle seas agree with SWAN. Postprocessing drops non-converged runs (last error >
  `MAX_ERROR` 0.01 in `wrapper_out.log`): metamodel cases are left out, dynamic
  hours are NaN (wind mode: `hywaves/WIND.md`).
- *Wind*: the reconstruction has no wind (the wind-correction code was removed
  on 2026-11-04); wind stays as dynamic-run forcing (`forcing.wind`, runs
  `_wind-on`/`_wind-off`). What was tried, the numbers and the options:
  `hywaves/WIND.md` (the one doc that keeps numbers).
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

## Where we are (2026-11-06)

hywaves **0.16.0** (no wind in the reconstruction; store names without
`wind-`) on `main` (HyGloW); bluemath_tk on `feature/more-hywaves` (into `develop`).

**Netherlands** (`Netherlands.json`): SnapWave, 05-12 re-run with the
`numerical_tests/` setup (open-boundary enclosure, `upwindref`, `niter` 200,
Hs up to 10 m). Mesh `inputs/GRID/Netherlands_mesh.nc` / `.grd`; CODEC water
level ends in 2018. Buoys: `utils/buoy_archive.py` (stations < 1 km merged,
hourly; Belgian ones Hs-only 2020-25). `portal_snapwave/` holds the results
before this re-run. 2013 complete: the four dynamic runs (WHACS, WHACS_cal x
wind on/off) and the reconstruction cache (mda600, `wl="forcing"`, every set,
both sources, bulk and spectral). First-iteration conclusions:
`Netherlands/analysis/README.md` (linear sum fine on average, high in the
biggest seas at 5 m; missing wind the largest error; spectral Tp/Dir/Spr poor;
calibration helps the storm tail). Notebooks 10 and 12 run on 2013 and
`portal/` is published (SnapWave); `portal_snapwave/` is the older snapshot.
Notebook 11 waits for buoy spectra (the processed buoys are bulk only).
`Netherlands.json` is switched to `model: swan` for the SWAN pass (uncommitted
work in notebook 08).

**Reconstruction internals** (bluemath_tk `waves/hywaves/store.py`,
`core/resources.py`, `waves/hywaves/resources.py`; details and open points in
`hywaves/RECONSTRUCTION.md`): bulk zarr sharded per month, and zarr 3.1.5
cannot write scattered sites into a shard, so writes read-modify-write the
month under a file lock; the setup `key` + input `fingerprint` in the attrs
rebuild the store on any change; hours with no energy are NaN. Rates are
measured on the 48-core node: re-measure there if the sum changes.

**Next (user)**: decide from `analysis/` what to fix or move into 09-12
(spectral Tp/Dir/Spr, the 5 m storm overshoot, wind: `hywaves/WIND.md`
options); deck from the new portal; buoy spectra for notebook 11; SWAN pass (05-12 with `model: swan`,
`tp` as PCA target, then snapshot `swan`); migrate Samoa and NewSouthWales
(`update` + config + re-run). Cluster: arrays <= 9999 tasks and 20000 job
records (bigger runs in `OFFSET` blocks).
