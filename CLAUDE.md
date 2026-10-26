# CLAUDE.md — HyGloW

Guide for maintainers and coding agents. User docs: `README.md`,
`hywaves/README.md` (the method, resources and timings), the case README. Case rules:
`hywaves/validate.py`.

## What it is

**HyGloW** (project) / **HyWaves** (method): offshore hindcast -> nearshore
waves without a spectral model every hour. Per offshore *goal*, a few hundred
stationary SnapWave or SWAN runs train an **MDA -> PCA -> GP** metamodel; the
nearshore series (bulk or spectra) is the **linear sum of the goal spectra**.

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
  needs `snapwave/b663c81-wind1`.
- Heavy data (`*.nc`, `*.zarr`, `*.tiff`, `*_CASES_*`, `outputs/`, `logs/`,
  `tmp/`) is git-ignored and huge: don't scan it recklessly.

```bash
python -m hywaves validate <Case>/ [--publish]
python -m hywaves publish  <Case>/                 # -> portal/, commit it (book/ is git-ignored)
python -m hywaves snapshot <Case>/ <tag> --note "…"  # portal/ -> portal_<tag>/ before a re-publish
python -m hywaves update   <Case>/ [--dry-run]     # case utils/ <- template
python -m unittest discover -s hywaves/tests -t .  # all hywaves tests (case_utils/ run inside a started case)
cd Netherlands && PYTHONPATH=. python -m unittest discover -s ../hywaves/tests/case_utils -t ../hywaves/tests/case_utils
cd ~/BlueMath_tk && python -m unittest tests.waves.test_hywaves tests.waves.test_hywaves_plots tests.waves.test_hywaves_store tests.waves.test_hywaves_resources tests.core.test_resources tests.core.test_core_metrics
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
   short GUI/deck READMEs. Method details go in bluemath_tk
   docstrings, maintainer facts here. No contract or method-note files.
8. **Style**: Python 3.12, type hints, `from __future__ import annotations`,
   NumPy docstrings, `pathlib`, loggers `hywaves.*`; sparse comments.
9. **Honest numbers**: quote what was computed, with period, MDA size and
   set; flag doubts about physics or data.

**Git**: PRs to `main` of `GeoOcean/HyGloW` from branches; 
messages `[JTH] short lowercase description`; commit only when asked;
after publishing, commit `portal/`.

## How a case works

- Notebooks run from the case folder; `utils.config` loads `<Case>/<Case>.json`
  on import (`PATHS`, `SIMULATIONS`, `METAMODEL`, `FORCING`, `SATELLITE`,
  `resolve()`, `load_goals()`, `open_offshore_waves()`); restart the kernel
  after editing the JSON. No hard-coded paths in notebooks.
- Pipeline: 01 GEBCO -> 02 mesh -> 03 observations + output sites -> 04 goals
  -> 05 training cases (05b wind correction) -> 06 fit + holdout -> 07a
  offshore (`07a_Offshore_WHACS` or `_ERA5`; 07b satellite calibration) ->
  08 dynamic runs -> 09 linear sum vs dynamic -> 10 buoys
  (`10_Validate_Buoys`) -> 11 spectra -> 12 GUI figures. Netherlands case studies in progress (storms, time lag) live untracked in `Netherlands/tmp/storm_validation/`.
  Save cells of 06/09/10/12 write the `outputs/` that `publish` bundles.
- Conventions: directions nautical, coming-from, clockwise; goal `angles` =
  `[left, right]` (left may be negative across North); goal ids are string
  ints. **Metrics reference first** (`bluemath_tk.core.metrics`): bias =
  estimate − reference, R² normalised by the reference, reference on the x-axis.
- Model: `simulations.model` (`snapwave`|`swan`) + `simulations.<model>`
  (`grid_file`, template and cases folders, `wind_cases`, `slurm`). Each model
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
  wind_correction=, save_spectra=, ...)`: reads `reconstruction_store(...)`
  (one zarr per setup in `tmp/reconstruct/{offshore}/{set}_mda{n}_{model}_
  {bulk|spectral}_wl-{wl}_wind-{on|off}[_spectra].zarr`, chunk = 1 site x
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
`min_depth`), `forcing`, `satellite`, `wind_correction` blocks; the sites CSV
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
(start, update, publish), `engine_files` (hashes of `utils/`). Cases carry
no `tests/`.

**Method facts**
- *Boundary forcing*: the sum equals an all-goals run only if each case puts
  a fraction w of its goal's energy along the boundary and the fractions add
  to 1. SnapWave: two nearest goals, inverse distance (bimodal between
  different directions; ends decay). SWAN: spectra interpolated linearly
  along the boundary (Hs ~ √w; one peak at the mean direction; ends take the
  nearest goal in full via `boundary_par_table`). Checked: Netherlands case
  01/0000 boundary Hs = 1.02 × the energy-linear profile.
- *SWAN*: ADCIRC `.grd` with counter-clockwise triangles (else it stops);
  ~32 min/case serial (SnapWave ~2 s); test case 01/0000 median Hs ratio to
  SnapWave 1.00. Xaver pilot (Dec 2013, 48 h, 7 buoys, no wind): Hs bias WHACS
  −0.60, SnapWave −1.42, SWAN −1.88 m; with 20 m/s wind −0.53 / −0.21 m.
- *Wind correction* (05b, `bluemath_tk.waves.hywaves.wind_correction`):
  cases stay without wind; `E_ws = Σ w·hs²` (w: steepness 0.02-0.03 ×
  spreading 20-25° ramps), K from a GBM on g·Hs_ws/U², g·F/U², g·d/U², g·d̄/U²,
  cos(angle), trained on ~100 full-domain cases with/without wind; adds
  `(K²−1)·E_ws` from the wind direction. Netherlands 05b held out: bias
  −0.52 -> −0.09, RMSE 0.85 -> 0.48 m vs SnapWave with wind. Limits: uniform
  training winds, flat beyond ±60°, SnapWave one period per node.
- *Satellite calibration* (07b, `bluemath_tk.waves.{altimetry,calibration}`):
  IMOS `/lustre/geocean/DATA/IMOS/processed/imos_altimeter_1985_2020.zarr`
  (also over HTTP); passes within `radius_km` of the hindcast node, depth >
  `min_depth`, flags 1-2, one median per pass; Mínguez 2011 directional
  energy fit, no intercept, sea/swell × 16 sectors. Netherlands WHACS (50 km):
  slope 0.81-0.92 -> 0.90-0.98, test bias −0.03..−0.12 -> −0.03..+0.08 m (the
  energy fit overshoots moderate seas: open). ERA5 (`EDH_API_KEY`; `ptp` are
  Tm-1,0) beats WHACS RMSE at 8/10 goals at 10 km, with few passes.

## Where we are (2026-10-16)

hywaves **0.15.0** on `main` (HyGloW) and `develop` (bluemath_tk).

**Netherlands** (`Netherlands.json`): SnapWave; 10 goals, 629 cases/goal;
mesh `inputs/GRID/Netherlands_mesh.nc` / `.grd` (153k nodes); 5156 ok sites
(5m 1175, 10m 1875, 20m 1927, DeenHaag 141, 38 buoys); site filter keeps 38 /
4971 / 139 in `buoys` / `isobaths` / `deenhaag`; `mda_candidates` [100, 550]
(fitted with 5000 epochs; the GP stops early, patience 30); wind correction
fitted; dynamic runs `Dynamic_Cases_WHACS_cal_2013.nc` (ERA5 wind, CODEC
water level, CODEC only to 2018). Buoys: `utils/buoy_archive.py` (stations
< 1 km merged, hourly; 27 Tp, 17 dir, 12 spr; Belgian ones Hs-only 2020-25).
Earlier results (2012-2013, MDA 550, `buoys`, median over 27 buoys, before
the 0.14 sites): bulk metamodel Hs bias −0.10 / RMSE 0.31 / R² 0.78, with
wind correction +0.08 / 0.29 / 0.83; WHACS closest goal +0.19 / 0.41; dynamic
SnapWave +0.19 / 0.42; Tp R² negative everywhere (`tp` raw).

**Reconstruction cache and resources** (user docs and timings:
`hywaves/README.md`). bluemath_tk `waves/hywaves/store.py`: one zarr per
setup on `(month, step, sites)`, chunk = 1 site x 1 month, bulk sharded per
month (zarr 3.1.5 cannot write scattered sites into a shard: writes
read-modify-write the month under a file lock), spectra one file per
site-month; setup `key` + input `fingerprint` in the attrs, any change
rebuilds; hours with no energy are NaN. `core/resources.py` (machine, 75%
cap, spawn pool) + `waves/hywaves/resources.py` (plans, memory model, rates
measured on the 48-core node; re-measure there if the sum changes). The sum
runs in site x hour blocks (`reconstruction.batch_shape`); fits run as
single-thread processes; the GP step runs per (goal, partition) in processes
(2 threads, 512-row kernels). Speed work in progress: `RECONSTRUCTION.md`.

**Next (user)**: re-run Netherlands 09-12, publish, snapshot; notebook 11
still has the template `TODO_Buoy`; decide on `WHACS_cal` (overshoot); SWAN
pass (05-12 with `model: swan`, `tp` as PCA target, then snapshot `swan`);
migrate Samoa and NewSouthWales (`update` + config + re-run). Cluster: arrays
<= 9999 tasks and 20000 job records (bigger runs in `OFFSET` blocks).
