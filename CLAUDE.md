# CLAUDE.md — HyGloW

Guide for maintainers and coding agents: the rules, how a case works, and the decisions
that are settled. History and numbers live in the docs below, not here.

## What it is

**HyGloW** (project) / **HyWaves** (method): offshore hindcast -> nearshore waves without
running a spectral model every hour. Per offshore *goal*, a few hundred stationary SnapWave
or SWAN runs train an **MDA -> PCA -> GP** metamodel; the nearshore series (bulk or spectra)
is the **linear sum of the goal spectra**. Dynamic runs (every hour, all goals, optional wind
and water level) validate the sum. No wind in the reconstruction.

| Repo / path | Role |
| --- | --- |
| `hywaves/` | One module per command (`start`, `validate` = the case rules, `publish`, `snapshot`, `update`), `common.py`, `cli.py`, the case `template/` (canonical) |
| `Netherlands/`, `Samoa/`, `NewSouthWales/` | The paper cases (Netherlands = reference, latest code). `NorthCarolina/`: parked |
| `GUI/` | Dash results explorer, reads only `{Case}/portal/` |
| `deck/` | Slide deck (GitHub Pages) built from the `portal/` bundles |
| `~/BlueMath_tk` | **bluemath_tk**: all generic code (`waves/hywaves/`, `waves/{altimetry,calibration}`, wrappers `wrappers/{snapwave,swan}/` = `_wrapper/_utils/_plots.py`) |
| `~/BlueMath` | Examples, branch `hywaves/netherlands`: `climate_services/hindcast/HyWaves_Netherlands`, `toolkit/data/` (IMOS, SWOT, WHACS notebooks on the public Zarr stores) |

## Docs (read the one you need)

| File | Holds |
| --- | --- |
| `README.md` | Users: install, start a case, offshore data (WHACS archive) |
| `hywaves/README.md` | The method (physics in one page), resources and timings table, commands |
| `hywaves/RECONSTRUCTION.md` | How the reconstruction uses the machine, the store, buoy timings, open points |
| `hywaves/WIND.md` | Wind: what was tried, the numbers, the options (the one doc with numbers) |
| `hywaves/SWOT_VALIDATION_PLAN.md` | Plan (not built): SWOT KaRIn validation of the reconstruction; removed when built |
| `<Case>/README.md` | One line per notebook, the JSON keys, case-only notes and how to resume runs |
| `Netherlands/TODO.md` | The one open method item: a consistent breaking correction |
| `*/numerical_tests/README.md` | Model-setup tests (Netherlands: templates, SnapWave vs SWAN; Samoa: linear sum vs dynamic) |
| `Netherlands/dashboard/README.md` | The 2013 dashboard and the first-iteration conclusions |

## Environment and commands

- Python: `~/miniforge3/envs/bluemath-312/bin/python` (bluemath_tk editable from
  `~/BlueMath_tk`). No GPU on the login node. After importing bluemath_tk, `gpd.read_file`
  needs `GDAL_DATA=$HOME/miniforge3/envs/bluemath-312/share/gdal`.
- Models run as SLURM arrays (`wrapper.run_cases_slurm(**sim.slurm_kwargs())`); arrays <= 9999
  tasks and 20000 job records (bigger runs in `OFFSET` blocks). SWAN: one serial `swan.exe` per
  task (module `swan/4151`; MPI misplaces `BOUNDSPEC ... VARIABLE PAR` on unstructured grids).
  SnapWave: module `snapwave/b663c81-wind2` (boundary spectra centred on each point's
  direction; the default since 2026-11-16), loaded by name in the case JSON. Draft upstream
  issue: `/lustre/geocean/WORK/users/javiert/software/snapwave/upstream_issue.md`.
- `/lustre/geocean/WORK` has two pools; if writes fail with a full disk, check `lfs df` and
  `lfs getstripe -d` (NorthCarolina README).
- Heavy data (`*.nc`, `*.zarr`, `*.tiff`, `*_CASES_*`, `outputs/`, `logs/`, `tmp/`) is
  git-ignored and huge: don't scan it recklessly.

```bash
python -m hywaves validate <Case>/ [--publish]
python -m hywaves publish  <Case>/                   # -> portal/, commit it (book/ is git-ignored)
python -m hywaves snapshot <Case>/ <tag> --note "…"  # portal/ -> portal_<tag>/ before a re-publish
python -m hywaves update   <Case>/ [--dry-run]       # case utils/ <- template
python -m unittest discover -s hywaves/tests -t .    # hywaves tests
cd Netherlands && PYTHONPATH=. python -m unittest discover -s ../hywaves/tests/case_utils -t ../hywaves/tests/case_utils
cd ~/BlueMath_tk && python -m unittest tests.waves.test_hywaves tests.waves.test_hywaves_plots tests.waves.test_hywaves_store tests.waves.test_hywaves_depth_cap tests.waves.test_hywaves_resources tests.waves.test_fetch tests.waves.test_altimetry tests.waves.test_calibration tests.core.test_resources tests.core.test_core_metrics
cd GUI && python app.py                              # http://localhost:8050
```

## Rules

1. **Latest working code only, zero legacy.** No backward compatibility, old keys, fallbacks or
   dead code in hywaves or `template/utils/`; migrate the template and the case being worked
   on. Old cases stay reproducible through their vendored `utils/` and the bluemath_tk commit
   in `.hywaves/manifest.json`. Small, simple functions; generic code goes to bluemath_tk,
   HyGloW keeps the JSON/case layer.
2. **`hywaves/template/` is canonical.** General fixes go there (tests in
   `hywaves/tests/case_utils/`), then `python -m hywaves update <Case>/`. Case-only code stays
   in the case (e.g. `Netherlands/utils/buoy_archive.py`). Say which copy you edit. Machine
   settings are passed as resources, never hard-coded.
3. **Case rules live in `hywaves/validate.py`.** Changing what a case provides: `validate.py`,
   `publish.py`, `hywaves/tests/` and the Reference below together.
4. **The GUI reads only `portal/`**: extend `publish.py`, never make the GUI read `outputs/`.
5. **Notebooks are thin** (logic in `utils/` or bluemath_tk). Edit sources only (a JSON
   round-trip that keeps outputs byte-for-byte); don't strip outputs unless asked. The user
   launches long runs (models, fits, reconstructions); prepare them.
6. **Refactors are checked numerically**: snapshot outputs on Netherlands data before, compare
   after (script in the scratchpad).
7. **Docs stay minimal**: only the files in the Docs table, notebook markdown and short
   GUI/deck READMEs. Method details go in bluemath_tk docstrings, maintainer facts here.
   **No metrics in docs** (they go stale): the conclusion in words and where to see it;
   timings and settings are fine. Exceptions: `WIND.md`, and a plan while it is a plan.
8. **Style**: Python 3.12, type hints, `from __future__ import annotations`, NumPy
   docstrings, `pathlib`, loggers `hywaves.*`; sparse comments.
9. **Honest numbers**: in answers, quote what was computed, with period, MDA size and set;
   flag doubts about physics or data.
10. **Data access stays general**: satellite stores are read from the public
    `https://geoocean.unican.es/zarr/...` URLs (the cluster address only as a local option).
    WHACS is read from a local copy of CSIRO's archive (users download it or ask GeoOcean).

**Git**: PRs to `main` of `GeoOcean/HyGloW` from branches; messages
`[JTH] short lowercase description`; commit only when asked; after publishing, commit `portal/`.

## How a case works

- Notebooks run from the case folder; `utils.config` loads `<Case>/<Case>.json` on import
  (`PATHS`, `SIMULATIONS`, `METAMODEL`, `FORCING`, `SATELLITE`, `resolve()`, `load_goals()`,
  `open_offshore_waves()`); restart the kernel after editing the JSON. No hard-coded paths.
- Pipeline: 01 topobathy (GEBCO, optionally merged with local sources) -> 02 mesh -> 03
  observations + output sites -> 04 goals -> 05 training cases -> 06 fit + holdout -> 07a
  offshore (`07a_Offshore_WHACS`: `utils/whacs.py` reads the goal nodes from the archive
  `/lustre/geocean/DATA/CSIRO/WHACS/global-smc`, 1979 to 2026-01, ~1 min/year with 16 workers;
  or ERA5) -> 07b satellite calibration -> 07c CODEC water level (ends 2018) + ERA5 wind ->
  08 dynamic runs (08b storms and maps) -> 09 linear sum vs dynamic -> 10 buoys -> 11 spectra
  -> 12 GUI figures. Save cells of 06/09/10/12 write the `outputs/` that `publish` bundles.
  Long chains are case scripts in `<Case>/tmp/run_*.sh` (the case README says which).
- Conventions: directions nautical, coming-from, clockwise; goal `angles` = `[left, right]`
  (left may be negative across North); goal ids are string ints. **Metrics reference first**
  (`bluemath_tk.core.metrics`): bias = estimate − reference, R² normalised by the reference,
  reference on the x-axis.
- Model: `simulations.model` (`snapwave`|`swan`) + `simulations.<model>` (`grid_file`,
  template and cases folders, `slurm`). Each model has its own cases folders; `split.json`
  records the model and artefacts refuse to mix. SWAN: predict `tp` (PCA target), not `"raw"`.
- **Output sites** (03, `geometry.output_sites`): isobaths traced on the topobathy blanked
  outside the mesh, mesh nodes in polygons, buoys; each gets `mesh_depth` and `status` (`ok`
  only if inside the mesh and `mesh_depth + bounds.wl[0] >= min_depth`). Saved once in
  `outputs/output_sites_{model}.csv` (`# grid:` header); everything reads its `ok` sites.
  `hs = 0` is a valid zero contribution. `site_filter` removes failing sites per goal
  (`split.json` `removed_sites`).
- **Metamodel** (`utils/metamodel.py`): inputs `hs, tp, dir, spr, wl` (LHS, steep seas dropped
  by `max_steepness`); predicts absolute nearshore fields. **Sets** (`metamodel.sets`:
  `buoys`, `isobaths`, ...) = PCA + GP per site group on the same cases; every call names its
  set. **Resources: the user decides**: `fit_resources()` / `reconstruction_resources(...)`
  print the machine and a proposal (<= 75 %, never SLURM) and return an editable object passed
  as `resources=` (required) to `fit_all` / `reconstruct` / `goal_contributions` /
  `sum_contributions`; notebooks have one resources cell before the first heavy call.
- **Reconstruction**: `reconstruct(time, n, metamodel=, spectral=, wl=0|"forcing",
  depth_cap=, ...)` reads a store per setup (`tmp/reconstruct/{offshore}/{set}_mda{n}_{model}_
  {bulk|spectral}_wl-{wl}[_spectra].zarr`, chunk 1 site x 1 month, rebuilt when the setup or
  an input changes), computes only missing site-months and returns the requested times.
  `reconstruction.variables` are wavespectra stats of the summed spectrum (bulk = one
  partition per goal), so `tm01` is a stat, never predicted; comparisons keep what both sides
  have. Internals: `hywaves/RECONSTRUCTION.md`.
- **Validation** (`utils/validation.py`): holdout `validate` / `sweep_all`; `compare(reference,
  estimate)` -> `Comparison`; Hs pairs outside 0-15 m are dropped. Site metadata:
  `geometry.site_meta`; ok buoys: `onshore_sites.list_snapwave_buoy_sites()`.

## Settled decisions (why: `numerical_tests/`, notebooks, the dashboard)

- *Boundary forcing*: the sum equals an all-goals run only if each goal case puts a fraction
  w of its energy along the boundary and the fractions add to 1. SnapWave: two nearest goals,
  inverse distance in energy; wind2 centres each boundary spectrum on its own direction (the
  old build used one direction for the whole boundary: wrong for dynamic runs on a closed
  ring, `Samoa/numerical_tests/`). SWAN: spectra interpolated linearly along the boundary,
  ends take the nearest goal (`boundary_par_table`).
- *SnapWave setup*: the enclosure holds only the open boundary (`open_boundary_nodes`;
  `simulations.snapwave.coast_depth` follows the outline to the coast for meshes built
  elsewhere); `upwindref = 1`, `niter = 200`; `sector = 360`; boundary `spr` <= `MAX_SPREAD`
  55° (SnapWave's `cos^ms` spreading inverts above 57.3°; only the boundary is clipped, so
  where the hindcast spread is broad the buoys are the referee, not the dynamic run).
  Non-converged runs (last error > 0.01) are dropped (cases) or NaN (dynamic hours).
- *SWAN*: ADCIRC `.grd` with counter-clockwise triangles; ~32 min/case serial (SnapWave
  ~2 s). Physics: dynamic `GEN3 KOMEN`; no-wind cases `OFF QUAD` + `OFF WCAP`; friction
  `JONSWAP CONSTANT 0.019`. Not for the paper (SnapWave is).
- *Partitions*: 07a writes them as the hindcast gives them; 07b closes them on the bulk
  (`satellite.close_partitions`, default true; `calibration.closed_hindcast`), so only `*_cal`
  is closed. WHACS writes the wind sea + 3 swells (WW3 `NOSWLL` = 3): the energy of further
  swells is in `hs` only (07b section 0: large in Samoa, negligible in the Netherlands).
- *Satellite calibration* (07b, IMOS passes, one median per pass): per goal, step 1 the
  Mínguez 2011 directional energy fit (sea/swell x 16 sectors), step 2 a `QuantileMapping`
  with a constant offset beyond the goal's tail quantile (`tail_quantile_for`, 5 passes above
  it). Judge it on the deep buoys (07b section 5), never on the passes; per goal is not judged
  yet (independent check with SWOT: plan step A). Use quantiles, not bins of observed Hs.
- *ERA5 partition periods*: Tp / Tm-1,0 per case from hours with one dominant partition.
- *Depth cap*: Hs <= ceiling x depth (ceiling = highest Hs/h of any training case, depth =
  mesh depth + mean goal level); the store keeps the plain sum. Open: `Netherlands/TODO.md`.
- *Wind*: not in the reconstruction; dynamic-run forcing only (`_wind-on`/`_wind-off`/
  `_wind-only`). If added: a wind-only term damped by the sea already there (`hywaves/WIND.md`).
- *SWOT KaRIn*: some quality-0 passes read 4-8 m in 2 m seas; screen each pass against the
  SWOT nadir of the same pass (IMOS satellite `SWOT`) before using it (plan, BlueMath notebooks).

## Reference

**`validate.py` checks**: `{Case}/{Case}.json` (`name`; `description` at publish; `paths`
strings; `TODO`s are gaps); goals JSON `{"goals": {"1": {"target": {"longitude",
"latitude"}, "angles"}}}`; observations GeoJSON (Points, unique `instrument_id`,
`measured_depth`); `offshore_waves` list (`name`, `data_path`, optional `hindcast` = goals-JSON
hindcast key, optional `archive` = local archive 07a reads); `outputs/goal_*/<set>/
validation_metrics.csv` (`goal, metamodel, mda_candidate, group, variable, n, bias, rmse,
r2`); the `metamodel`, `simulations` (bounds, `max_steepness`, model block,
`output_sites_config`, `min_depth`), `forcing`, `satellite` blocks; the sites CSV (present,
grid matches, every set selector names ok sites); `portal/` staleness.

**`portal/`** (`publish`): `case.json`, `goals.json`, `observations.json`, `aoi.geojson`,
`isobaths.json`, `output_sites.json` (ok non-buoy sites), `validation_metrics.csv`,
`metamodel_validation/`, `buoy_validation/` (`*_summary_*.csv`, prefix `onshore`/`offshore` =
the GUI's buoy type; `figures/<site>.png`), `gui_figures/` (fig01-fig04), `book/` (Jupyter
Book, git-ignored, copied to the server by hand), `manifest.json` (`hywaves_version`,
`bluemath_tk`, `case`, `model`, `published_at`, `skipped`, `files`, `sources`). Missing
sources skip a layer. Snapshots `portal_<tag>/` are never re-published. Case manifest
`.hywaves/manifest.json`: `hywaves_version`, `source`, `created`/`updated`, `bluemath_tk
{version, commit, branch, dirty}`, `notebooks`, `engine_files` (hashes of `utils/`). Cases
carry no `tests/`.

## Where we are (2026-11-23)

hywaves **0.17.0** on `main`; bluemath_tk on `feature/hywaves-depth-cap` (into `develop`).

- All three paper cases have WHACS 1979 to 2026-01 from the archive and the per-goal
  `WHACS_cal` (07a/07b re-run 2026-11-23).
- **Netherlands**: ready to re-run with the current setup: `tmp/run_chain.sh` (05 -> 06 || 08
  -> 2013 reconstruction -> 08b-12), then rebuild the dashboard store, snapshot + publish.
  Buoys have no `tm01`.
- **Samoa**: re-run with the current setup finished 2026-11-16; check 08b-12, then publish.
- **NewSouthWales**: ready, `tmp/run_chain.sh` (case README); dynamic year 2016.
- **NorthCarolina**: parked (holdout sweep failed, 08 not run; case README).
- Chains wait on the whole SLURM queue of the user: one case at a time.
