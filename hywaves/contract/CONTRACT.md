# HyGloW case contract

A **case** is a self-contained folder describing one nearshore wave-downscaling
region. Cases can live in any repository and be produced by any modelling
pipeline. What makes a folder a HyGloW case — and lets the portal (GUI),
CI, and global aggregation consume it — is conformance to this contract.

The contract has **two levels**:

- **Level 1 — Portal contract (mandatory).** Model-agnostic. Covers case
  identity, goals, observations, and validation outputs. Any case that
  validates at Level 1 can be published in the portal, regardless of which
  wave model or emulator produced the numbers.
- **Level 2 — Engine contract (conditional).** Applies only to cases that use
  the vendored HyGloW engine (SnapWave or SWAN + MDA→PCA→GPR metamodel), i.e. whose
  config contains a `metamodel` section.

Run the validator (stdlib-only, copy it anywhere):

```bash
python -m hywaves validate MyCase/             # development mode
python -m hywaves validate MyCase/ --publish   # strict: ready for the portal
python -m hywaves publish MyCase/              # export the portal bundle (§6)
# or standalone, without the hywaves package:
python validate.py MyCase/ --publish
```

---

## Level 1 — Portal contract

### 1. Case config JSON

`{Case}/{Case}.json` (file name matches the folder name; case-insensitive
fallback accepted).

| Key                      | Type | Required | Notes                                   |
| ------------------------ | ---- | -------- | --------------------------------------- |
| `name`                   | str  | yes      | Display name                            |
| `description`            | str  | publish  | One-line summary                        |
| `paths`                  | dict | no       | Declared input paths (existence warned) |
| `simulations.goals_file` | str  | yes      | Relative path to the goals JSON         |
| `onshore_waves`          | str  | publish  | Relative path to observations GeoJSON   |
| `offshore_waves`         | list | no       | Each entry: `name` (+ `data_path`)      |

Values containing `TODO` are allowed in development and rejected at publish.

### 2. Goals JSON (`simulations.goals_file`)

```json
{ "goals": { "1": { "target": {"longitude": ..., "latitude": ...},
                     "angles": [left, right] } } }
```

- `goals` — non-empty dict keyed by goal id (stringified integers).
- Each goal: `target.longitude` / `target.latitude` numeric,
  `angles` a 2-element numeric list (nautical clockwise).
- Optional `hindcast` block per goal (offshore source snap points).

### 3. Observations GeoJSON (`onshore_waves`)

A GeoJSON `FeatureCollection` of `Point` features. Each feature's
`properties` must contain `instrument_id` (unique) and should contain
`name`, `institution`, `recording`, `type`.

### 4. Validation outputs

`outputs/goal_*/<set>/validation_metrics.csv` — one per goal and metamodel
set, produced by the case's holdout validation. Required columns (extra
columns are fine):

```text
goal, metamodel, mda_candidate, group, variable, n, bias, rmse, r2
```

- `goal` — goal id matching the goals JSON.
- `metamodel` — metamodel set (a key of `metamodel.sets`, e.g. `buoys`,
  `isobaths`; any single label for an emulator without sets).
- `mda_candidate` — training-set size (or emulator size parameter).
- `group` — site group label (e.g. `10m`, `Apia`, `buoy`).
- `variable` — one of the case's validated variables (e.g. `hs`, `tp`).
- `n`, `bias`, `rmse`, `r2` — sample count and holdout metrics.

Metrics follow `bluemath_tk.core.metrics`: computed as
`metric(reference, estimate)`, with `bias = mean(estimate − reference)`
(positive = overestimation) and `r2` normalised by the reference variance.
The reference is what is trusted (the model runs in the holdout, the dynamic runs
for linear summation, the buoy for buoy validation). Cases may add optional
columns such as `rrmse` (RMSE / mean of the reference), `si` (Mentaschi et
al. 2013 scatter index) and `r` (Pearson); they are `NaN` for directional
variables.

At publish, at least one metrics file must exist.

### 5. Further validation outputs (optional)

Beyond the per-goal holdout metrics in §4, the reference GUI reads two more
optional, similarly-shaped outputs — a case without them still validates and
publishes, that tab in the portal just shows a "run this notebook" message:

- `outputs/metamodel_validation/*.csv` — domain-wide linear-summation-vs-dynamic
  comparison (no `goal` column; one reconstructed period and metamodel set per
  file, `linear_summation_<set>_<period>.csv`). Same required columns as §4
  minus `goal`, plus `period`.
- `outputs/buoy_validation/*_summary_*.csv` — buoy-vs-model comparison, one
  row per `(buoy, model, variable)`; same columns as §4 plus `model` and
  `period`, with `group` naming the buoy. A matching
  `outputs/buoy_validation/figures/<site>.png` (one combined scatter +
  time-series image per buoy) is shown when a viewer clicks that buoy's
  marker on the map.

`outputs/gui_figures/fig*.png` — case showcase figures, from the case's own
`12_Final_Plots.ipynb`. Rendered by the reference GUI's "Final Plots" tab as
clickable thumbnails (full-size on click) — degrades per-figure for a case
that hasn't run that notebook, or ran an older version producing fewer
figures. The standard shape (not required, but what the reference GUI's
captions expect): `fig01_domain.png`, `fig02_map_comparison.png`,
`fig03_metamodel.png`, `fig04_spectra_reconstruction.png`. Also valid to
embed in a case's own Jupyter Book report or other consumers — the notebook
displays each figure inline (not just saving it) for exactly that reason.

### 6. Portal bundle (`{Case}/portal/`)

The portal never reads a case's working files directly. Instead,

```bash
python -m hywaves publish MyCase/
```

runs the strict (`--publish`) validation above and, only if it passes,
exports everything the portal needs into `{Case}/portal/` — the one folder a
deployment ships and a case author commits. Contract files are copied under
fixed names; heavy sources are pre-computed into light files at publish
time, so neither the portal nor its server needs the case's raw data or
scientific stack.

| Bundle file                                       | From                                                                      |
| ------------------------------------------------- | ------------------------------------------------------------------------- |
| `case.json`                                       | the case config JSON (§1), as is                                           |
| `goals.json`                                      | `simulations.goals_file` (§2), as is                                       |
| `observations.json`                               | `onshore_waves` (§3), as is                                                |
| `aoi.geojson`                                     | `paths.aoi_file` shapefile → GeoJSON `FeatureCollection` (Polygon/MultiPolygon, EPSG:4326) |
| `isobaths.json`                                   | `paths.topobathy_file` raster, contoured at each `simulations.output_sites_config.isobaths[].level` (after that entry's `smooth_m` / `min_length_m`, as for the output sites) → `[{"name", "level", "segments": [[[lon, lat], …], …]}]` |
| `output_sites.json`                               | first `output_sites_*.nc` under `simulations.<model>.metamodel_cases` (or, if that path isn't reachable, a same-named folder inside the case), buoy sites dropped → `{"lon": […], "lat": […], "source": […], "label": […]}` |
| `validation_metrics.csv`                          | every `outputs/goal_*/*/validation_metrics.csv` (§4), merged — each row keeps its `goal` and `metamodel` |
| `metamodel_validation/*.csv`                      | `outputs/metamodel_validation/*.csv` (§5), as is                           |
| `buoy_validation/*_summary_*.csv`                 | `outputs/buoy_validation/*_summary_*.csv` (§5), as is                      |
| `buoy_validation/figures/*.png`                   | `outputs/buoy_validation/figures/*.png` (§5), downscaled                   |
| `gui_figures/*.png`                               | `outputs/gui_figures/*.png` (§5), downscaled                               |
| `book/`                                           | the case's report: a Jupyter Book of `README.md` + every numbered notebook (`NN*_*.ipynb`, saved outputs, not re-executed), built by `publish` in the git-ignored `{Case}/tmp/book/` (needs `jupyter-book`; `--no-book` to skip) — **not** committed |
| `manifest.json`                                   | see below                                                                  |

- Coordinates are rounded to 5 decimals (~1 m). PNGs whose longest side
  exceeds `--max-image-px` (default 2400) are downscaled; smaller ones are
  copied byte-for-byte. Re-publishing unchanged sources reproduces the same
  files (only `manifest.json`'s timestamp changes), so re-commits stay clean.
- Every optional layer degrades independently: a missing source file or
  Python package skips just that file, with the reason recorded in the
  manifest, and the portal shows that layer/tab as "not published yet".
- The bundle is built in `portal.tmp/` and swapped in whole — a failed
  publish never leaves a partial or stale `portal/`.

**`manifest.json`** — `schema` (currently `1`), `hywaves_version`, `case`,
`model` (`simulations.model`: the numerical model behind the results),
`published_at` (UTC ISO-8601), `max_image_px`, `book` (bool), `book_url`
(the route the report's absolute links were built for,
`[/<HYGLOW_URL_BASE>]/case-book/<case id>`; `--url-base` / `--case-id`), `skipped`
(`{layer: reason}`), `files` (`{path: {size, sha256}}`, excluding `book/`),
`sources` (`{case-relative path: {size, mtime_ns, sha256}}` of every input
read) and `source_patterns` (the globs that found them).

**Staleness.** `python -m hywaves validate` compares `sources` against the
case folder (size and mtime first, re-hashing only files whose mtime moved)
and re-runs `source_patterns`, warning `portal.stale` if anything changed,
appeared or disappeared since the last publish — re-run `publish`.

**Version control.** Commit `{Case}/portal/`; ignore `{Case}/portal/book/`
(a built static site, tens of MB — deploy it separately) and the transient
`portal.tmp/` / `portal.old/`.

**Snapshots.** `python -m hywaves snapshot {Case}/ <tag> --note "…"` copies
the current `portal/` (book included; its links rewritten to
`…/case-book/<case id>@<tag>`, where the GUI serves it) to `{Case}/portal_<tag>/` — `<tag>` is
lowercase letters, digits and `-` — and adds `snapshot: {tag, note, created}`
to that copy's `manifest.json`. Snapshots are frozen versions of a case (e.g.
before re-running it with a new metamodel): never re-published, not checked
for staleness, committed like `portal/` (their `book/` git-ignored). The
reference GUI lists them as versions of the case.

---

## Level 2 — Engine contract (`metamodel` section present)

| Key                                                     | Rule                                                                                                            |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `sets`                                                  | non-empty dict; each value `{"sites": selector}` — `null` (all sites), a site group / umbrella / id, or a list   |
| `mda_candidates`                                        | non-empty list of positive ints (last = biggest)                                                                |
| `mda_inputs`                                            | non-empty list of strings (`hs`, `tp`, `dir`, `spr`, `wl`)                                                      |
| `mda_directional_variables`                             | list, subset of `mda_inputs`                                                                                    |
| `vars_to_predict`                                       | non-empty dict; each value `"raw"` or a dict with `vars_to_stack` (non-empty list) and `pca_variance` in (0, 1] |
| `site_filter`                                           | dict; keys `{var}_max`, `{var}_min` or `{var}_ratio_max` (`var / {var}_forcing`)                                 |
| `pca_preprocessing`                                     | optional dict of dicts                                                                                          |
| `validation.metrics_variables`                          | non-empty list, subset of `vars_to_predict`                                                                     |
| `reconstruction.variables`                              | non-empty dict; keys subset of `vars_to_predict`; values valid wavespectra stats                                |
| `reconstruction.save_spectra`                           | bool, required                                                                                                  |
| `paths.pca_dir` / `paths.gp_dir`                        | template strings with `{goal}`, `{metamodel}` and `{mda_candidate}`                                             |
| `simulations.bounds`                                    | `[low, high]` for every non-directional `mda_inputs` entry                                                      |
| `simulations.max_steepness`                             | positive number; LHS samples with `hs / L0` above it are dropped                                                |
| `simulations.model`                                     | `"snapwave"` or `"swan"`, with a `simulations.<model>` block                                                   |
| `simulations.<model>`                                   | `grid_file`, `metamodel_templates`, `dynamic_templates`, `metamodel_cases`, `dynamic_cases` (paths; templates folders and grid should exist), optional `wind_cases` (training pairs of the wind correction) and `slurm` dict |
| `wind_correction` (optional)                            | dict; `wind` `"inferred"` or `"forcing"` (needs `forcing.wind`); `added_tp` `"wind_sea"` or `"growth"`; ranges (`speed_ratio`, `wind_wave_angle`, `u10_range`, `steepness_range`, `spr_range`) increasing pairs |
| `forcing.wind` / `forcing.water_level` (optional)       | `{name, data_path}` (water level: optional `variable`, default `waterlevel`); data should exist — warning      |
| `vars_to_predict.tp` with SWAN                          | should be a PCA target, not `"raw"` (SWAN changes Tp nearshore) — warning                                       |

The wavespectra-stats check is skipped (with a note) when `wavespectra` is
not importable — the validator itself never requires third-party packages.

---

## Changelog

Every change to this document is recorded here.

- **2026-10** (hywaves 0.11.0): the numerical cases run with SnapWave or SWAN.
  `simulations.model` chooses the model, and a `simulations.<model>` block
  holds its `grid_file` (SnapWave: UGRID `.nc`; SWAN: ADCIRC `.grd` with
  counterclockwise triangles), `metamodel_templates` / `dynamic_templates`,
  `metamodel_cases` / `dynamic_cases` and optional `slurm` options, replacing
  the flat `simulations.metamodel_templates` / `dynamic_templates` /
  `metamodel_cases` / `dynamic_cases`. Scaffolded template folders are
  `templates-{snapwave,swan}-{metamodel,dynamic}/`. Both models write the same
  `output_sites_{goal}.nc`; the fitted artefacts record the model in
  `outputs/goal_*/split.json` (`model`) and refuse to mix with the other one.
  The portal `manifest.json` records `model`. `publish` builds the report
  itself (README + numbered notebooks → `portal/book/`, in `{Case}/tmp/book/`;
  no `{Case}/book/` folder any more) and records its `book_url`; `snapshot`
  points a copied report at its own route. New optional `forcing` block:
  `forcing.wind` (10 m `u10`/`v10` on `(time, latitude, longitude)`, e.g.
  ERA5) and `forcing.water_level` (on `(time, goal)`, e.g. CODEC) feed the
  dynamic runs; `reconstruct(wl="forcing")` uses the water-level series at
  each goal instead of a constant. Optional `metamodel.wind_correction`: the
  metamodel cases stay without wind, and the reconstruction adds the local
  wind sea before the spectral sum — `K = Hs(with wind) / Hs(without wind)`
  from a model fitted on full-domain training pairs
  (`simulations.<model>.wind_cases`), the fetch per site and direction, and
  the wind (`"inferred"` from the wind-sea contributions, or `"forcing"`).
  Artefacts in `outputs/wind/` (`fetch_table_{model}.nc`,
  `wind_correction_{model}.pkl`); reconstructions gain `wind_k`.
- **2026-10** (hywaves 0.10.0): Hs is a metamodel input. SnapWave metamodel
  cases sample `hs` too (`simulations.bounds.hs`; the `hs.txt` template writes
  `{{ hs }}` at `active_node`), samples steeper than
  `simulations.max_steepness` are dropped, and the metamodel predicts absolute
  nearshore Hs (no more unit-Hs transfer coefficient). `metamodel.sets` defines
  metamodel sets — PCA + GP models on a subset of the output sites, sharing
  the SnapWave cases and MDA of each goal. Holdout metrics move to
  `outputs/goal_*/<set>/validation_metrics.csv` with a required `metamodel`
  column (§4); `paths.pca_dir` / `gp_dir` take `{metamodel}`; `site_filter`
  accepts `{var}_ratio_max`. Portal snapshots (§6): `hywaves snapshot` freezes
  `portal/` as `portal_<tag>/`, shown as case versions in the GUI.
- **2026-10** (hywaves 0.9.0): isobath output sites are placed on a smoothed
  bathymetry: each `simulations.output_sites_config.isobaths` entry takes
  `smooth_m` (Gaussian width, m), `min_length_m` (drop shorter contour pieces)
  and `spacing_m` (metres between sites along the contour), replacing the
  degree-based `spacing`. `isobaths.json` follows the same smoothing.
- **2026-10** (hywaves 0.8.0): SnapWave cases are driven by
  `bluemath_tk.wrappers.snapwave`. The case templates (`templates-metamodel/`,
  `templates-dynamic/`) keep the same SnapWave settings but use its context
  names: `n_nodes`, `active_node`, `hs_nodes`/`tp_nodes`/`dir_nodes`/`spr_nodes`
  (formerly `ngoals`, `goal`, `*_goals`).
- **2026-10** (hywaves 0.6.0): metrics follow the reference-first convention
  of `bluemath_tk.core.metrics` (§4): `bias = mean(estimate − reference)`,
  `r2` normalised by the **reference** variance. Earlier engines normalised
  `r2` by the estimate, so pre-0.6.0 metrics CSVs (no `rrmse` column) report
  different R²; `validate` flags them as `outputs.metrics_legacy`. The engine
  now also writes optional `rrmse`, `si` and `r` columns, and floors the
  metamodel's predicted `hs` at 0 (calm-site negatives used to be dropped
  from the metrics).
- **2026-09** (hywaves 0.5.0): added §6, the portal bundle —
  `python -m hywaves publish` exports `{Case}/portal/` (contract files under
  fixed names, pre-computed AoI/isobath/output-site layers, merged metrics,
  web-sized figures, `manifest.json`), and the reference GUI now reads only
  that folder. `validate` gained the `portal.missing` / `portal.ok` /
  `portal.stale` findings. No change to the §1–§5 inputs themselves.
- **2026-09** (hywaves 0.4.0): `outputs/gui_figures/fig*.png` is rendered by
  the reference GUI again, as a "Final Plots" tab of clickable thumbnails —
  corrects the note below saying it wasn't (that tab was retired, then
  reinstated in this shape). Documented the standard 4-figure naming
  (`fig01_domain` / `fig02_map_comparison` / `fig03_metamodel` /
  `fig04_spectra_reconstruction`) `12_Final_Plots.ipynb` now produces.
- **2026-09**: documented `outputs/metamodel_validation/` and
  `outputs/buoy_validation/` (§5, optional) — feed the reference GUI's
  Metamodel/Buoy Validation tabs and per-buoy map popups. Noted
  `outputs/gui_figures/` is no longer rendered there (its Gallery tab was
  retired in favor of these three validation tabs).
- **2026-07**: initial contract, extracted from the Samoa reference case.
