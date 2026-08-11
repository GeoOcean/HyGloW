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
  the vendored HyGloW engine (SnapWave + MDA→PCA→GPR metamodel), i.e. whose
  config contains a `metamodel` section.

Run the validator (stdlib-only, copy it anywhere):

```bash
python -m hyglow validate MyCase/             # development mode
python -m hyglow validate MyCase/ --publish   # strict: ready for the portal
# or standalone, without the hyglow package:
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

`outputs/goal_*/validation_metrics.csv` — one per goal, produced by the
case's holdout validation. Required columns (extra columns are fine):

```text
goal, mda_candidate, group, variable, n, bias, rmse, r2
```

- `goal` — goal id matching the goals JSON.
- `mda_candidate` — training-set size (or emulator size parameter).
- `group` — site group label (e.g. `10m`, `Apia`, `buoy`).
- `variable` — one of the case's validated variables (e.g. `hs`, `tp`).
- `n`, `bias`, `rmse`, `r2` — sample count and holdout metrics.

At publish, at least one metrics file must exist.

### 5. Portal figures (optional)

`outputs/gui_figures/fig*.png` — showcase figures for the portal gallery.

---

## Level 2 — Engine contract (`metamodel` section present)

| Key                                                     | Rule                                                                                                            |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `mda_candidates`                                        | non-empty list of positive ints (last = biggest)                                                                |
| `mda_inputs`                                            | non-empty list of strings                                                                                       |
| `mda_directional_variables`                             | list, subset of `mda_inputs`                                                                                    |
| `vars_to_predict`                                       | non-empty dict; each value `"raw"` or a dict with `vars_to_stack` (non-empty list) and `pca_variance` in (0, 1] |
| `site_filter`                                           | dict; keys end with `_max` or `_min`                                                                            |
| `pca_preprocessing`                                     | optional dict of dicts                                                                                          |
| `validation.metrics_variables`                          | non-empty list, subset of `vars_to_predict`                                                                     |
| `reconstruction.variables`                              | non-empty dict; keys subset of `vars_to_predict`; values valid wavespectra stats                                |
| `reconstruction.save_spectra`                           | bool, required                                                                                                  |
| `paths.pca_dir` / `paths.gp_dir`                        | template strings                                                                                                |
| `simulations.metamodel_templates` / `dynamic_templates` | folders should exist                                                                                            |

The wavespectra-stats check is skipped (with a note) when `wavespectra` is
not importable — the validator itself never requires third-party packages.

---

## Changelog

Every change to this document is recorded here.

- **2026-07**: initial contract, extracted from the Samoa reference case.
