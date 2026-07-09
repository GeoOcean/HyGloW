# Samoa metamodel & validation — technical debt and improvement backlog

Last reviewed: 2026-07-09  
Scope: `utils/model/metamodel.py`, `utils/evaluation/validation.py`, `Samoa.json`, notebooks 05–06, related viz/simulation helpers.

This document captures known inconsistencies, design tensions, and suggested refactors identified during a deep review of the training + holdout validation pipeline. Items are grouped by priority; within each group, order is roughly recommended execution order.

---

## Current pipeline (reference)

```
SnapWave LHS (output_sites_{goal}.nc)
    → MDA on forcing (tp, dir, spr, wl)
    → PCA per variable (hs, dir, spr) on stacked sites
    → GP: forcing → stacked PCs
    → predict (test cases or hindcast time)
    → validation vs SnapWave reference (metrics + plots)
```

**Key files**

| File | ~Lines | Role |
|------|--------|------|
| `utils/model/metamodel.py` | 1900 | Train, predict, reconstruct, cache |
| `utils/evaluation/validation.py` | 1540 | Holdout metrics/plots, buoy comparison, sweep |
| `Samoa.json` | — | `vars_to_predict`, `site_filter`, `metrics_variables` |
| `06_Validate_Metamodel.ipynb` | — | Fit, sweep, plot workflow |

---

## What works well (keep)

- [x] Config-driven `vars_to_predict` / `mda_inputs` / `metrics_variables`
- [x] MDA centroids = train, remaining LHS cases = test (`split.json`)
- [x] Site filter persisted in `split.json` (`removed_sites`)
- [x] Ergonomic validation API: `validate()` → `result.plot()` / `result.plot_all()`
- [x] `spr` integrated in config, PCA, metrics, and 2×2 validation plots
- [x] `_available_metrics_variables()` skips vars missing from estimate/reference (buoy-safe)

---

## P0 — Design decisions (resolve before more refactors)

### P0.1 NaN handling policy

**Status:** Unresolved design tension.

**Current behaviour** (`metamodel.py`):

```python
VALUE_TO_REPLACE_NANS = {"hs": 0.0, "dir_u": 0.0, "dir_v": 0.0, "spr": 0.0}
NAN_THRESHOLD_TO_DROP = {"hs": 0.99, "dir_u": 0.99, "dir_v": 0.99, "spr": 0.99}
```

BlueMath PCA behaviour:

1. `nan_threshold_to_drop` — drops **site columns** where too many cases are NaN (not whole cases).
2. `value_to_replace_nans` — fills **remaining** NaNs with **0** before sklearn PCA.

**Observed on goal 1 (450 train cases):**

- ~8.1% of site×case values are NaN (consistent dry/invalid sites across cases).
- Every case has at least one NaN somewhere → **case-level “drop if any NaN” would remove all cases**.
- `nan_threshold=0.99` keeps ~668/727 sites; residual NaNs are zero-filled.

**Preferred direction (from user):** drop bad **cases**, avoid fake zeros.  
**Pragmatic reality:** NaNs are mostly **site-level** (dry mesh nodes), so site-drop is the meaningful lever; case-drop needs a threshold (e.g. max NaN fraction per case), not “any NaN”.

**Action items:**

- [ ] **Decide policy** and document in `Samoa.json` + README:
  - Option A: site-drop only (`nan_threshold_to_drop`), no zero-fill (may need BlueMath support or pre-filter cases in xarray).
  - Option B: site-drop + zero-fill (current; document as explicit compromise).
  - Option C: site-drop + drop cases above NaN fraction threshold; warn when `n < mda_candidates[k]`.
- [ ] Move `VALUE_TO_REPLACE_NANS` and `NAN_THRESHOLD_TO_DROP` from hardcoded constants → `Samoa.json` (`metamodel.pca_preprocessing`).
- [ ] Log clearly at fit time: sites dropped per variable, cases dropped (if any), final matrix shape.
- [ ] Persist in `split.json`: `pca_sites_kept`, `pca_nan_policy`, `n_cases_requested`, `n_cases_used`.

### P0.2 README vs config mismatch (`spr` modelling)

**`README.md` (line ~170)** says:

> `hs` and `dir` via PCA; **`tp` and `spr` passed through**

**`Samoa.json` actual:**

| Variable | Mode |
|----------|------|
| `hs` | PCA |
| `dir` | PCA (`dir_u`, `dir_v`) |
| `tp` | **raw** (passthrough) |
| `spr` | **PCA** |

- [ ] Update `README.md` to match `Samoa.json`.
- [ ] Add a short comment block in `Samoa.json` explaining raw vs PCA per variable.

---

## P1 — Correctness & traceability (medium risk)

### P1.1 PCA site mask not persisted or enforced at predict time

PCA internally stores `not_nan_positions` (sites kept per stacked variable). Training can silently drop ~8% of sites. Prediction uses all sites from `load_cases_goal()` (after `site_filter` only).

- [ ] After PCA fit, record retained site indices/labels per variable in artefacts or `split.json`.
- [ ] In `predict()`, align output sites with PCA-retained sites (or document that inverse_transform only returns kept sites).
- [ ] Validation metrics should use the same site set the model was trained on (or report both).

### P1.2 GP ↔ PCA row alignment not verified

```python
cases = load_cases_goal(goal).isel(case_num=train_idx[:n])
# PCA on cases...

gp.fit(
    subset_data=mda.centroids[MDA_INPUTS].iloc[:n],
    target_data=target,  # from pca.pcs_df
)
```

Assumes row `i` of centroids == case `train_idx[i]` == row `i` of `pcs_df`. Likely true with BlueMath MDA but **not asserted**.

- [ ] Add alignment check at end of `_fit_models()` (compare forcing indices / case_num).
- [ ] Fail loudly or warn if mismatch.

### P1.3 `_holdout_pair` reference variable selection inconsistent

```python
vars_ = [v for v in _metrics_variables(None) if v in cases.data_vars]
```

Uses `_metrics_variables`, not `_available_metrics_variables(estimate, reference)`. If estimate and reference diverge, metrics/plots can disagree.

- [ ] Use `_available_metrics_variables(estimate, cases, ...)` when building reference.
- [ ] Warn if a configured metric variable is missing from SnapWave cases.

### P1.4 `n` (MDA candidate) vs effective training size

Artefact paths use `mda_candidates` (e.g. `450`) even if fewer cases/sites survive preprocessing. No warning when effective training size &lt; `n`.

- [ ] Log: `Goal G / n=450: trained on 450 cases × 668 sites (59 sites dropped by PCA threshold)`.
- [ ] Optional: store `n_cases_used`, `n_sites_used` in `split.json` or a `fit_summary.json`.

### P1.5 Scatter plots can crash on small / degenerate samples

`density_scatter` → `gaussian_kde` raises `LinAlgError` when:

- Very few points in a validation group.
- Estimate ≈ reference (common in good holdout fits).

- [ ] Fallback to plain `ax.scatter` when `n < N_min` or KDE fails.
- [ ] Apply in `_scatter_panel()` in `validation.py`.

---

## P2 — Resource hygiene & robustness (low–medium risk)

### P2.1 NetCDF file handles not closed

| Location | Pattern |
|----------|---------|
| `metamodel.load_cases_goal()` | `xr.open_dataset(cases_path(goal))` — never closed |
| `metamodel._load_reconstruction_cache()` | `xr.open_dataset(path)` |
| `snapwave_case_plots.py` | opens sites/map per plot (does close at end) |
| `snapwave_wrappers.py` | `open_dataset` in postprocess |

- [ ] Use context managers or `.load()` + `.close()` in `load_cases_goal()`.
- [ ] Consider `@lru_cache` for loaded case datasets within a notebook session (with explicit `clear_cases_cache()`).

### P2.2 `force_retrain` partial staleness

- `fit(goal, force_retrain=True)` re-fits MDA + models.
- Loading cached MDA while forcing only GP/PCA (or vice versa) is not a separate flag.

- [ ] Document `force_retrain` behaviour in README and notebook 06.
- [ ] Optional: split flags `force_retrain_mda`, `force_retrain_models`.

### P2.3 Config reload / staged workflow

`utils.config` loads JSON at import; notebooks remind to `reload()` after edits. Metamodel keys are not validated at import (staged workflow is intentional).

- [ ] Optional: `validate_metamodel_config()` called from `fit()` / `validate()` with clear errors for missing keys.
- [ ] Optional: warn (not raise) when `offshore_waves` empty but reconstruction notebook is run.

---

## P3 — Config & metadata consolidation (maintainability)

### P3.1 Variable metadata duplicated in three places

| Source | Contents |
|--------|----------|
| `Samoa.json` → `vars_to_predict` | PCA vs raw, `vars_to_stack`, `pca_variance` |
| `Samoa.json` → `validation.metrics_variables` | What to score |
| `validation.PARAMETERS` | label, units, `circular` flag |

Adding a new variable (e.g. replacing `ra` with `spr`) required edits in all three.

- [ ] Propose unified `metamodel.variables` schema, e.g.:

```json
"variables": {
  "hs":  {"predict": "pca", "stack": ["hs"], "pca_variance": 0.99, "label": "Hs",  "units": "m",  "circular": false, "validate": true},
  "tp":  {"predict": "raw", "label": "Tp", "units": "s", "circular": false, "validate": true},
  "dir": {"predict": "pca", "stack": ["dir_u","dir_v"], "pca_variance": 0.99, "label": "Dir", "units": "deg", "circular": true,  "validate": true},
  "spr": {"predict": "pca", "stack": ["spr"], "pca_variance": 0.99, "label": "Spr", "units": "deg", "circular": false, "validate": true}
}
```

- [ ] Derive `PARAMETERS`, `_pca_variables()`, and `METRICS_VARIABLES` from one source.

### P3.2 `site_filter` incomplete for `spr`

Current filter: `hs_max`, `tp_min` only. Forcing bounds allow `spr` 1–45°.

- [ ] Consider `spr_min` / `spr_max` in `site_filter` (mirror hs/tp logic).
- [ ] Update notebook 06 info cell (still prints only Hs/Tp filters).
- [ ] Extend `detect_removed_sites()` if new keys added.

### P3.3 PCA preprocessing not in config

`VALUE_TO_REPLACE_NANS` and `NAN_THRESHOLD_TO_DROP` are module-level constants.

- [ ] Move to `Samoa.json` under `metamodel.pca_preprocessing`.
- [ ] Allow per-variable overrides.

---

## P4 — Documentation & stale strings (low risk, high clutter)

### P4.1 Stale docstrings in `metamodel.py`

| Location | Issue |
|----------|-------|
| `_wavespectra_sum` | Says output bulk `hs/tp/dir`; uses `spr` in spectra construction; bulk stats omit `spr` |
| `reconstruct()` return doc | Lists `hs`, `tp`, `dir` only |
| `FORCING_COLS` comment | Says “MDA inputs passed through GP” but excludes `wl`; GP uses full `MDA_INPUTS` |
| `detect_removed_sites` | Doc mentions only `hs` / `tp` |
| `SnapWaveDynamicModelWrapper` module doc | Says outputs `hs`, `tp`, `dir` on sites (no `spr`) |

- [ ] Pass over docstrings after P0/P1 decisions are fixed.

### P4.2 Stale strings in `snapwave_case_plots.py`

- [ ] `TOP_VARS = ("hs", "tp", "dir")` — case diagnostic plots omit `spr` (may be intentional for layout; document or add optional 2×2).
- [ ] `resolve_wave_variable` error message: “expected hs/tp/dir” — should mention `spr` if supported elsewhere.

### P4.3 Notebook / README alignment

- [ ] `06_Validate_Metamodel.ipynb` — mention `spr` in intro and filter info cell.
- [ ] `09_Validate_Linear_Summation.ipynb` — still compares `[["hs", "tp", "dir"]]` only; consider `spr` if dynamic output includes it.
- [ ] `10_Validate_Buoy_Observations.ipynb` — explicitly `variables = ["hs", "tp", "dir"]` (correct if buoys lack `spr`; add comment).

---

## P5 — Module structure (larger refactors, optional)

### P5.1 Split `metamodel.py` (~1900 lines)

Suggested modules:

```
utils/model/
  metamodel.py      # thin public API re-exports
  training.py       # _fit_mda, _fit_models, fit, fit_all
  predict.py        # predict, load_cases_goal, forcing_dataframe
  reconstruct.py    # reconstruct, cache, _wavespectra_sum, offshore forcing
  paths.py          # artefact path helpers
```

- [ ] Split when next major feature touches training + reconstruction together.

### P5.2 Split `validation.py` (~1540 lines)

Suggested modules:

```
utils/evaluation/
  metrics.py        # compute_metrics, metrics_table
  holdout.py        # validate, sweep, _holdout_pair, HoldoutValidationResult
  buoy.py           # load_buoy_observations, compare_all, offshore at buoys
  plots.py          # plot_group, plot_scatter, plot_sweep, grid helpers
  validation.py     # re-exports for notebooks
```

- [ ] Keep `from utils.evaluation import validation as val` working via re-exports.

### P5.3 Simplify `HoldoutValidationResult` plot overrides

Subclass only overrides default labels (`Metamodel` / `SnapWave`) for `plot`, `plot_scatter`, `plot_all`.

- [ ] Replace with `default_estimate_label` / `default_reference_label` on base `ComparisonResult`.

---

## P6 — Validation & plotting enhancements (nice to have)

### P6.1 Plot layout consistency

Holdout validation: map + 2×2 (`hs`, `tp`, `dir`, `spr`) — **done**.

- [ ] Apply same 2×2 idea to `plot_scatter()` standalone default `figsize` tuning per layout.
- [ ] `plot_sweep()` — already adapts columns to variable count; verify 4-column layout readability.

### P6.2 Metrics for directional spread

`spr` uses linear bias/RMSE/R² (`circular: false`). Directional spreading is not periodic like `dir`.

- [ ] Confirm with domain experts that linear metrics are appropriate (likely yes for spread in degrees).
- [ ] Document in `PARAMETERS` why `spr` is not circular.

### P6.3 Pre-fit diagnostic helper

- [ ] Add `mm.report_training_data(goal, n=450)`:
  - site filter removals
  - NaN counts per variable (cases × sites)
  - projected PCA site retention at configured threshold
  - list of cases that would be dropped under each policy

### P6.4 Tests (minimal, high value)

No automated tests exist for metamodel/validation.

- [ ] `test_scatter_grid_shape()` — pure function
- [ ] `test_available_metrics_variables()` — missing `spr` on buoy-like dataset
- [ ] `test_holdout_pair_alignment()` — mock small dataset
- [ ] `test_metrics_table_circular_dir()` — known dir bias case

---

## P7 — Cross-notebook / simulation consistency

### P7.1 Post-run case plots vs metamodel variables

`snapwave_case_plots.py` is metamodel-oriented (hs 0–1 white→blue) but generic `TOP_VARS` excludes `spr`.

- [ ] Decide: add `spr` panel to case diagnostic plots or keep 3-panel + map.
- [ ] Share variable metadata with validation `PARAMETERS` if unified config is adopted.

### P7.2 `save_map` / `map_file` in generated cases

Some cases have empty `map_file` in `snapwave.inp`; plotting falls back to `output_map.nc`.

- [ ] Ensure `build_case()` + notebook 05 `save_map` column reliably set `map_file` for new cases.
- [ ] Document fallback behaviour in case plot module.

### P7.3 Goals JSON vs metamodel goal sectors

Reconstruction filters timesteps by goal nautical sector (`_goal_sector`). MDA train cases are per-goal LHS — consistent.

- [ ] Add sanity check: goal sector in `Samoa_Goals.json` overlaps sampled `dir` in LHS forcing for that goal.

---

## P8 — Hindcast / reconstruction (related, lower urgency for notebook 06)

### P8.1 Bulk reconstruction output variables

`_wavespectra_sum` returns `hs`, `tp`, `dir` after partition sum; `spr` is used internally but not in bulk output stats.

- [ ] Document that reconstructed bulk fields are `hs/tp/dir` only.
- [ ] If `spr` hindcast is needed, extend wavespectra stats or document workaround.

### P8.2 Reconstruction cache invalidation

Cache paths encode many parameters (`reconstruct-v2__...`). Changing PCA variance or site filter may not invalidate old caches.

- [ ] Include `pca_variances` and site-filter hash in cache key (partially done via `pca-` label).
- [ ] Document `recompute=True` when retraining models.

---

## Quick reference — open questions for the team

1. **NaN policy:** site-drop only, case-drop with threshold, or keep zero-fill after site-drop?
2. **`spr` in case diagnostic plots:** 2×2 with hs/tp/dir/spr or keep 3-panel?
3. **Unified variable schema:** worth the migration now or after next case (e.g. NewSouthWales)?
4. **Module split:** do now or after NaN policy is settled?

---

## Suggested execution order

| Phase | Items | Effort |
|-------|-------|--------|
| **A** | P0.2 README fix, P4 stale docstrings, P2.5 KDE fallback, P2.1 close datasets, fit logging | Small |
| **B** | P0.1 NaN policy decision + config move + split.json persistence | Medium |
| **C** | P1 PCA site mask, P1 GP alignment assert, P1 holdout reference fix | Medium |
| **D** | P3 unified variable metadata, P5 module splits | Large |

---

## Changelog

| Date | Notes |
|------|-------|
| 2026-07-09 | Initial backlog from deep review of metamodel training + validation after `spr` integration |
