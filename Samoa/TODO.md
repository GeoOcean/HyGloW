# Samoa metamodel & validation — backlog

Last reviewed: 2026-07-13  
Scope: `utils/model/metamodel.py`, `utils/evaluation/validation.py`, `Samoa.json`, notebooks 05–10, viz/simulation helpers.

Active items only. Completed work is summarised in [Done](#done) and the [changelog](#changelog).

---

## Pipeline (reference)

```
SnapWave LHS (output_sites_{goal}.nc)
    → site_filter (config: hs_max, tp_min, …)
    → MDA on forcing (tp, dir, spr, wl)
    → PCA per variable (hs, dir, spr) + tp raw
    → GP: forcing → stacked PCs
    → predict (test cases or hindcast time)
    → holdout validation vs SnapWave (metrics + 2×2 plots)
```

| File | Role |
|------|------|
| `Samoa.json` | All case config (`vars_to_predict`, `site_filter`, `pca_preprocessing`, …) |
| `utils/model/metamodel.py` | Train, predict, reconstruct |
| `utils/evaluation/validation.py` | Holdout metrics/plots, buoy comparison |
| `06_Validate_Metamodel.ipynb` | Fit, sweep, plot workflow |
| `README.md` | User-facing documentation (keep in sync with config) |

---

## After refit — sanity checks

Run after `mm.fit_all(force_retrain=True)` (or per-goal `mm.fit`):

- [ ] `split.json` contains `pca_nan_policy` and `fit_by_mda["450"]` (or each candidate you fit)
- [ ] `mm.load_fit_summary("1", 450)` shows `pca_sites_kept`, `n_cases_used`, `pca_sites_dropped`
- [ ] `logs/metamodel.log` reports `PCA hs kept N/M sites` per goal
- [ ] `val.sweep(goal)` writes `validation_metrics.csv` with `hs`, `tp`, `dir`, `spr`
- [ ] `val.validate(goal, n=450).plot_all()` — 2×2 scatter layout per site group
- [ ] Review whether dry-site PCA zero-fill affects shallow isobaths (decide later if site mask is needed)

---

## Next up — documentation & cleanup (P4)

Small diffs, no behaviour change. **Recommended before larger refactors.**

### Docstrings & comments

- [ ] `metamodel.py` — `_wavespectra_sum`, `reconstruct()` (bulk output is `hs/tp/dir` only; `spr` internal)
- [ ] `metamodel.py` — `SnapWaveDynamicModelWrapper` cross-ref if doc lives elsewhere
- [ ] `snapwave_case_plots.py` — document why `TOP_VARS = ("hs", "tp", "dir")` omits `spr` (layout choice)
- [ ] `validation.PARAMETERS` — note why `spr` uses linear (not circular) metrics

### Notebooks

- [ ] `06_Validate_Metamodel.ipynb` — intro markdown: mention `spr` in metrics and modelling
- [ ] `09_Validate_Linear_Summation.ipynb` — comment if `spr` excluded from comparison
- [ ] `10_Validate_Buoy_Observations.ipynb` — comment that buoys use `hs/tp/dir` only (no `spr` obs)

### README / config docs

- [ ] Document `force_retrain` behaviour (re-fits MDA + all PCA/GP for that goal)
- [ ] Document `mm.reconstruct` cache + `recompute=True` after retraining

---

## Deferred — PCA site mask at predict (P1.1)

**Decision:** wait until post-refit validation review.

PCA fit drops ~5–10% of sites (dry nodes). Sites kept are persisted in `split.json` → `fit_by_mda[n].pca_sites_kept`. Prediction still returns all sites after `site_filter`; inverse_transform only fills PCA-kept sites.

- [ ] After reviewing refit metrics: decide whether to align predict/validation to `pca_sites_kept`
- [ ] If yes: subset in `predict()` and/or validation metrics (or report both full-site and PCA-site metrics)

---

## Config & maintainability (P3)

### Unified variable schema

Variable metadata is duplicated in three places:

| Source | Contents |
|--------|----------|
| `Samoa.json` → `vars_to_predict` | PCA vs raw, `vars_to_stack`, `pca_variance` |
| `Samoa.json` → `validation.metrics_variables` | What to score |
| `validation.PARAMETERS` | label, units, `circular` |

- [ ] Propose unified `metamodel.variables` schema (see old P3.1 draft in git history)
- [ ] Derive `PARAMETERS`, `_pca_variables()`, `METRICS_VARIABLES` from one source
- [ ] **When:** after next case clone (e.g. NewSouthWales) or when adding a new variable

### Optional site_filter extensions

- [ ] Add `spr_min` / `spr_max` to `site_filter` if refit shows bad spread at specific sites
- [ ] `detect_removed_sites()` already supports any `{var}_max` / `{var}_min` key

### Training diagnostics

- [ ] `mm.report_training_data(goal, n)` — NaN counts, projected site retention, case-drop preview (pre-fit helper)

---

## Resource hygiene (P2)

- [ ] `metamodel._load_reconstruction_cache()` — close NetCDF handles or use context manager
- [ ] `snapwave_wrappers.py` — audit `open_dataset` in postprocess
- [ ] Optional: `@lru_cache` for `load_cases_goal()` with `clear_cases_cache()`
- [ ] Optional: split `force_retrain` into `force_retrain_mda` / `force_retrain_models`
- [ ] Warn (not raise) when `offshore_waves` empty but reconstruction notebook is run

---

## Module structure (P5) — large refactors

Do when training + reconstruction are touched together, not urgently.

### Split `metamodel.py` (~2000 lines)

```
utils/model/
  metamodel.py      # thin public API re-exports
  training.py       # _fit_mda, _fit_models, fit, fit_all
  predict.py        # predict, load_cases_goal, forcing_dataframe
  reconstruct.py    # reconstruct, cache, _wavespectra_sum
  paths.py          # artefact path helpers
```

### Split `validation.py` (~1500 lines)

```
utils/evaluation/
  metrics.py, holdout.py, buoy.py, plots.py
  validation.py     # re-exports for notebooks
```

- [ ] Simplify `HoldoutValidationResult` plot overrides → `default_estimate_label` on base class

---

## Validation & plotting enhancements (P6)

- [ ] `plot_scatter()` standalone — tune `figsize` for 2×2 layout
- [ ] `plot_sweep()` — verify 4-column readability with `spr`
- [ ] Confirm with domain experts: linear metrics for `spr` (spread in degrees)
- [ ] Minimal tests: `test_scatter_grid_shape`, `test_available_metrics_variables`, `test_holdout_pair_alignment`, `test_metrics_table_circular_dir`

---

## Cross-notebook consistency (P7)

- [ ] **Case diagnostic plots:** add `spr` panel (2×2) or document 3-panel + map as intentional (`snapwave_case_plots.py`)
- [ ] **`save_map` / `map_file`:** ensure notebook 05 sets `map_file` reliably; document fallback to `output_map.nc`
- [ ] **Goals vs LHS sectors:** sanity check goal `angles` overlap sampled `dir` per goal

---

## Hindcast / reconstruction (P8)

- [ ] Document bulk `mm.reconstruct` output: `hs`, `tp`, `dir` only (`spr` used in spectra, not bulk stats)
- [ ] Cache key: ensure site-filter / PCA policy changes invalidate `tmp/reconstruct` caches
- [ ] Document `recompute=True` after model retrain

---

## Open questions

1. **`spr` in case diagnostic plots** — 2×2 with map or keep 3-panel layout?
2. **PCA site mask at predict** — defer until refit metrics reviewed (see [Deferred](#deferred--pca-site-mask-at-predict-p11))
3. **Unified variable schema** — migrate now or with next case folder?
4. **Module split** — after P4 cleanup and refit review?

---

## Suggested order

| Step | Focus | Effort |
|------|-------|--------|
| **Now** | Refit + sanity checks (above) | — |
| **1** | P4 docstrings & notebook comments | Small |
| **2** | Review refit → decide on PCA site mask | Decision |
| **3** | P2 resource hygiene | Small |
| **4** | P6 tests (minimal) | Medium |
| **5** | P3 unified variables + P5 module split | Large |

---

## Done

Resolved in Phases A–B (2026-07-09 – 2026-07-13):

- NaN policy documented; `pca_preprocessing` in `Samoa.json`
- Generic `site_filter` (`{var}_max` / `{var}_min`)
- `spr` as PCA variable; `tp` raw; README aligned
- `validate_metamodel_config()` at `fit()` / `validate()`
- `split.json`: `pca_nan_policy`, `fit_by_mda`, `pca_sites_kept`, case/site counts
- Fit logging (cases × sites, sites dropped per PCA)
- GP ↔ MDA centroid alignment assert
- `_holdout_pair` uses `_available_metrics_variables`; missing-var warning
- KDE fallback in validation scatter plots
- `load_cases_goal()` closes NetCDF handles
- Holdout validation 2×2 plots (`hs`, `tp`, `dir`, `spr`)
- Notebook 06 info cell: dynamic site_filter + metrics list

---

## Changelog

| Date | Notes |
|------|-------|
| 2026-07-13 | Backlog trimmed to active items; refit checklist added |
| 2026-07-13 | Phase B: `validate_metamodel_config`, `fit_by_mda`, PCA sites kept |
| 2026-07-09 | Phase A: README NaN policy, KDE fallback, GP alignment, generic site_filter |
| 2026-07-09 | Initial backlog after `spr` integration review |
