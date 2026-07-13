# Samoa — hybrid wave downscaling case

Samoa is a **hybrid downscaling** workflow for nearshore waves around the Samoan
archipelago. SnapWave LHS simulations feed a **GPR metamodel** (MDA → PCA → GP)
so full offshore hindcasts can be reconstructed without running the model at every
timestep.

- **Config:** `Samoa.json` (auto-loaded by `utils.config`)
- **Code:** `utils/` + numbered notebooks
- **Backlog:** `TODO.md` (deferred refactors only)

**Environment:** conda `bluemath-311-dev`; run notebooks with `Samoa/` as cwd.

---

## Quick start

```python
from utils.model import metamodel as mm
from utils.evaluation import validation as val
from utils.config import METAMODEL, load_goals, validate_metamodel_config
```

After editing `Samoa.json`: restart kernel or `utils.config.reload()`.

---

## Pipeline (notebooks)

| Step | Notebook | Purpose |
|------|----------|---------|
| 01–04 | Mesh, observations, goals | Static setup |
| 05 | `05_Metamodel_Cases_Generation.ipynb` | 500 LHS cases / goal |
| 06 | `06_Validate_Metamodel.ipynb` | Fit, holdout validation, MDA sweep |
| 07 | `07_Download_Process_WHACS.ipynb` | Offshore zarr |
| 08 | `08_Dynamic_Cases_Generation.ipynb` | Dynamic reference runs |
| 09 | `09_Validate_Linear_Summation.ipynb` | Spectral reconstruct vs dynamic |
| 10 | `10_Validate_Buoy_Observations.ipynb` | Buoy comparison |

```
Goals → LHS cases (05) → fit metamodel (06) → WHACS (07) → dynamic (08) → validate (09–10)
                                              → mm.reconstruct() hindcast
```

---

## Metamodel (notebook 06)

### Variables

| Variable | Training (`vars_to_predict`) | Holdout metrics |
|----------|---------------------------|-----------------|
| `hs` | PCA | yes |
| `tp` | **raw** (GP passthrough) | yes |
| `dir` | PCA (`dir_u`, `dir_v`) | yes |
| `spr` | PCA | yes |

Buoy validation (10) uses `hs`, `tp`, `dir` only — no `spr` observations.

### Fit

```python
mm.fit_all(gp_epochs=1500)                         # all goals × mda_candidates
mm.fit("1", force_retrain=True)                    # one goal, overwrite artefacts
mm.fit("1", candidates=[450], force_retrain=True)  # one MDA size
```

**`force_retrain=True`** re-fits MDA (when needed) and all PCA/GP models for
that goal. Updates `split.json` → `fit_by_mda`. Use after changing metamodel
config or upgrading code.

**Diagnostics (read-only):**

```python
mm.report_training_data("1", n=450)   # site filter + NaN fractions
mm.load_fit_summary("1", 450)         # pca_sites_kept, n_cases_used, …
mm.load_split("1")                    # train_idx, test_idx
```

### Validate

```python
val.sweep("1")                    # → outputs/goal_01/validation_metrics.csv
result = val.validate("1", n=450)
result.plot("10m")                # map + 2×2 scatters (hs, tp, dir, spr)
result.plot_all()
```

Validation loads saved models only. `validate_metamodel_config()` runs at
`fit()` and `validate()`.

### Training stages (per goal)

1. **Site filter** — drop numerically inconsistent sites (`site_filter`)
2. **MDA** — centroids on `(tp, dir, spr, wl)`; largest size = `mda_max_points()`
3. **Split** — centroids = train; other LHS cases = holdout test
4. **PCA** — `hs`, `dir`, `spr` on stacked sites; `tp` raw
5. **GP** — centroids → stacked PCs (alignment checked at fit time)

### Outputs (`outputs/goal_XX/`)

| File | Contents |
|------|----------|
| `mda.pkl` | MDA object |
| `split.json` | Indices, `removed_sites`, `pca_nan_policy`, `fit_by_mda` |
| `pca_{n}/`, `gp_{n}/` | Fitted models |
| `validation_metrics.csv` | Holdout sweep |

**`split.json` → `fit_by_mda["450"]`:** `n_cases_used`, `pca_sites_kept`,
`pca_sites_dropped`, `pca_variances`, `n_gp_components`.

Logs: `logs/metamodel.log`, `logs/validation.log`.

---

## Configuration (`Samoa.json`)

| Section | Role |
|---------|------|
| `simulations` | LHS bounds, case dirs, `output_sites_config` |
| `metamodel.mda_candidates` | Train sizes, e.g. `[50, 100, 150, 250, 450]` |
| `metamodel.vars_to_predict` | PCA vs raw per variable |
| `metamodel.site_filter` | Pre-MDA rules (`{var}_max`, `{var}_min`) |
| `metamodel.pca_preprocessing` | PCA NaN thresholds + zero-fill |
| `metamodel.validation.metrics_variables` | Holdout scoring |

---

## Data quality & NaN handling

> Two mechanisms in `metamodel` config — see `TODO.md` if changing behaviour.

### 1. Site filter (before MDA/PCA)

Numerical inconsistencies from extreme LHS forcing.

| Key | Rule |
|-----|------|
| `hs_max: 10.0` | drop site if any case has Hs > 10 m |
| `tp_min: 0.5` | drop site if any case has Tp < 0.5 s |

Extend with `{var}_max` / `{var}_min`. Removed sites → `split.json` → `removed_sites`.

### 2. PCA preprocessing (during PCA fit)

Dry / low water level NaNs at shallow nodes.

1. `nan_threshold_to_drop` — drop site columns with too many case NaNs (default 0.99)
2. `value_to_replace_nans` — fill remaining NaNs with `0.0`

Sites kept per PCA variable → `fit_by_mda[n].pca_sites_kept`. **Predict does not
yet subset to PCA-kept sites** (under review after refit).

---

## Hindcast reconstruction

```python
ds = mm.reconstruct(
    "2020-05",
    n=450,
    spectral=True,
    offshore_name="WHACS",
    sites="10m",
    recompute=True,   # after retraining or config change
)
```

- **Output:** bulk `hs`, `tp`, `dir` on `(time, sites)` — not `spr`
- **`spr`** is used inside wavespectra construction; bulk sum omits spread stats
- **Cache:** `tmp/reconstruct/`; filename includes PCA settings + config fingerprint
  (`site_filter` + `pca_preprocessing`). Use `recompute=True` after refit.
- Warns if `offshore_waves` is empty in config

Same `pca_variance` override as at fit time.

---

## Case diagnostic plots

`utils.viz.snapwave_case_plots` — post-run SnapWave case check:

- Layout: **hs / tp / dir** panels + full-width map (`spr` omitted by design)
- Map file: `snapwave.inp` → `map_file`, else `output_map.nc`

---

## Python modules (`utils/`)

| Module | Role |
|--------|------|
| `config.py` | JSON load, `resolve()`, `validate_metamodel_config()` |
| `model.metamodel` | Fit, predict, reconstruct, `report_training_data` |
| `evaluation.validation` | Holdout metrics/plots, buoy comparison |
| `simulation.snapwave_wrappers` | SnapWave case build/run |
| `viz.snapwave_case_plots` | Case folder diagnostics |
| `geometry.sites` | Site groups (`10m`, `Apia`, buoys) |

---

## Tests

From `Samoa/`:

```bash
python -m unittest discover -s tests -v
```

Covers scatter grid shape, metrics variable filtering, site filter, PCA config.

---

## Commit & full re-run checklist

Use this after pulling these changes or before a clean validation commit.

### 1. Config

- [ ] `Samoa.json` reviewed (`mda_candidates`, `site_filter`, `pca_preprocessing`)
- [ ] `utils.config.reload()` or kernel restart

### 2. Prerequisites (if not already done)

- [ ] Notebooks 01–05: mesh, goals, LHS cases exist
- [ ] Notebook 07: WHACS zarr (for 08–10)

### 3. Refit metamodel

```python
mm.fit_all(gp_epochs=1500, force_retrain=True)
```

Per goal only: `mm.fit("1", force_retrain=True)`

Verify:

```python
summary = mm.load_fit_summary("1", 450)
assert summary["n_cases_used"] == 450
assert "hs" in summary["pca_sites_kept"]
mm.report_training_data("1", n=450)
```

### 4. Holdout validation

```python
val.sweep_all()                          # all goals → validation_metrics.csv
result = val.validate("1", n=450)
result.plot_all()
```

Check `logs/validation.log` for missing-variable warnings.

### 5. Hindcast validation (optional)

- [ ] Notebook 08: dynamic cases for chosen period
- [ ] Notebook 09: `mm.reconstruct(..., recompute=True)` vs dynamic
- [ ] Notebook 10: buoy comparison (`hs/tp/dir`)

### 6. Before git commit

- [ ] Run unit tests: `python -m unittest discover -s tests`
- [ ] Clear large notebook outputs if desired
- [ ] Do **not** commit `tmp/reconstruct/` caches or huge NetCDF outputs unless intentional

### Suggested commit scope

Code + config + docs:

- `utils/model/metamodel.py`, `utils/evaluation/validation.py`, `utils/config.py`
- `utils/viz/snapwave_case_plots.py`, `utils/simulation/snapwave_wrappers.py`
- `Samoa.json`, `README.md`, `TODO.md`
- `tests/`, notebook intro cells (06, 09, 10)

Artefacts (`outputs/`, `logs/`) — commit only if you want frozen validation results in the repo.

---

## Tips

- **MDA sweep:** compare all `mda_candidates`; largest ≠ always best
- **PCA variance:** same override in fit, predict, reconstruct, validate
- **Period alignment:** notebook 10 requires matching buoy / dynamic / reconstruct periods
- **Site selectors:** `resolve_sites(ds, "10m")`, `"buoy"`, `"Apia"`, or site id

---

## Dependencies

bluemath_tk, wavespectra, SnapWave (external), xarray, geopandas, rioxarray, pymesh2d
