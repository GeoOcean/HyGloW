# Samoa — hybrid wave downscaling case

Samoa is a **hybrid downscaling** workflow for nearshore waves around the Samoan
archipelago. SnapWave LHS simulations feed a **GPR metamodel** (MDA → PCA → GP)
so full offshore hindcasts can be reconstructed without running the model at every
timestep.

- **Config:** `Samoa.json` (auto-loaded by `utils.config`)
- **Code:** `utils/` + numbered notebooks
- **Observations catalogue:** `inputs/Observations/Samoa_observations.json` (GeoJSON; notebook 03)

**Environment:** conda `bluemath-311-dev`; run notebooks with `Samoa/` as cwd.

---

## Quick start

```python
from utils.model import metamodel as mm
from utils.evaluation import validation as val
from utils.config import METAMODEL, load_goals, validate_metamodel_config
from utils.observation import onshore_sites as obs
```

After editing `Samoa.json`: restart kernel or `utils.config.reload()`.

---

## Pipeline (notebooks)

| Step  | Notebook                                       | Purpose                                      |
| ----- | ---------------------------------------------- | -------------------------------------------- |
| 00    | `00_Course_Figures.ipynb`                      | Slide-ready PNGs for teaching                |
| 01–04 | Mesh, observations, goals                      | Static setup                                 |
| 03    | `03_Generate_Observations_Configuration.ipynb` | Onshore buoy/site GeoJSON                    |
| 05    | `05_Metamodel_Cases_Generation.ipynb`          | 500 LHS cases / goal                         |
| 06    | `06_Validate_Metamodel.ipynb`                  | Fit, holdout validation, MDA sweep           |
| 07    | `07_Download_Process_WHACS.ipynb`              | Offshore zarr                                |
| 08    | `08_Dynamic_Cases_Generation.ipynb`            | Dynamic reference runs                       |
| 09    | `09_Validate_Linear_Summation.ipynb`           | Spectral reconstruct vs dynamic (bulk stats) |
| 10    | `10_Validate_Buoy_Observations.ipynb`          | Buoy bulk comparison                         |
| 11    | `11_Validate_Spectra.ipynb`                    | Buoy vs model spectra (in progress)          |

```
Goals → LHS (05) → fit (06) → WHACS (07) → dynamic (08)
                              → mm.reconstruct() hindcast
                              → validate bulk (09–10) → validate spectra (11)
```

---

## Metamodel (notebook 06)

### Variables

| Variable | Training (`vars_to_predict`) | Holdout metrics |
| -------- | ---------------------------- | --------------- |
| `hs`     | PCA                          | yes             |
| `tp`     | **raw** (GP passthrough)     | yes             |
| `dir`    | PCA (`dir_u`, `dir_v`)       | yes             |
| `spr`    | PCA                          | yes             |

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
`fit()`, `validate()`, and `reconstruct()`.

### Training stages (per goal)

1. **Site filter** — drop numerically inconsistent sites (`site_filter`)
2. **MDA** — centroids on `(tp, dir, spr, wl)`; largest size = `mda_max_points()`
3. **Split** — centroids = train; other LHS cases = holdout test
4. **PCA** — `hs`, `dir`, `spr` on stacked sites; `tp` raw
5. **GP** — centroids → stacked PCs (alignment checked at fit time)

### Outputs (`outputs/goal_XX/`)

| File                     | Contents                                                 |
| ------------------------ | -------------------------------------------------------- |
| `mda.pkl`                | MDA object                                               |
| `split.json`             | Indices, `removed_sites`, `pca_nan_policy`, `fit_by_mda` |
| `pca_{n}/`, `gp_{n}/`    | Fitted models                                            |
| `validation_metrics.csv` | Holdout sweep                                            |

**`split.json` → `fit_by_mda["450"]`:** `n_cases_used`, `pca_sites_kept`,
`pca_sites_dropped`, `pca_variances`, `n_gp_components`.

Logs: `logs/metamodel.log`, `logs/validation.log`.

---

## Configuration (`Samoa.json`)

| Section                                  | Role                                               |
| ---------------------------------------- | -------------------------------------------------- |
| `onshore_waves`                          | Path to observations GeoJSON (notebook 03)         |
| `simulations`                            | LHS bounds, case dirs, `output_sites_config`       |
| `metamodel.mda_candidates`               | Train sizes, e.g. `[50, 100, 150, 250, 450]`       |
| `metamodel.vars_to_predict`              | PCA vs raw per variable                            |
| `metamodel.site_filter`                  | Pre-MDA rules (`{var}_max`, `{var}_min`)           |
| `metamodel.pca_preprocessing`            | PCA NaN thresholds + zero-fill                     |
| `metamodel.validation.metrics_variables` | Holdout scoring                                    |
| `metamodel.reconstruction.variables`     | HyGloW name → wavespectra stat for `reconstruct()` |
| `metamodel.reconstruction.save_spectra`  | Cache summed `efth` spectra (bool, required)       |

`validate_metamodel_config()` enforces required reconstruction keys — no code defaults.

---

## Data quality & NaN handling

Two mechanisms in `metamodel` config:

### 1. Site filter (before MDA/PCA)

Numerical inconsistencies from extreme LHS forcing.

| Key            | Rule                                 |
| -------------- | ------------------------------------ |
| `hs_max: 10.0` | drop site if any case has Hs > 10 m  |
| `tp_min: 0.5`  | drop site if any case has Tp < 0.5 s |

Extend with `{var}_max` / `{var}_min`. Removed sites → `split.json` → `removed_sites`.

### 2. PCA preprocessing (during PCA fit)

Dry / low water level NaNs at shallow nodes.

1. `nan_threshold_to_drop` — drop site columns with too many case NaNs (default 0.99)
2. `value_to_replace_nans` — fill remaining NaNs with `0.0`

Sites kept per PCA variable → `fit_by_mda[n].pca_sites_kept`.

---

## Hindcast reconstruction

```python
ds = mm.reconstruct(
    "2020-07",
    n=450,
    spectral=True,
    offshore_name="WHACS",
    sites="buoy_Aunuu_Buoy",
    recompute=True,   # after retraining or config change
)
```

- **Output:** HyGloW variables on `(time, sites)` from required
  `metamodel.reconstruction.variables` (e.g. `hs→hs`, `tp→tp`, `dir→dm`, `spr→dspr`).
  All wavespectra bulk stats are computed per site; only the mapped names are returned.
- **`save_spectra`:** when `true`, cache summed `efth` on `(time, sites, freq, dir)` as
  `{bulk_cache}__spectra.nc`. Load with `mm.load_reconstruction_spectra(...)`.
- **Cache:** `tmp/reconstruct/` (`reconstruct-v5`); use `recompute=True` after refit or
  config change.
- Warns if `offshore_waves` is empty in config

Same `pca_variance` override as at fit time.

---

## Buoy observations (notebooks 03, 10, 11)

1. **Catalogue** — `03_Generate_Observations_Configuration.ipynb` writes
   `inputs/Observations/Samoa_observations.json` (`instrument_id`, lon/lat, …).
2. **SnapWave site ids** — `obs.snapwave_site_id("Aunuu_Buoy")` → `buoy_Aunuu_Buoy`
   (same convention as mesh output sites).
3. **Processed files** — user prepares NetCDF under `inputs/Observations/processed/`
   (bulk or spectral). Notebook 10 loads, derives bulk stats if needed, and builds a
   `(time, sites)` dataset for `val.compare_all()`.
4. **Site metadata** — `obs.attach_buoy_site_meta(ds, site_id)` adds map/plot coords
   from the GeoJSON.

Buoy validation requests `hs`, `tp`, `dir`, `spr`; `val.resolve_comparison_variables()`
keeps only variables present in buoy and all models.

---

## Pre-run diagnostics (notebooks 05, 08)

Before building or running SnapWave cases, both wrapper classes expose:

```python
wrapper.plot_configuration(...)       # map + per-goal direction bars + text summary
wrapper.plot_configuration_map(...)   # same map alone, bigger — for reading goal labels closely
```

The per-goal bar panel means different things depending on the wrapper:

- **Metamodel** (`SnapWaveMetaModelWrapper`) — the crisp outline bounds the
  *actual sampled* direction interval, verifying the LHS design fills its
  configured sector.
- **Dynamic** (`SnapWaveDynamicModelWrapper`) — the crisp outline bounds the
  goal's *configured* sector instead; the shaded density is the real
  hindcast, which is free to (and does) extend past it — that gap is exactly
  the "% outside sector" figure in the text panel.

Colour is assigned by goal order (not a repeating qualitative palette), so
it never runs out or repeats regardless of goal count.

---

## Case diagnostic plots

`utils.viz.snapwave_case_plots` — post-run SnapWave case check:

- Layout: full-width domain map + 2×2 site panels (`hs`, `tp`, `dir`, `spr`)
- Map file: `snapwave.inp` → `map_file`, else `output_map.nc`

---

## Python modules (`utils/`)

| Module                         | Role                                                  |
| ------------------------------ | ----------------------------------------------------- |
| `config.py`                    | JSON load, `resolve()`, `validate_metamodel_config()` |
| `model.metamodel`              | Fit, predict, reconstruct, `report_training_data`     |
| `evaluation.validation`        | Holdout metrics/plots, buoy comparison                |
| `observation.onshore_sites`    | Observations GeoJSON I/O, buoy site ids               |
| `simulation.snapwave_wrappers` | SnapWave case build/run                               |
| `viz.snapwave_case_plots`      | Case folder diagnostics                               |
| `geometry.sites`               | Site groups (`10m`, `Apia`, buoys)                    |

---

## Tests

From `Samoa/`:

```bash
python -m unittest discover -s tests -v
```

Covers scatter grid shape, comparison variable resolution, site filter, PCA config,
onshore buoy site helpers.

---

## Commit & full re-run checklist

### 1. Config

- [ ] `Samoa.json` reviewed (`mda_candidates`, `site_filter`, `pca_preprocessing`, `reconstruction`)
- [ ] `utils.config.reload()` or kernel restart

### 2. Prerequisites (if not already done)

- [ ] Notebooks 01–05: mesh, goals, LHS cases exist
- [ ] Notebook 03: observations GeoJSON
- [ ] Notebook 07: WHACS zarr (for 08–11)

### 3. Refit metamodel

```python
mm.fit_all(gp_epochs=1500, force_retrain=True)
```

### 4. Holdout validation

```python
val.sweep_all()
result = val.validate("1", n=450)
result.plot_all()
```

### 5. Hindcast validation

- [ ] Notebook 08: dynamic cases for chosen period
- [ ] Notebook 09: `mm.reconstruct(..., spectral=True, recompute=True)` vs dynamic
- [ ] Notebook 10: process buoy NetCDF → `val.compare_all()`
- [ ] Notebook 11: spectral comparison (buoy `Efth` vs `mm.load_reconstruction_spectra()`)

### 6. Before git commit

- [ ] Run unit tests: `python -m unittest discover -s tests`
- [ ] Clear large notebook outputs if desired
- [ ] Do **not** commit `tmp/reconstruct/` caches or huge NetCDF outputs unless intentional

### Suggested commit scope

Code + config + docs:

- `utils/`, `Samoa.json`, `README.md`, `tests/`
- Notebook intro/config cells (06, 09, 10, 11)

Artefacts (`outputs/`, `logs/`) — commit only if you want frozen validation results in the repo.

---

## Tips

- **MDA sweep:** compare all `mda_candidates`; largest ≠ always best
- **PCA variance:** same override in fit, predict, reconstruct, validate
- **Period alignment:** use the same `period` in notebooks 08–11
- **Site selectors:** `resolve_sites(ds, "10m")`, `"buoy"`, `"Apia"`, or site id
- **New case:** copy `Samoa/` folder, rename JSON, update `onshore_waves` and paths

---

## Dependencies

bluemath_tk, wavespectra, SnapWave (external), xarray, geopandas, rioxarray, bluemesh2d (pip)
