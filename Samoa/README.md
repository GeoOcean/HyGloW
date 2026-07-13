# Samoa — hybrid wave downscaling case

Samoa is a **hybrid downscaling** workflow for nearshore waves around the Samoan
archipelago. A small set of **SnapWave** numerical simulations is combined with
a **GPR metamodel** (MDA → PCA → Gaussian Process) so that full historical
offshore hindcasts can be reconstructed without running the model hundreds of
thousands of times.

All case-specific paths and parameters live in **`Samoa.json`**. Python helpers
are in **`utils/`**; each stage has a numbered Jupyter notebook.

Technical backlog (refactors, deferred items): **`TODO.md`**.

---

## Quick start

1. Work from the `Samoa/` directory (notebooks assume this as the working directory).
2. Run notebooks **in order** (see pipeline below).
3. Config auto-discovery: `utils.config` loads `<case-folder-name>.json` from the
   current case directory (e.g. `Samoa/Samoa.json`).
4. After editing `Samoa.json`, restart the kernel or call `utils.config.reload()`.
5. SnapWave cases are heavy — use `nohup` or a cluster for steps 05 and 08.

**Imports** (consistent across validation notebooks):

```python
from utils.model import metamodel as mm
from utils.evaluation import validation as val
from utils.config import load_goals, METAMODEL, validate_metamodel_config
```

`validate_metamodel_config()` is also called automatically inside `mm.fit()` and
`val.validate()`.

---

## Pipeline (notebooks)

| Step | Notebook | Purpose |
|------|----------|---------|
| 01 | `01_Generate_tif_From_GEBCO.ipynb` | Download GEBCO, clip to AoI, write topobathy GeoTIFF |
| 02 | `02_Create_Mesh_By_pymesh2d.ipynb` | Gradient-limited mesh sizing → SnapWave mesh |
| 03 | `03_Generate_Observations_Configuration.ipynb` | Onshore sites GeoJSON (buoys, gauges, …) |
| 04 | `04_Select_Goals_by_Hindcasts.ipynb` | Place metamodel goals in AoI buffer ring → `Samoa_Goals.json` |
| 05 | `05_Metamodel_Cases_Generation.ipynb` | LHS propagation cases (500 per goal) → per-goal NetCDF |
| 06 | `06_Validate_Metamodel.ipynb` | Fit MDA/PCA/GP, holdout validation, MDA sweep plots |
| 07 | `07_Download_Process_WHACS.ipynb` | Offshore WHACS → processed zarr `(time, goal)` |
| 08 | `08_Dynamic_Cases_Generation.ipynb` | SnapWave forced by offshore time series (reference runs) |
| 09 | `09_Validate_Linear_Summation.ipynb` | Spectral metamodel vs dynamic SnapWave (linear sum) |
| 10 | `10_Validate_Buoy_Observations.ipynb` | Bulk/spectral/dynamic/offshore vs buoy records |

### Data flow

```
GEBCO + AoI  →  mesh  →  output sites (isobaths, mesh, buoys)
                              ↓
Goals (04)  →  LHS cases (05)  →  train metamodel (06)
                              ↓
WHACS (07)  →  dynamic cases (08)  →  hindcast validation (09, 10)
                              ↓
                    mm.reconstruct()  →  nearshore time series
```

---

## Metamodel training (notebook 06)

### Fit workflow

```python
mm.fit_all(gp_epochs=1500)                    # all goals × all mda_candidates
mm.fit("1", force_retrain=True)               # one goal, re-fit everything
mm.fit("1", candidates=[450], force_retrain=True)  # one goal, one MDA size
```

**`force_retrain=True`** re-fits MDA (if needed) and all PCA/GP models for that
goal, overwriting pickles and updating `split.json` → `fit_by_mda`. Use after
changing `Samoa.json` metamodel settings or when upgrading fit-summary fields.

**Validation** loads saved models only (no refitting):

```python
metrics = val.sweep("1")                      # CSV per goal
result = val.validate("1", n=450)             # metrics + plot helpers
result.plot("10m")                            # map + 2×2 scatters per group
result.plot_all()
```

### Training stages (per goal)

1. **Site filter** — drop sites with unrealistic LHS extremes (`site_filter` in config).
2. **MDA** — `num_centers = mda_max_points()` centroids on forcing `(tp, dir, spr, wl)`.
3. **Split** — centroid case indices = **train**; remaining LHS cases = **holdout test**.
4. **PCA** — per variable on stacked sites (`hs`, `dir` via u/v, `spr`); `tp` is **raw** (GP passthrough).
5. **GP** — MDA centroids → stacked principal components.

### Modelling vs validation variables

| Variable | Metamodel (`vars_to_predict`) | Holdout metrics (`metrics_variables`) |
|----------|-------------------------------|---------------------------------------|
| `hs` | PCA (`hs`) | yes |
| `tp` | **raw** (passthrough) | yes |
| `dir` | PCA (`dir_u`, `dir_v`) | yes |
| `spr` | PCA (`spr`) | yes |

Buoy validation (notebook 10) typically scores `hs`, `tp`, `dir` only — observations
may not include directional spread.

### Outputs (`outputs/goal_XX/`)

| Artefact | Contents |
|----------|----------|
| `mda.pkl` | Fitted MDA object |
| `split.json` | Train/test indices, site filter, fit summaries (see below) |
| `pca_{n}/` | Per-variable PCA pickles |
| `gp_{n}/` | GP pickle |
| `validation_metrics.csv` | Holdout sweep (`val.sweep`) |

### `split.json` structure

Written at MDA fit; extended at PCA/GP fit:

```json
{
  "train_idx": […],
  "test_idx": […],
  "num_centers": 500,
  "removed_sites": ["…"],
  "site_filter": { "hs_max": 10.0, "tp_min": 0.5 },
  "pca_nan_policy": {
    "nan_threshold_to_drop": { "hs": 0.99, … },
    "value_to_replace_nans": { "hs": 0.0, … }
  },
  "fit_by_mda": {
    "450": {
      "n_cases_requested": 450,
      "n_cases_used": 450,
      "n_sites_before_pca": 727,
      "pca_sites_kept": { "hs": ["…"], "dir": ["…"], "spr": ["…"] },
      "pca_sites_dropped": { "hs": 59, "dir": 59, "spr": 59 },
      "pca_variances": { "hs": 0.99, "dir": 0.99, "spr": 0.99 },
      "n_gp_components": 42
    }
  }
}
```

Inspect after fit:

```python
mm.load_fit_summary("1", 450)
mm.load_split("1")   # train_idx, test_idx only
```

Logs: `logs/metamodel.log`, `logs/validation.log`.

---

## Configuration (`Samoa.json`)

`utils.config` discovers `<CASE_DIR>/<CASE_DIR.name>.json`.

| Section | Role |
|---------|------|
| `paths` | AoI, topobathy, mesh |
| `offshore_waves` | Processed hindcast zarrs |
| `onshore_waves` | Observations GeoJSON |
| `simulations` | LHS bounds, templates, case dirs, `output_sites_config` |
| `metamodel` | MDA, PCA targets, site filter, preprocessing, validation metrics |

### Key metamodel fields

| Field | Role |
|-------|------|
| `mda_candidates` | Train sizes to try, e.g. `[50, 100, 150, 250, 450]`; largest = `mda_max_points()` |
| `mda_inputs` | Forcing passed to MDA and GP: `tp`, `dir`, `spr`, `wl` |
| `mda_directional_variables` | Circular handling for `dir` in MDA |
| `vars_to_predict` | What the GP predicts (PCA or raw) — see table above |
| `site_filter` | Pre-MDA numerical consistency rules |
| `pca_preprocessing` | PCA NaN thresholds and fill values for dry sites |
| `validation.metrics_variables` | Variables scored in holdout validation |
| `paths.pca_dir`, `paths.gp_dir` | Artefact directory templates |

---

## Data quality & NaN handling

> **Two separate mechanisms** at different stages, both in `metamodel` config.

### 1. Pre-PCA site filter (`site_filter`)

Applied **before** MDA and PCA. Catches numerical inconsistencies from extreme
LHS combinations (not dry-bed physics).

Current Samoa rules:

| Key | Rule | Rationale |
|-----|------|-----------|
| `hs_max: 10.0` | drop site if **any** case has Hs > 10 m | unrealistic height at a node |
| `tp_min: 0.5` | drop site if **any** case has Tp < 0.5 s | unrealistic period at a node |

**Extending:** add `{variable}_max` or `{variable}_min` keys (e.g. `spr_max`).
`detect_removed_sites()` applies every rule automatically. Removed sites →
`split.json` → `removed_sites`.

### 2. Post-filter PCA NaN handling (`pca_preprocessing`)

Applied **during** PCA fit. Addresses **dry / low water level** cases where
SnapWave returns NaN at shallow mesh nodes.

BlueMath PCA (in order):

1. **`nan_threshold_to_drop`** — drop **site columns** where too large a fraction
   of training cases are NaN (default `0.99` per stacked variable).
2. **`value_to_replace_nans`** — fill **remaining** NaNs with `0.0` so sklearn PCA runs.

This is an intentional compromise: NaNs are mostly site-level (dry nodes), not
whole-case failures. Zero-fill after site-drop encodes calm/dry nodes as zeros in
the PCA input space.

**Sites kept** per PCA variable are persisted in `fit_by_mda[n].pca_sites_kept`.
Prediction currently returns all sites after `site_filter` (PCA-kept subset not
enforced at predict time — under review after refit; see `TODO.md`).

---

## Holdout validation plots

`val.validate()` produces a map plus **2×2 scatter panels** per site group:
`hs`, `tp`, `dir`, `spr`. KDE density colouring is used when possible; falls
back to plain scatter on small or degenerate samples.

Configured metrics missing from estimate or reference trigger a **warning** in
the validation log.

---

## Step-by-step detail

### 01 — Topobathy GeoTIFF

Produces `inputs/GEBCO/Samoa_GEBCO.tiff` (`paths.topobathy_file`) and the AoI
shapefile used everywhere else.

### 02 — Computational mesh

`utils.mesh.mesh_sizing.precalc_hfuns` → pymesh2d → `inputs/GRID/Samoa_mesh.nc`.

### 03 — Observations configuration

Onshore sites GeoJSON (`onshore_waves`). Buoys included as SnapWave output sites
when `include_buoys: true`.

### 04 — Metamodel goals

`utils.geometry.sectors` places targets in the inland buffer ring, snaps to
hindcast nodes, saves `inputs/Goals/Samoa_Goals.json` with **nautical** direction
sectors (`utils/directions.py`).

### 05 — Metamodel LHS cases

500 Latin Hypercube samples per goal of `(tp, dir, spr, wl)` within `simulations.bounds`.
Output: `SnapWave_CASES_Metamodel/output_sites_{goal}.nc`.

### 07 — Offshore WHACS

Processed zarr: `inputs/WHACS/processed/Samoa_WHACS.zarr`, dims `(time, goal)`.
Open via `utils.config.open_offshore_waves("WHACS")`.

### 08 — Dynamic SnapWave cases

Full offshore time series forcing. Reference runs for hindcast validation.

### 09 — Linear summation validation

Spectral `mm.reconstruct` (partitions 0–3, wavespectra sum) vs dynamic SnapWave.
Comparison variables: `hs`, `tp`, `dir` (bulk reconstruction does not output `spr`).

### 10 — Buoy validation

Compares buoy obs to offshore hindcast, bulk/spectral metamodel, and dynamic SnapWave.
Typically `variables = ["hs", "tp", "dir"]`. Align **period** across all sources.

---

## Python modules (`utils/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | Load `Samoa.json`, `resolve()`, `validate_metamodel_config()`, offshore zarr |
| `directions.py` | Nautical direction and sector helpers |
| `mesh.mesh_sizing` | Mesh sizing (gradient-limited h) |
| `geometry.sectors` | Goal placement, buffer ring |
| `geometry.output_sites` | Isobaths + mesh + buoys → output sites |
| `geometry.sites` | Site metadata, groups (`10m`, `Apia`, buoys) |
| `simulation.lhs_cases` | LHS case generation |
| `simulation.snapwave_wrappers` | SnapWave build/run/postprocess |
| `model.metamodel` | Fit, predict, `reconstruct`, `load_fit_summary` |
| `evaluation.validation` | Holdout metrics/plots, buoy comparison |
| `viz.snapwave_case_plots` | Post-run case diagnostics (hs/tp/dir + map) |

---

## Hindcast reconstruction

```python
ds = mm.reconstruct(
    "2020-05",
    n=450,
    spectral=True,
    offshore_name="WHACS",
    sites="10m",
    pca_variance={"hs": 0.99, "dir": 0.99, "spr": 0.99},
)
# Bulk output: hs, tp, dir on (time, sites)
```

- Use the same `pca_variance` override as at fit time.
- After retraining models, use `recompute=True` if cached reconstruction exists.
- Bulk reconstruction returns **`hs`, `tp`, `dir` only**; `spr` is used inside
  spectral construction but not in summed bulk stats.

---

## Tips

- **Working directory** — `Samoa/` as cwd for `utils` imports.
- **Config edits** — `utils.config.reload()` or restart kernel before `mm.fit()`.
- **MDA sweep** — `val.sweep` / compare candidates in `mda_candidates`; larger
  is not always better.
- **PCA variance** — same override in `fit`, `predict`, `reconstruct`, `validate`.
- **Site selectors** — `mm.resolve_sites(ds, "10m")`, `"buoy"`, `"Apia"`, or site id.
- **Refit** — after changing metamodel config, `mm.fit_all(force_retrain=True)`.
- **Git** — clear large notebook outputs before committing.

---

## Outputs layout

```
Samoa/
├── inputs/
├── templates-metamodel/ / templates-dynamic/
├── SnapWave_CASES_Metamodel/ / SnapWave_CASES_Dynamic/
├── outputs/goal_01/ … goal_16/
│   ├── mda.pkl
│   ├── split.json          # train/test, site filter, fit_by_mda
│   ├── validation_metrics.csv
│   ├── pca_450/
│   └── gp_450/
├── logs/
│   ├── metamodel.log
│   └── validation.log
├── README.md               # this file
└── TODO.md                 # active backlog
```

---

## Dependencies

- **bluemath_tk** — MDA, PCA, GP, metrics, SnapWave wrappers
- **wavespectra** — linear partition summation in `mm.reconstruct`
- **SnapWave** — numerical wave model (external)
- **xarray**, **geopandas**, **rioxarray**, **pymesh2d**

Conda env used in development: `bluemath-311-dev`.
