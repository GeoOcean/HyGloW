# Samoa — hybrid wave downscaling case

Samoa is a **hybrid downscaling** workflow for nearshore waves around the Samoan
archipelago. A small set of **SnapWave** numerical simulations is combined with
a **GPR metamodel** (MDA → PCA → Gaussian Process) so that full historical
offshore hindcasts can be reconstructed without running the model hundreds of
thousands of times.

All case-specific paths and parameters live in **`Samoa.json`**. Python helpers
are in **`utils/`**; each stage has a numbered Jupyter notebook.

---

## Quick start

1. Work from the `Samoa/` directory (notebooks assume this as the working directory).
2. Run notebooks **in order** (see pipeline below).
3. Config auto-discovery: `utils.config` loads `<case-folder-name>.json` from the
   current case directory (e.g. `Samoa/Samoa.json`, `NewSouthWales/NewSouthWales.json`).
4. After editing your case JSON, restart the kernel or call `utils.config.reload()`.
5. SnapWave cases are heavy — use `nohup` or a cluster for steps 05 and 08.

**Imports** (consistent across validation notebooks):

```python
from utils.model import metamodel as mm
from utils.evaluation import validation as val
from utils.config import resolve, load_goals, METAMODEL, SIMULATIONS
```

---

## Pipeline (notebooks)

| Step | Notebook | Purpose |
|------|----------|---------|
| 01 | `01_Generate_tif_From_GEBCO.ipynb` | Download GEBCO, clip to AoI, write topobathy GeoTIFF |
| 02 | `02_Create_Mesh_By_pymesh2d.ipynb` | Gradient-limited mesh sizing (`utils.mesh.mesh_sizing`) → SnapWave mesh |
| 03 | `03_Generate_Observations_Configuration.ipynb` | Onshore sites GeoJSON (buoys, gauges, …) |
| 04 | `04_Select_Goals_by_Hindcasts.ipynb` | Place metamodel goals in AoI buffer ring → `Samoa_goals.json` |
| 05 | `05_Metamodel_Cases_Generation.ipynb` | LHS propagation cases (500 per goal) → per-goal NetCDF |
| 06 | `06_Validate_Metamodel.ipynb` | Fit MDA/PCA/GP, holdout validation, MDA sweep plots |
| 07 | `07_Download_Process_WHACS.ipynb` | Offshore WHACS → processed zarr `(time, goal)` |
| 08 | `08_Dynamic_Cases_Generation.ipynb` | SnapWave forced by offshore time series (reference runs) |
| 09 | `09_Validate_Linear_Summation.ipynb` | Spectral metamodel vs dynamic SnapWave (linear sum) |
| 10 | `10_Validate_Buoy_Observations.ipynb` | Bulk/spectral/dynamic/offshore vs Lauri buoy records |

### Data flow (conceptual)

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

## Step-by-step detail

### 01 — Topobathy GeoTIFF

Produces `inputs/GRID/Samoa_300m.tif` (path in `Samoa.json` → `paths.topobathy_file`)
and the AoI shapefile used everywhere else.

### 02 — Computational mesh

Uses `utils.mesh.mesh_sizing.precalc_hfuns` to limit bathymetry gradients, then pymesh2d
to build `inputs/GRID/Samoa_mesh.nc`. The mesh resolution drives SnapWave cost
for all later runs.

### 03 — Observations configuration

Builds `inputs/LauriBuoys/LauriBuoys_coordinates.json` (or your chosen path in
`onshore_waves`). Buoys listed here are automatically included as SnapWave output
sites when `include_buoys: true` in `output_sites_config`.

### 04 — Metamodel goals

`utils.geometry.sectors` places random targets in the inland buffer ring, snaps each goal
to hindcast catalog nodes (WHACS, etc.), and saves:

- `inputs/goals/Samoa_goals.json` — targets, **nautical** direction sectors (`[left, right]` clockwise from N; see `utils/directions.py`), hindcast snaps
- `inputs/goals/Samoa_goals.gpkg` — QGIS preview

Goals define **where** offshore forcing enters the domain (boundary nodes + sectors).

### 05 — Metamodel LHS cases

For each goal, `utils.simulation.lhs_cases.generate_cases` builds 500 Latin Hypercube samples
of `(tp, dir, spr, wl)` within bounds in `Samoa.json`. Wave direction `dir` is sampled
in nautical degrees inside each goal's sector (`angles` in the goals JSON), then
normalized to `[0, 360)`. `SnapWaveMetaModelWrapper` runs propagation and writes
`SnapWave_CASES_Metamodel/output_sites_{goal}.nc`.

These cases are the **training library** for the GPR metamodel.

### 06 — Fit and holdout validation

```python
mm.fit_all(gp_epochs=1500)           # train all goals × MDA sizes
metrics = val.sweep("10")            # CSV per goal
result = val.validate("10", n=450)   # plots for one MDA size
result.plot("10m")
```

Outputs under `outputs/goal_XX/`:

- `mda.pkl`, `split.json` — MDA object and train/test case indices
- `pca_{n}/`, `gp_{n}/` — fitted models per MDA candidate size
- `validation_metrics.csv` — sweep results

**Site filter** (`metamodel.site_filter`): sites with unrealistic LHS extremes
(e.g. Hs > 10 m) are dropped before PCA; list stored in `split.json`.

### 07 — Offshore WHACS

Downloads/processes bulk + partition spectra onto goal nodes. Result:

`inputs/WHACS/processed/Samoa_WHACS.zarr` with dims `(time, goal)` and variables
`hs/tp/dir/spr` plus `phs0`–`phs3`, etc.

Open via `utils.config.open_offshore_waves("WHACS")`.

### 08 — Dynamic SnapWave cases

`utils.simulation.snapwave_wrappers.SnapWaveDynamicModelWrapper` forces SnapWave with the full offshore time series
for a chosen `period`. Output: joined NetCDF under `SnapWave_CASES_Dynamic/`
(e.g. `Dynamic_Cases_WHACS_fullaso.nc`).

Used as **reference** for hindcast validation (expensive but exact).

### 09 — Linear summation validation

Compares **spectral metamodel reconstruction** (`mm.reconstruct`, partitions 0–3
summed with wavespectra) against the dynamic run for the same period and sites.

### 10 — Buoy validation

Compares buoy observations to:

- Closest-goal offshore hindcast
- Bulk metamodel (`spectral=False`)
- Spectral metamodel (`spectral=True`)
- Dynamic SnapWave output

Use the **same `period`** for buoy data, reconstruction, and dynamic NC.

---

## Configuration (`Samoa.json`)

`utils.config` discovers config as `<CASE_DIR>/<CASE_DIR.name>.json`. In this
case folder that is `Samoa/Samoa.json`.

| Section | Role |
|---------|------|
| `paths` | AoI, topobathy, mesh |
| `offshore_waves` | Processed hindcast zarrs |
| `onshore_waves` | Observations GeoJSON |
| `simulations` | LHS bounds, templates, case dirs, **output_sites_config** |
| `metamodel` | MDA sizes, PCA targets, site filter, validation metrics |

Key metamodel fields:

- **`mda_candidates`** — train sizes to try (largest = `mda_max_points()`)
- **`vars_to_predict`** — `hs` and `dir` via PCA; `tp` and `spr` passed through
- **`site_filter`** — remove unstable sites before fitting

---

## Python modules (`utils/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | Load `Samoa.json`, `resolve()`, offshore zarr I/O |
| `directions.py` | Nautical direction and sector convention helpers |
| `mesh.mesh_sizing` | Mesh sizing function (gradient-limited h) |
| `geometry.sectors` | Goal placement, buffer ring, catalog snaps |
| `observation.onshore_sites` | Onshore GeoJSON I/O |
| `geometry.output_sites` | Isobaths + mesh nodes + buoys → output sites table |
| `simulation.lhs_cases` | LHS case generation |
| `simulation.snapwave_wrappers` | SnapWave case build/run/postprocess |
| `simulation.offshore_forcing` | Offshore → dynamic template parameters |
| `geometry.sites` | Site metadata, groups (`10m`, `Apia`, buoys), `resolve_sites` |
| `model.metamodel` | Fit, predict, `reconstruct` (hindcast) |
| `evaluation.validation` | Metrics, holdout/buoy comparison, plots |
| `viz.case_plots` | Case maps and validation site maps |

---

## Tips

- **Working directory** — run notebooks with `Samoa/` as cwd so `utils` imports work.
- **Goal consistency** — `Samoa_goals.json` goal ids must match the `goal` dimension
  in the WHACS zarr and in `output_sites_{goal}.nc`.
- **MDA size** — use `val.sweep` / `aggregate_sweep` to pick `n`; larger is not
  always better (diminishing returns vs training cost).
- **PCA variance** — if you override at fit time (`pca_variance={...}`), use the
  same override in `predict`, `reconstruct`, and validation.
- **Period alignment** — notebook 10 will silently mis-compare if buoy, dynamic,
  and reconstruction periods differ.
- **Site selectors** — `mm.resolve_sites(ds, "10m")`, `"buoy"`, `"Apia"`, or a
  site id; see `utils.geometry.sites.metric_groups`.
- **Clear notebook outputs** before committing large validation notebooks (keeps
  git diffs small).

---

## Things worth a closer look

These are not blockers — suggestions from the code review:

1. **Output sites spacing** — isobath `spacing: 0.1`° in `Samoa.json` can produce
   many points per depth; check SnapWave cost and whether coarser spacing is enough
   for validation.

2. **`geometry.output_sites.get_isobath_points`** — uses Matplotlib contouring (headless-safe).
   For very high-resolution rasters, consider `skimage` or `contourpy` if profiling
   shows a bottleneck (unlikely compared to SnapWave itself).

3. **SnapWave launcher path** — hard-coded in `simulation/snapwave_wrappers.py`
   (`available_launchers`); consider moving to `Samoa.json` for portability.

4. **`Samoa_goals copy.json`** — duplicate goals file in `inputs/goals/`; remove or
   rename to avoid confusion.

5. **Notebook 06 last cell** — comment says “10m isobath” but query filters
   `site_label == 'Apia'`; align comment or query when reviewing sweep results.

6. **Dynamic case storage** — full-period dynamic runs grow quickly; document which
   `period` slices you keep for validation vs archive.

7. **Enclosure files** — `SnapWaveModelWrapper.build_case` writes a 4-corner
   ``enclosure.txt`` from the AoI exterior buffered by 10 km (``enclosure_buffer_m``).
   A static ``templates-*/enclosure.txt`` is optional and will be overwritten.

---

## Outputs layout (typical)

```
Samoa/
├── inputs/          # Static data, goals, WHACS zarr, buoys
├── templates-metamodel/
├── templates-dynamic/
├── SnapWave_CASES_Metamodel/
├── SnapWave_CASES_Dynamic/
├── outputs/goal_01/ … goal_11/
│   ├── mda.pkl
│   ├── split.json
│   ├── validation_metrics.csv
│   ├── pca_450/
│   └── gp_450/
└── logs/
    ├── metamodel.log
    └── validation.log
```

---

## Reproducing hindcast reconstruction in code

```python
from utils.model import metamodel as mm

ds = mm.reconstruct(
    "2020-05",           # period for offshore zarr
    n=450,               # MDA train size (models must exist)
    spectral=True,       # WHACS partitions 0–3 + wavespectra sum
    offshore_name="WHACS",
    sites="10m",         # or "buoy", site id, etc.
    pca_variance={"hs": 0.99, "dir": 0.99},
)
# ds: (time, sites) with hs, tp, dir
```

---

## Dependencies

- **bluemath_tk** — MDA, PCA, GP, metrics, SnapWave wrappers
- **wavespectra** — linear partition summation in `mm.reconstruct`
- **SnapWave** — numerical wave model (external install)
- **xarray**, **geopandas**, **rioxarray**, **pymesh2d** (mesh notebook)

Ensure the Python environment used in notebooks has `bluemath_tk` and can import
`utils` from the `Samoa/` directory.
