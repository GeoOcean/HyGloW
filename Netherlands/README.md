# Netherlands — HyGloW hybrid wave downscaling case

Nearshore waves along the Belgian and Dutch coasts, up to the Ems, downscaled
from the WHACS offshore hindcast with the HyWaves method. This is the **current
reference case** of the repository: it runs on the latest engine (hywaves
0.11.0) and is where new features land first (SWAN as an alternative model,
wind and water-level forcing, the wind correction). All results so far were
made with **SnapWave**; SWAN is set up for this case and was compared with
SnapWave in the Xaver pilot (`SWAN_test/`).

- **Config:** `Netherlands.json` (auto-loaded by `utils.config`). Never
  hard-code paths in notebooks; add them to the JSON and `resolve()` them.
- **Working directory:** run every notebook with `Netherlands/` as cwd.
- **Environment:** conda `bluemath-312`, with `bluemath_tk` from its
  `develop` branch (`pip install "git+https://github.com/GeoOcean/BlueMath_tk.git@develop"`), on the GeoOcean cluster.

---

## How the method works here, in short

1. **10 goals** (offshore points on the open boundary, `inputs/Goals/`) each get
   their own **metamodel**: 1000 Latin Hypercube sea states (`hs, tp, dir, spr,
   wl`), of which the 629 plausible ones (`hs / L0 <= 0.06`) are run with
   SnapWave, Hs only at that goal.
2. Per goal and per **metamodel set** (`buoys`: the 38 buoys; `isobaths`: 369
   sites on the smoothed 5/10/20 m isobaths), MDA picks the training cases
   (50, 100, 200, 350 or 550), PCA compresses the site fields and a GP maps
   the offshore sea state to them. The rest of the cases are the holdout.
   Before that, the site filter removes 48 isobath sites (from hywaves
   0.12.0): 39 outside the mesh (estuaries, harbours, Wadden inlets) and 9
   that fall dry in more than 1 % of the cases (`"hs_nan_max": 0.01`); see
   `hywaves/docs/site_filter.md`.
3. Any offshore time series (WHACS, bulk or its 4 spectral partitions) is
   reconstructed by predicting every goal's contribution and **summing them
   as spectra**.
4. Optional: the **wind correction** adds the local wind sea before the sum
   (the metamodel cases have no wind).
5. **Validation:** holdout (06), against SnapWave run on the full forcing at
   every hour (the "dynamic" runs, 09), and against the buoys (10, 11).

---

## Status (2026-10-10)

| Stage | State |
| --- | --- |
| AoI, bathymetry, meshes (01–02) | done; computational mesh `inputs/GRID/Netherlands_mesh.nc` (SnapWave) / `.grd` (SWAN), 153k nodes, same nodes |
| Observations (03) | done; **38 buoys** (see *Buoys* below) |
| Goals + sectors (04) | done; 10 goals |
| Metamodel cases (05) | done; SnapWave, 629 cases/goal, 407 output sites (369 isobath + 38 buoy) |
| Wind-correction training (05b) | done; 100 cases with and 100 without wind, `K` fitted (`outputs/wind/`) |
| Metamodel fit (06) | done; 10 goals × 2 sets × 5 MDA sizes (50–550) |
| Offshore hindcast (07a) | done; WHACS 1979–2023 at the goals, bulk + 4 partitions |
| Satellite calibration (07b) | set up (2026-10-13); IMOS 1985–2020, 50 km, depth > 15 m: 2145–4549 passes per goal; raw WHACS bias −0.03 to −0.12 m, slope 0.81–0.92 (test passes); calibrated zarr not written yet |
| Dynamic runs (08) | run for **2012–2014** with ERA5 wind + CODEC water level; 2012–2013 postprocessed (`Dynamic_Cases_WHACS_2012-2013.nc`) |
| Linear summation (09) | done, `buoys` set, 2012–2013, MDA 550 |
| Buoy validation (10) | done, 2012–2013, MDA 550, with and without the wind correction (27 of the 38 buoys have data in those years) |
| Spectra (11), final plots (12) | pending (`outputs/gui_figures/` not written yet) |
| Portal | `portal/` = previous results; `portal_hs1/` = unit-Hs metamodel (hywaves 0.5.0). Re-publish now |

### Results so far (2012–2013, MDA 550, `buoys` set)

Medians over the buoys with data. Reference first; bias = estimate − reference.

| Comparison | Hs bias (m) | Hs RMSE (m) | Hs R² |
| --- | --- | --- | --- |
| 09: linear summation vs dynamic SnapWave (38 sites) | −0.08 | 0.36 | 0.66 |
| 10: WHACS at the closest goal vs buoys (27) | +0.19 | 0.41 | 0.62 |
| 10: dynamic SnapWave (ERA5 + CODEC) vs buoys | +0.19 | 0.42 | 0.45 |
| 10: bulk metamodel vs buoys | −0.10 | 0.31 | 0.78 |
| 10: bulk metamodel + wind correction vs buoys | +0.08 | 0.29 | 0.83 |
| 10: spectral metamodel vs buoys | −0.08 | 0.30 | 0.79 |
| 10: spectral metamodel + wind correction vs buoys | +0.11 | 0.28 | 0.83 |

Two caveats:
- **09 mixes two differences.** The dynamic runs have ERA5 wind and the
  reconstruction in 09 has no correction, so 09's Hs bias also contains the
  missing wind.
- **Tp is weak everywhere.** Median R² is negative for every estimate (the
  metamodel passes the offshore Tp through, `tp: "raw"`), and directions have
  ~25–35° RMSE.

---

## Inputs

| Data | Path | Notes |
| --- | --- | --- |
| AoI | `inputs/AoI_Polygons/Netherlands_AoI.shp` | |
| Bathymetry | `inputs/GRID/UGdcsm-fm_100m2coast3d_v3_era5bnd_net.tiff` | DCSM; isobath sites are drawn on it |
| Computational grid | `inputs/GRID/Netherlands_mesh.nc` / `.grd` | BlueMesh2D; the `.grd` triangles are counterclockwise (SWAN requires it). 39 isobath sites fall outside it |
| Plotting mesh | `inputs/GRID/UGdcsm-fm_100m2coast3d_v3_era5bnd_net.nc` | `paths.mesh_file`, maps only |
| Offshore waves | `inputs/WHACS/processed/Netherlands_WHACS_goal.zarr` | `(time, goal)`, 1979–2023 |
| Wind | `inputs/ERA5/raw/Netherlands_wind.zarr` | ERA5 `u10`/`v10`, hourly 1979–2023, 0.25°, covers the mesh |
| Water level | `inputs/CODEC/processed/Netherlands_CODEC_goal.zarr` | `waterlevel` at the goals, **1979–2018 only** (0 afterwards, with a warning) |
| Buoys | `inputs/Observations/Netherlands_observations.json` + `processed/<id>.nc` | built in notebook 03 |

### Buoys

`utils/buoy_archive.py` (case-specific) builds the buoy set from Paulina's
delivery (`inputs/Observations/raw/paulina/`: the original transfer zip,
`paulina.csv`, Waterinfo and CMEMS archives) plus the earlier Rijkswaterstaat
exports in `inputs/Observations/raw/`:

- stations **inside the AoI and the model grid** only (no model output
  elsewhere);
- stations within **1 km** of each other are **one buoy** (co-located sensors,
  renamed stations, the same buoy in both sources); earlier ids are kept, e.g.
  `AmelanderZeegat_1_2` is merged into `AmelanderZeegat_1_1`;
- per variable, every source is resampled to **hourly means** (directions as
  vector means) and merged **longest record first**, the others filling gaps;
  Waterinfo Hm0 over 30–1000 mHz is only a last fallback;
- processed files: `hs`, `tp` (from the peak frequency), `dir`, `spr` where
  measured; each variable's `sources` attribute lists what was merged.

Result: 38 buoys, all with Hs, 27 with Tp, 17 with direction and 12 with
spreading. The Belgian buoys inside the AoI are Hs-only from CMEMS (mostly 2020–2025). `PettenMeetraai01` comes only
from the old export; `ZandmoterBoeiNoord` was dropped (598 h).

---

## Notebooks

| Step | Notebook | What it does |
| --- | --- | --- |
| 03 | `03_Generate_Observations_Configuration.ipynb` | Builds the 38 processed buoys + the observations JSON |
| 04 | `04_Select_Goals_by_Hindcasts.ipynb` | Goals and propagation sectors |
| 05 | `05_Metamodel_Cases_Generation.ipynb` | LHS cases per goal, build / run (SLURM) / postprocess |
| 05b | `05b_Wind_Correction.ipynb` | Fetch table, wind-correction training cases (with / without wind), fit `K` |
| 06 | `06_Validate_Metamodel.ipynb` | `mm.fit_all()`, holdout metrics per goal, set and MDA size |
| 07a | `07a_Download_ERA5.ipynb` | ERA5 waves (Earth Data Hub) at the goal nodes, 1979–2023, all partitions → `ERA5` |
| 07a | `07a_Download_Process_WHACS.ipynb` | Raw WHACS → seapoint zarr (dask on SLURM) → goal zarr |
| 07b | `07b_Satellite_Calibration.ipynb` | WHACS vs IMOS altimetry at the goals; directional calibration per goal → `WHACS_cal` |
| 08 | `08_Dynamic_Cases_Generation.ipynb` | One full-forcing case per hour (WHACS + ERA5 + CODEC); forcing diagnostics |
| 09 | `09_Validate_Linear_Summation.ipynb` | Reconstruction vs dynamic runs |
| 10 | `10_Validate_Onshore_Buoys.ipynb` | Reconstruction, dynamic runs and WHACS vs the 38 buoys |
| 11 | `11_Validate_Spectra.ipynb` | Spectra at the buoys |
| 12 | `12_Final_Plots.ipynb` | Figures for the GUI (`outputs/gui_figures/`) |
| 13 | `13_Storm_Analysis.ipynb` | Storm peaks: model vs buoys (case-specific) |
| 14 | `14_TimeLag_Exploration.ipynb` | Propagation time lag in the reconstruction (case-specific) |

Notebooks 06, 09 and 10 end with a save cell writing `outputs/` in the shape
`publish` and the GUI expect. Most notebooks set `mda_n`; it must be one of the
fitted sizes (**50, 100, 200, 350, 550**).

---

## Model, forcing and wind correction (hywaves 0.11.0)

- **Model.** `simulations.model` is `"snapwave"` (current). The `swan` block is
  ready (grid `Netherlands_mesh.grd`, `templates-swan-*`, `SWAN_CASES_*`). Each
  model writes to its own cases folders; the fitted artefacts in `outputs/`
  record the model and refuse to mix with the other one.
- **Forcing of the dynamic runs.** Each hourly case gets the ERA5 field at its
  time and CODEC at every goal. SnapWave then needs its wind-fixed build
  (`snapwave/b663c81-wind1`, set in `simulations.snapwave.slurm.setup`) and
  switches to `sector = 360`, `niter = 200`; SWAN to `GEN3`. Passing
  `wind=None` in notebook 08 runs without wind (write to a separate cases
  folder then).
- **Wind correction** (explained in [`hywaves/docs/wind_correction.md`](../hywaves/docs/wind_correction.md)).
  `metamodel.wind_correction` (`enabled: false` for now,
  `wind: "forcing"` = ERA5). `mm.reconstruct(..., wind_correction=True)` adds
  `(K² − 1) · E_ws` as one more spectral contribution, where `E_ws` is the
  wind-sea energy of the goals' contributions at the site and `K` comes from
  the fitted model (fetch, depth, wind–wave angle). Output gains `wind_k`.
- **Compare like with like in 09.** Reconstruction without the correction ↔
  dynamic runs **without** wind; with the correction ↔ dynamic runs **with**
  wind. The current dynamic runs have ERA5 wind, so the uncorrected
  reconstruction differs from them by the missing wind too (a negative Hs bias,
  largest in storms).
- **Water level.** `mm.reconstruct(..., wl="forcing")` uses CODEC at each goal;
  the default is a constant 0.

### Comparing SnapWave and SWAN

```bash
# pass 1 (SnapWave): notebooks 05–12, then
python -m hywaves publish  Netherlands/
python -m hywaves snapshot Netherlands/ snapwave --note "SnapWave, hywaves 0.11.0"
mv Netherlands/outputs Netherlands/outputs_snapwave
# pass 2: "model": "swan" and vars_to_predict.tp as a PCA target, notebooks 05–12, then
python -m hywaves publish  Netherlands/
python -m hywaves snapshot Netherlands/ swan --note "SWAN, hywaves 0.11.0"
```

The GUI then lists both versions of the case.

---

## Running on the GeoOcean cluster

- Cases run as SLURM arrays: `wrapper.run_cases_slurm(**sim.slurm_kwargs(max_parallel=200))`,
  which writes `case_dirs.txt` and the array script in the cases folder.
- The cluster allows arrays up to **9999 tasks** and **20000 job records in
  total**. Larger runs (the 26,304 hourly cases of 2012–2014) go in blocks with
  an offset, and never more than two full arrays queued at once:

  ```bash
  sbatch --array=1-9999%200                              snapwave_array.sh   # lines 1–9999
  sbatch --array=1-9999%200 --export=ALL,OFFSET=9999      snapwave_array.sh   # lines 10000–19998
  sbatch --array=1-6306%200 --export=ALL,OFFSET=19998 --dependency=afterany:<first> snapwave_array.sh
  ```

  (`case_dirs.txt` has no trailing newline: count its lines in Python, not with `wc -l`.)
- `wrapper.monitor_cases(value_counts="simple")` reports what finished; rerun
  failures with `run_cases_slurm(cases_to_run=[...])`.
- Long Python jobs (GP training) on the login node: `nohup python script.py > logs/x.log 2>&1 &`.

---

## Case-specific code

| File | Role |
| --- | --- |
| `utils/buoy_archive.py` | Waterinfo + CMEMS + old exports → processed buoys (notebook 03) |
| `utils/rijkswaterstaat.py` | Reads the earlier Rijkswaterstaat raw exports |
| `utils/metamodel.py` | The template's, with GPU/RAM settings tuned to this host (12-month GPU batches, 64 GiB summation budget) |
| `SWAN_test/` | The SWAN and wind pilot (Xaver, December 2013) that led to 0.11.0; see its README |

Everything else in `utils/` is the vendored template engine: fix bugs in
`hywaves/template/utils/` and bring them here with `python -m hywaves update
Netherlands/` (it keeps the locally tuned `metamodel.py`).

## Checks

```bash
python -m unittest discover -s tests -v          # from Netherlands/
python -m hywaves validate Netherlands/          # from the repository root
```
