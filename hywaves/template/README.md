# __CASE_NAME__ — HyGloW hybrid wave downscaling case

Scaffolded with `python -m hywaves start`. This folder is **self-contained**:
the engine code (`utils/`), the pipeline notebooks, SnapWave and SWAN
templates, and
tests are all vendored here — it runs on any machine (laptop or HPC) without
installing the `hywaves` package itself.

**Config:** `__CASE_NAME__.json` (auto-loaded by `utils.config`).
Run notebooks with `__CASE_NAME__/` as the working directory.

## 0. Install dependencies

- The following scientific stack is expected to already be in your (conda)
  environment: `bluemesh2d`, `wavespectra`, `xarray`, `geopandas`,
  `rioxarray`, `pandas`, `scipy`, `matplotlib`, and `bluemath_tk` from its
  `develop` branch (`pip install "git+https://github.com/GeoOcean/BlueMath_tk.git@develop"`).
- The nearshore propagation model that actually runs each case, chosen with
  `simulations.model`: [**SnapWave**](https://github.com/danoroelvink/snapwave/)
  (`"snapwave"`, UGRID `.nc` mesh) or **SWAN** (`"swan"`, ADCIRC `.grd` grid
  with counterclockwise triangles). Both are external binaries. You don't need
  them for notebooks 01–04, but you do from 05 onward. Notebooks run them from
  your PATH (`launcher="default"`), or on the GeoOcean cluster as SLURM job
  arrays with `wrapper.run_cases_slurm(**sim.slurm_kwargs())` (module and
  resources from `simulations.<model>.slurm`). Runs with wind need SnapWave's
  wind-fixed build (`snapwave/b663c81-wind1`, the template default).

## Getting started

1. **Fill in the config** — open `__CASE_NAME__.json` and replace the `TODO`
   values. Review the LHS `bounds` (incl. `hs`), `max_steepness` (implausible
   steep samples are dropped, so each goal keeps fewer than
   `num_samples_per_goal` cases), `output_sites_config` isobaths (`smooth_m`,
   `min_length_m`, `spacing_m` in metres), and the `metamodel` settings for
   your region — `sets` fits one PCA + GP per group of output sites (e.g.
   `buoys`, `isobaths`) on the same model cases. `simulations.model` picks
   SnapWave or SWAN and `simulations.<model>` its grid, templates
   (`templates-<model>-metamodel/`, `templates-<model>-dynamic/`), cases
   folders and SLURM options; with SWAN, predict `tp` (a PCA target) instead
   of `"raw"`. Optional `forcing` (wind, e.g. ERA5; water level, e.g. CODEC)
   feeds the dynamic runs and `reconstruct(wl="forcing")`. `site_filter`
   decides which output sites are trained: keep `"hs_nan_max": 0.01` so sites
   outside the mesh or often dry are removed (see the hywaves
   `docs/site_filter.md`).
2. **Area of interest** — put your AoI shapefile in `inputs/AoI_Polygons/`.
3. **Work through the notebooks in order:**

   | Step  | Notebook                                       | Purpose                        |
   | ----- | ---------------------------------------------- | ------------------------------ |
   | 01    | `01_Generate_tiff_from_GEBCO.ipynb`            | Clip GEBCO bathymetry to AoI   |
   | 02    | `02_Create_mesh_by_bluemesh2d.ipynb`           | Unstructured mesh              |
   | 03    | `03_Generate_Observations_Configuration.ipynb` | Buoy/site GeoJSON catalogue    |
   | 04    | `04_Select_Goals_by_Hindcasts.ipynb`           | Goal locations + sectors       |
   | 05    | `05_Metamodel_Cases_Generation.ipynb`          | LHS model cases per goal       |
   | 05b   | `05b_Wind_Correction.ipynb`                    | Optional: wind correction (K)  |
   | 06    | `06_Validate_Metamodel.ipynb`                  | Fit + holdout validation       |
   | 07    | `07_Download_Process_WHACS.ipynb`              | Offshore hindcast zarr         |
   | 08    | `08_Dynamic_Cases_Generation.ipynb`            | Dynamic reference runs         |
   | 09    | `09_Validate_Linear_Summation.ipynb`           | Metamodel vs dynamic runs      |
   | 10a   | `10a_Validate_Offshore_Buoys.ipynb`            | Offshore buoy validation       |
   | 10b   | `10b_Validate_Onshore_Buoys.ipynb`             | Onshore buoy validation        |
   | 11    | `11_Validate_Spectra.ipynb`                    | Buoy vs reconstructed spectra  |
   | 12    | `12_Final_Plots.ipynb`                         | Showcase figures for the GUI   |

   Notebooks 06, 09, and 10a/10b each end with a save cell that writes to
   `outputs/` in the shape the GUI's Goals/Metamodel/Buoy Validation tabs
   expect (see `GUI/README.md`) — run them (in order) before checking a case
   in the portal. 10a/10b also save one combined scatter+time-series PNG per
   buoy (`outputs/buoy_validation/figures/<site>.png`), shown in the GUI when
   you click that buoy's star on the map.

4. **Check your progress against the case contract**:

   ```bash
   python -m hywaves validate __CASE_NAME__/            # while developing
   python -m hywaves validate __CASE_NAME__/ --publish  # portal-ready check
   ```

5. **Publish to the portal.** The GUI never reads `inputs/` or `outputs/`
   directly — it reads a small bundle that `publish` builds (validating
   first) in `portal/`:

   ```bash
   python -m hywaves publish __CASE_NAME__/
   ```

   Commit `__CASE_NAME__/portal/` (a few MB: configs, merged metrics,
   pre-computed map layers, web-sized figures), but git-ignore
   `__CASE_NAME__/portal/book/`: the case's report, a Jupyter Book of this
   README and the numbered notebooks that `publish` builds (needs
   `pip install jupyter-book`; `--no-book` skips it; `--url-base` matches a
   GUI served under a prefix). It is deployed separately.
   Re-run `publish` after re-running notebooks; `validate` warns
   (`portal.stale`) whenever the bundle has fallen behind.

## Model, wind and water level

- **SnapWave or SWAN** (both ready; the published reference results use
  SnapWave). Set `simulations.model`; everything else (wrappers,
  templates, grids, cases folders, case plots) follows from the
  `simulations.<model>` block, through `utils.simulation.HyWavesWrapperSetup`.
  Both models write the same `output_sites_{goal}.nc`, so 06 onward is the
  same. SWAN changes Tp nearshore: predict it (`vars_to_predict.tp` as a PCA
  target) instead of `"raw"`.
- **Wind and water level (optional).** With `forcing.wind` and
  `forcing.water_level` in the JSON, notebook 08 gives every hourly case the
  wind field and the water level at the goals, and shows them first
  (`setup.plot_dynamic_forcing`). `wind=None` in 08 runs without wind; then
  write to a separate cases folder.
- **Wind correction (optional, notebook 05b).** The metamodel cases have no
  wind. 05b computes the fetch at every output site, runs ~100 full-domain
  cases with and without wind, and fits the growth factor `K`;
  `mm.reconstruct(..., wind_correction=True)` then adds the local wind sea.
  In 09, compare like with like: without the correction against dynamic runs
  without wind, with it against dynamic runs with wind. The method, its
  assumptions and its results are explained in `hywaves/docs/wind_correction.md`
  in the main HyGloW repository.
- **Comparing models.** Run the pipeline with one model, `publish`, keep it
  with `python -m hywaves snapshot __CASE_NAME__/ <tag>`, move `outputs/`
  aside, switch `simulations.model` and run again. The GUI lists both
  versions.

## Stuck, or want to see it done first?

**Netherlands** (in the main HyGloW repository) is the reference case on the
current engine, including SWAN, wind and water level. **NewSouthWales** and
**Samoa** are complete worked examples on an older engine; their READMEs
document the metamodel workflow and the re-run checklist in detail. All are
real, runnable cases you can read notebook-by-notebook alongside your own.

## For developers

- Run the vendored unit tests from this folder: `python -m unittest discover -s tests -v`
- This folder was created from the versioned template in `hywaves/template/`.
  `.hywaves/manifest.json` records which engine snapshot it came from — treat
  `utils/` as vendored code, and if you fix a bug in it here, please also
  PR the fix to `hywaves/template/` in the main HyGloW repository so
  `python -m hywaves update` can carry it to every other case.
