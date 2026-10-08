# Netherlands — HyGloW hybrid wave downscaling case

Scaffolded with `python -m hywaves start`. This folder is **self-contained**:
the engine code (`utils/`), the pipeline notebooks, SnapWave templates, and
tests are all vendored here — it runs on any machine (laptop or HPC) without
installing the `hywaves` package itself.

**Config:** `Netherlands.json` (auto-loaded by `utils.config`).
Run notebooks with `Netherlands/` as the working directory.

## 0. Install dependencies

- The following scientific stack is expected to already be in your (conda)
  environment: `bluemath_tk`, `bluemesh2d`, `wavespectra`, `xarray`,
  `geopandas`, `rioxarray`, `pandas`, `scipy`, `matplotlib`.
- [**SnapWave**](https://github.com/danoroelvink/snapwave/) — the nearshore
  propagation model that actually runs each case. It's an external binary:
  clone that repo and compile it yourself (see its README for build
  instructions). You don't need it for notebooks 01–04, but you do from
  05 onward. Notebooks 05/08 run it as `snapwave` from your PATH
  (`launcher="default"`), or on the GeoOcean cluster as a SLURM job array
  with `wrapper.run_cases_slurm()` (loads the `snapwave` module for you).

## Getting started

1. **Fill in the config** — open `Netherlands.json` and replace the `TODO`
   values. Review the LHS `bounds` (incl. `hs`), `max_steepness` (implausible
   steep samples are dropped, so each goal keeps fewer than
   `num_samples_per_goal` cases), `output_sites_config` isobaths (`smooth_m`,
   `min_length_m`, `spacing_m` in metres), and the `metamodel` settings for
   your region — `sets` fits one PCA + GP per group of output sites (e.g.
   `buoys`, `isobaths`) on the same SnapWave cases.
2. **Area of interest** — put your AoI shapefile in `inputs/AoI_Polygons/`.
3. **Work through the notebooks in order:**

   | Step  | Notebook                                       | Purpose                        |
   | ----- | ---------------------------------------------- | ------------------------------ |
   | 01    | `01_Generate_tiff_from_GEBCO.ipynb`            | Clip GEBCO bathymetry to AoI   |
   | 02    | `02_Create_mesh_by_bluemesh2d.ipynb`           | Unstructured mesh              |
   | 03    | `03_Generate_Observations_Configuration.ipynb` | Buoy/site GeoJSON catalogue    |
   | 04    | `04_Select_Goals_by_Hindcasts.ipynb`           | Goal locations + sectors       |
   | 05    | `05_Metamodel_Cases_Generation.ipynb`          | LHS SnapWave cases per goal    |
   | 06    | `06_Validate_Metamodel.ipynb`                  | Fit + holdout validation       |
   | 07    | `07_Download_Process_WHACS.ipynb`              | Offshore hindcast zarr         |
   | 08    | `08_Dynamic_Cases_Generation.ipynb`            | Dynamic reference runs         |
   | 09    | `09_Validate_Linear_Summation.ipynb`           | Metamodel vs dynamic SnapWave  |
   | 10    | `10_Validate_Onshore_Buoys.ipynb`              | Buoy validation (all 9 buoys)  |
   | 11    | `11_Validate_Spectra.ipynb`                    | Buoy vs reconstructed spectra  |
   | 12    | `12_Final_Plots.ipynb`                         | Showcase figures for the GUI   |

   This case has a single (nearshore) buoy network, so the template's
   `10a_Validate_Offshore_Buoys.ipynb`/`10b_Validate_Onshore_Buoys.ipynb` split
   was collapsed into one `10_Validate_Onshore_Buoys.ipynb` covering all 9
   configured buoys.

   Notebooks 06, 09, and 10 each end with a save cell that writes to
   `outputs/` in the shape the GUI's Goals/Metamodel/Buoy Validation tabs
   expect (see `GUI/README.md`) — run them (in order) before checking a case
   in the portal. 10 also saves one combined scatter+time-series PNG per
   buoy (`outputs/buoy_validation/figures/<site>.png`), shown in the GUI when
   you click that buoy's star on the map.

4. **Check your progress against the case contract**:

   ```bash
   python -m hywaves validate Netherlands/            # while developing
   python -m hywaves validate Netherlands/ --publish  # portal-ready check
   ```

5. **Publish to the portal** — the GUI reads only the `portal/` bundle this
   writes (validating first); commit it, `portal/book/` is git-ignored:

   ```bash
   python -m hywaves publish Netherlands/
   ```

## Stuck, or want to see it done first?

**Samoa** (in the main HyGloW repository) is the most complete worked
example — its README documents the full pipeline, the metamodel workflow,
and the commit/re-run checklist in detail. **NewSouthWales** shows another
use case. Both are real, runnable cases you can read notebook-by-notebook
alongside your own.

## For developers

- Run the vendored unit tests from this folder: `python -m unittest discover -s tests -v`
- This folder was created from the versioned template in `hywaves/template/`.
  `.hywaves/manifest.json` records which engine snapshot it came from — treat
  `utils/` as vendored code, and if you fix a bug in it here, please also
  PR the fix to `hywaves/template/` in the main HyGloW repository so
  `python -m hywaves update` can carry it to every other case.
