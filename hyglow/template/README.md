# __CASE_NAME__ — HyGloW hybrid wave downscaling case

Scaffolded with `python -m hyglow start`. This folder is **self-contained**:
the engine code (`utils/`), the pipeline notebooks, SnapWave templates, and
tests are all vendored here — it runs on any machine (laptop or HPC) without
installing the `hyglow` package itself.

**Config:** `__CASE_NAME__.json` (auto-loaded by `utils.config`).
Run notebooks with `__CASE_NAME__/` as the working directory.

## 0. Install dependencies

- `pip install -r requirements.txt` — installs `bluemesh2d` (the mesher used
  by `02_Create_mesh_by_bluemesh2d.ipynb`; it's pip-only, not on conda-forge).
- The rest of the scientific stack is expected to already be in your (conda)
  environment: `bluemath_tk`, `wavespectra`, `xarray`, `geopandas`,
  `rioxarray`, `pandas`, `scipy`, `matplotlib`
  (`conda install -c conda-forge bluemath-tk` covers the first one).
- [**SnapWave**](https://github.com/danoroelvink/snapwave/) — the nearshore
  propagation model that actually runs each case. It's an external binary:
  clone that repo and compile it yourself (see its README for build
  instructions). You don't need it for notebooks 01–04, but you do from
  05 onward.

## Getting started

1. **Fill in the config** — open `__CASE_NAME__.json` and replace the `TODO`
   values. Review the LHS `bounds`, `output_sites_config` isobaths, and
   `metamodel` settings for your region (the defaults come from the Samoa
   reference case).
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
   | 09–11 | Validation notebooks                           | Reconstruct vs dynamic / buoys |

4. **Check your progress against the case contract** (this is also the
   publish gate — the portal only ever reads contract files):

   ```bash
   python -m hyglow validate __CASE_NAME__/            # while developing
   python -m hyglow validate __CASE_NAME__/ --publish  # portal-ready check
   ```

## Stuck, or want to see it done first?

**Samoa** (in the main HyGloW repository) is the most complete worked
example — its README documents the full pipeline, the metamodel workflow,
and the commit/re-run checklist in detail. **NewSouthWales** shows a case
mid-pipeline (mesh, goals, and observations done; fit and validation
pending). Both are real, runnable cases you can read notebook-by-notebook
alongside your own.

## For developers

- Run the vendored unit tests from this folder: `python -m unittest discover -s tests -v`
- This folder was created from the versioned template in `hyglow/template/`.
  `.hyglow/manifest.json` records which engine snapshot it came from — treat
  `utils/` as vendored code, and if you fix a bug in it here, please also
  PR the fix to `hyglow/template/` in the main HyGloW repository so
  `python -m hyglow update` can carry it to every other case.
