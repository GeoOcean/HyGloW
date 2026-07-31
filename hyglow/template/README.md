# __CASE_NAME__ — HyGloW hybrid wave downscaling case

Scaffolded with `python -m hyglow start`. This folder is **self-contained**:
the engine code (`utils/`), the pipeline notebooks, SnapWave templates, and
tests are all vendored here — it runs on any machine (laptop or HPC) without
installing the `hyglow` package itself. The one dependency that isn't
vendored is `bluemesh2d` (the mesher), installed from PyPI — see
`requirements.txt`.

**Config:** `__CASE_NAME__.json` (auto-loaded by `utils.config`).
Run notebooks with `__CASE_NAME__/` as the working directory.

## Getting started

0. **Install dependencies** — `pip install -r requirements.txt` (pulls in
   `bluemesh2d`; the rest of the scientific stack — xarray, geopandas,
   rioxarray, bluemath_tk, wavespectra, etc. — is assumed to already be in
   your environment, see Dependencies below).
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

4. **Check your progress against the case contract** (the portal reads only
   contract files, so this is also the publish gate):

   ```bash
   python -m hyglow validate __CASE_NAME__/            # while developing
   python -m hyglow validate __CASE_NAME__/ --publish  # portal-ready check
   ```

5. **Run the unit tests** (from this folder):

   ```bash
   python -m unittest discover -s tests -v
   ```

## Reference

This folder was created from the versioned template in `hyglow/template/`
(the canonical reference). The Samoa case in the main HyGloW repository is
the most complete worked example — its README documents the full pipeline,
the metamodel workflow, and the commit/re-run checklist in detail.

`.hyglow/manifest.json` records which engine snapshot this case was created
from. Treat `utils/` as vendored code: if you fix a bug there, please PR it
to the main HyGloW repository so everyone benefits.

## Dependencies

`pip install -r requirements.txt` installs `bluemesh2d` (the mesher used by
`02_Create_mesh_by_bluemesh2d.ipynb`). Everything else is expected to already
be in your (conda) environment: bluemath_tk, wavespectra, xarray, geopandas,
rioxarray, pandas, scipy, matplotlib — plus SnapWave (external binary).
