# __CASE_NAME__

A HyGloW case: nearshore waves from an offshore hindcast (the method:
`hywaves/README.md` in the HyGloW repository).

## How to run it

1. Fill in `__CASE_NAME__.json`: replace every `TODO`. Put your area of
   interest (a polygon shapefile) in `inputs/AoI_Polygons/`.
2. Open each notebook **from this folder** and run it top to bottom, in order.
   The first cell of each one says what it needs and what it writes.
3. At any point, `python -m hywaves validate __CASE_NAME__/` (from the HyGloW
   folder) lists what is still missing.

| | Notebook | Does | Needs in the JSON |
| --- | --- | --- | --- |
| 01 | `01_Generate_tiff_from_GEBCO` | Bathymetry of the area | `paths` |
| 02 | `02_Create_mesh_by_bluemesh2d` | Model mesh | `paths.mesh_file` (= `simulations.<model>.grid_file`) |
| 03 | `03_Generate_Observations_Configuration` | Buoys, and the output sites on the mesh | `onshore_waves`, `simulations.output_sites_config` |
| 04 | `04_Select_Goals_by_Hindcasts` | Goals (offshore points) and their sectors | `simulations.goals_file` |
| 05 | `05_Metamodel_Cases_Generation` | Training cases per goal (model runs) | `simulations` (`model`, `bounds`, `num_samples_per_goal`) |
| 05b | `05b_Wind_Correction` | *Optional:* wind growth factor; runs vs correction, K response | `metamodel.wind_correction`, `forcing.wind` |
| 06 | `06_Validate_Metamodel` | Fit the metamodel, holdout check | `metamodel` |
| 07a | `07a_Offshore_WHACS` **or** `07a_Offshore_ERA5` | Offshore waves at the goals (ERA5: partition Tm-1,0 -> Tp) | `offshore_waves` |
| 07b | `07b_Satellite_Calibration` | *Optional:* check and calibrate them | `satellite` |
| 08 | `08_Dynamic_Cases_Generation` | Reference runs, every hour, all goals | `forcing` (optional) |
| 09 | `09_Validate_Linear_Summation` | Reconstruction vs reference runs | |
| 10 | `10_Validate_Buoys` | Everything vs the buoys | |
| 10b | `10b_Storm_Event` | One storm in detail: forcing, buoys, wind correction, maps | |
| 11 | `11_Validate_Spectra` | Spectra at the buoys | |
| 12 | `12_Final_Plots` | Figures for the results portal | |

The wave model (`simulations.model`: `"snapwave"` or `"swan"`) is needed from
05 on, on your PATH or through SLURM (`simulations.<model>.slurm`).

## Share it

```bash
python -m hywaves publish __CASE_NAME__/
```

writes `portal/`, what the HyGloW results portal shows; commit it. To keep a
version before re-running: `python -m hywaves snapshot __CASE_NAME__/ <tag>`.

## The code

`utils/` is the case layer on top of bluemath_tk, copied from the HyGloW
template; `python -m hywaves update __CASE_NAME__/` brings it to the latest
template. `.hywaves/manifest.json` records the bluemath_tk version it works
with.
