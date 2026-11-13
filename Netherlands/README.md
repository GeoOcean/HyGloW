# Netherlands

Nearshore waves along the Belgian and Dutch coasts, up to the Ems, from the
WHACS offshore hindcast: the reference case of HyGloW (method:
[`hywaves/README.md`](../hywaves/README.md)). Run with SnapWave; SWAN is set
up as well (`simulations.swan`).

- 10 goals; training cases: `simulations.num_samples_per_goal` LHS samples
  of Hs, Tp, direction, spreading and water level, fewer after the steepness
  filter (`max_steepness`).
- Output sites on the model mesh: 5, 10 and 20 m isobaths, nodes off Den Haag
  and the buoys. Metamodel sets: `buoys`, `isobaths`, `deenhaag`.
- `numerical_tests/`: how the SnapWave and SWAN templates were chosen, and
  SnapWave vs SWAN on the same cases (its own README).
- `dashboard/`: a web app with every 2013 result on the same sites and hours: error
  budget, linear sum vs dynamic, wind, storms, metamodel, calibration, numerical tests
  (`build_store.py`, then `app.py`; its own README, with the first-iteration conclusions).
- Forcing of the reference runs: ERA5 wind and CODEC water level (CODEC ends
  in 2018). Optional: satellite-calibrated WHACS (`WHACS_cal`).
- Results: the HyGloW results portal (`portal/`).

## How to run it

As any case: fill in `Netherlands.json`, run the notebooks from this folder in
order, `python -m hywaves validate Netherlands/` to see what is missing.

| | Notebook | Does |
| --- | --- | --- |
| 03 | `03_Generate_Observations_Configuration` | Processed buoys and their GeoJSON, then the output sites on the mesh |
| 04 | `04_Select_Goals_by_Hindcasts` | Goals and sectors |
| 05 | `05_Metamodel_Cases_Generation` | Training cases (SLURM) |
| 06 | `06_Validate_Metamodel` | Fit (MDA sizes `metamodel.mda_candidates`) and holdout check |
| 07a | `07a_Offshore_WHACS` / `07a_Offshore_ERA5` | Offshore waves at the goals (ERA5: partition Tm-1,0 -> Tp) |
| 07b | `07b_Satellite_Calibration` | WHACS / ERA5 vs IMOS altimetry, `WHACS_cal`, `ERA5_cal` |
| 08 | `08_Dynamic_Cases_Generation` | Hourly reference runs with water level, one folder per forcing (`<dynamic_cases>_<offshore>_wind-<on|off>`) |
| 09–12 | validation and figures | Reconstruction vs reference runs, buoys, spectra, portal figures; `metamodel.reconstruction.depth_cap` caps Hs at the training ceiling x depth (`hywaves/README.md`) |

Notebook 11 is not run yet: it compares spectra against buoy spectra, and the processed
buoys (`utils/buoy_archive.py`) hold bulk parameters only.

## Case-specific code

| File | |
| --- | --- |
| `utils/buoy_archive.py` | Builds the processed buoys (Waterinfo, CMEMS and older Rijkswaterstaat exports) |
| `utils/rijkswaterstaat.py` | Reads the older Rijkswaterstaat exports |
| `reconstruct_buoys.py` | Fills the reconstruction cache of notebook 10 (every offshore source, 1979-2018) |

The rest of `utils/` is the template's: `python -m hywaves update Netherlands/`
keeps it current.
