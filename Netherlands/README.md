# Netherlands

Nearshore waves along the Belgian and Dutch coasts, up to the Ems, from the
WHACS offshore hindcast: the reference case of HyGloW (method:
[`hywaves/README.md`](../hywaves/README.md)). Run with SnapWave; SWAN is set
up as well (`simulations.swan`).

- 10 goals, 629 training cases each (Hs, Tp, direction, spreading, water
  level).
- 5156 output sites on the model mesh: 5, 10 and 20 m isobaths, 141 nodes off
  Den Haag, and 38 buoys. Metamodel sets: `buoys`, `isobaths`, `deenhaag`.
- Forcing of the reference runs: ERA5 wind and CODEC water level (CODEC ends
  in 2018). Optional: wind correction, satellite-calibrated WHACS (`WHACS_cal`).
- Results: the HyGloW results portal (`portal/`).

## How to run it

As any case: fill in `Netherlands.json`, run the notebooks from this folder in
order, `python -m hywaves validate Netherlands/` to see what is missing.

| | Notebook | Does |
| --- | --- | --- |
| 03 | `03_Generate_Observations_Configuration` | Processed buoys and their GeoJSON, then the output sites on the mesh |
| 04 | `04_Select_Goals_by_Hindcasts` | Goals and sectors |
| 05 | `05_Metamodel_Cases_Generation` | Training cases (SLURM) |
| 05b | `05b_Wind_Correction` | Wind growth factor K; runs vs correction, K response |
| 06 | `06_Validate_Metamodel` | Fit (MDA sizes 100, 550) and holdout check |
| 07a | `07a_Offshore_WHACS` / `07a_Offshore_ERA5` | Offshore waves at the goals (ERA5: partition Tm-1,0 -> Tp) |
| 07b | `07b_Satellite_Calibration` | WHACS / ERA5 vs IMOS altimetry, `WHACS_cal`, `ERA5_cal` |
| 08 | `08_Dynamic_Cases_Generation` | Hourly reference runs with wind and water level |
| 09–12 | validation and figures | Reconstruction vs reference runs, buoys, spectra, portal figures |
| 10b | `10b_Storm_Event` | Xaver (Dec 2013) in detail: forcing, buoys, wind correction, maps |

## Case-specific code

| File | |
| --- | --- |
| `utils/buoy_archive.py` | Builds the processed buoys (Waterinfo, CMEMS and older Rijkswaterstaat exports) |
| `utils/rijkswaterstaat.py` | Reads the older Rijkswaterstaat exports |
| `reconstruct_buoys.py` | Fills the reconstruction cache of notebook 10 (`WHACS_cal`, `ERA5_cal` x wind off/on, 1979-2018) |

The rest of `utils/` is the template's: `python -m hywaves update Netherlands/`
keeps it current.
