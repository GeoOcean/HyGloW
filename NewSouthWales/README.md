# NewSouthWales

Nearshore waves along the New South Wales coast, Australia, from Byron Bay to
Eden, from the WHACS offshore hindcast (method:
[`hywaves/README.md`](../hywaves/README.md)). Run with SnapWave.

- 26 goals, 500 training cases each (Tp, direction, spreading, water level;
  Hs was not yet a metamodel input).
- 3,860 output sites: the 5, 10, 20 and 50 m isobaths and 35 buoys (7
  offshore MHL Waveriders, 28 nearshore DCCEEW Spotters). Only Sydney, Byron
  Bay and Batemans Bay have directional spectra; Sydney's raw spectral file
  is corrupted, so its spectra keep only the normalised shape.
- MDA sizes 50 to 450 (the portal shows 450).
- Results: the HyGloW results portal (`portal/`).

**Older code.** This case runs the code it was built with (hywaves 0.9.0, in
its own `utils/`); its notebooks and published results match that version.
To re-run it with the current code: `python -m hywaves update NewSouthWales/`,
then bring `NewSouthWales.json` to the current keys (see
[Netherlands](../Netherlands/) and `python -m hywaves validate
NewSouthWales/`).

## Notebooks

Run from this folder, in order.

| | Notebook | Does |
| --- | --- | --- |
| 00 | `00_Buoys_Processing`, `00_Buoys_Processing_Spectra`, `00_Buoys_Inspection`, `00_Download_Wind` | Raw buoys (bulk and spectra) and ERA5 wind |
| 01–02 | `01_Generate_tiff_from_GEBCO`, `02_Create_mesh_by_bluemesh2d` | Bathymetry and mesh |
| 03 | `03_Generate_Observations_Configuration` | Buoys |
| 04 | `04_Select_Goals_by_Hindcasts` | Goals and sectors |
| 05 | `05_Metamodel_Cases_Generation` | Training cases |
| 06 | `06_Validate_Metamodel` | Fit and holdout check |
| 07 | `07_Download_Process_WHACS` | Offshore waves at the goals |
| 08 | `08_Dynamic_Cases_Generation` | Hourly reference runs |
| 09 | `09_Validate_Linear_Summation` | Reconstruction vs reference runs |
| 10a, 10b | `10a_Validate_Offshore_Buoys`, `10b_Validate_Onshore_Buoys` | Offshore buoys 2001–2011 (bulk and spectra), nearshore buoys 2013, 2017–2023 |
| 11–12 | `11_Validate_Spectra`, `12_Final_Plots` | Spectra at the buoys, portal figures |

Case-specific code: `utils/raw_buoys.py` and `utils/raw_spectra.py` read the
raw MHL and DCCEEW buoy archives.
