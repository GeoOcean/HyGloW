# Samoa

Nearshore waves around the Samoan islands from the WHACS offshore hindcast
(method: [`hywaves/README.md`](../hywaves/README.md)). Run with SnapWave.

- 16 goals, 500 training cases each (Tp, direction, spreading, water level;
  Hs was not yet a metamodel input).
- Output sites on the 5, 10, 20 and 50 m isobaths, the Apia area, and 3 buoys.
- MDA sizes 50 to 450.
- Results: the HyGloW results portal (`portal/`).

**Older code.** This case runs the code it was built with (hywaves 0.9.0, in
its own `utils/`); its notebooks and published results match that version.
To re-run it with the current code: `python -m hywaves update Samoa/`, then
bring `Samoa.json` to the current keys (see [Netherlands](../Netherlands/)
and `python -m hywaves validate Samoa/`).

## Notebooks

Run from this folder, in order.

| | Notebook | Does |
| --- | --- | --- |
| 01–02 | `01_Generate_tiff_from_GEBCO`, `02_Create_mesh_by_bluemesh2d` | Bathymetry and mesh |
| 03 | `03_Generate_Observations_Configuration` | Buoys |
| 04 | `04_Select_Goals_by_Hindcasts` | Goals and sectors |
| 05 | `05_Metamodel_Cases_Generation` | Training cases |
| 06 | `06_Validate_Metamodel` | Fit and holdout check |
| 07 | `07_Download_Process_WHACS` | Offshore waves at the goals |
| 08 | `08_Dynamic_Cases_Generation` | Hourly reference runs |
| 09–12 | validation and figures | Reconstruction vs reference runs, buoys, spectra, portal figures |
