# Samoa

Nearshore waves around Savai'i, Upolu and Tutuila from the WHACS offshore
hindcast, satellite-calibrated (method: [`hywaves/README.md`](../hywaves/README.md)).
Run with SnapWave; dynamic runs forced with CODEC water level and ERA5 wind.

- 16 goals on a closed ring around the islands; 1500 training cases each
  (Hs, Tp up to 22 s for the long Pacific swells, direction, spreading, water level).
- Bathymetry: GEBCO 2026 + the Samoa project bathymetry (Upolu, Savai'i) + NOAA
  Pago Pago 3" and Tutuila 1/3" DEMs (MHW -> MSL), merged at 1 arc-second
  (notebook 01). The mesh (2026-07) keeps its geometry; its node depths come
  from that raster (notebook 02).
- Buoys with data: CDIP 189 Aunu'u (two mooring sites, 2014 onwards) and
  CDIP 273 King Poloa (2024 onwards).
- Output sites on the 5, 10, 20 and 50 m isobaths, the Apia area, and the buoys.
- `numerical_tests/`: why the linear sum is high against the dynamic run here
  (SnapWave forces the boundary with one direction; broad bulk spreads; breaking).

## Notebooks

Run from this folder, in order.

| | Notebook | Does |
| --- | --- | --- |
| 01 | `01_Generate_tiff_from_GEBCO` | Merged topobathy raster (GEBCO 2026 + local sources) |
| 02 | `02_Create_mesh_by_bluemesh2d` | Mesh (2026-07) and its node depths from the merged raster |
| 03 | `03_Generate_Observations_Configuration` | Buoys (CDIP hourly records) and output sites |
| 04 | `04_Select_Goals_by_Hindcasts` | Goals and sectors (with manual direction overrides) |
| 05 | `05_Metamodel_Cases_Generation` | Training cases |
| 06 | `06_Validate_Metamodel` | Fit and holdout check |
| 07a | `07a_Offshore_WHACS` | WHACS at the goals (spurious Tp cleaned) |
| 07b | `07b_Satellite_Calibration` | IMOS calibration -> `WHACS_cal` |
| 07c | `07c_Water_Level_and_Wind` | CODEC water level at the goals, ERA5 wind |
| 08 | `08_Dynamic_Cases_Generation` | Hourly reference runs (2018: CODEC and Aunu'u) |
| 09-12 | validation and figures | Reconstruction vs reference runs, buoys, spectra, portal figures |

## JSON keys

`paths` (AoI, merged raster, mesh), `onshore_waves`, `offshore_waves`
(`WHACS`, `WHACS_cal`), `satellite` (IMOS), `forcing` (ERA5 wind, CODEC water
level), `simulations` (SnapWave; `bounds`: Tp to 22 s, water level -1 to 1.25 m
from CODEC), `metamodel` (sets `buoys`, `isobaths`, `apia`).
