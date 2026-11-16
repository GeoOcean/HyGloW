# Tyndall

Planned case (not started): nearshore waves around Tyndall Air Force Base, Florida
Panhandle (Gulf of Mexico), hit by Hurricane Michael on 10 October 2018. The case is
created with `python -m hywaves start --name Tyndall` once the questions below are
answered; this folder holds only this note until then (`start` refuses an existing
folder: move the README aside, start, put it back).

## What was found (2026-11-16)

**Domain.** The shelf edge is close and steep off Panama City and Tyndall: the 50, 100
and 200 m isobaths are all 31-47 km from the coast (GEBCO 2026, nearest distance), so an
offshore boundary ~50 km out and one on the 200 m isobath are the same thing there. East
of Cape San Blas the shelf widens (200 m is 113 km off St George Island; at 50 km due
south of Mexico Beach it is only ~17 m deep). Proposal: offshore boundary on the ~200 m
isobath from Panama City Beach (or Destin) to Cape San Blas, closing at Cape San Blas.

**Bathymetry** (NOAA NCEI THREDDS, `https://www.ngdc.noaa.gov/thredds/dodsC/regional/`):

| File | Resolution | Extent | Datum |
| --- | --- | --- | --- |
| `panama_city_13_mhw_2010.nc` | 1/3" | -86.10 to -85.20, 29.55 to 30.50 (all of Tyndall) | MHW |
| `destin_13_navd88_2016.nc` | 1/3" | -86.80 to -86.07, 30.00 to 30.53 | NAVD88 |
| `northern_gulf_1_mhw_2010.nc` | 1" | -90.75 to -85.00, 28.50 to 31.25 (past the shelf edge) | MHW |
| `apalachicola_bay_g100_30m.nc` | 30 m (UTM) | Apalachicola Bay | - |

Offsets to MSL from the Panama City gauge (NOS 8729108, epoch 1983-2001): MHW is 0.19 m
above MSL (`offset = +0.19` for the MHW DEMs), NAVD88 is 0.03 m below MSL. Microtidal
(mean range ~0.4 m). Merge them over GEBCO 2026 in notebook 01
(`bluemath_tk.topo_bathy.merge`), as in Samoa.

**Offshore hindcasts** (`QGIS/HyWorld/hindcasts/`): WHACS nodes every ~6 km, ERA5 waves
every 0.5°, CODEC stations every 7-22 km: enough for goals ~50 km out. CODEC ends in
2018 and includes Michael.

**Observations.**

| Station | Where | Record | Note |
| --- | --- | --- | --- |
| NDBC 42039 | 115 NM SSE of Pensacola, deep | 1995-2023 | ~85 km outside a 50 km domain: offshore check only |
| NDBC 42036 | 112 NM WNW of Tampa | 1994-2023 | far south-east, outside |
| NDBC 42012 | Orange Beach, 44 NM SE of Mobile | 1983-2023 | far west, outside |
| NDBC 42027 / 42028 | USF COMPS, 45 m / 58 m isobaths | - | no wave data at NDBC |
| NOS 8729108 / 8729210 | Panama City / Panama City Beach | tide gauges | water level and Michael's surge |
| NDBC sgof1 | Tyndall AFB Tower C (N4) | met only | |

The local NDBC archive is `/lustre/geocean/DATA/OBSERVATIONS/WAVES/BUOYS/NOAA/`. No wave
buoy with a long record lies inside the proposed domain.

**Dynamic year**: 2018 (Hurricane Michael, inside CODEC), as in Samoa and North Carolina.

## Questions (user)

1. **Coastline extent**: only Tyndall's shore (Shell Island to Mexico Beach, ~30 km), or
   Panama City Beach to Cape San Blas (the proposal above)?
2. **Nearshore wave data**: are there local measurements (Spotters, ADCPs, the Michael
   deployments, the Tyndall rebuild projects)? Without them, validation rests on the
   dynamic run and the satellite.
3. **Mesh**: bluemesh2d (notebook 02), or the DistMesh scripts that built the North
   Carolina mesh (`NorthCarolina/inputs/delivery_2026-11-15/nc_mesh/`, DEM straight to a
   SnapWave mesh with the open boundary and output points)?
4. **Caveats to accept**: the Gulf is fetch-limited and Michael was wind-driven, so the
   missing wind in the reconstruction (`hywaves/WIND.md`) will be the largest error here;
   CODEC (GTSM) probably underestimates Michael's surge (about 4 m at Mexico Beach), which
   also sets `bounds.wl`.
