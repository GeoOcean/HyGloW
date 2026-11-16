# NorthCarolina

Nearshore waves along the North Carolina coast, from Long Bay to Virginia Beach
(Capes Fear, Lookout and Hatteras), from the WHACS offshore hindcast (method:
[`hywaves/README.md`](../hywaves/README.md)). Run with SnapWave on a mesh delivered
with the ShoreShop work (`inputs/delivery_2026-11-15/`).

- **Mesh**: delivered, not built here (242,110 nodes, about 80 m at the shore, sounds
  closed, flattened at about 100 m offshore; `nc_mesh/` holds the scripts that built it
  from a 3 arc-second DEM, which was not delivered). Its vertical datum is the DEM's.
- **Bathymetry raster**: GEBCO 2026 with the mesh depths on top, 3 arc-seconds, so the
  isobaths are the ones the model sees.
- **Open boundary**: the west, offshore and north sides of the mesh, from coast to
  coast (`simulations.snapwave.coast_depth`). 14 goals along it, lateral sides included;
  the two at the coast ends were dropped (shallow).
- **Output sites**: the 8 m isobath every 250 m (set `transects`, the line of the
  delivered `nc_obs_8m.txt`), the 5, 10, 20 and 50 m isobaths (set `isobaths`) and 19
  NDBC buoys (set `buoys`).
- **Buoys**: hourly Hs, Tp and mean direction from the ShoreShop NDBC tables
  (`utils/shoreshop_buoys.py`), no spreading. Ten of them are CDIP stations, whose
  spreading and spectra are still to be downloaded (notebook 11 waits for them).
- **Reference in the delivery**: `snapwave_NC_his.nc`, a daily SnapWave run 1985-2023
  forced with ERA5 at the 8 m points, to compare the reconstruction against.
- **Dynamic year**: 2018 (inside CODEC; Hurricane Florence in September).
- **Checked against Dano's run** (2026-11-16, `tmp/dano_check/`: `build.py`, `run*.sh`,
  `compare.py`; two days of 2018, Florence and a median day). His input rerun with the
  wind1 build reproduces his output exactly. Our setup comes out lower in Hs, and almost
  all of it is two deliberate choices: the wind2 build (each boundary point keeps its own
  direction; largest locally, where the boundary directions differ) and full convergence
  (`niter` 200 + `upwindref`; his `crit` 0.01 stops early and the median day does not
  converge). The enclosure changes nothing and `dtheta` 9 -> 10 almost nothing; forcing
  the 14 goals instead of his 24 support points is unbiased on average, with local
  differences.

- **Offshore**: WHACS at the goals 1979-2023 and its satellite calibration (`WHACS_cal`),
  CODEC water level to 2018. Training bounds from them: Hs to 17 m (Hurricane Isabel,
  2003), steepness to 0.07 (the calibrated storms are steeper than 0.06), water level
  -1.25 to 2.25 m. On the deep buoys the calibration does not clearly improve the storm
  tail (07b, section 5): check before choosing the offshore source.

## Where it stands and what to do next

**Status (2026-11-16, 22:30).** Done: notebooks 01-04, 07a (WHACS), 07b, the CODEC part
of 07c, and 05: the 14,770 training cases ran and were postprocessed (one, goal case
2913, gave no output and is left out). `tmp/run_chain.sh` stopped at 08: building the
2018 dynamic cases hit a full disk (`/lustre/geocean/WORK`, its data servers at 100%),
so `SnapWave_CASES_Dynamic_WHACS_cal_wind-off/` is half-built (step 2: it is the SATA pool
that is full, not the whole filesystem). The 06 fit, started
before that, was still running on its own (`logs/chain_06.log`, `logs/chain_nb_06.log`).
The first chain's log is kept as `logs/chain_2026-11-16.log`.

1. **Check 06 finished**: `logs/chain_06.log` ends with `end notebook 06`. If not,
   re-run notebook 06 (the fit restarts from the artefacts it already saved).
2. **Make space** on `/lustre/geocean/WORK` (`lfs df -h /lustre/geocean/WORK`): the 2018
   dynamic cases need about 3 GB, the reconstruction cache a few more.
   `SnapWave_CASES_Metamodel/` (34 GB) is no longer needed once 06 is done, except to
   refit.
   Where the space is: `/lustre/geocean/WORK` reports free space overall, but our folders
   write to the **SATA** pool (inherited from `/lustre/geocean/WORK/users/javiert`;
   `lfs getstripe -d <folder>`), whose OSTs were at 100% on 2026-11-16; the free space was
   in the **SAS-HA** pool (`lfs df -h /lustre/geocean/WORK`, `lfs pool_list GEOWORK.SAS-HA`).
   Either free space on SATA, or (after asking the cluster admins whether SAS-HA may be
   used) point the case at it: `lfs setstripe -p SAS-HA <Case>/` before the run (only new
   files go there; `lfs migrate -p SAS-HA <file>` moves existing ones).
3. **Choose the offshore source** of the dynamic run and the reconstruction: `WHACS_cal`
   is set, as in Samoa. On the deep buoys it does not clearly beat WHACS in the storm
   tail here (07b, section 5). To use WHACS: `OFFSHORE` in `tmp/run_08.py`,
   `offshore_name` in `tmp/reconstruct_2018.py` and in notebooks 08, 08b, 09, 11, and
   `primary` in 10.
4. **Resume from 08**, with an empty SLURM queue (the script waits on every job of the
   user), from this folder:
   `rm -rf SnapWave_CASES_Dynamic_WHACS_cal_wind-off`, then
   `setsid nohup bash tmp/run_chain_from08.sh > logs/chain.log 2>&1 < /dev/null &`
   It does 08 (2018, hourly) -> reconstruction of 2018 (every set, bulk and spectral)
   -> 08b, 09, 10, 12. Progress in `logs/chain.log`, each step in `logs/chain_*.log`.
5. **Then**: compare the `transects` set with the delivered daily run
   (`inputs/delivery_2026-11-15/snapwave_NC_his.nc`, ERA5-forced, no wind, no water
   level); `python -m hywaves publish NorthCarolina/`. New South Wales runs after this
   case (its README).

Still missing, not needed for the run above:

- ERA5 waves (`07a_Offshore_ERA5`) and the wind of 07c: set `EDH_API_KEY` first. Wind
  is only a forcing of the dynamic runs; the reconstruction has none
  (`hywaves/WIND.md`).
- Spreading and spectra of the ten CDIP buoys (`bluemath_tk.waves.cdip`, as in Samoa's
  notebook 03): notebook 11 needs them.

Known limits: bulk WHACS spreading is above SnapWave's 55° limit in part of the hours
(as in Samoa); the mesh datum is the unknown one of its DEM; buoy 41025 has been
moored at more than one position (the current one is used).

## Notebooks

Run from this folder, in order; `tmp/` holds them as scripts for the cluster
(`whacs_seapoints.py`, `run_inputs.sh`, `run_chain.sh`).

| | Notebook | Does |
| --- | --- | --- |
| 01 | `01_Generate_tiff_from_GEBCO` | AoI from the mesh open boundary; raster GEBCO 2026 + mesh depths |
| 02 | `02_Create_mesh_by_bluemesh2d` | Checks the delivered mesh and its open boundary (nothing is meshed) |
| 03 | `03_Generate_Observations_Configuration` | NDBC buoys, their hourly records, the output sites |
| 04 | `04_Select_Goals_by_Hindcasts` | Goals along the open boundary and their sectors |
| 05 | `05_Metamodel_Cases_Generation` | Training cases |
| 06 | `06_Validate_Metamodel` | Fit and holdout check |
| 07a | `07a_Offshore_WHACS` | WHACS at the goals (`tmp/whacs_seapoints.py` reads the global archive first) |
| 07a | `07a_Offshore_ERA5` | ERA5 waves at the goals |
| 07b | `07b_Satellite_Calibration` | IMOS calibration -> `WHACS_cal` |
| 07c | `07c_Water_Level_and_Wind` | CODEC water level at the goals, ERA5 wind |
| 08 | `08_Dynamic_Cases_Generation` | Hourly reference run of 2018 |
| 08b | `08b_Storm_Maps` | The largest 2018 storms: dynamic maps vs the reconstruction and the buoys |
| 09-12 | validation and figures | Reconstruction vs the reference run, buoys, spectra, portal figures |

## JSON keys

`paths` (AoI, merged raster, mesh), `onshore_waves`, `offshore_waves` (`WHACS`,
`WHACS_cal`, `ERA5`), `satellite` (IMOS), `forcing` (ERA5 wind, CODEC water level),
`simulations` (SnapWave with `coast_depth`; `bounds` from WHACS at the goals; the
isobaths of `output_sites_config`), `metamodel` (sets `buoys`, `transects`, `isobaths`).
