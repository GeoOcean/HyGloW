# NewSouthWales

Nearshore waves along the New South Wales coast, Australia, from Byron Bay to
Eden, from the WHACS offshore hindcast, satellite-calibrated (method:
[`hywaves/README.md`](../hywaves/README.md)). Run with SnapWave; dynamic runs
forced with CODEC water level.

- 26 goals along the open boundary (north lateral, offshore side, south lateral);
  950 training samples each (Hs, Tp, direction, spreading, water level), 701 cases
  after the steepness filter: 18,226 in all, two SLURM arrays under the cluster's
  20,000-record limit. MDA sizes up to 600.
- Mesh (OCSmesh, 241k nodes) and its depths from the NSW 50 m blended bathymetry
  (`inputs/GEBCO/NewSouthWales_MAO.tiff`, the `paths.topobathy_file`); GEBCO only
  outside it.
- Output sites: the 5, 10, 20 and 50 m isobaths every 1 km (smoothed 200 m for
  5-10 m: the shallow isobaths lie within a few hundred metres of this steep coast)
  and 35 buoys: 7 offshore MHL Waveriders (65-100 m) and 28 nearshore NSW DCCEEW
  Spotters (6-33 m), each deployment its own site. Buoy records are bulk `hs`, `tp`,
  `dir` (no spreading); 2D spectra for Byron Bay and Batemans Bay to 2011 (Sydney's
  raw spectral file is corrupted: only the normalised shape).
- Dynamic year 2016: inside CODEC (ends 2018), the six MHL buoys, five nearshore
  Spotters, and the June 2016 East Coast Low (Collaroy-Narrabeen), which `Collaroy2`
  recorded.
- Offshore: WHACS and `WHACS_cal`. Only one deep buoy (Eden) is near a goal for the
  07b check, and there the calibration raises the mean bias while fixing the extreme
  tail: one buoy, so the choice stays open (notebook 10 compares both).
- The 0.9.0 results are kept: `portal_v0-9-0/` (shown by the GUI as a version),
  `SnapWave_CASES_*_v0.9.0/`, `outputs_v0.9.0/`, `inputs/WHACS/processed/*_goal_v0.9.0.zarr`.

## Where it stands and what to do next

**Status (2026-11-16, 22:30).** Migrated to hywaves 0.17 (template `utils/`, notebooks 03
and 05-12, current JSON keys). Done: 03 (sites), 07a, 07b, the CODEC part of 07c. Nothing
has been submitted: `tmp/after_northcarolina.sh`, which was to start the run after North
Carolina, was stopped by hand when `/lustre/geocean/WORK` filled up (its log:
`logs/after_northcarolina.log`).

1. **Wait for North Carolina** to finish its run (its README) and for an empty SLURM
   queue: `tmp/run_chain.sh` waits on every job of the user, and the two cases would
   compete for the same slots.
2. **Make space** on `/lustre/geocean/WORK` (`lfs df -h /lustre/geocean/WORK`): the
   18,226 training cases will take about 34 GB, as North Carolina's did on a similar mesh.
   Where the space is: `/lustre/geocean/WORK` reports free space overall, but our folders
   write to the **SATA** pool (inherited from `/lustre/geocean/WORK/users/javiert`;
   `lfs getstripe -d <folder>`), whose OSTs were at 100% on 2026-11-16; the free space was
   in the **SAS-HA** pool (`lfs df -h /lustre/geocean/WORK`, `lfs pool_list GEOWORK.SAS-HA`).
   Either free space on SATA, or (after asking the cluster admins whether SAS-HA may be
   used) point the case at it: `lfs setstripe -p SAS-HA <Case>/` before the run (only new
   files go there; `lfs migrate -p SAS-HA <file>` moves existing ones).
3. **Choose the offshore source** (`WHACS_cal` is set, see above). To use WHACS:
   `OFFSHORE` in `tmp/run_08.py`, `offshore_name` in `tmp/reconstruct_2016.py` and in
   notebooks 08, 08b, 09, 11, 12, and `primary` in 10.
4. **Launch the run** from this folder:
   `setsid nohup bash tmp/run_chain.sh > logs/chain.log 2>&1 < /dev/null &`
   It does 05 (18,226 training cases) -> 06 fit, with 08 (2016, hourly) alongside ->
   reconstruction of 2016 (both sets, bulk and spectral) -> 08b, 09, 10, 11, 12.
   Progress in `logs/chain.log`, each step in `logs/chain_*.log`. North Carolina's 05
   took about 11.5 h for 14,770 cases on the free slots of the day.
5. **Then**: `python -m hywaves publish NewSouthWales/` (the 0.9.0 portal stays as
   `portal_v0-9-0/`).

Still missing, not needed for the run: ERA5 waves and wind (`EDH_API_KEY`).

## Notebooks

Run from this folder. 00, 01, 02 and 04 are the records of how the existing inputs
were made (hywaves 0.9.0; not re-run, their code is older than `utils/`).

| | Notebook | Does |
| --- | --- | --- |
| 00 | `00_Buoys_Processing`, `00_Buoys_Processing_Spectra`, `00_Buoys_Inspection` | Raw MHL and DCCEEW buoys -> `inputs/Observations/processed/` (`utils/raw_buoys.py`, `utils/raw_spectra.py`, case-only) |
| 01-02 | `01_Generate_tiff_from_GEBCO`, `02_Create_mesh_by_bluemesh2d` | Bathymetry and mesh (record) |
| 03 | `03_Generate_Observations_Configuration` | Buoys GeoJSON and the output sites |
| 04 | `04_Select_Goals_by_Hindcasts` | Goals and sectors (record) |
| 05 | `05_Metamodel_Cases_Generation` | Training cases |
| 06 | `06_Validate_Metamodel` | Fit and holdout check |
| 07a | `07a_Offshore_WHACS` | WHACS at the goals (from the case's seapoint extract) |
| 07a | `07a_Offshore_ERA5` | ERA5 waves at the goals |
| 07b | `07b_Satellite_Calibration` | IMOS calibration -> `WHACS_cal` |
| 07c | `07c_Water_Level_and_Wind` | CODEC water level at the goals, ERA5 wind |
| 08 | `08_Dynamic_Cases_Generation` | Hourly reference run of 2016 |
| 08b | `08b_Storm_Maps` | The largest 2016 storms: dynamic maps vs the reconstruction and the buoys |
| 09 | `09_Validate_Linear_Summation` | Reconstruction vs the reference run |
| 10 | `10_Validate_Buoys` | Everything vs the buoys, 2000-2018 |
| 11 | `11_Validate_Spectra` | Spectra at Byron Bay and Batemans Bay (July 2011) |
| 12 | `12_Final_Plots` | Portal figures |

## JSON keys

`paths` (AoI, the 50 m raster, mesh), `onshore_waves`, `offshore_waves` (`WHACS`,
`WHACS_cal`), `satellite` (IMOS), `forcing` (ERA5 wind, CODEC water level),
`simulations` (SnapWave; `bounds` from WHACS and CODEC at the goals: Hs to 12 m,
Tp 2.5-22 s, water level -1.25 to 1.5 m), `metamodel` (sets `buoys`, `isobaths`;
`tm01` as a stat of the summed spectrum).
