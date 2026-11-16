# Numerical tests (Samoa, November 2026): why the linear sum is high against the dynamic run

On 2018 (WHACS_cal, CODEC, mda800) the linear sum came out high against the dynamic
SnapWave run at every site set, bulk and spectral, with the gap growing with the
offshore directional spread (notebook 09). These tests find out why. In short: **mostly
the dynamic run, not the sum.** SnapWave forces the whole boundary with one direction and
cannot represent the broad bulk spreads of Samoa; the remaining true nonlinearity is
depth-induced breaking near the coast.

The numbers are frozen in `results/*.csv`; the runs are in `tmp/linearity_test/`
(git-ignored).

## Findings

1. **SnapWave forces every boundary point with one direction.** Each run builds all
   boundary spectra around the Hs-weighted mean direction of the boundary points
   (`build_boundary_support_points_spectra`, `wdmean_bwv`; upstream at commit `b663c81`
   it was `thetamean`, the same value without wind), cut at ±90° of it, and the solver
   sector (`sector = 180`) is centred on that mean too. A goal case has one direction on
   every point, so it is right; the dynamic run of a closed ring, with 16 goals looking
   in every direction, pushes much of the boundary energy out of or along the boundary.
   With the spectra centred on each point's own direction (test build, variant E/F) the
   exact sum and the dynamic run agree. This is the main cause.
2. **SnapWave cannot take the bulk spreads of Samoa.** Its directional distribution is
   `cos^ms`, `ms = 1/σ² − 1` (σ in rad): for σ > 57.3° `ms < 0` and the distribution
   peaks at ±90° of the mean direction; even a uniform half circle has a spread of about
   49°. WHACS bulk `spr` in Samoa (1979-2023): median 53°, 38 % of goal-hours above
   57.3°, 64 % above 49°; the partitions never above 57.3° (a broad bulk spread here is a
   mix of seas from several directions). So every bulk-forced SnapWave run in Samoa (the
   dynamic run, and the bulk goal cases if their bounds were widened) gets a wrong
   directional spectrum. **Do not widen `bounds.spr` above ~55° for SnapWave**; use the
   spectral (partition) mode, whose spreads sit inside the training range.
3. **The GP extrapolates in bulk mode** (`bounds.spr` [1, 45], 76 % of the bulk goal-hours
   above it): on top of 1-2, the GP sum is higher than the exact sum of the same runs at
   spreads above 60° (`results/exact_vs_gp_sum.csv`). It disappears with the spectral
   mode (point 2), not by widening the bounds.
4. **Depth-induced breaking is a true nonlinearity of the sum.** A goal case carries a
   fraction of the energy, breaks less, and the sum stays high at 5 m (and somewhat at
   10 m) even with correct boundary directions (variant F). The depth cap barely acts in
   Samoa (09: 3 % of the Apia site-hours, 0.02 % of the isobath ones), so it does not
   remove this. Bottom friction (`fw`) changes nothing measurable (variant A).
5. **Against the buoys, the dynamic run is the low one.** Aunu'u buoys, 2018
   (`results/dynamic_vs_buoys.csv`): the dynamic run is further below the buoys than the
   linear sum (bulk or spectral). The buoys are deep (55-79 m) and WHACS_cal has its own
   errors, but it agrees with 1.
6. **Why it worked in the Netherlands:** an open coast with waves from one side, so the
   goal directions are close (circular std ~11° per hour; Samoa ~30°) and bulk spreads
   above 57.3° are rare (3.6 % of the 2013 goal-hours). The same dynamic hours rerun with
   the per-point build and `sector = 360` (25 random 2013 hours + Xaver,
   `Netherlands/numerical_tests/runs/perpoint_check/`, git-ignored) come out on average
   equal to the stock runs (slightly lower, with a ±10-20 % hour-to-hour scatter); in
   Samoa the stock runs lose a quarter to a third of the energy against the fixed ones
   (`results/stock_vs_perpoint_dynamic.csv`). The method and the cases behave the same;
   the ring geometry is what the one-direction boundary cannot take. A fuller Netherlands
   recheck with the fixed build is still worth doing.

## Test design

- **Hours** (`results/goal_cases.csv`): 25 hours of 2018, 8 random ones per class of the
  goal-mean offshore spread (< 45°, 50-58°, > 62°) plus the Cyclone Gita peak
  (2018-02-09 21:00).
- **Exact sum:** per hour, 16 single-goal SnapWave cases with the real offshore forcing of
  their goal (Hs, Tp, Dir, Spr and CODEC level at the goal), same templates and wrapper as
  the training cases, no GP; summed in energy over the goals. All 400 converged.
- **Ratio** = Σ Hs² of the estimate / Σ Hs² of the dynamic run over the sites of a group,
  median over the hours (and split by the spread class). 1 = the sum is exact.
- **Variants** (`scripts/03`, `scripts/04`; goal cases and the dynamic hours rerun with
  the same change):

  | Variant | Change |
  | --- | --- |
  | baseline | the templates as they are |
  | A | `fw = 0` (no bottom friction) |
  | B | A + `gamma = 10` (practically no depth-induced breaking) |
  | C | `sector = 360` |
  | D | C + B, goal cases forced with the hour's mean Tp and level (SnapWave runs one domain Tp = the boundary mean, and the mean level) |
  | E | D, dynamic hours with the test build (spectra centred on each boundary point's direction) |
  | F | C, dynamic hours with the test build (realistic dissipation) |

  E has every known nonlinearity removed: it is the check that the chain is understood.

## Results (energy ratio, exact sum / dynamic run, median over the 25 hours)

From `results/variants.csv`:

| Group | baseline | A | B | C | D | E | F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 m | 1.95 | 1.96 | 1.31 | 2.02 | 1.30 | 1.07 | 1.71 |
| 10 m | 1.33 | 1.32 | 1.27 | 1.32 | 1.27 | 1.02 | 1.11 |
| 20 m | 1.32 | 1.32 | 1.31 | 1.28 | 1.26 | 1.00 | 1.04 |
| 50 m | 1.26 | 1.26 | 1.25 | 1.23 | 1.22 | 1.00 | 1.03 |
| Apia (polygon) | 1.35 | 1.36 | 1.28 | 1.35 | 1.22 | 1.07 | 1.13 |
| Buoys | 1.07 | 1.07 | 1.07 | 1.09 | 1.06 | 1.00 | 1.00 |

The spread split is in the CSV: in the baseline the 50 m ratio goes from 1.10 (spread
< 45°) to 1.38 (> 60°); in E it is 1.00 and 1.04.

## The test build of SnapWave

`scripts/snapwave_perpoint_direction.patch` (one change in `snapwave_boundaries.f90`:
`wdmean_bwv` -> `wdt_bwv(ib)` in the boundary spectra), applied to the source of the
module `snapwave/b663c81-wind1` and built in `tmp/snapwave_perpoint` (gcc 12.5 and the
netcdf of the module, from spack):

```bash
S=/nfs/software/geocean/spack/opt/spack/linux-icelake
export PATH=$S/binutils-2.46.0-fxir3exizduhk4rh4xy7kiif3clrnbqt/bin:$S/gcc-12.5.0-fugh7gguopsl4pefyn462am24futrpm6/bin:$PATH
export LIBRARY_PATH=$S/netcdf-c-4.9.3-iwl7lhirpgzoljyqmjixw4l7pdmn7fpi/lib:$S/hdf5-1.14.6-qnia67pknbhvnnnvlysn7x2dxavnnsxv/lib
make FC=gfortran CC=gcc NF_CONFIG=$S/netcdf-fortran-4.6.2-mv2q6ehezbmkqcgr6fmesytvk7htzt7t/bin/nf-config STRIP_BUILD=1
# run: module load snapwave; LD_LIBRARY_PATH=$S/gcc-12.5.0-fugh7gguopsl4pefyn462am24futrpm6/lib64:$LD_LIBRARY_PATH
```

On a goal case (one direction everywhere) it gives the same output as the module (checked,
identical). With it the dynamic runs need `sector = 360` (the solver sector is still
centred on the mean direction). It is not installed as a module.

## Scripts (run from the case folder, `PYTHONPATH=.`)

- `01_build_goal_cases.py`: the 25 hours and their 400 goal cases, submitted to SLURM.
- `02_compare_exact_gp.py`: dynamic vs exact sum vs GP sum (`results/exact_vs_gp_sum.csv`).
- `03_physics_variants.py <A|B|C|D>`: the variants on the cluster. E and F were run by
  hand: copies of the D and C dynamic hours run with the test build.
- `04_compare_variants.py`: all variants (`results/variants.csv`).
- `05_dynamic_vs_buoys.py`: dynamic run and linear sum against the Aunu'u buoys.
- `06_swan_storm.py <grid|submit|wind1|postprocess>`: SWAN and SnapWave wind1 dynamic runs
  of the Cyclone Gita window (`results/swan_storm_gita2018.csv`; notebook 08b section 5).

## Applied (2026-11-15) and open

Applied in the template and Samoa: the per-point build as `snapwave/b663c81-wind2`
(cluster module; goal cases unchanged by it), `sector = 360` in both
SnapWave templates (goal cases too: with 180° the energy refraction turns past 90°
around the islands is lost, +9 % at 50 m to x2 at the buoys in energy on the 400 test
cases), spread at most 55° (`MAX_SPREAD`) with `bounds.spr` [1, 55], partitions scaled to
the bulk energy in 07a. Samoa 05-12 re-run with them (finished 2026-11-16: all 17,728
training cases and all 8,760 dynamic hours converged; 46 % of the dynamic boundary
`spr` values were clipped at 55°).

## Re-run with the fixes (2018, WHACS_cal, mda800, `wl="forcing"`, depth cap on)

Linear sum (bulk) vs the dynamic run, Hs (`results/linear_sum_vs_dynamic_rerun.csv`,
from the 09 metrics before and after):

| Group | bias before (m) | R² before | bias after (m) | R² after |
| --- | --- | --- | --- | --- |
| 5 m | +0.36 | -0.02 | +0.21 | 0.23 |
| 10 m | +0.34 | 0.44 | +0.08 | 0.87 |
| 20 m | +0.25 | 0.61 | +0.02 | 0.91 |
| 50 m | +0.22 | 0.63 | +0.005 | 0.89 |
| Apia | +0.31 | 0.33 | +0.07 | 0.85 |
| Buoy sites | +0.16 to +0.25 | 0.52-0.74 | +0.03 to +0.08 | 0.89-0.95 |

Against the Aunu'u buoys, energy ratio model / buoy (`results/dynamic_vs_buoys_rerun.csv`
vs `dynamic_vs_buoys.csv`): dynamic run 0.81 -> 1.07 (Aunuu_1) and 0.85 -> 0.95 (Aunuu_2);
spectral sum 0.90 -> 1.04 and 0.89 -> 0.99; bulk sum 1.00 -> 1.08 and 0.94 -> 1.00.
The one-direction boundary was the cause: the sum now matches the dynamic run from 10 to
50 m and both are close to the buoys. Left: breaking at 5 m (and some at 10 m), the
depth cap still barely acts (4.5 % of the Apia site-hours, 0.05 % of the isobath ones).

## SWAN check of wind2 (Cyclone Gita, 2018-02-08 12:00 to 02-11 12:00, 73 hours)

Same forcing as the 2018 dynamic run (WHACS_cal bulk at the 16 goals, `spr` clipped at
55°, goal-mean CODEC level, no wind), run with SWAN (the outer ring as one open boundary
from goal 1 back to goal 1, islands as land, `inputs/GRID/Samoa_mesh.grd`; all 73 hours
reach 99.5 % accuracy) and with SnapWave wind1 (copies of the wind2 case folders, only
the build changes). SWAN as the reference, Hs (`results/swan_storm_gita2018.csv`):

| Group | RMSE wind1 (m) | RMSE wind2 (m) | R² wind1 | R² wind2 |
| --- | --- | --- | --- | --- |
| 5 m | 0.42 | 0.40 | 0.32 | 0.38 |
| 10 m | 0.55 | 0.51 | 0.58 | 0.63 |
| 20 m | 0.55 | 0.51 | 0.69 | 0.72 |
| 50 m | 0.45 | 0.39 | 0.84 | 0.88 |
| Apia | 0.28 | 0.19 | 0.92 | 0.96 |
| Aunuu (both sites) | 0.29 | 0.17-0.19 | 0.89-0.91 | 0.95-0.97 |
| King_Poloa | 0.45 | 0.13 | 0.69 | 0.97 |

wind2 is closer to SWAN in every group (Hs and direction; Tp is the same in both
builds); peak Hs at Aunuu_1: SWAN 5.58, wind2 5.49, wind1 5.01 m. All three are well
below the buoy (7.0 m): the gap at Gita is the forcing (WHACS_cal at a cyclone, no local
wind), not the boundary. Away from the buoys SnapWave and SWAN still differ (R² 0.4-0.9 on
the isobaths): model physics, plus the boundary itself. SWAN imposes the boundary spectra
only on the directions entering the domain, SnapWave the whole spectrum on every node: on
the lee side of the ring (goals 6-12 for Gita) SWAN carries the interior field out
(goals 9-10, behind Tutuila, 1.1-1.5 m below the forcing at the peak), SnapWave keeps the
offshore Hs there.

Still open:

- `b663c81-wind2` is the module default since 2026-11-16; report the issues upstream
  (draft: `/lustre/geocean/WORK/users/javiert/software/snapwave/upstream_issue.md`).
- The dynamic reference cannot take partitions in SnapWave (one spectrum per boundary
  point): the clean reference for the spectral sum is a SWAN dynamic run with partition
  spectra at the goals, or the buoys.
- Breaking near the coast (point 4): recheck the depth cap against the new dynamic run.
- The Netherlands with the same setup (goal cases and dynamic runs).
