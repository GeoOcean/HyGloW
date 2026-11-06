# Numerical tests (Netherlands, November 2026)

The first reference for the numerical models behind HyGloW: why the SWAN and SnapWave
templates in `hywaves/template/` look the way they do, and how the two models compare.
HyGloW lets a case choose its nearshore model; whichever model a case runs must ship its
best-tested configuration. Every option here was tested on the Netherlands case,
against buoys where possible, before going into the templates.

- `01_SWAN_Decisions.ipynb`: source terms, whitecapping, forcing, numerics, friction, wind.
- `02_SnapWave_Decisions.ipynb`: boundary enclosure, refraction scheme, iterations,
  friction, wind, non-converging runs.
- `03_Compare_Models_Metamodel.ipynb`: SnapWave vs SWAN on the same training cases.
- `04_Compare_Models_Dynamic.ipynb`: SnapWave vs SWAN on the hourly reference runs and
  against the buoys.

01 and 02 read only `results/*.csv`, so they keep working after `runs/` is deleted. 03 and
04 read the case runs (they put the case folder on `sys.path` for `utils`, and use
`model_comparison.py` from here). **03 and 04 are not a clean comparison yet**: they ran
on the runs available at the time, before both models had the final templates on the
same hours; rerun them once both models have the full 05 and 08 runs.

## Decisions

The numbers behind each decision are in `results/*.csv` (frozen with the tests), notebooks
01 and 02, and the dashboard's Numerical tests page.

| Model | Setting | Decision | Why (Hs at 27 buoys, or the other model) |
| --- | --- | --- | --- |
| SWAN | Source terms (dynamic, with wind) | `GEN3 KOMEN` | Best in storms; Westhuysen and ST6 lower, and ST6 needs 16 GB and twice the time |
| SWAN | Whitecapping in no-wind cases (goal cases) | `OFF WCAP` | Without it steep goal cases lose a quarter of their Hs against SnapWave |
| SWAN | Bottom friction | `FRICTION JONSWAP CONSTANT 0.019` | Better above 3 m on the 39 and the 120 independent hours, slightly high below 1 m |
| SWAN | Breaking, triads, iterations, directions, water level | Unchanged: `BREAKING CONSTANT 1.0 0.73`, `TRIAD`, `STOPC` defaults with 50 iterations, `CIRCLE 36`, one level per case | `BKD`, no triads, 100 iterations, 5° and a water-level field change nothing at the buoys, or make it worse |
| SWAN | Wind input | ERA5 wind as is, no `AGROW` | `AGROW` changes nothing; wind × 1.1 helps Xaver but spoils calm hours |
| SWAN | Outputs | `TMM10` in the table, 2D spectra at the buoys in dynamic runs | For Tp and notebook 11 |
| SnapWave | Boundary enclosure | Thin polygon around the open boundary (`open_boundary_nodes`, `open_boundary_enclosure`) | The AoI rectangle also forced the coast (6,329 boundary nodes, now 768) and was too high nearshore |
| SnapWave | Refraction scheme | `upwindref = 1` | Every goal case converges, and gentle seas agree with SWAN |
| SnapWave | Iterations | `niter = 200` (sweeps) | Upwind runs need ~13 iterations = 52 sweeps; no cost once converged |
| SnapWave | Friction, sector | Defaults (`fw` 0.01, sector 180 / 360 with wind) | `fw` 0.02 and 0.04 trade calm hours for storms; sector 360 gains nothing |
| SnapWave | Non-converging runs | Dropped in postprocessing (final error > 0.01, `MAX_ERROR`) | A few % of the hours of 2013, with Hs maxima far above the offshore Hs |
| Both | Offshore forcing | WHACS_cal with satellite quantile mapping (07b); dynamic forcing still open | Better Xaver than raw WHACS; ERA5_cal better still |
| Both | Hs bounds of the goal cases | `hs` [0.05, 10] m, 1,500 samples per goal (864 kept) | The mapped WHACS_cal reaches about 10 m; forcing steepness stays below 0.06 |

## Test design

- **Buoys:** 27 Netherlands and Belgian buoys with Hs in 2013 (`utils/buoy_archive.py`).
  Bias = model − buoy, n = hour-buoy pairs.
- **39 test hours** (`inputs/dynamic_hours.csv`): 8 calm, 20 storm (January–February),
  11 Xaver (December 5–6). Each test changes one thing on the same hours.
- **120 other hours** (`inputs/hours_6.csv`): random hours from a 2013 sample
  stratified by offshore Hs (`inputs/no_wind_hours.csv`, 500 hours), to check friction
  and forcing away from the tuning hours. They still include 8 Xaver and 3 St Jude
  hours, so the storm classes above 5 m remain thin.
- **40 goal cases** (`inputs/metamodel_cases.csv`): 8 per steepness class
  (0–0.07), compared with the other model. The ratio is `sqrt(Σ a² / Σ b²)` over sites
  with at least 25 % of the offshore Hs, split at 18 m depth.
- **Reading by observed Hs:** binning by the observation regresses to the mean, so the
  top bins look worse than the model is. Compare variants within a bin, not bins with
  each other.

## What was learned, in order

1. **SWAN looked worse than SnapWave in big waves** (notebook 04). On the 39 hours
   Westhuysen, SWAN's default, was low in storms. Komen beat it, and ST6 was worse
   still.
2. **The goal cases had whitecapping with no wind** (`GEN3` off but `WCAP` on): steep
   cases lost a quarter of their Hs. `OFF WCAP` fixed it.
3. **Most of the storm deficit was the forcing.** The old WHACS_cal was too low in the
   tail at the deep buoys. A satellite-only quantile mapping (IMOS, all goals pooled,
   constant beyond p99.9) fixed most of it. ERA5_cal is better still in Xaver.
4. **Ruled out as the cause of the deficit:** the boundary (Hs at the goal nodes equals
   the forcing), the 50-iteration limit, breaking, friction, the water level, and the
   stationary mode (a non-stationary Xaver run gained nothing).
5. **Friction 0.019** (the SWAN manual's value for smooth seafloors) helps above 3 m
   on both the 39 and the 120 hours, at a small cost below 1 m.
6. **Wind matters**, most in calm hours, less in storms. How to add it to the linear sum
   is open (`hywaves/WIND.md`).
7. **SnapWave forced its coast.** `build_enclosure_polygon` took the minimum rectangle
   around the AoI, which held the whole mesh, so every outline node became a boundary
   node. The new enclosure follows the outline through the goals out to the lateral
   corners; on this mesh it matches SWAN's `.grd` open boundary exactly.
8. **SnapWave is high in calm and moderate seas because of its wind input.** Without
   wind it matches SWAN; with wind it adds about twice what SWAN adds. That lives in the
   source's growth curves, not in any setting, and any wind correction learned from
   SnapWave inherits it.
9. **SnapWave wind mode fails in a few % of hours** (long swell under light wind). No
   setting fixes it (`dt`, `niter`, `jadcgdx`, upwind refraction, the enclosure), but the
   final error separates these runs cleanly, so they are dropped.

## Open points

- **Dynamic forcing:** ERA5_cal wins in the biggest seas, but mostly in Xaver. ERA5
  partition periods are doubtful (some partition energy at very long periods).
- **SnapWave wind input:** the calm-sea overestimate needs a fix in the source, not
  in the template.
- **SnapWave dynamic runs** centre every goal's spectrum on one domain-mean direction.
  This affects only the dynamic reference, not the hybrid.
- **ERA5 storm winds:** Xaver stays low in every setup at the biggest buoy seas.
- **Water level in dynamic runs:** both models use one level per case. SnapWave
  averages the boundary nodes (`depth = max(mean(zs_boundary) - zb, hmin)` in
  `snapwave_boundaries.f90`), and the SWAN template sets one `SET LEVEL`. In surge
  storms the goals' levels differ by metres (Xaver peak) but both run at one mean
  level. Goal cases are unaffected (one level at every node), and the hybrid with
  `wl="forcing"` keeps each goal's own level. A SWAN water-level field changed nothing
  at the buoys, but they are mostly deeper than 15 m, so the 5 m sites in surge storms
  are not checked. A field in SnapWave needs a source patch.
- **Breaking is not validated:** the 27 buoys sit outside the surf zone. At the Xaver
  peak breaking dominates the dissipation at 5 m, matters at 10 m and is nil at 20 m
  and at the buoys. So `gamma` (SnapWave 0.7, SWAN 0.73) and `BREAKING BKD` were only
  weakly tested.
- **Unused SnapWave outputs:** every history file also holds `point_dw` (breaking
  dissipation, W/m²), `point_df` (bottom-friction dissipation), `point_zs`, `point_zb`
  and, with wind, `point_Sw` / `point_St` (wind input). Postprocessing keeps only Hs,
  Tp, direction and spreading. Dissipation is not additive across goals, so the
  linear sum cannot reconstruct it, but dynamic runs could keep it as a diagnostic
  (where waves break; wind input in the calm hours where SnapWave runs high).

## Layout

| Path | What | Git |
| --- | --- | --- |
| `README.md`, `01_*.ipynb` - `04_*.ipynb` | This summary and the four notebooks | tracked |
| `decisions.py` | Tables and plots for 01 and 02 | tracked |
| `model_comparison.py` | SnapWave vs SWAN readers, metrics and plots for 03 and 04 | tracked |
| `results/*.csv` | Tidy scores: `buoy_scores`, `metamodel_ratios`, `snapwave_unstable_hours`, `snapwave_convergence_2013`, `swan_directions` | tracked |
| `results/raw/` | The printed outputs the CSVs come from | tracked |
| `results/figures/` | `snapwave_enclosure.png` | tracked |
| `inputs/` | Hour and case lists; `snapwave_production_2013_hours.csv` (final error, offshore Hs, wind for all 8,760 production hours) | tracked |
| `scripts/swan/`, `scripts/snapwave/` | The build, run (SLURM) and scoring scripts, as run | tracked |
| `scripts/collect_results.py`, `scripts/build_notebooks.py` | `results/raw` → `results/*.csv`; writes and runs the notebooks | tracked |
| `runs/` | Case folders, SLURM logs, the October pilots (`pilot_2026-10/`), ~58 GB | ignored |

```bash
cd Netherlands/numerical_tests
python scripts/collect_results.py   # results/*.csv from results/raw/
python scripts/build_notebooks.py   # writes and executes the two notebooks
```

**Running the scripts:** the scripts in `scripts/swan` and `scripts/snapwave` ran from
`Netherlands/` when this folder was `SWAN_test/` (and imported `utils.model_comparison`,
now `model_comparison.py` here). Their paths say so, and their
reference runs (the production SnapWave and SWAN runs of 2026-11) have been replaced.
They are kept to show exactly what was run, not to be run again as they are.

- **Models:** `swan/4151`, serial `swan.exe` per task; `snapwave/b663c81-wind1`.
- **Reproducing a test:** build cases with `utils.simulation.HyWavesWrapperSetup`,
  edit the input file of each variant, and run them as a SLURM array.
