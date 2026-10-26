# Reconstruction: how it runs, what was measured, what is left

Working notes on the speed and resources of the HyWaves reconstruction (state
on 2026-10-16). User-facing summary and timings: `hywaves/README.md`.

## Where the code is

Uncommitted at the time of writing:

| Repo, branch | Files |
| --- | --- |
| bluemath_tk `hywaves/fast-reconstruct` (from `hywaves/reconstruct-store`, 09fcd23) | `interpolation/gps.py` (`predict(batch_size=)`), `waves/hywaves/metamodel.py` (`predict_pcs`, `fields_from_pcs`, `PREDICT_BATCH_ROWS`), `waves/hywaves/resources.py` (`sum_workers`, `POOL_SECONDS`, new GP time constants), `tests/waves/test_hywaves_resources.py` |
| HyGloW `reconstruct-store` | `hywaves/template/utils/metamodel.py` (= `Netherlands/utils/metamodel.py`), `hywaves/README.md`, this file |

Already committed before: the reconstruction cache (`waves/hywaves/store.py`),
the resources proposals (`core/resources.py`, `waves/hywaves/resources.py`),
the site x hour sum blocks.

## How a reconstruction runs

`mm.reconstruction_resources(...)` proposes, the user decides, then
`mm.reconstruct(..., resources=res)`:

1. **Request** (`_request`): sites of the set, the cache stores (bulk, and
   spectra with `save_spectra`), contributions per site (`parts` = goals x
   partitions + 1 with the wind correction). `todo` = site-months not stored.
2. **Calls**: the missing months in batches of `res.months` (up to 12), each
   with the missing sites (in groups of `res.sites` if memory is short). A
   call that does not fit `res.memory` stops before computing
   (`call_memory`).
3. **GP step** (`goal_contributions`): every goal x partition with valid
   hours (energy, inside the goal's sector) is one job; `run_processes` runs
   them in `res.threads // 2` processes of 2 threads (`GP_THREADS`), each
   loading its GP and predicting the stacked PCs 512 rows at a time
   (`predict_pcs`, `PREDICT_BATCH_ROWS`). The water level is read once per
   goal. On a GPU the jobs run in the main process.
4. **PCA step**: back in the main process, threaded: PCs -> fields at the
   sites (`fields_from_pcs`), site metadata, wind-sea columns.
5. **Sum** (`sum_contributions` -> `sum_partition_spectra_batched`): JONSWAP
   x Cartwright spectra summed per site and hour, only the requested stats.
   Work items = site groups x time spans (time is split too when there are
   fewer sites than workers); each worker sums blocks of `batch_shape` within
   its budget (`sum_budget`: up to 2 GiB per worker). `sum_workers` runs the
   sum in the main process when a pool would cost more than it saves
   (`POOL_SECONDS` = 10 s; e.g. one site-year).
6. **Store**: whole months written per site (bulk sharded per month, spectra
   one file per site-month), then the request is read back.

## Resources rules

- At most 75% of the cores and of the free memory (`USAGE`); never SLURM.
- Planner (`plan_reconstruction`): one sum worker per usable core; largest
  call that fits, preferring every site and up to 12 months, then fewer
  months, then fewer sites, then fewer workers.
- Memory per call (`call_memory`): contributions 4 fields x float64 x 2
  copies; saved spectra x 3 copies; per worker 0.75 GiB + 0.5-2 GiB of blocks.
  Sum peak per site = float32 spectrum x (8 per contribution + 4)
  (measured 6.35x per part + 2.3x).
- Fits (`plan_fit`, `fit_all`): MDAs, then each (goal, set, n) as a
  single-thread process, largest sets first; ~1 GiB + 8x the goal's cases.

## Measurements

48-core Xeon Gold 5418Y node, shared (other users' xbeach/dimr ran during
2026-10-16, load 30-58), Netherlands, WHACS_cal.

**2026-10-16, MDA 550, water level CODEC + wind correction, empty cache,
new code** (`Netherlands/tmp/benchmarks/rec.log`):

| Case | Total | GP step | Sum | Peak |
| --- | --- | --- | --- | --- |
| 38 buoys, 2013, bulk | 44 s | 16 s | 26 s | 53 GiB |
| 38 buoys, 2013, spectral | 84 s | 29 s | 52 s | 76 GiB |
| 38 buoys, 2012, spectral, spectra saved | 97 s | 26 s | 57 s | 77 GiB |
| 1 site, 2011-2013, bulk (3 calls) | 45 s | 17-18 s | 4 s | 11 GiB |
| 1 site, 2011-2013, spectral (3 calls) | 83 s | 26-30 s | 12 s | 26 GiB |
| 4,971 isobath sites, Nov 2013, bulk | 149 s | 17 s | 116 s | 52 GiB |
| 1 site x 1 year from the cache | 1 s | | | 1 GiB |

Extrapolated: one site 1979-2023 ~17 min bulk, ~31 min spectral (45 calls);
all 4,971 isobath sites 1979-2023 bulk ~22 h (540 x 149 s); spectral ~50 h
(from the MDA 100 spectral month, 337 s; not re-measured at 550).

**Same requests with the code before today** (GP threaded in one process,
whole-period kernels, pool for every sum): 38 buoys 2013 bulk 105 s (GP 81 s,
sum 22 s), spectral 173 s (GP 122 s, sum 49 s); 1 site per year bulk ~60 s
(GP 73-76 s per call, sum 11-18 s), spectral ~97 s (GP 127-138 s, sum
11-12 s).

**GP step experiments** (one year, buoys, MDA 550, 40 goal-partitions):
- One goal-partition, one thread: whole year in one kernel 154 s; 4096 rows
  76 s; 2048 48 s; 1024 29 s; 512 20 s (12 threads: 53 -> 15 s; 256 rows
  slower). The kernel of a few hundred rows x 550 training points stays in
  cache.
- All 40, with 512-row chunks: one process x 36 threads 121 s; 36 x 1
  thread 31 s; 18 x 2, 12 x 3, 9 x 4: 23-24 s; 6 x 6 28 s; 40 x 1 26 s.
  2-4 threads reproduce the threaded PCs exactly; 1 thread differs by up to
  3e-2 (float32 summation order), hence `GP_THREADS = 2`.
- The GP does not depend on the number of sites; the PCA inverse does
  (isobaths: most of the 17 s).

**Sum experiments** (2026-10-15, MDA 100, no wind):
- Peak memory 6.55x the float32 spectra array, for every shape tried.
- One core: 22 us per step x site x contribution + 104 us per step x site
  (stats; computing only the requested stats made the sum 1.4x faster).
- 8 / 18 / 36 workers: 7.1 / 10.9 / 14.1x one core (memory bandwidth).
- Blocks per worker: 1 site 92 s, 2 sites 81 s, 5 sites 84 s, 17 sites 113 s
  (isobaths month) -> ~2 GiB per worker.
- Worker pool start ~6 s (36 workers); with sending the inputs, ~10 s.

**Fits** (2026-10-15): one GP fit (38 buoys, MDA 550, 100 epochs) 108 s on
1 thread, 58 s on 36; one fit <= 1.2 GiB (isobaths, 4,977 sites, 179 PCs);
1 MDA + 30 fits (MDA 50, 30 epochs) 16 s in 30 processes; a pooled fit equals
the in-process fit exactly. The GP stops each PC early (patience 30), so 5000
epochs is a ceiling, not the cost.

## Numerical checks

- Store + resources vs the earlier code (MDA 100, 5 Dec 2013, buoys): bulk
  and spectral within 1.5e-4 m Hs.
- GP processes + chunks vs the old code (MDA 550, 2013, 38 buoys, bulk, wind
  on, against the notebook-09 store): Hs within 3.8 mm wherever the wind
  factor K is unchanged. **K changed by > 1e-3 at 1.2% of site-hours (up to
  2.6)**, which explains all 256 site-hours with Hs differences > 5 cm: the
  GBM of K is a step function, so mm-level input changes can flip it. Tp
  jumped > 1 s at 18 site-hours (two nearly equal spectral peaks).

## Left for Monday

1. **Not measured yet**: isobaths 2013 bulk and spectral at MDA 550 with
   full forcing (the planner proposed 11-month calls), and `fit_all` at MDA
   550 / 5000 epochs for all goals and sets (10 MDAs + 30 fits).
2. **Wind correction K**: smooth it (e.g. a smoother regressor, monotone
   constraints, or averaging K over a small neighbourhood of the inputs)
   before trusting wind-corrected extremes.
3. **One-site spectral sum**: 12 s per year on one core (a pool costs ~10 s);
   a persistent pool across calls, or a few workers on time spans, could
   halve it.
4. **Time model** (`hours`): GP constants fitted on a loaded node (estimates
   22 / 40 / 108 s vs measured 44 / 41 / 149 s); re-fit on a quiet node.
5. **GPU**: untested (no GPU here); the GP jobs then run in the main
   process, `MONTHS_PER_CALL["cuda"] = 4`.
6. Commit the branches above; update `CLAUDE.md` if anything here becomes
   permanent.

## Re-running the benchmarks

Scripts in `Netherlands/tmp/benchmarks/` (git-ignored): `bench.py` (one
reconstruction with timing and peak memory; `label time set spectral [site|-]
[save]`), `compare_store.py` (new vs an old store, read-only), `fit_bench.py`
(all goals at MDA 550), `run_all.sh`, `rec.log` (today's output).

They run in a **scratch copy** of the case so your stores and models are not
touched: a folder named `Netherlands/` holding a copy of `Netherlands.json`,
`utils/` and `outputs/` (~150 MB), with `inputs/` (and, for fits,
`SnapWave_CASES_Metamodel/`) as symlinks to the real case; for fits copy only
`outputs/output_sites_snapwave.csv`. Run from that folder with
`PYTHONPATH=.`. Scripts that start processes need `if __name__ ==
"__main__":` (spawn re-runs the script otherwise).
