# Reconstruction: how it runs, how long it takes, what is open

How `mm.reconstruct` uses the machine (state on 2026-10-26, bluemath_tk
`develop`). The method and the user summary: [`README.md`](README.md).

## How a reconstruction runs

`mm.reconstruction_resources(...)` prints the machine and proposes settings
(at most 75% of its cores and free memory, never SLURM); you edit them if
you want and pass them to `mm.reconstruct(..., resources=res)`.

1. **Cache.** Results are stored per site and calendar month, one zarr per
   setup in `tmp/reconstruct/{offshore}/`. Only the missing site-months are
   computed; a change of setup or of an input file rebuilds the store.
2. **Calls.** The missing months go in calls of up to 12 months x every site
   (fewer months, then fewer sites, if memory is short).
3. **GP step.** Every goal x offshore partition is one job (it depends on the
   offshore sea states only, not on the sites), run in processes of 2
   threads, predicting 512 hours at a time.
4. **PCA step.** Principal components -> Hs, Tp, direction and spreading at
   the sites, in the main process.
5. **Sum.** Per site and hour, the JONSWAP x cos²ˢ spectra of all goals are
   added and the bulk parameters computed, in worker processes on site x hour
   blocks of 0.5-2 GiB each. A small sum (one site) runs in the main process.
6. **Store.** Whole months are written per site (and the spectra, with
   `save_spectra=True`), then the request is read back.

## Timings: the buoys

Netherlands (10 goals, 38 buoys), WHACS, MDA 550, water level from the
forcing; one 48-core Xeon Gold 5418Y node, 36 cores used, shared with other
jobs.

| Request | Time | Peak memory |
| --- | --- | --- |
| 38 buoys, one year, bulk | 44 s | 53 GiB |
| 38 buoys, one year, spectral | 84 s | 76 GiB |
| 38 buoys, one year, spectral, spectra saved | ~97 s (GP 28 s, sum 43 s, writing the rest) | 77 GiB |
| 38 buoys, 1979-2023, spectral, spectra saved | ~73 min (45 calls of ~97 s), ~90 GB of spectra | 77 GiB |
| One buoy, 1980-2023, bulk | 14 min (44 calls of ~19 s) | 11 GiB |
| One buoy and year read back from the cache | 1 s | 1 GiB |

The 45-year run is extrapolated from its first 11 years (2026-10-26); the
rest was measured on 2026-10-16 (one-year bulk and spectral with the wind
correction on). Every call takes about the same time, so a long period costs
the number of years times one year.

## Open

- **Time estimate.** The proposal's time is fitted on the buoys and runs
  short for long calls over many sites (isobaths, 11 months: estimated 18
  min, the sum alone took ~37). In those calls the sum blocks hold one site
  over ~2,600 hours and run ~1.7x slower per site-hour than the 3 sites x
  720 hours of a one-month call; capping the hours per block may fix both.
- **Wind correction K.** The K regressor is a step function: mm-level input
  changes flip it at ~1% of site-hours (up to 2.6). Smooth it before
  trusting wind-corrected extremes.
- **One site, spectral.** ~12 s of sum per year on one core; a worker pool
  kept between calls would roughly halve it.
- **GPU.** Untested (the GP jobs then run in the main process).
