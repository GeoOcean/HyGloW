# NewSouthWales — hybrid wave downscaling case

A **hybrid downscaling** workflow for nearshore waves along the New South
Wales coast, Australia — from Byron Bay in the north to Eden in the south.
SnapWave LHS simulations feed a **GPR metamodel** (MDA → PCA → GP) so full
offshore hindcasts can be reconstructed without running the model at every
timestep, following the same pipeline as the [Samoa](../Samoa/README.md)
reference case.

- **Config:** `NewSouthWales.json` (auto-loaded by `utils.config`)
- **Code:** `utils/` + numbered notebooks
- **Domain:** 26 goal points along the coast, offshore hindcast forcing from WHACS
- **Reconstruction catalogue:** 3,860 reachable sites — 35 buoys + isobath
  points across the 5m/10m/20m/50m depth bands (`mm.common_model_sites()`)
- **Observations catalogue:** `inputs/Observations/NewSouthWales_observations.json`
  — 35 wave buoys (7 offshore MHL Waverider buoys + 28 nearshore DCCEEW
  Spotter deployments), built in notebook 03 and inspected in notebook 00.
  Only 3 of the 7 offshore buoys (Sydney, Byron Bay, Batemans Bay) have raw
  directional spectra processed (notebook 00's spectral pass) — the other 4
  offshore buoys and all 28 nearshore buoys are bulk-only. Sydney's own SD
  (spectral density) file is corrupted in the raw MHL archive (entirely
  null bytes), so Sydney's processed spectra carry the peak-normalized
  shape only (`efth_norm`), not an absolute-scale spectrum (`efth`) — see
  `utils/observation/raw_spectra.py`'s module docstring.

**Environment:** conda `bluemath-311-dev`; run notebooks with
`NewSouthWales/` as cwd.

---

## Status

Checked with `python -m hywaves validate NewSouthWales/` (run from the repo root):

| Stage | State |
| --- | --- |
| AoI, GEBCO clip, mesh (01–02) | ✅ done — `inputs/GRID/NewSouthWales_mesh.nc` |
| Observations catalogue (03) | ✅ done — 35 buoys, see below |
| Goals + propagation sectors (04) | ✅ done — 26 goals |
| Metamodel LHS cases, built + run (05) | ✅ done — 500 samples/goal |
| Metamodel fit + holdout validation (06) | ✅ done — all 26 goals, holdout R² ≈ 0.95–0.98 for hs/tp/dir at 5m |
| Offshore hindcast — WHACS (07) | ✅ done — `inputs/WHACS/processed/*.zarr` |
| Dynamic reference runs, built + run (08) | ✅ done |
| Reconstruct vs dynamic, bulk, all sites (09) | ✅ done — all 26 goals, all 3,860 sites, `workers=` parallel reconstruction (see below) |
| Offshore/spectral buoy validation (10a) | ✅ done — 3 buoys with real spectra (Sydney, Byron Bay, Batemans Bay), 2001–2011, bulk + spectral |
| Nearshore buoy validation (10b) | ✅ done — 18 of the 28 nearshore buoys with usable coverage, 2013 & 2017–2023, bulk + spectral |
| Spectral shape validation (11) | ✅ done — buoy vs bulk-reconstruction vs spectral-reconstruction, same 3 offshore buoys, 2001–2011 |

Notebooks 06, 09, and 10a/10b end with a save cell (`outputs/goal_*/validation_metrics.csv`,
`outputs/metamodel_validation/*.csv`, `outputs/buoy_validation/*_summary_*.csv` +
`.../figures/<site>.png`) — the contract files the [GUI](../GUI/README.md)'s
Goals/Metamodel/Buoy Validation tabs and per-buoy map popups read directly.
`catalog/cases.yaml` pins this case's production MDA size at 450
(`metamodel.mda_candidate`) so all three tabs read a consistent train size.

There's also `09_extra_Reconstruction_Timing.ipynb` — not part of the
numbered pipeline, just a quick wall-clock benchmark for demoing how fast
inference is once a case is fitted: bulk vs. spectral metamodel, one buoy's
output site, **all 26 active goals** (the real production path — restricting
to just the buoy's closest goal, as a first pass at this notebook did, is an
easier but different computation, not this one). Fresh compute
(`use_cache=False`), one representative year measured directly and
extrapolated to the full ~45-year WHACS record (fair — cost scales with
timesteps processed, not which ones):

| Model | 1-year sample | Extrapolated (~45 years) | Per timestep |
| --- | --- | --- | --- |
| Bulk metamodel | 5.8 min | 262.2 min (~4.4 h) | 39.89 ms |
| Spectral metamodel | 15.1 min | 680.5 min (~11.3 h) | 103.50 ms |
| **Total** | **20.9 min** | **943 min (~15.7 h)** | — |

That's the real production cost — reconstructing the entire WHACS record for
one site, but every one of the 26 goals along the way, not just the one that
happens to cover that site (see "Reconstructing at scale" below for why that
distinction is roughly a 20x difference, not a rounding error).

Full pipeline has been run end-to-end. Built SnapWave case folders
(`SnapWave_CASES_Metamodel`, `SnapWave_CASES_Dynamic`) live outside the repo
at the path set in `NewSouthWales.json` → `simulations.metamodel_cases` /
`dynamic_cases` (this machine points them at `/hyglowdata/NewSouthWales/` —
change that to wherever you want them for your own run). Notebook outputs
aren't currently committed, so don't expect populated `outputs/` cells if
you pull this fresh — re-run the notebooks to regenerate.

---

## Reconstructing at scale

`mm.reconstruct()` now takes a `workers=` parameter (default `1`, unchanged
sequential behaviour) that parallelizes the wavespectra summation step —
the actual bottleneck at this case's full 3,860-site scale, not GP
prediction (which costs about the same regardless of how many sites are
requested). At full scale, one month of bulk-mode reconstruction across all
3,860 sites took ~54 minutes single-threaded; `workers=N` splits that
across a process pool, each worker capped to a fair share of both memory
and CPU threads so N workers don't oversubscribe the host. Pick `N` with
some cores left over for everything else running on the machine — e.g.
`workers=os.cpu_count() // 2`.

**GP inference itself is capped by GPU memory, not compute.** `reconstruct()`
batches consecutive months into as few GP prediction calls as possible — one
call is much cheaper than several, since each call pays a large fixed
overhead regardless of how many months it covers. The cap
(`RECONSTRUCT_GPU_BATCH_MONTHS_CAP` in `metamodel.py`) is a hard VRAM ceiling
on this machine's 6 GiB GPU, not a compute-speed one — it doesn't shrink just
because the GPU is fast, only with more VRAM. Re-measured directly on GPU
with **all 26 active goals** (the case that actually matters — an earlier
pass here only tested 1-2 goals, which turned out to have much more VRAM
headroom than the real workload): bulk tolerates at least 8 months/call
(18.9 s/month), but spectral (4x bulk's partition data per call) is the
binding constraint — clean at 4 months/call (73.3 s/month), thrashing near
the OOM edge at 6 (>300s for what should be a ~2 min call), outright CUDA OOM
at 8. **4 is now the default** — the largest value safe for both modes in one
call, and a real ~1.2-1.5x speedup over the old default of 3 for both. A GPU
with meaningfully more VRAM would let this rise further — spectral-at-26-goals
is specifically what to re-test on better hardware, not bulk (which already
has headroom to spare).

**Goal count is close to a direct multiplier, too — restrict `goals=` to
what you actually need.** `reconstruct()` loops goals one at a time (loads
that goal's GP, predicts, moves on) — no sharing of work across goals, so
cost scales with how many are requested, not flat. Measured on a single
month: 1 goal 6.4s, 2 goals 9.8s, 4 goals 20.4s — roughly linear (not exactly,
there's a small fixed cost per call on top, so 4 goals is ~3.2x 1 goal's
time, not 4x). A validation reconstruction that only needs one buoy's answer
is genuinely cheaper restricted to that buoy's closest goal (see
`val.closest_goals_table`) than reconstructed across all 26 — that's not an
approximation of cost, it's a smaller real computation. `09_extra` (see
"Status" above) intentionally uses all 26 anyway, because it's benchmarking
the production path (notebook 09's domain-wide reconstruction), not one
buoy's own validation.

**Only pass `save_spectra=True` when you actually need the full spectra.**
The case JSON defaults it to `true` (for notebook 11), but a full `(time,
sites, freq, dir)` array is ~1000x bigger than the bulk stats most
notebooks actually use, and results currently accumulate in memory across
every batch before the final write — at full site-count scale this is
enough to threaten OOM. Notebook 09 explicitly passes `save_spectra=False`
for exactly this reason.

The validation buoys' periods are already reconstructed and cached under
`tmp/reconstruct/` (bulk + spectral both), so notebooks 10a/10b/11 should
mostly hit cache and run fast:

- Offshore/spectral buoys (Sydney, Byron Bay, Batemans Bay): 2001–2011
- Nearshore buoys (18 of 28, see notebook 10b): 2013, 2017–2023

A fresh full-catalogue reconstruction for a *new* period will pay the real
cost above — budget accordingly, and consider `workers=` before running one.

---

## Happy Coding ;)

Just the Aoi.shp file is uploaded in inputs folder, go to notebook 1 and start coding!!
