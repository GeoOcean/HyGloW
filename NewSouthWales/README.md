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

Checked with `python -m hyglow validate NewSouthWales/` (run from the repo root):

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
