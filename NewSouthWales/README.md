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
- **Observations catalogue:** `inputs/Observations/NewSouthWales_observations.json`
  — 35 wave buoys (7 offshore MHL Waverider buoys + 28 nearshore Spotter
  deployments), built in notebook 03 and inspected in notebook 00

**Environment:** conda `bluemath-311-dev`; run notebooks with
`NewSouthWales/` as cwd.

---

## Status

Checked with `python -m hyglow validate NewSouthWales/` (run from the repo root):

| Stage                                                       | State                                                             |
| ----------------------------------------------------------- | ----------------------------------------------------------------- |
| AoI, GEBCO clip, mesh (01–02)                               | ✅ done — `inputs/GRID/NewSouthWales_mesh.nc`                      |
| Observations catalogue (03)                                 | ✅ done — 35 buoys, see below                                      |
| Goals + propagation sectors (04)                            | ✅ done — 26 goals                                                 |
| Metamodel LHS cases, built + run (05)                       | ✅ done — 500 samples/goal                                         |
| Metamodel fit + holdout validation (06)                     | ✅ done — all 26 goals, holdout R² ≈ 0.95–0.98 for hs/tp/dir at 5m |
| Offshore hindcast — WHACS (07)                              | ✅ done — `inputs/WHACS/processed/*.zarr`                          |
| Dynamic reference runs, built + run (08)                    | ✅ done                                                            |
| Reconstruct-vs-dynamic / buoy / spectral validation (09–11) | ✅ run at least once — under active review, see note below         |

Full pipeline has been run end-to-end. Built SnapWave case folders
(`SnapWave_CASES_Metamodel`, `SnapWave_CASES_Dynamic`) live outside the repo
at the path set in `NewSouthWales.json` → `simulations.metamodel_cases` /
`dynamic_cases` (this machine points them at `/hyglowdata/NewSouthWales/` —
change that to wherever you want them for your own run). Notebook outputs
aren't currently committed (kept local while the buoy-processing work below
is still being checked), so don't expect populated `outputs/` cells if you
pull this fresh — re-run the notebooks to regenerate.

---

## Happy Coding ;)

Just the Aoi.shp file is uploaded in inputs folder, go to notebook 1 and start coding!!
