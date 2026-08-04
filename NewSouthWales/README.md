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

Checked with `python -m hyglow validate NewSouthWales/`:

| Stage                                  | State                                       |
| --------------------------------------- | -------------------------------------------- |
| AoI, GEBCO clip, mesh (01–02)          | ✅ done — `inputs/GRID/NewSouthWales_mesh.nc` |
| Observations catalogue (03)            | ✅ done — 35 buoys, see below                |
| Goals + propagation sectors (04)       | ✅ done — 26 goals                           |
| Metamodel LHS cases (05)               | ✅ configured — 500 samples/goal, not yet built/run |
| Metamodel fit + holdout validation (06) | ⬜ pending                                    |
| Offshore hindcast — WHACS (07)         | ✅ done — `inputs/WHACS/processed/*.zarr`     |
| Dynamic reference runs (08)            | ✅ configured, not yet built/run              |
| Hindcast / buoy / spectra validation (09–11) | ⬜ pending                              |

In short: the domain is fully set up (mesh, goals, observations, offshore
forcing) and both SnapWave case generators (`05`, `08`) have been reviewed
via their pre-run diagnostics (`wrapper.plot_configuration()` /
`plot_configuration_map()`) — see below. Building the case folders, running
SnapWave, fitting the metamodel, and validating are the next steps.

---

## Quick start

```python
from utils.config import PATHS, SIMULATIONS, load_goals, resolve
from utils.observation import onshore_sites as obs
from utils.simulation.snapwave_wrappers import SnapWaveMetaModelWrapper, SnapWaveDynamicModelWrapper
```

After editing `NewSouthWales.json`: restart kernel or `utils.config.reload()`.

---

## Pipeline (notebooks)

| Step  | Notebook                                       | Purpose                                       |
| ----- | ----------------------------------------------- | ---------------------------------------------- |
| 00    | `00_Buoys_Inspection.ipynb`                     | Raw buoy inventory: coverage, gaps, completeness |
| 00    | `00_Course_Figures.ipynb`                       | Slide-ready PNGs for teaching                  |
| 01    | `01_Generate_tiff_from_GEBCO.ipynb`             | Clip GEBCO bathymetry to AoI                   |
| 02    | `02_Create_mesh_by_bluemesh2d.ipynb`            | Unstructured mesh                              |
| 03    | `03_Generate_Observations_Configuration.ipynb`  | Buoy/site GeoJSON catalogue                    |
| 04    | `04_Select_Goals_by_Hindcasts.ipynb`            | Goal locations + propagation sectors           |
| 05    | `05_Metamodel_Cases_Generation.ipynb`           | LHS SnapWave cases per goal                    |
| 06    | `06_Validate_Metamodel.ipynb`                   | Fit + holdout validation                       |
| 07    | `07_Download_Process_WHACS.ipynb`               | Offshore hindcast zarr                         |
| 08    | `08_Dynamic_Cases_Generation.ipynb`             | Dynamic reference runs                         |
| 09–11 | Validation notebooks                            | Reconstruct vs dynamic / buoys                 |

```
Goals → LHS (05) → fit (06) → WHACS (07) → dynamic (08)
                              → mm.reconstruct() hindcast
                              → validate bulk (09–10) → validate spectra (11)
```

---

## Buoy observations (notebooks 00, 03)

Raw buoy data lives under `inputs/Observations/raw/` in two networks:

| Network                      | Count | Source                                                | Sampling |
| ------------------------------ | ----- | ------------------------------------------------------ | -------- |
| Offshore MHL Waverider buoys | 7     | Manly Hydraulics Laboratory (Sydney, Port Kembla, Crowdy Head, Coffs Harbour, Byron Bay, Eden, Batemans Bay) | hourly, since 1976 |
| Nearshore Spotter buoys      | 28    | NSW DCCEEW / SEED nearshore wave buoy program (per-deployment) | 30 min |

`00_Buoys_Inspection.ipynb` matches every raw file to its catalogue entry and
reports record spans, sampling completeness, directional coverage, and which
years have the most combined data (useful for picking a validation
`period`) — inventory saved to `outputs/buoys_raw_inventory.csv`.

`03_Generate_Observations_Configuration.ipynb` builds the actual catalogue
(`inputs/Observations/NewSouthWales_observations.json`) that the rest of the
pipeline reads: `lon`, `lat`, `depth`, `institution`, `recording` period per
`instrument_id`. Offshore buoy coordinates come from public AODN /
researchdata.edu.au metadata (deployment-area centroids, accurate to
~station-watch-circle precision, not survey-grade); nearshore buoy
coordinates come from the NSW SEED deployment shapefile (survey-grade).

---

## Pre-run diagnostics (notebooks 05, 08)

Before building or running SnapWave cases, both wrapper classes expose:

```python
wrapper.plot_configuration(...)       # map + per-goal direction bars + text summary
wrapper.plot_configuration_map(...)   # same map alone, bigger — for reading goal labels on a 26-goal coastline
```

The per-goal bar panel means different things depending on the wrapper:

- **Metamodel** (`SnapWaveMetaModelWrapper`) — the crisp outline bounds the
  *actual sampled* direction interval, verifying the LHS design fills its
  configured sector.
- **Dynamic** (`SnapWaveDynamicModelWrapper`) — the crisp outline bounds the
  goal's *configured* sector instead; the shaded density is the real
  hindcast, which is free to (and does) extend past it — that gap is exactly
  the "% outside sector" figure in the text panel.

With 26 goals along a folded coastline, colour is assigned by goal order
(not a repeating qualitative palette) and labels/markers shrink and fan out
across four offsets automatically once a case has more than ~15 goals.

---

## Configuration (`NewSouthWales.json`)

Same sections as [Samoa](../Samoa/README.md#configuration-samoajson); the
values actually set for this case:

| Section                      | Value                                                    |
| ------------------------------ | --------------------------------------------------------- |
| `offshore_waves`             | `WHACS` → `inputs/WHACS/processed/NewSouthWales_WHACS_goal.zarr` |
| `simulations.num_samples_per_goal` | 500                                                 |
| `simulations.bounds`         | `tp [1, 20]`, `spr [1, 45]`, `wl [-2.0, 2.0]`            |
| `metamodel.mda_candidates`   | `[50, 100, 150, 250, 450]`                               |
| `output_sites_config.isobaths` | `5m, 10m, 20m, 50m`, clipped to AoI, buoys included    |

---

## Python modules (`utils/`)

| Module                          | Role                                                   |
| ---------------------------------- | --------------------------------------------------------- |
| `config.py`                      | JSON load, `resolve()`, `validate_metamodel_config()`    |
| `model.metamodel`                | Fit, predict, reconstruct                                 |
| `evaluation.validation`          | Holdout metrics/plots, buoy comparison                    |
| `observation.onshore_sites`      | Observations GeoJSON I/O, buoy site ids                   |
| `simulation.snapwave_wrappers`   | SnapWave case build/run, pre-run diagnostics              |
| `simulation.lhs_cases`           | Latin Hypercube design per goal                            |
| `simulation.offshore_forcing`    | WHACS → SnapWave boundary forcing                           |
| `geometry.output_sites`          | Isobath/mesh/buoy output-site catalogue                     |
| `viz.case_plots`                 | Case/domain maps, pre-run configuration diagnostics         |
| `viz.snapwave_case_plots`        | Post-run per-case diagnostics                                |

---

## Tests

From `NewSouthWales/`:

```bash
python -m unittest discover -s tests -v
```

Covers direction handling, logging config, metamodel helpers, offshore
forcing, onshore site helpers, output sites, and site groups.

---

## Next steps

- [ ] `05_Metamodel_Cases_Generation.ipynb` — build and run the LHS cases
- [ ] `06_Validate_Metamodel.ipynb` — `mm.fit_all()`, holdout sweep
- [ ] `08_Dynamic_Cases_Generation.ipynb` — build and run the dynamic reference
- [ ] `09`–`11` — reconstruct-vs-dynamic, buoy, and spectral validation

---

## Dependencies

Conda environment with: `bluemath_tk`, `wavespectra`, `xarray`, `geopandas`,
`rioxarray`, `pandas`, `scipy`, `matplotlib` — plus
[SnapWave](https://github.com/danoroelvink/snapwave/) (external binary).
