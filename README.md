# HyGloW — Hybrid Global Wave downscaling

HyGloW downscales offshore wave hindcasts to nearshore sites without the cost
of running a full spectral wave model at every timestep. For each region, a
small number of physics-based **SnapWave** propagation runs train a
statistical **metamodel** (MDA → PCA → Gaussian Process) that then
reconstructs full time series — bulk parameters or spectra — at nearshore
buoys and output sites cheaply.

A **case** is a self-contained folder for one region: its own area of
interest, mesh, goal points, observations catalogue, and results. Cases can
live in this repository or in a contributor's own repo — what makes a folder
a HyGloW case is conformance to the [case contract](hyglow/contract/CONTRACT.md).

## Layout

| Path             | What it is                                                                          |
| ---------------- | ------------------------------------------------------------------------------------ |
| `hyglow/`        | The engine, contract, and CLI (`python -m hyglow ...`) — scaffolds and validates cases |
| `Samoa/`         | Reference case — Samoan archipelago                                                |
| `NewSouthWales/` | Reference case — NSW coast, Australia                                              |
| `GUI/`           | *HyGloW Results Explorer* — a Dash portal that reads any case's contract files       |
| `EuropePolygons/` | Candidate area-of-interest polygons for future European cases ([details](EuropePolygons/README.md)) |

## Quickstart

```bash
# Scaffold a new self-contained case (engine + notebooks + config skeleton)
python -m hyglow start --name MyCase

# Check a case against the contract as you go
python -m hyglow validate MyCase/            # development mode: gaps are warnings
python -m hyglow validate MyCase/ --publish  # strict: portal-ready check

# Browse existing cases
cd GUI && python app.py   # http://localhost:8050
```

Full CLI reference (including `update`, which re-syncs a case's engine code
from the template): [`hyglow/README.md`](hyglow/README.md).

## Cases

| Case                                   | Region             | Status                                                                                     |
| --------------------------------------- | ------------------- | -------------------------------------------------------------------------------------------- |
| [Samoa](Samoa/README.md)               | Samoan archipelago | Full pipeline: mesh → LHS → metamodel fit → offshore hindcast → buoy/spectra validation      |
| [NewSouthWales](NewSouthWales/README.md) | NSW coast, Australia | Mesh, 26 goals, and a 35-buoy observations catalogue done; metamodel fit and validation pending |

## The pipeline, in one paragraph

For a region, the numbered notebooks walk through: clipping bathymetry to an
area of interest → building an unstructured mesh → cataloguing nearshore wave
observations (buoys, tide gauges, ...) → choosing offshore "goal" points and
their propagation sectors → sampling each sector with a Latin Hypercube
design and running SnapWave once per sample (the **metamodel** training set)
→ fitting an MDA → PCA → GP emulator per goal → downloading an offshore
hindcast and running SnapWave once per hindcast timestep (the **dynamic**
reference run) → validating the metamodel against that dynamic run and
against real buoy records.

Step-by-step notebook table: [`hyglow/template/README.md`](hyglow/template/README.md).

## Environment

Conda environment with `bluemath_tk`, `wavespectra`, `xarray`, `geopandas`,
`rioxarray`, `pandas`, `scipy`, `matplotlib`, plus
[SnapWave](https://github.com/danoroelvink/snapwave/) (external binary,
compiled separately). See a case's README for its exact package list, and
`GUI/requirements.txt` for the portal's (lighter) dependencies.

## Contributing a case

1. `python -m hyglow start --name YourCase`
2. Fill in `YourCase/YourCase.json` and work through the numbered notebooks.
3. `python -m hyglow validate YourCase/ --publish`
4. Open a PR here, or host it in your own repo and point the GUI catalog
   (`GUI/catalog/cases.yaml`) at it — the contract is the only thing that
   needs to match.

Engine code (`utils/`) is **vendored** into every case so each one runs
standalone. If you fix a bug in your case's copy, please also land it in
[`hyglow/template/`](hyglow/template/) so `python -m hyglow update` can carry
the fix to every other case.
