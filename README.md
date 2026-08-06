# HyGloW — Hybrid Global Wave downscaling

HyGloW turns an offshore wave hindcast into nearshore wave conditions —
buoys, tide gauges, any point you care about — without running a full
spectral wave model at every timestep. For your region, a small number of
physics-based **SnapWave** runs train a statistical **metamodel** that then
reconstructs the whole time series (bulk parameters or full spectra) cheaply.

If you have an area of interest, a bathymetry source, and an offshore wave
hindcast, you can have nearshore wave conditions there. That's what this
repo helps you do.

## Quickstart

**0. Install dependencies** (once):

- A conda/Python environment with `bluemath_tk`, `wavespectra`, `xarray`,
  `geopandas`, `rioxarray`, `pandas`, `scipy`, `matplotlib`.
- [**SnapWave**](https://github.com/danoroelvink/snapwave/) — the nearshore
  propagation model. It's an external binary: clone that repo and compile it
  yourself (see its own README for build instructions). You'll need it
  before the notebooks that actually run cases (step 05 onward below).
- `bluemesh2d` (the mesher) — installed per case with `pip`, see step 1.

**1. Create your case:**

```bash
python -m hyglow start --name MyCase
cd MyCase
pip install -r requirements.txt
```

This gives you a self-contained folder: engine code, numbered notebooks,
and a config skeleton. Nothing is downloaded for you — you fill in your
region's data as you go.

**2. Open `MyCase/README.md`** and work through the numbered notebooks in
order — that README is your main guide from here on, with the full
notebook-by-notebook walkthrough.

**3. Check your progress anytime:**

```bash
python -m hyglow validate MyCase/            # development mode: gaps are warnings
python -m hyglow validate MyCase/ --publish  # strict: is this ready to share?
```

## Want to see a finished example first?

| Case                                     | Region               | Status                                                                                          |
| ---------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------------- |
| [Samoa](Samoa/README.md)                 | Samoan archipelago   | Full pipeline: mesh → LHS → metamodel fit → offshore hindcast → buoy/spectra validation         |
| [NewSouthWales](NewSouthWales/README.md) | NSW coast, Australia | Mesh, 26 goals, and a 35-buoy observations catalogue done; metamodel fit and validation pending |

Or browse every published case in the results portal:

```bash
cd GUI && python app.py   # http://localhost:8050
```

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

## Repository layout

| Path              | What it is                                                                                          |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| `hyglow/`         | The engine, contract, and CLI (`python -m hyglow ...`) — scaffolds and validates cases              |
| `Samoa/`          | Reference case — Samoan archipelago                                                                 |
| `NewSouthWales/`  | Reference case — NSW coast, Australia                                                               |
| `GUI/`            | *HyGloW Results Explorer* — a Dash portal that reads any case's contract files                      |
| `EuropePolygons/` | Candidate area-of-interest polygons for future European cases ([details](EuropePolygons/README.md)) |

Full CLI reference (including `update`, which re-syncs a case's engine code
from the template): [`hyglow/README.md`](hyglow/README.md).

## Contributing a case back

A **case** is a self-contained folder for one region — its own area of
interest, mesh, goal points, observations catalogue, and results. Cases can
live in this repository or in your own repo; what makes a folder a HyGloW
case is conformance to the [case contract](hyglow/contract/CONTRACT.md).

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
