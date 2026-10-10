<p align="center">
  <img src="assets/logo.svg" alt="HyGloW — Hybrid Global Wave downscaling" width="100%">
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-active%20development-orange">
  <img alt="Contributions" src="https://img.shields.io/badge/contributions-welcome-brightgreen">
  <img alt="Built for" src="https://img.shields.io/badge/built%20for-oceanographers-1f93b3">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-GPLv3-blue"></a>
</p>

HyGloW turns an offshore wave hindcast into nearshore wave conditions —
buoys, tide gauges, any point you care about — without running a full
spectral wave model at every timestep. For your region, a few hundred
stationary runs of a nearshore wave model (**SnapWave** or **SWAN**) per
offshore "goal" point train a statistical **metamodel** that then
reconstructs the whole time series (bulk parameters or full spectra) cheaply.
Wind and water level can be added on top (see *The pipeline* below).

If you have an area of interest, a bathymetry source, and an offshore wave
hindcast, you can have nearshore wave conditions there. That's what this
repo helps you do.

## What this repository is

HyGloW is under active development. This repo is the **base code for the
whole HyGloW ecosystem** — the engine, the case contract, and the results
portal (`GUI/`) all live here, and every regional use case is built from it.
There are two ways to use it:

- **Just want nearshore waves for your region?** Fork or clone this repo,
  run `hywaves start`, and work in your own case folder from there —
  Quickstart below. Once scaffolded, a case is self-contained: you don't
  need to keep up with this repo unless you want engine improvements later
  (`hywaves update`).
- **Want to help build the toolkit itself** — the engine, the GUI, the case
  contract? Fork it and open a PR against `hywaves/` or `GUI/`, same as any
  open-source project.

Expect rough edges while this is young. Issues and PRs are welcome either way.

## Quickstart

**0. Install dependencies** (once):

- A conda/Python environment (Python ≥ 3.11) with `bluemesh2d`, `wavespectra`,
  `xarray`, `geopandas`, `rioxarray`, `pandas`, `scipy`, `matplotlib` and
  **`bluemath_tk` from its `develop` branch**. The released `bluemath-tk`
  packages don't have the HyWaves code yet:

  ```bash
  pip install "git+https://github.com/GeoOcean/BlueMath_tk.git@develop"
  ```
- The nearshore wave model, an external binary:
  [**SnapWave**](https://github.com/danoroelvink/snapwave/) (default; compile
  it yourself, see its README) or **SWAN**. The case JSON picks one
  (`simulations.model`). You'll need it from the notebooks that run cases
  (step 05 onward): as `snapwave` / `swan.exe` on your PATH, or, on the
  GeoOcean cluster, through SLURM job arrays (`wrapper.run_cases_slurm()`),
  which load the cluster modules for you. Both models are ready to use; the
  results published so far were made with SnapWave.

**1. Create your case:**

```bash
python -m hywaves start --name MyCase
cd MyCase
```

This gives you a self-contained folder: engine code, numbered notebooks,
and a config skeleton. Nothing is downloaded for you — you fill in your
region's data as you go.

**2. Open `MyCase/README.md`** and work through the numbered notebooks in
order — that README is your main guide from here on, with the full
notebook-by-notebook walkthrough.

**3. Check your progress anytime:**

```bash
python -m hywaves validate MyCase/            # development mode: gaps are warnings
python -m hywaves validate MyCase/ --publish  # strict: is this ready to share?
```

**4. Publish it** — `python -m hywaves publish MyCase/` validates, then
writes `MyCase/portal/`: the small bundle (a few MB) the results portal
reads. Commit that folder.

## Want to see a finished example first?

| Case                                     | Region                   | Status                                                                                                                                  |
| ---------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| [Netherlands](Netherlands/README.md)     | Belgian and Dutch coasts | **Reference case**, latest engine (hywaves 0.11.0): 10 goals, 38 buoys, ERA5 wind + CODEC water level, wind correction. SnapWave results; SWAN set up and piloted |
| [NewSouthWales](NewSouthWales/README.md) | NSW coast, Australia     | Full pipeline done and published on an older engine (≤ 0.9.0): 26 goals, 35 buoys                                                       |
| [Samoa](Samoa/README.md)                 | Samoan archipelago       | Full pipeline done and published on an older engine (≤ 0.9.0)                                                                           |

The two older cases still run their own vendored engine; they move to the
current one (`python -m hywaves update` + the config changes in the
[contract changelog](hywaves/contract/CONTRACT.md#changelog)) the next time
they are re-run.

Or browse every published case (each one's `portal/` bundle) in the results portal:

```bash
cd GUI && python app.py   # http://localhost:8050
```

## The pipeline, in one paragraph

For a region, the numbered notebooks walk through: clipping bathymetry to an
area of interest → building an unstructured mesh → cataloguing nearshore wave
observations (buoys, tide gauges, ...) → choosing offshore "goal" points and
their propagation sectors → sampling each sector with a Latin Hypercube
design (Hs, Tp, direction, spreading, water level) and running the model once
per sample, with waves only at that goal (the **metamodel** training set) →
fitting an MDA → PCA → GP emulator per goal and per group of output sites →
downloading an offshore hindcast, reconstructing it by summing the goals'
predicted spectra, and running the model once per hindcast hour with every
goal, the wind and the water level (the **dynamic** reference run) →
validating the reconstruction against that dynamic run and against real buoy
records.

Optional, since hywaves 0.11.0: the metamodel cases have no wind, so a **wind
correction** adds the locally generated wind sea in the reconstruction. Its
growth factor is learned from paired model runs with and without wind, using
the fetch and depth at each site (notebook `05b`). How and why:
[`hywaves/docs/wind_correction.md`](hywaves/docs/wind_correction.md).

For the details, read in this order: a case README
([Netherlands](Netherlands/README.md)), the engine README
([`hywaves/README.md`](hywaves/README.md)), and the
[case contract](hywaves/contract/CONTRACT.md) (what a case must provide and
what each config key means; its changelog explains every engine version).

## Repository layout

| Path             | What it is                                                                             |
| ---------------- | -------------------------------------------------------------------------------------- |
| `hywaves/`       | The engine template, contract, and CLI (`python -m hywaves ...`) — scaffolds, validates, updates, and publishes cases |
| `Netherlands/`   | Reference case — Belgian and Dutch coasts (latest engine)                              |
| `NewSouthWales/` | Worked case — NSW coast, Australia                                                     |
| `Samoa/`         | Worked case — Samoan archipelago                                                       |
| `GUI/`           | *HyGloW Results Explorer* — a Dash portal that reads each case's `portal/` bundle(s)   |
| `presentation/`  | Static HTML deck about HyGloW/HyWaves, its numbers built from the `portal/` bundles   |

Heavy data (`*.nc`, `*.zarr`, `*.tiff`, model case folders, `outputs/`,
`logs/`, `tmp/`) is git-ignored: a clone has the code, configs, notebooks and
published `portal/` bundles, not the raw inputs or model runs.

## Related repositories

| Repository | Role |
| --- | --- |
| [**bluemath_tk**](https://github.com/GeoOcean/BlueMath_tk) | The generic toolkit this repo builds on: the HyWaves method (`waves/hywaves/`: sectors, output points, metamodel, reconstruction, wind correction), wave and wind tools (`waves/steepness.py`, `waves/wind.py`, `waves/fetch.py`), and the SnapWave and SWAN wrappers (`wrappers/snapwave/`, `wrappers/swan/`). The current features need its `develop` branch (no release includes them yet) |
| [**BlueMath**](https://github.com/GeoOcean/BlueMath) | Worked examples, incl. `climate_services/hindcast/HyWaves_Netherlands` (plain-Python notebooks) |

Code that is generic goes to bluemath_tk; HyGloW keeps the case layer
(JSON-driven `utils/`, notebooks, contract, CLI, GUI).

Full CLI reference (including `update`, which re-syncs a case's engine code
from the template): [`hywaves/README.md`](hywaves/README.md).

## Contributing a case back

A **case** is a self-contained folder for one region — its own area of
interest, mesh, goal points, observations catalogue, and results. Cases can
live in this repository or in your own repo; what makes a folder a HyGloW
case is conformance to the [case contract](hywaves/contract/CONTRACT.md).

1. `python -m hywaves start --name YourCase`
2. Fill in `YourCase/YourCase.json` and work through the numbered notebooks.
3. `python -m hywaves publish YourCase/` (validates in strict mode, then writes
   `YourCase/portal/` — commit it)
4. Open a PR here (using your fork), or host it in your own repo and point the GUI catalog
   (`GUI/catalog/cases.yaml`) at it — the contract is the only thing that
   needs to match.

Engine code (`utils/`) is **vendored** into every case so each one runs
standalone. If you fix a bug in your case's copy, please also land it in
[`hywaves/template/`](hywaves/template/) so `python -m hywaves update` can carry
the fix to every other case.

---

## Authors

Built by the [**GeoOcean**](https://github.com/GeoOcean) research group
(Universidad de Cantabria). Questions, doubts, or just want to say hi?
Reach out to Javier Tausía ([@tausiaj](https://github.com/tausiaj),
[tausiaj@unican.es](mailto:tausiaj@unican.es)).

Licensed under [GPLv3](LICENSE).
