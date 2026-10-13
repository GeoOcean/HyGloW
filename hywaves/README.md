# hywaves — engine, contract, and CLI

Source of truth for the HyGloW project tooling. `hywaves` itself never needs
to be pip-installed: everything runs as `python -m hywaves` from the
repository root, and case folders receive a **vendored copy** of `utils/` so
they stay fully self-contained (laptop, HPC, or someone else's repo).

Current version: **0.13.0** (`hywaves/__init__.py`). What changed in each
version, and the config keys it introduced, is in the
[contract changelog](contract/CONTRACT.md#changelog).

## Commands

```bash
# Scaffold a new self-contained case (engine + notebooks + templates + config skeleton)
python -m hywaves start --name MyCase

# Check a case against the case contract
python -m hywaves validate MyCase/             # development mode: gaps are warnings
python -m hywaves validate MyCase/ --publish   # strict: portal-ready check

# Export the portal bundle the GUI reads (validates first) → MyCase/portal/
python -m hywaves publish MyCase/              # commit MyCase/portal/ (its book/ is git-ignored)
python -m hywaves publish MyCase/ --no-book --max-image-px 1600
python -m hywaves publish MyCase/ --url-base /hyglow/   # report links for a GUI served under /hyglow/

# Keep the current portal as a version before re-running/re-publishing → MyCase/portal_<tag>/
python -m hywaves snapshot MyCase/ hs1 --note "unit Hs metamodel"   # commit it; its book/ is git-ignored

# Re-sync a case's vendored engine from the template
python -m hywaves update MyCase/ --dry-run     # report only
python -m hywaves update MyCase/               # sync untouched files, keep local edits
python -m hywaves update MyCase/ --force       # overwrite local edits too

# Run the hywaves package's own test suite (scaffold/update/validate/config)
python -m unittest discover -s hywaves/tests -t . -v
```

`start` is deliberately minimal — it creates the structure and the user takes
over with the notebooks (GEBCO clipping, mesh, goals, LHS, fit, validation).
No data is downloaded.

## Layout

| Path                   | Role                                                        |
| ---------------------- | ----------------------------------------------------------- |
| `contract/CONTRACT.md` | The case contract (Level 1 portal + Level 2 engine)         |
| `contract/validate.py` | Stdlib-only validator — copyable standalone into any repo   |
| `publish.py`           | `publish` implementation — builds `{Case}/portal/` (CONTRACT.md §6) |
| `snapshot.py`          | `snapshot` implementation — freezes `portal/` as `portal_<tag>/`    |
| `config.py`            | `CaseConfig` class — load any case folder programmatically  |
| `scaffold.py`          | `start` implementation (vendoring + manifest)               |
| `update.py`            | `update` implementation (re-sync vendored `utils/` and `tests/`) |
| `template/`            | The canonical case reference (see below)                    |
| `docs/`                | Method notes: [`boundary_forcing.md`](docs/boundary_forcing.md), [`satellite_calibration.md`](docs/satellite_calibration.md), [`site_filter.md`](docs/site_filter.md), [`wind_correction.md`](docs/wind_correction.md) |
| `tests/`               | Tests for `hywaves` itself — scaffold/update/validate/config |

## The model

- **Cases are folders, not packages.** Each case carries its own engine
  snapshot (`utils/`), stamped in `.hywaves/manifest.json`. `bluemesh2d` is a
  regular environment dependency (see a case's README), not part of the
  snapshot.
- **`template/` is the canonical reference.** It holds the versioned engine
  code (`utils/`), case-agnostic pipeline notebooks (`notebooks/`), SnapWave/SWAN
  templates, tests, and the config skeleton. Engine improvements and
  notebook fixes land there. Existing cases
  are snapshots that `hywaves update` re-syncs: untouched engine files are
  updated, locally modified ones are kept (overridden by `--force`), and
  notebooks and model templates are never touched.
- **The contract is the interface.** `publish` turns a case's contract files
  (case JSON, goals, observations, metrics CSVs, figures) plus pre-computed
  map layers into a small `{Case}/portal/` bundle, and the GUI reads only
  that. What a case does internally is its author's business. `validate`
  warns when a bundle has fallen behind its sources.
- **Community cases live in their authors' repos.** This repo hosts the
  engine, the contract, the GUI, and three worked cases: Netherlands (on the
  current engine) and Samoa and NewSouthWales (on older ones, until re-run).
- **Generic code lives in bluemath_tk.** The HyWaves method, the wind and
  fetch tools and the SnapWave/SWAN wrappers are in
  [bluemath_tk](https://github.com/GeoOcean/BlueMath_tk); install it from
  its `develop` branch (`pip install "git+https://github.com/GeoOcean/BlueMath_tk.git@develop"`), as no release includes this code yet.
  `template/utils/` is the JSON/case layer on top of it.

## What a case configures (0.13.0)

All in the case JSON (`{Case}/{Case}.json`); the template's `case.json` has
every key, and the contract documents them.

| Block | What it sets |
| --- | --- |
| `paths` | AoI, bathymetry raster, plotting mesh |
| `offshore_waves` | Offshore hindcast zarr(s) at the goals, `(time, goal)` |
| `satellite` (optional) | Altimeter zarr (path or URL) and matching choices (`radius_km`, `min_depth`, `train_fraction`, ...) to validate and calibrate the offshore hindcast at the goals (notebook 07b); see [`docs/satellite_calibration.md`](docs/satellite_calibration.md) |
| `forcing` (optional) | `wind` (10 m `u10`/`v10`, e.g. ERA5) and `water_level` (per goal, e.g. CODEC) for the dynamic runs and `reconstruct(wl="forcing")` |
| `simulations.model` | `"snapwave"` or `"swan"` |
| `simulations.<model>` | Its grid (`grid_file`), template folders, cases folders (`metamodel_cases`, `dynamic_cases`, `wind_cases`) and SLURM options; how each model spreads the goals along its open boundary: [`docs/boundary_forcing.md`](docs/boundary_forcing.md) |
| `simulations` (rest) | Goals file, LHS size and bounds, steepness filter, output sites (isobaths, polygons, buoys) |
| `metamodel` | Inputs, sets (groups of output sites), MDA sizes, predicted variables, PCA/GP options, reconstruction variables |
| `metamodel.site_filter` | Which output sites are trained: bounds, ratio to the forcing, and `hs_nan_max` (outside the grid or dry in too many cases); see [`docs/site_filter.md`](docs/site_filter.md) |
| `metamodel.wind_correction` (optional) | Wind-correction training design and use (`enabled`, `wind`: `"inferred"` or `"forcing"`); see [`docs/wind_correction.md`](docs/wind_correction.md) |

Switching the model: the cases folders are separate per model, and fitted
artefacts in `outputs/` record the model that produced them, so a second
model needs a fresh `outputs/`. Keep the first model's results with
`snapshot` before re-running.

## Versioning

- **Current: 0.13.0** (offshore hindcast validated and calibrated against
  satellite altimetry; 0.12.0: dry and outside-grid sites handled by the site
  filter). Both models (SnapWave and SWAN) are ready to use; the results
  published so far were made with SnapWave (0.11.0).
- **While in 0.x:** the minor version goes up (0.12, 0.13, …) whenever the
  case JSON or the contract changes, and the patch version (0.11.1, …) for
  fixes only. Every change is in the
  [contract changelog](contract/CONTRACT.md#changelog), and each case records
  the engine it was scaffolded or updated from in `.hywaves/manifest.json`
  (`python -m hywaves update` moves it forward).
- **Releases are git tags** on `main` (`v0.11.0`, …), after the PR is merged
  and the reference case is published.
- **bluemath_tk** is used from its `develop` branch until a release includes
  the HyWaves code; then the README will name the minimum version.
- **1.0.0** comes when the contract is stable enough that 1.x only adds
  things (breaking changes after that mean 2.0). Before it:
  1. a bluemath_tk release with this code;
  2. a fresh case scaffolded with `hywaves start` running cleanly through the
     template notebooks;
  3. the SWAN pass run and validated in a case (today SWAN is validated in
     the Netherlands pilot only);
  4. Samoa and NewSouthWales migrated, or kept on purpose as 0.9.0 examples.

  A `1.0.0-rc1` tag can mark the point where 1 and 2 are done.

## For tooling authors

```python
from hywaves.config import CaseConfig
from hywaves.contract import validate_case

case = CaseConfig("Samoa")
goals = case.load_goals()
findings = validate_case("Samoa", publish=True)

from hywaves.publish import publish_case
result = publish_case("Samoa")        # → Samoa/portal/; result.skipped lists any skipped layers
```
