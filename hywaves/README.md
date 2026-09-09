# hywaves — engine, contract, and CLI

Source of truth for the HyGloW project tooling. `hywaves` itself never needs
to be pip-installed: everything runs as `python -m hywaves` from the
repository root, and case folders receive a **vendored copy** of `utils/` so
they stay fully self-contained (laptop, HPC, or someone else's repo).

## Commands

```bash
# Scaffold a new self-contained case (engine + notebooks + templates + config skeleton)
python -m hywaves start --name MyCase

# Check a case against the case contract
python -m hywaves validate MyCase/             # development mode: gaps are warnings
python -m hywaves validate MyCase/ --publish   # strict: portal-ready check

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
| `config.py`            | `CaseConfig` class — load any case folder programmatically  |
| `scaffold.py`          | `start` implementation (vendoring + manifest)               |
| `template/`            | The canonical case reference (see below)                    |
| `tests/`               | Tests for `hywaves` itself — scaffold/update/validate/config |

## The model

- **Cases are folders, not packages.** Each case carries its own engine
  snapshot (`utils/`), stamped in `.hywaves/manifest.json`. `bluemesh2d` is a
  regular environment dependency (see a case's README), not part of the
  snapshot.
- **`template/` is the canonical reference.** It holds the versioned engine
  code (`utils/`), case-agnostic pipeline notebooks (`notebooks/`), SnapWave
  templates, tests, and the config skeleton. Engine improvements and
  notebook fixes land there. Existing cases
  are snapshots that `hywaves update` re-syncs: untouched engine files are
  updated, locally modified ones are kept (overridden by `--force`), and
  notebooks/SnapWave templates are never touched.
- **The contract is the interface.** The GUI and any global aggregation read
  only contract files (case JSON, goals, observations, metrics CSVs,
  figures). What a case does internally is its author's business.
- **Community cases live in their authors' repos.** This repo hosts the
  engine, the contract, the GUI, and two worked examples: Samoa and
  NewSouthWales, both carried through goals/metamodel/buoy validation.

## For tooling authors

```python
from hywaves.config import CaseConfig
from hywaves.contract import validate_case

case = CaseConfig("Samoa")
goals = case.load_goals()
findings = validate_case("Samoa", publish=True)
```
