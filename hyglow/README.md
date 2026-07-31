# hyglow — engine, contract, and CLI

Source of truth for the HyGloW project tooling. `hyglow` itself never needs
to be pip-installed: everything runs as `python -m hyglow` from the
repository root, and case folders receive a **vendored copy** of `utils/` so
they stay fully self-contained (laptop, HPC, or someone else's repo). The one
exception is `bluemesh2d` (the mesher), which ships on PyPI and is installed
as a regular dependency (`pip install -r requirements.txt` inside a case)
rather than vendored.

## Commands

```bash
# Scaffold a new self-contained case (engine + notebooks + templates + config skeleton)
python -m hyglow start --name Cantabria

# Check a case against the case contract
python -m hyglow validate Cantabria/             # development mode: gaps are warnings
python -m hyglow validate Cantabria/ --publish   # strict: portal-ready check

# Re-sync a case's vendored engine from the template
python -m hyglow update Cantabria/ --dry-run     # report only
python -m hyglow update Cantabria/               # sync untouched files, keep local edits
python -m hyglow update Cantabria/ --force       # overwrite local edits too
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

## The model

- **Cases are folders, not packages.** Each case carries its own engine
  snapshot (`utils/`), stamped in `.hyglow/manifest.json`. `bluemesh2d` is a
  pip dependency (`requirements.txt`), not part of the snapshot.
- **`template/` is the canonical reference.** It holds the versioned engine
  code (`utils/`), case-agnostic pipeline notebooks (`notebooks/`), SnapWave
  templates, tests, the `requirements.txt`, and the config skeleton.
  Engine improvements and notebook fixes land there. Existing cases
  are snapshots that `hyglow update` re-syncs: untouched engine files are
  updated, locally modified ones are kept (overridden by `--force`), and
  notebooks/SnapWave templates are never touched.
- **The contract is the interface.** The GUI and any global aggregation read
  only contract files (case JSON, goals, observations, metrics CSVs,
  figures). What a case does internally is its author's business.
- **Community cases live in their authors' repos.** This repo hosts the
  engine, the contract, the GUI, and a few worked examples (Samoa is the
  most complete one).

## For tooling authors

```python
from hyglow.config import CaseConfig
from hyglow.contract import validate_case

case = CaseConfig("Samoa")
goals = case.load_goals()
findings = validate_case("Samoa", publish=True)
```
