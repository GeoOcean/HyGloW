"""Scaffold a new self-contained HyGloW case folder.

``python -m hywaves start --name Cantabria`` creates a case from the built-in
template (``hywaves/template/``): vendored engine code (``utils/``),
case-agnostic pipeline notebooks, SnapWave templates, tests, and a fresh
config skeleton. The result runs anywhere — laptop or HPC — as long as the
scientific stack (``bluemath_tk``, ``bluemesh2d``, ``wavespectra``, ...) is
in your environment; see the generated ``README.md`` for the exact list.

``--from-case Samoa`` clones an existing case's engine and notebooks instead
(outputs stripped), for when you want to continue someone's working setup.

Deliberately minimal: no data download, no tiff generation. The user starts,
then fights (with the notebooks as the guide).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION, __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = Path(__file__).resolve().parent / "template"

# Vendored into every new case; whichever exist in the source are copied
# (older cases, e.g. Samoa, still carry a vendored bluemesh2d/ or legacy
# pymesh2d/ from before bluemesh2d became a regular environment dependency —
# see the generated README.md's Dependencies section).
ENGINE_DIRS = ("utils", "bluemesh2d", "pymesh2d")
SUPPORT_DIRS = ("tests", "templates-metamodel", "templates-dynamic")
# Hashed in the manifest and kept in sync by `hywaves update`. SnapWave
# templates and notebooks are user-owned after scaffolding — never synced.
SYNCED_DIRS = ENGINE_DIRS + ("tests",)
INPUT_SUBDIRS = ("AoI_Polygons", "GEBCO", "GRID", "Goals", "Observations", "WHACS")
EMPTY_DIRS = ("outputs", "logs")

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")
_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


class ScaffoldError(RuntimeError):
    """Raised when a new case cannot be created."""


def _resolve_source_case(from_case: str) -> Path:
    """Locate an existing case folder named *from_case* for ``--from-case`` cloning.

    Looks under the current working directory first, then the repo root.

    Raises
    ------
    ScaffoldError
        If no directory named *from_case* with a ``utils/`` subfolder is found.
    """
    for base in (Path.cwd(), REPO_ROOT):
        candidate = (base / from_case).resolve()
        if (candidate / "utils").is_dir():
            return candidate
    raise ScaffoldError(
        f"Reference case {from_case!r} not found (looked in {Path.cwd()} and {REPO_ROOT})"
    )


def _strip_notebook(source: Path, target: Path) -> None:
    """Copy a notebook with all outputs and execution counts removed."""
    nb = json.loads(source.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    nb.get("metadata", {}).pop("widgets", None)
    target.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def _render_template(template_name: str, target: Path, name: str) -> None:
    """Copy ``TEMPLATE_DIR / template_name`` to *target*, substituting ``__CASE_NAME__``."""
    text = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    target.write_text(text.replace("__CASE_NAME__", name), encoding="utf-8")


def _file_hashes(root: Path, dirs: tuple[str, ...]) -> dict[str, str]:
    """sha256 per vendored engine file, used by `hywaves update` for diffing."""
    hashes: dict[str, str] = {}
    for dirname in dirs:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            if "__pycache__" in rel.parts or path.suffix == ".pyc":
                continue
            hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def start_case(
    name: str,
    dest: str | Path = ".",
    *,
    from_case: str | None = None,
    notebooks: bool = True,
) -> Path:
    """Create the case folder ``dest/name`` and return its path.

    Sources from ``hywaves/template/`` by default; from an existing case's
    folder when *from_case* is given (clone mode).

    Parameters
    ----------
    name
        Case name; becomes the folder name and the config JSON stem. Must
        start with a letter and contain only letters, digits, ``_`` or ``-``.
    dest
        Parent directory the case folder is created under.
    from_case
        Name of an existing case to clone the engine and notebooks from
        (outputs stripped), instead of the built-in template.
    notebooks
        Copy the pipeline notebooks. ``False`` skips them entirely.

    Returns
    -------
    Path
        The new case folder, ``dest/name``.

    Raises
    ------
    ScaffoldError
        If *name* is invalid, the target already exists, *from_case* can't
        be resolved, or source and target would be the same folder.
    """
    if not _NAME_RE.fullmatch(name):
        raise ScaffoldError(
            f"Invalid case name {name!r}: use letters, digits, '_' or '-', "
            "starting with a letter"
        )
    source = _resolve_source_case(from_case) if from_case else TEMPLATE_DIR
    target = (Path(dest) / name).resolve()
    if target.exists():
        raise ScaffoldError(f"Target already exists: {target}")
    if source == target:
        raise ScaffoldError("Source and target case are the same folder")

    target.mkdir(parents=True)

    for dirname in ENGINE_DIRS + SUPPORT_DIRS:
        src = source / dirname
        if src.is_dir():
            shutil.copytree(src, target / dirname, ignore=_IGNORE)

    copied_notebooks: list[str] = []
    if notebooks:
        if from_case:
            # Clone mode: strip outputs from the source case's notebooks.
            for nb_path in sorted(source.glob("[0-9][0-9]_*.ipynb")):
                _strip_notebook(nb_path, target / nb_path.name)
                copied_notebooks.append(nb_path.name)
        else:
            # Template notebooks are stored agnostic and already stripped.
            for nb_path in sorted((TEMPLATE_DIR / "notebooks").glob("*.ipynb")):
                shutil.copy2(nb_path, target / nb_path.name)
                copied_notebooks.append(nb_path.name)

    _render_template("case.json", target / f"{name}.json", name)
    _render_template("README.md", target / "README.md", name)

    for sub in INPUT_SUBDIRS:
        (target / "inputs" / sub).mkdir(parents=True)
        (target / "inputs" / sub / ".gitkeep").touch()
    for sub in EMPTY_DIRS:
        (target / sub).mkdir()
        (target / sub / ".gitkeep").touch()

    manifest = {
        "hywaves_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "source": source.name if from_case else f"hywaves-template@{__version__}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notebooks": copied_notebooks,
        "engine_files": _file_hashes(target, SYNCED_DIRS),
    }
    manifest_dir = target / ".hywaves"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    return target
