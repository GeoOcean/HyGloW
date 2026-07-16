"""Scaffold a new self-contained HyGloW case folder.

``python -m hyglow start --name Cantabria`` creates a case from the built-in
template (``hyglow/template/``): vendored engine code, case-agnostic pipeline
notebooks, SnapWave templates, tests, and a fresh config skeleton. The result
runs anywhere — laptop or HPC — with no package installation.

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
# (the template ships bluemesh2d, older cases like Samoa still carry pymesh2d).
ENGINE_DIRS = ("utils", "bluemesh2d", "pymesh2d")
SUPPORT_DIRS = ("tests", "templates-metamodel", "templates-dynamic")
INPUT_SUBDIRS = ("AoI_Polygons", "GEBCO", "GRID", "Goals", "Observations", "WHACS")
EMPTY_DIRS = ("outputs", "logs")

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")
_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


class ScaffoldError(RuntimeError):
    """Raised when a new case cannot be created."""


def _resolve_source_case(from_case: str) -> Path:
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
    text = (TEMPLATE_DIR / template_name).read_text(encoding="utf-8")
    target.write_text(text.replace("__CASE_NAME__", name), encoding="utf-8")


def _file_hashes(root: Path, dirs: tuple[str, ...]) -> dict[str, str]:
    """sha256 per vendored engine file, for future `hyglow update` diffing."""
    hashes: dict[str, str] = {}
    for dirname in dirs:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                hashes[str(path.relative_to(root))] = digest
    return hashes


def start_case(
    name: str,
    dest: str | Path = ".",
    *,
    from_case: str | None = None,
    notebooks: bool = True,
) -> Path:
    """Create the case folder ``dest/name`` and return its path.

    Sources from ``hyglow/template/`` by default; from an existing case's
    folder when *from_case* is given (clone mode).
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
    _render_template("gitignore", target / ".gitignore", name)

    for sub in INPUT_SUBDIRS:
        (target / "inputs" / sub).mkdir(parents=True)
        (target / "inputs" / sub / ".gitkeep").touch()
    for sub in EMPTY_DIRS:
        (target / sub).mkdir()
        (target / sub / ".gitkeep").touch()

    manifest = {
        "hyglow_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "source": source.name if from_case else f"hyglow-template@{__version__}",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notebooks": copied_notebooks,
        "engine_files": _file_hashes(target, ENGINE_DIRS),
    }
    manifest_dir = target / ".hyglow"
    manifest_dir.mkdir()
    (manifest_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    return target
