"""The case manifest, ``{Case}/.hywaves/manifest.json``: what a case runs with.

A case vendors its own ``utils/``, so it keeps working while the template and
bluemath_tk move on. The manifest records the pair that works for it: the
hywaves version its ``utils/`` came from (with the shipped file hashes that
``hywaves update`` diffs against) and the bluemath_tk revision. ``start`` and
``update`` write it; ``publish`` refreshes ``bluemath_tk``, since a published
case is one that ran.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_REL = Path(".hywaves") / "manifest.json"


def now() -> str:
    """UTC timestamp, ISO-8601 to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(folder: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(folder), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def bluemath_tk_revision() -> dict[str, Any]:
    """The installed bluemath_tk: version, and git commit, branch and dirty flag.

    The commit comes from the source tree of an editable install, or from the
    VCS record of a ``pip install git+…``; ``None`` when neither exists.

    Raises
    ------
    RuntimeError
        If bluemath_tk is not installed.
    """
    spec = importlib.util.find_spec("bluemath_tk")
    if spec is None or spec.origin is None:
        raise RuntimeError("bluemath_tk is not installed in this environment")
    revision: dict[str, Any] = {
        "version": importlib.metadata.version("bluemath_tk"),
        "commit": None,
        "branch": None,
        "dirty": None,
    }
    source = Path(spec.origin).parent
    try:
        revision["commit"] = _git(source, "rev-parse", "HEAD")
        revision["branch"] = _git(source, "rev-parse", "--abbrev-ref", "HEAD")
        revision["dirty"] = bool(_git(source, "status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        direct_url = importlib.metadata.distribution("bluemath_tk").read_text("direct_url.json")
        vcs = json.loads(direct_url).get("vcs_info", {}) if direct_url else {}
        revision["commit"] = vcs.get("commit_id")
    return revision


def read_manifest(case_dir: Path) -> dict[str, Any] | None:
    """The case manifest, or ``None`` if the case has none."""
    path = case_dir / MANIFEST_REL
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_manifest(case_dir: Path, **fields: Any) -> dict[str, Any]:
    """Merge *fields* into the case manifest (created if missing) and return it."""
    manifest = {**(read_manifest(case_dir) or {}), **fields}
    path = case_dir / MANIFEST_REL
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def describe_revision(revision: dict[str, Any]) -> str:
    """One line for the CLI, e.g. ``bluemath_tk 1.3.dev49 @ 0a1e3c8 (develop, uncommitted changes)``."""
    text = f"bluemath_tk {revision['version']}"
    if revision.get("commit"):
        text += f" @ {revision['commit'][:7]}"
    extra = [revision["branch"]] if revision.get("branch") else []
    if revision.get("dirty"):
        extra.append("uncommitted changes")
    return text + (f" ({', '.join(extra)})" if extra else "")
