"""Freeze a case's published portal under a tag: ``{Case}/portal/`` → ``{Case}/portal_<tag>/``.

A snapshot keeps the results of one attempt (e.g. ``hs1`` = the unit-Hs
metamodel) before the case is re-run and re-published, so the GUI can show
every version side by side (CONTRACT.md §6). The copy is complete, built
Jupyter Book included; its ``manifest.json`` gains a ``snapshot`` block with
the tag, a free-text note and the snapshot date. Snapshots are never
re-published or checked for staleness.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

#: Allowed snapshot tags (folder ``portal_<tag>``).
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")


class SnapshotError(Exception):
    """Raised when a snapshot cannot be taken (no portal, bad tag, tag exists)."""


def snapshot_dir(case_dir: str | Path, tag: str) -> Path:
    """Folder of the snapshot *tag*: ``{Case}/portal_<tag>``."""
    return Path(case_dir) / f"portal_{tag}"


def list_snapshots(case_dir: str | Path) -> list[Path]:
    """Snapshot folders of a case (``portal_*`` with a ``manifest.json``), sorted by name."""
    return sorted(
        p
        for p in Path(case_dir).glob("portal_*")
        if p.is_dir() and (p / "manifest.json").is_file()
    )


def snapshot_case(case_dir: str | Path, tag: str, note: str = "") -> Path:
    """Copy ``{Case}/portal/`` (book included) to ``{Case}/portal_<tag>/``.

    Parameters
    ----------
    case_dir
        Case folder with a published ``portal/``.
    tag
        Short name of the version: lowercase letters, digits and ``-``.
    note
        Free text describing the version (shown in the GUI dropdown).

    Returns
    -------
    Path
        The snapshot folder.

    Raises
    ------
    SnapshotError
        If the tag is invalid or taken, or the case has no published portal.
    """
    case_dir = Path(case_dir)
    if not TAG_PATTERN.match(tag):
        raise SnapshotError(
            f"invalid tag {tag!r}: use lowercase letters, digits and '-' (e.g. 'hs1')"
        )
    source = case_dir / "portal"
    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise SnapshotError(
            f"{source} has no manifest.json — run `python -m hywaves publish {case_dir}` first"
        )
    target = snapshot_dir(case_dir, tag)
    if target.exists():
        raise SnapshotError(f"{target} already exists — pick another tag")

    shutil.copytree(source, target)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot"] = {
        "tag": tag,
        "note": note,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return target
