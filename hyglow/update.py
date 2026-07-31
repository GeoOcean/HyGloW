"""Re-sync a case's vendored engine from the canonical template.

``python -m hyglow update MyCase/`` compares the case's engine files
(``utils/``, the mesher, ``tests/``) against ``hyglow/template/`` using the
hashes recorded in ``.hyglow/manifest.json``:

- files the user never touched are updated to the template version;
- files with local modifications are reported and left alone
  (``--force`` overwrites them — the polite message is: if your fix is
  good, PR it to the template so everyone benefits);
- files present in the case but gone from the template (e.g. a legacy
  ``pymesh2d/``) are reported as orphaned, never deleted.

Cases created before the manifest existed are handled conservatively:
without a shipped-state record, every differing file counts as locally
modified. Notebooks and SnapWave templates are user-owned and never synced.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .scaffold import SYNCED_DIRS, TEMPLATE_DIR, _file_hashes

MANIFEST_REL = Path(".hyglow") / "manifest.json"


class UpdateError(RuntimeError):
    """Raised when a case cannot be updated."""


def _load_manifest(case_dir: Path) -> dict | None:
    path = case_dir / MANIFEST_REL
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def plan_update(case_dir: Path) -> dict[str, list[str]]:
    """Classify every synced engine file. Returns lists keyed by action."""
    template = _file_hashes(TEMPLATE_DIR, SYNCED_DIRS)
    if not template:
        raise UpdateError(f"Template has no engine files under {TEMPLATE_DIR}")
    case = _file_hashes(case_dir, SYNCED_DIRS)
    manifest = _load_manifest(case_dir)
    shipped = (manifest or {}).get("engine_files", {})

    plan: dict[str, list[str]] = {
        "up_to_date": [],
        "update": [],   # differs, but untouched since shipping → safe to sync
        "add": [],      # new in template, absent in case
        "modified": [], # differs AND locally modified (or unknown provenance)
        "orphaned": [], # in case, not in template (legacy dirs, local additions)
    }
    for rel, template_hash in template.items():
        case_hash = case.get(rel)
        if case_hash is None:
            plan["add"].append(rel)
        elif case_hash == template_hash:
            plan["up_to_date"].append(rel)
        elif shipped.get(rel) == case_hash:
            plan["update"].append(rel)
        else:
            plan["modified"].append(rel)
    plan["orphaned"] = sorted(set(case) - set(template))
    return plan


def _summarize_orphans(orphaned: list[str]) -> list[str]:
    """Collapse orphaned top-level dirs with many files (e.g. pymesh2d/) to one line."""
    by_top: dict[str, int] = {}
    for rel in orphaned:
        top = rel.split("/", 1)[0]
        by_top[top] = by_top.get(top, 0) + 1
    collapsed = {top for top, count in by_top.items() if count > 3}
    lines = [f"{top}/ ({by_top[top]} file(s))" for top in sorted(collapsed)]
    lines += [rel for rel in orphaned if rel.split("/", 1)[0] not in collapsed]
    return lines


def _write_manifest(case_dir: Path, synced_hashes: dict[str, str]) -> None:
    manifest = _load_manifest(case_dir) or {}
    manifest.setdefault(
        "created", datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    manifest.update(
        {
            "hyglow_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "source": f"hyglow-template@{__version__}",
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "engine_files": synced_hashes,
        }
    )
    path = case_dir / MANIFEST_REL
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def update_case(
    case_dir: str | Path, *, force: bool = False, dry_run: bool = False
) -> int:
    """Sync the case's engine from the template; returns a CLI exit code."""
    case_dir = Path(case_dir).resolve()
    if not case_dir.is_dir():
        raise UpdateError(f"Not a directory: {case_dir}")
    if case_dir == TEMPLATE_DIR:
        raise UpdateError("Refusing to update the template from itself")

    has_manifest = _load_manifest(case_dir) is not None
    if not has_manifest:
        print(
            "note: no .hyglow/manifest.json — this case predates scaffolding, "
            "so every differing file is treated as locally modified."
        )

    plan = plan_update(case_dir)
    to_sync = plan["update"] + plan["add"] + (plan["modified"] if force else [])

    print(f"Template : hyglow-template@{__version__}")
    print(f"Case     : {case_dir.name}")
    print(f"  up to date : {len(plan['up_to_date'])}")
    for action in ("update", "add"):
        for rel in plan[action]:
            print(f"  {action:<10} : {rel}")
    for rel in plan["modified"]:
        tag = "force-sync" if force else "KEPT (locally modified)"
        print(f"  {tag:<10} : {rel}")
    for line in _summarize_orphans(plan["orphaned"]):
        print(f"  orphaned   : {line} (not in template; left alone)")

    if dry_run:
        print("\nDry run — nothing written.")
        return 0

    for rel in to_sync:
        source = TEMPLATE_DIR / rel
        target = case_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # Manifest records shipped-state hashes only for files that now match the
    # template; skipped modified files stay unrecorded → flagged again next run.
    template_hashes = _file_hashes(TEMPLATE_DIR, SYNCED_DIRS)
    case_hashes = _file_hashes(case_dir, SYNCED_DIRS)
    synced = {
        rel: h for rel, h in template_hashes.items() if case_hashes.get(rel) == h
    }
    _write_manifest(case_dir, synced)

    kept = 0 if force else len(plan["modified"])
    print(f"\nSynced {len(to_sync)} file(s); kept {kept} locally modified file(s).")
    if kept:
        print(
            "Re-run with --force to overwrite them — and if your local fix is "
            "good, please PR it to hyglow/template/ so everyone benefits."
        )
    return 0
