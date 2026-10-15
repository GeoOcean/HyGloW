"""Re-sync a case's vendored engine from the canonical template.

``python -m hywaves update MyCase/`` compares the case's engine files
(``utils/``) against ``hywaves/template/`` using the hashes
recorded in ``.hywaves/manifest.json``:

- files the user never touched are updated to the template version;
- files with local modifications are reported and left alone
  (``--force`` overwrites them — if your fix is good, PR it to the template);
- files present in the case but gone from the template are reported as
  orphaned, never deleted.

Notebooks and model (SnapWave/SWAN) templates are user-owned and never synced.
The manifest then records the new hywaves version and the bluemath_tk
revision of the environment (see :mod:`hywaves.common`).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import __version__
from .common import (
    ENGINE_DIRS,
    TEMPLATE_DIR,
    bluemath_tk_revision,
    describe_revision,
    file_hashes,
    now,
    read_manifest,
    write_manifest,
)


class UpdateError(RuntimeError):
    """Raised when a case cannot be updated."""


def plan_update(case_dir: Path) -> dict[str, list[str]]:
    """Classify every synced engine file by the action ``update_case`` would take.

    Parameters
    ----------
    case_dir
        Case folder (with a manifest) to compare against ``hywaves/template/``.

    Returns
    -------
    dict[str, list[str]]
        Case-relative paths keyed by action: ``up_to_date``, ``update``
        (differs but untouched since shipping — safe to sync), ``add`` (new
        in the template), ``modified`` (differs and locally edited),
        ``orphaned`` (in the case but gone from the template — never synced
        or deleted).

    Raises
    ------
    UpdateError
        If the case has no manifest.
    """
    manifest = read_manifest(case_dir)
    if manifest is None:
        raise UpdateError(
            f"{case_dir.name} has no .hywaves/manifest.json; scaffold a new case with "
            "`python -m hywaves start` and move its inputs there"
        )
    template = file_hashes(TEMPLATE_DIR, ENGINE_DIRS)
    case = file_hashes(case_dir, ENGINE_DIRS)
    shipped = manifest.get("engine_files", {})

    plan: dict[str, list[str]] = {
        "up_to_date": [], "update": [], "add": [], "modified": [], "orphaned": []
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


def update_case(
    case_dir: str | Path, *, force: bool = False, dry_run: bool = False
) -> int:
    """Sync the case's engine from the template, printing the plan.

    Parameters
    ----------
    case_dir
        Case folder to update.
    force
        Overwrite locally modified files too (normally left alone).
    dry_run
        Print the plan without writing anything.

    Returns
    -------
    int
        CLI exit code (always ``0``; sync problems are reported per file).

    Raises
    ------
    UpdateError
        If *case_dir* is not a directory, is the template itself, or has no
        manifest.
    """
    case_dir = Path(case_dir).resolve()
    if not case_dir.is_dir():
        raise UpdateError(f"Not a directory: {case_dir}")
    if case_dir == TEMPLATE_DIR:
        raise UpdateError("Refusing to update the template from itself")

    plan = plan_update(case_dir)
    to_sync = plan["update"] + plan["add"] + (plan["modified"] if force else [])
    revision = bluemath_tk_revision()

    print(f"Template : hywaves-template@{__version__}")
    print(f"Case     : {case_dir.name}")
    print(f"Toolkit  : {describe_revision(revision)}")
    print(f"  up to date : {len(plan['up_to_date'])}")
    for action in ("update", "add"):
        for rel in plan[action]:
            print(f"  {action:<10} : {rel}")
    for rel in plan["modified"]:
        tag = "force-sync" if force else "KEPT (locally modified)"
        print(f"  {tag:<10} : {rel}")
    for rel in plan["orphaned"]:
        print(f"  orphaned   : {rel} (not in template; left alone)")

    if dry_run:
        print("\nDry run — nothing written.")
        return 0

    for rel in to_sync:
        target = case_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TEMPLATE_DIR / rel, target)

    # Shipped-state hashes only for files that now match the template; kept
    # modified files stay unrecorded, so they are flagged again next run.
    template_hashes = file_hashes(TEMPLATE_DIR, ENGINE_DIRS)
    case_hashes = file_hashes(case_dir, ENGINE_DIRS)
    write_manifest(
        case_dir,
        hywaves_version=__version__,
        source=f"hywaves-template@{__version__}",
        updated=now(),
        bluemath_tk=revision,
        engine_files={rel: h for rel, h in template_hashes.items() if case_hashes.get(rel) == h},
    )

    kept = 0 if force else len(plan["modified"])
    print(f"\nSynced {len(to_sync)} file(s); kept {kept} locally modified file(s).")
    if kept:
        print(
            "Re-run with --force to overwrite them — and if your local fix is "
            "good, please PR it to hywaves/template/ so everyone benefits."
        )
    return 0
