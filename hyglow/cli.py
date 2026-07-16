"""HyGloW command-line interface: ``python -m hyglow <command>``."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_start(args: argparse.Namespace) -> int:
    from .scaffold import ScaffoldError, start_case

    try:
        target = start_case(
            args.name,
            args.dest,
            from_case=args.from_case,
            notebooks=not args.no_notebooks,
        )
    except ScaffoldError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Created case: {target}")
    print(
        f"""
Next steps (see {target.name}/README.md):
  1. Edit {target.name}/{target.name}.json — fill in the TODOs
  2. Put your AoI shapefile in inputs/AoI_Polygons/
  3. Work through the numbered notebooks (01 → 11), starting with
     01_Generate_tiff_from_GEBCO.ipynb
  4. Check progress anytime with:
     python -m hyglow validate {target.name}
"""
    )
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    from .update import UpdateError, update_case

    try:
        return update_case(args.case_dir, force=args.force, dry_run=args.dry_run)
    except UpdateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_validate(args: argparse.Namespace) -> int:
    from .contract.validate import main as validate_main

    argv = [args.case_dir] + (["--publish"] if args.publish else [])
    return validate_main(argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hyglow",
        description="HyGloW — hybrid global wave downscaling toolkit",
    )
    parser.add_argument("--version", action="version", version=f"hyglow {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser(
        "start",
        help="Scaffold a new self-contained case folder",
        description=(
            "Create a new case folder with the vendored engine, notebooks "
            "(outputs stripped), SnapWave templates, tests, and a config "
            "skeleton. No data is downloaded — fill in inputs via the notebooks."
        ),
    )
    p_start.add_argument("--name", required=True, help="Case name (becomes the folder name)")
    p_start.add_argument("--dest", default=".", help="Parent directory (default: current)")
    p_start.add_argument(
        "--from-case",
        default=None,
        help=(
            "Clone an existing case's engine and notebooks instead of the "
            "built-in template (e.g. --from-case Samoa)"
        ),
    )
    p_start.add_argument(
        "--no-notebooks", action="store_true", help="Skip copying the pipeline notebooks"
    )
    p_start.set_defaults(func=_cmd_start)

    p_update = sub.add_parser(
        "update",
        help="Re-sync a case's vendored engine from the template",
        description=(
            "Update the case's engine files (utils/, mesher, tests/) to the "
            "current hyglow/template/ versions. Locally modified files are "
            "kept unless --force. Notebooks and SnapWave templates are never "
            "touched. Orphaned files (e.g. a legacy pymesh2d/) are reported, "
            "never deleted."
        ),
    )
    p_update.add_argument("case_dir", help="Path to the case folder")
    p_update.add_argument(
        "--force", action="store_true", help="Overwrite locally modified files too"
    )
    p_update.add_argument(
        "--dry-run", action="store_true", help="Report the plan without writing"
    )
    p_update.set_defaults(func=_cmd_update)

    p_validate = sub.add_parser(
        "validate",
        help="Check a case folder against the case contract",
        description="Validate a case against hyglow/contract/CONTRACT.md.",
    )
    p_validate.add_argument("case_dir", help="Path to the case folder")
    p_validate.add_argument(
        "--publish",
        action="store_true",
        help="Strict mode: completeness gaps become errors (portal-ready check)",
    )
    p_validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
