"""HyGloW command-line interface: ``python -m hywaves <command>``."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_start(args: argparse.Namespace) -> int:
    """Handle ``hywaves start``: scaffold a new case folder and print next steps."""
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
  3. Work through the numbered notebooks (01 → 12), starting with
     01_Generate_tiff_from_GEBCO.ipynb
  4. Check progress anytime with:
     python -m hywaves validate {target.name}
"""
    )
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    """Handle ``hywaves update``: re-sync a case's vendored engine from the template."""
    from .update import UpdateError, update_case

    try:
        return update_case(args.case_dir, force=args.force, dry_run=args.dry_run)
    except UpdateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle ``hywaves validate``: run the case contract validator."""
    from .contract.validate import main as validate_main

    argv = [args.case_dir] + (["--publish"] if args.publish else [])
    return validate_main(argv)


def _cmd_publish(args: argparse.Namespace) -> int:
    """Handle ``hywaves publish``: validate, then export the portal bundle."""
    from .publish import PublishError, publish_case

    print(f"Publishing {args.case_dir} …")
    try:
        result = publish_case(
            args.case_dir, book=not args.no_book, max_image_px=args.max_image_px
        )
    except PublishError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"\nPortal bundle written: {result.portal}\n"
        f"  {len(result.files)} file(s), {result.size_bytes / 1e6:.1f} MB"
        + (" (+ book/, not counted — git-ignored)" if result.book else "")
    )
    if result.skipped:
        print(f"  {len(result.skipped)} layer(s) skipped — see the reasons above")
    print(f"\nNext: commit it with `git add {result.portal.parent.name}/portal`")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m hywaves`` — parses argv and dispatches to a subcommand."""
    parser = argparse.ArgumentParser(
        prog="hywaves",
        description="HyGloW — hybrid global wave downscaling toolkit",
    )
    parser.add_argument("--version", action="version", version=f"hywaves {__version__}")
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
            "Update the case's engine files (utils/, tests/) to the current "
            "hywaves/template/ versions. Locally modified files are kept "
            "unless --force. Notebooks and SnapWave templates are never "
            "touched. Orphaned files (e.g. a legacy vendored bluemesh2d/ or "
            "pymesh2d/, now pip dependencies) are reported, never deleted."
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
        description="Validate a case against hywaves/contract/CONTRACT.md.",
    )
    p_validate.add_argument("case_dir", help="Path to the case folder")
    p_validate.add_argument(
        "--publish",
        action="store_true",
        help="Strict mode: completeness gaps become errors (portal-ready check)",
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_publish = sub.add_parser(
        "publish",
        help="Export the case's portal bundle to {Case}/portal/",
        description=(
            "Run `validate --publish`, then build {Case}/portal/: the small "
            "set of files the portal (GUI) reads — contract files, "
            "pre-computed map layers (AoI, depth contours, output sites), "
            "merged metrics, web-sized figures, and a manifest. Replaces any "
            "previous bundle. See CONTRACT.md §6."
        ),
    )
    p_publish.add_argument("case_dir", help="Path to the case folder")
    p_publish.add_argument(
        "--no-book",
        action="store_true",
        help="Don't copy the built Jupyter Book into portal/book/",
    )
    p_publish.add_argument(
        "--max-image-px",
        type=int,
        default=2400,
        help="Downscale published PNGs to this longest side in pixels (default: 2400)",
    )
    p_publish.set_defaults(func=_cmd_publish)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
