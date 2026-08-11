"""HyGloW case contract validator (see CONTRACT.md).

Self-contained and stdlib-only on purpose: copy this single file into any
repository and run it against a case folder, no installs required::

    python validate.py path/to/MyCase [--publish]

Or through the CLI when the hyglow package is available::

    python -m hyglow validate MyCase/ --publish

Development mode reports incomplete-but-expected gaps (missing inputs,
TODO placeholders, no outputs yet) as warnings. ``--publish`` escalates
them to errors: a publish-clean case is ready for the portal.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_METRICS_COLUMNS = {
    "goal",
    "mda_candidate",
    "group",
    "variable",
    "n",
    "bias",
    "rmse",
    "r2",
}

ERROR = "ERROR"
WARN = "WARN"
INFO = "INFO"


@dataclass
class Finding:
    """One validator result: a severity level, a stable dotted code, and a message."""

    severity: str
    code: str
    message: str

    def __str__(self) -> str:
        icon = {ERROR: "✗", WARN: "!", INFO: "·"}[self.severity]
        return f"{icon} [{self.severity}] {self.code}: {self.message}"


class _Report:
    """Collects findings; ``gap()`` severity depends on publish mode."""

    def __init__(self, publish: bool):
        self.publish = publish
        self.findings: list[Finding] = []

    def error(self, code: str, message: str) -> None:
        self.findings.append(Finding(ERROR, code, message))

    def warn(self, code: str, message: str) -> None:
        self.findings.append(Finding(WARN, code, message))

    def info(self, code: str, message: str) -> None:
        self.findings.append(Finding(INFO, code, message))

    def gap(self, code: str, message: str) -> None:
        """Expected-in-development gap: WARN normally, ERROR at publish."""
        self.findings.append(Finding(ERROR if self.publish else WARN, code, message))


# ----------------------------------------------------------------------
# Config discovery
# ----------------------------------------------------------------------


def find_case_config(case_dir: Path) -> Path | None:
    """Locate the case config JSON inside *case_dir*.

    Convention: ``{Case}/{Case}.json`` (case-insensitive fallback). If exactly
    one other top-level JSON exists that is not a goals/observations file, it
    is accepted as a fallback. Returns ``None`` if no config can be resolved.
    """
    name = case_dir.name
    for candidate in (case_dir / f"{name}.json", case_dir / f"{name.lower()}.json"):
        if candidate.is_file():
            return candidate
    fallbacks = sorted(
        p
        for p in case_dir.glob("*.json")
        if "goals" not in p.name.lower() and "observations" not in p.name.lower()
    )
    return fallbacks[0] if len(fallbacks) == 1 else None


def _load_json(path: Path, report: _Report, code: str) -> Any | None:
    """Parse *path* as JSON, reporting *code* as an error (not raising) on failure."""
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        report.error(code, f"{path.name} is not valid JSON: {exc}")
        return None


def _scan_todos(value: Any, trail: str, hits: list[str]) -> None:
    """Recursively collect dotted-path *trail*s of any string containing ``"TODO"``."""
    if isinstance(value, str) and "TODO" in value:
        hits.append(trail)
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan_todos(item, f"{trail}.{key}" if trail else str(key), hits)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _scan_todos(item, f"{trail}[{i}]", hits)


def _is_number(value: Any) -> bool:
    """``True`` for ``int``/``float``, excluding ``bool`` (a ``bool`` is technically an ``int``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ----------------------------------------------------------------------
# Level 1 — portal contract
# ----------------------------------------------------------------------


def _check_identity(config: dict, report: _Report) -> None:
    """Check ``name``, ``description``, and scan the whole config for TODOs."""
    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        report.error("config.name", "'name' must be a non-empty string")
    description = config.get("description")
    if not isinstance(description, str) or not description.strip():
        report.gap("config.description", "'description' is missing or empty")

    todos: list[str] = []
    _scan_todos(config, "", todos)
    if todos:
        report.gap(
            "config.todos",
            f"config still contains TODO placeholders at: {', '.join(todos[:8])}",
        )


def _check_paths(case_dir: Path, config: dict, report: _Report) -> None:
    """Check ``paths`` is a dict of strings, and warn about relative entries that don't exist yet."""
    paths = config.get("paths", {})
    if not isinstance(paths, dict):
        report.error("config.paths", "'paths' must be a dict")
        return
    for key, value in paths.items():
        if not isinstance(value, str):
            report.error("config.paths", f"paths.{key} must be a string")
            continue
        if "TODO" in value:
            continue  # already reported by the TODO scan
        if not (case_dir / value).exists() and not Path(value).is_absolute():
            report.warn("paths.missing", f"paths.{key} does not exist yet: {value}")


def _check_goals(case_dir: Path, config: dict, report: _Report) -> None:
    """Check ``simulations.goals_file`` is set, parses, and each goal has valid target/angles."""
    goals_file = config.get("simulations", {}).get("goals_file")
    if not isinstance(goals_file, str) or not goals_file:
        report.gap("goals.declared", "simulations.goals_file is not set")
        return
    path = case_dir / goals_file
    if not path.is_file():
        report.gap("goals.file", f"goals file does not exist yet: {goals_file}")
        return
    data = _load_json(path, report, "goals.parse")
    if data is None:
        return
    goals = data.get("goals")
    if not isinstance(goals, dict) or not goals:
        report.error(
            "goals.structure", f"{path.name}: 'goals' must be a non-empty dict"
        )
        return
    for gid, goal in goals.items():
        target = goal.get("target", {})
        if not (
            _is_number(target.get("longitude")) and _is_number(target.get("latitude"))
        ):
            report.error(
                "goals.target",
                f"goal {gid}: target.longitude/latitude must be numeric",
            )
        angles = goal.get("angles")
        if (
            not isinstance(angles, list)
            or len(angles) != 2
            or not all(_is_number(a) for a in angles)
        ):
            report.error("goals.angles", f"goal {gid}: 'angles' must be [left, right]")
    report.info("goals.ok", f"{len(goals)} goal(s) in {path.name}")


def _check_observations(case_dir: Path, config: dict, report: _Report) -> None:
    """Check ``onshore_waves`` is set and points to a valid GeoJSON FeatureCollection."""
    onshore = config.get("onshore_waves")
    if not isinstance(onshore, str) or not onshore:
        report.gap("observations.declared", "'onshore_waves' is not set")
        return
    path = case_dir / onshore
    if not path.is_file():
        report.gap(
            "observations.file", f"observations file does not exist yet: {onshore}"
        )
        return
    data = _load_json(path, report, "observations.parse")
    if data is None:
        return
    if data.get("type") != "FeatureCollection":
        report.error(
            "observations.type", f"{path.name} must be a GeoJSON FeatureCollection"
        )
        return
    features = data.get("features", [])
    seen_ids: set[str] = set()
    for i, feat in enumerate(features):
        props = feat.get("properties", {})
        geometry = feat.get("geometry", {})
        coords = geometry.get("coordinates", [])
        if geometry.get("type") != "Point" or len(coords) < 2:
            report.error(
                "observations.geometry", f"feature {i}: must be a Point with [lon, lat]"
            )
        instrument = props.get("instrument_id")
        if not isinstance(instrument, str) or not instrument:
            report.error(
                "observations.instrument", f"feature {i}: missing 'instrument_id'"
            )
        elif instrument in seen_ids:
            report.error(
                "observations.instrument", f"duplicate instrument_id {instrument!r}"
            )
        else:
            seen_ids.add(instrument)
    report.info(
        "observations.ok", f"{len(features)} observation site(s) in {path.name}"
    )


def _check_offshore(config: dict, report: _Report) -> None:
    """Check ``offshore_waves`` is a list (or single dict) of entries each with a ``name``."""
    block = config.get("offshore_waves", [])
    entries = [block] if isinstance(block, dict) else block
    if not isinstance(entries, list):
        report.error("offshore.structure", "'offshore_waves' must be a list")
        return
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("name"):
            report.error("offshore.name", f"offshore_waves[{i}] must have a 'name'")


def _check_outputs(case_dir: Path, report: _Report) -> None:
    """Check ``outputs/goal_*/validation_metrics.csv`` files have the required columns.

    In ``--publish`` mode, also warns if no portal figures exist under
    ``outputs/gui_figures/``.
    """
    metrics_files = sorted((case_dir / "outputs").glob("goal_*/validation_metrics.csv"))
    if not metrics_files:
        report.gap(
            "outputs.metrics",
            "no outputs/goal_*/validation_metrics.csv yet — run the holdout validation sweep",
        )
    for path in metrics_files:
        try:
            with path.open(encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
        except OSError as exc:
            report.error("outputs.metrics", f"cannot read {path}: {exc}")
            continue
        missing = REQUIRED_METRICS_COLUMNS - set(header)
        if missing:
            report.error(
                "outputs.metrics",
                f"{path.parent.name}/{path.name} missing required columns: {sorted(missing)}",
            )
    if metrics_files:
        report.info(
            "outputs.ok", f"{len(metrics_files)} validation_metrics.csv file(s)"
        )

    figures = sorted((case_dir / "outputs" / "gui_figures").glob("fig*.png"))
    if not figures and report.publish:
        report.warn("outputs.figures", "no portal figures in outputs/gui_figures/")


# ----------------------------------------------------------------------
# Level 2 — engine contract
# ----------------------------------------------------------------------


def _check_engine(case_dir: Path, config: dict, report: _Report) -> None:
    """Level 2 checks: validate the ``metamodel`` section against the engine contract.

    Skipped entirely (with an info finding) when ``metamodel`` is absent —
    only cases that opt into the metamodel workflow need to satisfy this.
    """
    mm = config.get("metamodel")
    if mm is None:
        report.info("engine.skip", "no 'metamodel' section — Level 2 checks skipped")
        return
    if not isinstance(mm, dict):
        report.error("engine.metamodel", "'metamodel' must be a dict")
        return

    candidates = mm.get("mda_candidates")
    if not isinstance(candidates, list) or not candidates:
        report.error(
            "engine.mda_candidates", "metamodel.mda_candidates must be a non-empty list"
        )
    else:
        for n in candidates:
            if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
                report.error(
                    "engine.mda_candidates",
                    f"mda_candidates entries must be positive ints, got {n!r}",
                )

    inputs = mm.get("mda_inputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not all(isinstance(c, str) for c in inputs)
    ):
        report.error(
            "engine.mda_inputs",
            "metamodel.mda_inputs must be a non-empty list of strings",
        )
        inputs = []

    directional = mm.get("mda_directional_variables", [])
    if not isinstance(directional, list):
        report.error(
            "engine.directional", "metamodel.mda_directional_variables must be a list"
        )
    else:
        unknown = set(directional) - set(inputs)
        if unknown:
            report.error(
                "engine.directional",
                f"mda_directional_variables not in mda_inputs: {sorted(unknown)}",
            )

    vars_ = mm.get("vars_to_predict")
    predict_names: set[str] = set()
    if not isinstance(vars_, dict) or not vars_:
        report.error(
            "engine.vars_to_predict",
            "metamodel.vars_to_predict must be a non-empty dict",
        )
    else:
        predict_names = set(vars_)
        for name, spec in vars_.items():
            if spec == "raw":
                continue
            if not isinstance(spec, dict):
                report.error(
                    "engine.vars_to_predict",
                    f"vars_to_predict[{name!r}] must be 'raw' or a dict",
                )
                continue
            stack = spec.get("vars_to_stack")
            if not isinstance(stack, list) or not stack:
                report.error(
                    "engine.vars_to_predict",
                    f"vars_to_predict[{name!r}].vars_to_stack must be a non-empty list",
                )
            variance = spec.get("pca_variance")
            if not _is_number(variance) or not 0 < float(variance) <= 1:
                report.error(
                    "engine.vars_to_predict",
                    f"vars_to_predict[{name!r}].pca_variance must be in (0, 1]",
                )

    site_filter = mm.get("site_filter", {})
    if not isinstance(site_filter, dict):
        report.error("engine.site_filter", "metamodel.site_filter must be a dict")
    else:
        for key in site_filter:
            if not (key.endswith("_max") or key.endswith("_min")):
                report.error(
                    "engine.site_filter",
                    f"site_filter key {key!r} must end with '_max' or '_min'",
                )

    metrics = mm.get("validation", {}).get("metrics_variables")
    if not isinstance(metrics, list) or not metrics:
        report.error(
            "engine.metrics_variables",
            "metamodel.validation.metrics_variables must be a non-empty list",
        )
    elif predict_names:
        missing = [v for v in metrics if v not in predict_names]
        if missing:
            report.error(
                "engine.metrics_variables",
                f"metrics_variables not in vars_to_predict: {missing}",
            )

    recon = mm.get("reconstruction")
    if not isinstance(recon, dict):
        report.error("engine.reconstruction", "metamodel.reconstruction must be a dict")
    else:
        var_map = recon.get("variables")
        if not isinstance(var_map, dict) or not var_map:
            report.error(
                "engine.reconstruction",
                "reconstruction.variables must be a non-empty dict (name → wavespectra stat)",
            )
        else:
            if predict_names:
                missing = [v for v in var_map if v not in predict_names]
                if missing:
                    report.error(
                        "engine.reconstruction",
                        f"reconstruction.variables keys not in vars_to_predict: {missing}",
                    )
            try:
                from wavespectra.construct import STATS  # type: ignore
            except ImportError:
                report.info(
                    "engine.wavespectra",
                    "wavespectra not installed — stat-name check skipped",
                )
            else:
                unknown = [s for s in var_map.values() if s not in STATS]
                if unknown:
                    report.error(
                        "engine.reconstruction",
                        f"unsupported wavespectra stats: {unknown}; allowed: {list(STATS)}",
                    )
        if "save_spectra" not in recon:
            report.error(
                "engine.reconstruction", "reconstruction.save_spectra is required"
            )
        elif not isinstance(recon["save_spectra"], bool):
            report.error(
                "engine.reconstruction", "reconstruction.save_spectra must be a bool"
            )

    paths = mm.get("paths", {})
    for key in ("pca_dir", "gp_dir"):
        if key not in paths or not isinstance(paths[key], str):
            report.error(
                "engine.paths", f"metamodel.paths.{key} must be a template string"
            )

    sim = config.get("simulations", {})
    for key in ("metamodel_templates", "dynamic_templates"):
        folder = sim.get(key)
        if isinstance(folder, str) and folder and not (case_dir / folder).is_dir():
            report.warn(
                "engine.templates", f"simulations.{key} folder missing: {folder}"
            )


# ----------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------


def validate_case(case_dir: str | Path, *, publish: bool = False) -> list[Finding]:
    """Validate one case folder against the contract (see ``CONTRACT.md``).

    Runs the Level 1 portal checks (identity, paths, goals, observations,
    offshore catalog, outputs) and, if the config declares a ``metamodel``
    section, the Level 2 engine checks.

    Parameters
    ----------
    case_dir
        Path to the case folder.
    publish
        Strict mode: expected-in-development gaps (missing inputs, TODO
        placeholders, no outputs yet) become errors instead of warnings.

    Returns
    -------
    list[Finding]
        Every finding produced, in check order. Empty only if *case_dir*
        can't be resolved to a config at all (that itself is a ``Finding``,
        so this is never actually empty in practice).
    """
    case_dir = Path(case_dir).resolve()
    report = _Report(publish)

    if not case_dir.is_dir():
        report.error("case.dir", f"not a directory: {case_dir}")
        return report.findings

    config_path = find_case_config(case_dir)
    if config_path is None:
        report.error(
            "config.file",
            f"no case config JSON found (expected {case_dir.name}/{case_dir.name}.json)",
        )
        return report.findings

    config = _load_json(config_path, report, "config.parse")
    if config is None:
        return report.findings
    if not isinstance(config, dict):
        report.error("config.parse", f"{config_path.name} must contain a JSON object")
        return report.findings

    _check_identity(config, report)
    _check_paths(case_dir, config, report)
    _check_goals(case_dir, config, report)
    _check_observations(case_dir, config, report)
    _check_offshore(config, report)
    _check_outputs(case_dir, report)
    _check_engine(case_dir, config, report)
    return report.findings


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: parse args, run :func:`validate_case`, print findings.

    Returns
    -------
    int
        ``0`` if there were no errors, ``1`` otherwise (warnings never fail).
    """
    parser = argparse.ArgumentParser(
        description="Validate a HyGloW case folder against the case contract."
    )
    parser.add_argument("case_dir", help="Path to the case folder")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Strict mode: completeness gaps become errors (portal-ready check)",
    )
    args = parser.parse_args(argv)

    findings = validate_case(args.case_dir, publish=args.publish)
    errors = [f for f in findings if f.severity == ERROR]
    warnings = [f for f in findings if f.severity == WARN]

    for finding in findings:
        print(finding)
    mode = "publish" if args.publish else "development"
    print(
        f"\n{Path(args.case_dir).name}: {len(errors)} error(s), "
        f"{len(warnings)} warning(s) [{mode} mode]"
    )
    if not errors:
        print("Contract check PASSED" + ("" if args.publish else " (development mode)"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
