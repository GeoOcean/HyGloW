"""North Carolina case configuration loader.

Reads ``NorthCarolina.json`` once when the module is imported and exposes
its sections as module-level variables that can be used directly from a
notebook::

    from utils.config import CONFIG, PATHS, GOALS, SIMULATIONS, METAMODEL

Helpers
-------
:func:`resolve`
    Resolve a path against the case directory.
:func:`get_buoy_coordinates`
    Read buoy GeoJSON, applying the ``buoys.roles_override`` from the
    main JSON on top of the per-feature ``role``/``type`` defaults.
:func:`list_institutions`
    Unique buoy institutions in order of first appearance.
:func:`write_buoy_coordinates_txt`
    Dump ``lon lat`` pairs to a SnapWave-style text file.
:func:`reload`
    Re-read the JSON from disk (useful from a running notebook kernel).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

UTILS_DIR: Path = Path(__file__).resolve().parent
CASE_DIR: Path = UTILS_DIR.parent
REPO_ROOT: Path = CASE_DIR.parent

CASE_CONFIG_PATH: Path = CASE_DIR / "NorthCarolina.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve(path: str | Path) -> Path:
    """Return an absolute path; relative inputs are resolved against ``CASE_DIR``."""

    p = Path(path)
    return p if p.is_absolute() else (CASE_DIR / p)


# ---------------------------------------------------------------------------
# Module-level config snapshot
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = _load_json(CASE_CONFIG_PATH)

NAME: str = CONFIG.get("name", "")
DESCRIPTION: str = CONFIG.get("description", "")
PATHS: dict[str, Any] = CONFIG.get("paths", {})
PERIODS: dict[str, Any] = CONFIG.get("periods", {})
GOALS: dict[str, Any] = CONFIG.get("goals", {})
OFFSHORE_WAVES: list[dict] = CONFIG.get("offshore_waves", [])
ONSHORE_WAVES: list[dict] = CONFIG.get("onshore_waves", [])
BUOYS: dict[str, Any] = CONFIG.get("buoys", {})
OUTPUT_SITES: dict[str, Any] = CONFIG.get("output_sites", {})
SIMULATIONS: dict[str, Any] = CONFIG.get("simulations", {})
CALIBRATION: dict[str, Any] = CONFIG.get("calibration", {})
METAMODEL: dict[str, Any] = CONFIG.get("metamodel", {})
VARIABLES: dict[str, Any] = CONFIG.get("variables", {})
VALIDATION: dict[str, Any] = CONFIG.get("validation", {})


def snapwave_var_map() -> dict[str, str]:
    """Canonical parameter → SnapWave variable name."""
    return dict(VARIABLES.get("snapwave", {}))


def bulk_vars() -> list[str]:
    """Canonical bulk variable names (GPR reconstruction output)."""
    return list(VARIABLES.get("bulk", ["hs", "tp", "wd", "ds"]))


def spectral_vars() -> list[str]:
    """Canonical spectral product variable names."""
    return list(VARIABLES.get("spectral", ["hs", "tp", "tm01", "dm"]))


def forcing_vars() -> list[str]:
    """WHACS forcing / MDA input variable names."""
    return list(VARIABLES.get("forcing", ["hs", "tp", "wd", "ds", "zs"]))


def mda_inputs() -> list[str]:
    """Case variables used as MDA/GP subset inputs."""
    return list(METAMODEL.get("mda_inputs", ["tp", "wd", "ds", "zs"]))


def validation_period_var() -> str:
    """Period variable used for validation comparisons (``tp`` or ``tm01``)."""
    return str(VALIDATION.get("period_var", "tm01"))


def reload() -> dict[str, Any]:
    """Re-read ``NorthCarolina.json`` from disk and refresh module globals."""

    global CONFIG, NAME, DESCRIPTION, PATHS, PERIODS, GOALS
    global OFFSHORE_WAVES, ONSHORE_WAVES, BUOYS, OUTPUT_SITES
    global SIMULATIONS, CALIBRATION, METAMODEL, VARIABLES, VALIDATION

    CONFIG = _load_json(CASE_CONFIG_PATH)
    NAME = CONFIG.get("name", "")
    DESCRIPTION = CONFIG.get("description", "")
    PATHS = CONFIG.get("paths", {})
    PERIODS = CONFIG.get("periods", {})
    GOALS = CONFIG.get("goals", {})
    OFFSHORE_WAVES = CONFIG.get("offshore_waves", [])
    ONSHORE_WAVES = CONFIG.get("onshore_waves", [])
    BUOYS = CONFIG.get("buoys", {})
    OUTPUT_SITES = CONFIG.get("output_sites", {})
    SIMULATIONS = CONFIG.get("simulations", {})
    CALIBRATION = CONFIG.get("calibration", {})
    METAMODEL = CONFIG.get("metamodel", {})
    VARIABLES = CONFIG.get("variables", {})
    VALIDATION = CONFIG.get("validation", {})
    return CONFIG


# ---------------------------------------------------------------------------
# Buoys
# ---------------------------------------------------------------------------


def _buoys_geojson_path() -> Path:
    """Locate the buoys GeoJSON via the ``Buoys`` entry in ``onshore_waves``."""

    for entry in ONSHORE_WAVES:
        if entry.get("name") == "Buoys" and entry.get("config_path"):
            return resolve(entry["config_path"])
    raise FileNotFoundError(
        "No 'Buoys' entry with a 'config_path' field found in onshore_waves."
    )


def _build_role_map(features: list[dict]) -> dict[str, str]:
    """Per-feature default role: ``properties.role`` first, then ``properties.type``."""

    role_map: dict[str, str] = {}
    for feat in features:
        props = feat.get("properties") or {}
        iid = props.get("instrument_id")
        if iid is None:
            continue
        default = props.get("role") or props.get("type")
        if isinstance(default, str):
            role_map[str(iid)] = default.strip().lower()
    return role_map


def get_buoy_coordinates(
    role: str | None = None,
    institution: str | None = None,
    *,
    coordinates_path: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return buoy coordinates from the JennBuoys GeoJSON.

    Parameters
    ----------
    role
        Filter by buoy role (``"calibration"``, ``"validation"``, ``"ignore"``,
        ...). Defaults come from the per-feature ``role``/``type`` GeoJSON
        properties and are then overridden by
        ``CONFIG["buoys"]["roles_override"][role]`` if set. ``None`` returns
        every buoy regardless of role.
    institution
        Case-insensitive filter on ``properties.institution`` (e.g. ``"NDBC"``).
    coordinates_path
        Override the GeoJSON path (defaults to the ``Buoys`` entry in
        ``onshore_waves``).

    Returns
    -------
    dict[str, dict[str, Any]]
        ``instrument_id -> {"lon", "lat", "role", "institution", "type",
        "recording", "depth", "name"}``.
    """

    path = (
        Path(coordinates_path)
        if coordinates_path is not None
        else _buoys_geojson_path()
    )
    fc = _load_json(path)
    features = fc.get("features", []) or []

    role_map = _build_role_map(features)
    overrides = (BUOYS or {}).get("roles_override", {}) or {}
    for r, ids in overrides.items():
        for iid in ids or []:
            role_map[str(iid)] = str(r).strip().lower()

    requested_role = role.strip().lower() if isinstance(role, str) else None
    requested_inst = (
        institution.strip().lower() if isinstance(institution, str) else None
    )

    buoys: dict[str, dict[str, Any]] = {}
    for feat in features:
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue

        iid = props.get("instrument_id")
        if iid is None:
            continue
        iid = str(iid)

        feat_role = role_map.get(iid)
        feat_inst = props.get("institution", "") or ""

        if requested_role is not None and feat_role != requested_role:
            continue
        if requested_inst is not None and feat_inst.strip().lower() != requested_inst:
            continue

        buoys[iid] = {
            "lon": float(coords[0]),
            "lat": float(coords[1]),
            "role": feat_role,
            "institution": feat_inst,
            "type": props.get("type"),
            "recording": props.get("recording"),
            "depth": props.get("depth"),
            "name": props.get("name"),
        }

    return buoys


def get_buoy_aliases() -> dict[str, str]:
    """Return ``alias_id -> canonical_id`` mapping from ``CONFIG.buoys.aliases``."""

    return dict((BUOYS or {}).get("aliases", {}) or {})


def list_institutions(
    coordinates_path: Path | str | None = None,
) -> list[str]:
    """Unique institution names in the buoys GeoJSON, first-appearance order."""

    path = (
        Path(coordinates_path)
        if coordinates_path is not None
        else _buoys_geojson_path()
    )
    fc = _load_json(path)

    seen: set[str] = set()
    ordered: list[str] = []
    for feature in fc.get("features", []):
        inst = (feature.get("properties") or {}).get("institution")
        if isinstance(inst, str) and inst not in seen:
            seen.add(inst)
            ordered.append(inst)
    return ordered


def write_buoy_coordinates_txt(
    buoys: Mapping[str, Mapping[str, Any]] | Iterable[tuple[float, float]],
    output_path: Path | str,
    *,
    precision: int = 6,
    overwrite: bool = True,
) -> Path:
    """Write buoy coordinates to a plain ``lon lat`` text file (SnapWave format).

    ``buoys`` may be either a mapping like the one returned by
    :func:`get_buoy_coordinates` or any iterable of ``(lon, lat)`` pairs.
    Relative ``output_path`` is resolved against :data:`CASE_DIR`.
    """

    if isinstance(buoys, Mapping):
        pairs: list[tuple[float, float]] = [
            (float(entry["lon"]), float(entry["lat"]))
            for entry in buoys.values()
            if "lon" in entry and "lat" in entry
        ]
    else:
        pairs = [(float(lon), float(lat)) for lon, lat in buoys]

    path = resolve(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"{{:.{precision}f}} {{:.{precision}f}}\n"
    with path.open("w", encoding="utf-8") as f:
        for lon, lat in pairs:
            f.write(fmt.format(lon, lat))

    return path


__all__ = [
    "UTILS_DIR",
    "CASE_DIR",
    "REPO_ROOT",
    "CASE_CONFIG_PATH",
    "CONFIG",
    "NAME",
    "DESCRIPTION",
    "PATHS",
    "PERIODS",
    "GOALS",
    "OFFSHORE_WAVES",
    "ONSHORE_WAVES",
    "BUOYS",
    "OUTPUT_SITES",
    "SIMULATIONS",
    "CALIBRATION",
    "METAMODEL",
    "VARIABLES",
    "VALIDATION",
    "bulk_vars",
    "forcing_vars",
    "mda_inputs",
    "resolve",
    "reload",
    "snapwave_var_map",
    "spectral_vars",
    "validation_period_var",
    "get_buoy_coordinates",
    "get_buoy_aliases",
    "list_institutions",
    "write_buoy_coordinates_txt",
]
