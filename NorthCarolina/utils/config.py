"""North Carolina case configuration loader.

Reads ``NorthCarolina.json`` once when the module is imported and exposes
its sections as module-level variables that can be used directly from a
notebook::

    from utils.config import NAME, GOALS, OFFSHORE_WAVES, METAMODEL

Also provides :func:`get_buoy_coordinates` to read buoy coordinates from
``inputs/JennBuoys/JennBuoys_coordinates.json`` filtered by institution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

UTILS_DIR: Path = Path(__file__).resolve().parent
CASE_DIR: Path = UTILS_DIR.parent
REPO_ROOT: Path = CASE_DIR.parent

CASE_CONFIG_PATH: Path = CASE_DIR / "NorthCarolina.json"
BUOYS_COORDINATES_PATH: Path = (
    CASE_DIR / "inputs" / "JennBuoys" / "JennBuoys_coordinates.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


CONFIG: dict[str, Any] = _load_json(CASE_CONFIG_PATH)

NAME: str = CONFIG.get("name", "")
DESCRIPTION: str = CONFIG.get("description", "")
GOALS: dict[str, dict[str, Any]] = CONFIG.get("goals", {})
OFFSHORE_WAVES: list[dict[str, Any]] = CONFIG.get("offshore_waves", [])
ONSHORE_WAVES: list[dict[str, Any]] = CONFIG.get("onshore_waves", [])
METAMODEL: dict[str, Any] = CONFIG.get("metamodel", {})


def reload() -> dict[str, Any]:
    """Re-read ``NorthCarolina.json`` from disk and refresh module globals.

    Useful from an already-running notebook kernel after editing the JSON
    without restarting the kernel.
    """

    global CONFIG, NAME, DESCRIPTION, GOALS, OFFSHORE_WAVES, ONSHORE_WAVES, METAMODEL

    CONFIG = _load_json(CASE_CONFIG_PATH)
    NAME = CONFIG.get("name", "")
    DESCRIPTION = CONFIG.get("description", "")
    GOALS = CONFIG.get("goals", {})
    OFFSHORE_WAVES = CONFIG.get("offshore_waves", [])
    ONSHORE_WAVES = CONFIG.get("onshore_waves", [])
    METAMODEL = CONFIG.get("metamodel", {})
    return CONFIG


def get_buoy_coordinates(
    institution: str | None = None,
    *,
    coordinates_path: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return buoy coordinates from the JennBuoys GeoJSON.

    Parameters
    ----------
    institution : str, optional
        Filter features by ``properties.institution`` (case-insensitive,
        whitespace-trimmed). If ``None`` (default), buoys from every
        institution are returned.
    coordinates_path : Path | str, optional
        Override the GeoJSON path. Defaults to
        :data:`BUOYS_COORDINATES_PATH`.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping ``instrument_id -> {"lon", "lat", "institution", "type",
        "recording", "depth", "name"}``. When several features share the
        same ``instrument_id``, the last one in the file wins.
    """

    path = (
        Path(coordinates_path)
        if coordinates_path is not None
        else BUOYS_COORDINATES_PATH
    )
    feature_collection = _load_json(path)

    requested = institution.strip().lower() if isinstance(institution, str) else None

    buoys: dict[str, dict[str, Any]] = {}
    for feature in feature_collection.get("features", []):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue

        feat_institution = props.get("institution", "") or ""
        if requested is not None and feat_institution.strip().lower() != requested:
            continue

        instrument_id = props.get("instrument_id")
        if instrument_id is None:
            continue

        buoys[str(instrument_id)] = {
            "lon": float(coords[0]),
            "lat": float(coords[1]),
            "institution": feat_institution,
            "type": props.get("type"),
            "recording": props.get("recording"),
            "depth": props.get("depth"),
            "name": props.get("name"),
        }

    return buoys


def write_buoy_coordinates_txt(
    buoys: Mapping[str, Mapping[str, Any]] | Iterable[tuple[float, float]],
    output_path: Path | str,
    *,
    precision: int = 6,
    overwrite: bool = True,
) -> Path:
    """Write buoy coordinates to a plain ``lon lat`` text file (one buoy per line).

    The output format is the one expected by SnapWave: each line contains
    ``<lon> <lat>`` separated by a single space, with no header.

    Parameters
    ----------
    buoys : mapping or iterable
        Either the dict returned by :func:`get_buoy_coordinates` (mapping
        ``instrument_id -> {"lon", "lat", ...}``) or any iterable of
        ``(lon, lat)`` pairs.
    output_path : Path | str
        Destination ``.txt`` file. Relative paths are resolved against
        :data:`CASE_DIR`.
    precision : int, default 6
        Number of decimal places written for each coordinate (~0.1 m at
        these latitudes).
    overwrite : bool, default True
        If ``False`` and ``output_path`` already exists, raise
        ``FileExistsError``.

    Returns
    -------
    Path
        The resolved path of the file that was written.
    """

    if isinstance(buoys, Mapping):
        pairs: list[tuple[float, float]] = [
            (float(entry["lon"]), float(entry["lat"]))
            for entry in buoys.values()
            if "lon" in entry and "lat" in entry
        ]
    else:
        pairs = [(float(lon), float(lat)) for lon, lat in buoys]

    path = Path(output_path)
    if not path.is_absolute():
        path = CASE_DIR / path

    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = f"{{:.{precision}f}} {{:.{precision}f}}\n"
    with path.open("w", encoding="utf-8") as f:
        for lon, lat in pairs:
            f.write(fmt.format(lon, lat))

    return path


def list_institutions(
    coordinates_path: Path | str | None = None,
) -> list[str]:
    """Return the unique institution names present in the buoys GeoJSON,
    preserving first-appearance order.
    """

    path = (
        Path(coordinates_path)
        if coordinates_path is not None
        else BUOYS_COORDINATES_PATH
    )
    feature_collection = _load_json(path)

    seen: set[str] = set()
    ordered: list[str] = []
    for feature in feature_collection.get("features", []):
        inst = (feature.get("properties") or {}).get("institution")
        if isinstance(inst, str) and inst not in seen:
            seen.add(inst)
            ordered.append(inst)
    return ordered


__all__ = [
    "UTILS_DIR",
    "CASE_DIR",
    "REPO_ROOT",
    "CASE_CONFIG_PATH",
    "BUOYS_COORDINATES_PATH",
    "CONFIG",
    "NAME",
    "DESCRIPTION",
    "GOALS",
    "OFFSHORE_WAVES",
    "ONSHORE_WAVES",
    "METAMODEL",
    "get_buoy_coordinates",
    "list_institutions",
    "write_buoy_coordinates_txt",
    "reload",
]
