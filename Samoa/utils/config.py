"""Load ``Samoa.json`` and expose its sections plus :func:`resolve`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xarray as xr

UTILS_DIR = Path(__file__).resolve().parent
CASE_DIR = UTILS_DIR.parent
CASE_CONFIG_PATH = CASE_DIR / "Samoa.json"


def resolve(path: str | Path) -> Path:
    """
    Return an absolute path under the Samoa case directory.

    Parameters
    ----------
    path
        Absolute path, or relative to ``CASE_DIR``.

    Returns
    -------
    Path
        Resolved filesystem path.
    """
    p = Path(path)
    return p if p.is_absolute() else CASE_DIR / p


def _load() -> dict[str, Any]:
    """Read and parse ``Samoa.json``."""
    with CASE_CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


CONFIG = _load()

PATHS: dict[str, Any] = CONFIG["paths"]
OFFSHORE_WAVES: list[dict] = CONFIG.get("offshore_waves", [])
ONSHORE_WAVES: str = CONFIG["onshore_waves"]
OUTPUT_SITES: dict[str, Any] = CONFIG.get("output_sites", {})
SIMULATIONS: dict[str, Any] = CONFIG.get("simulations", {})
OUTPUT_SITES_CONFIG: dict[str, Any] = SIMULATIONS.get("output_sites_config", {})
METAMODEL: dict[str, Any] = CONFIG.get("metamodel", {})


def load_goals(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """
    Load the goals mapping from ``simulations.goals_file``.

    Parameters
    ----------
    path
        Optional override for the goals JSON path.

    Returns
    -------
    dict[str, dict]
        ``goals`` section keyed by goal id.
    """
    goals_path = resolve(path or SIMULATIONS["goals_file"])
    with goals_path.open(encoding="utf-8") as f:
        return json.load(f)["goals"]


def goals_for_snapwave(
    goals: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Slim goals dict for SnapWave wrappers.

    Parameters
    ----------
    goals
        Full goals mapping from :func:`load_goals`.

    Returns
    -------
    dict[str, dict]
        ``{id: {lon, lat, angles}}`` per goal. ``angles`` is nautical
        clockwise ``[left, right]`` (see :mod:`utils.directions`).
    """
    return {
        key: {
            "lon": goal["target"]["longitude"],
            "lat": goal["target"]["latitude"],
            "angles": goal["angles"],
        }
        for key, goal in goals.items()
    }


def mda_max_points() -> int:
    """
    Largest MDA candidate size (= train cases at full fit).

    Returns
    -------
    int
        Last entry in ``metamodel.mda_candidates``.

    Raises
    ------
    ValueError
        If ``mda_candidates`` is empty.
    """
    candidates = METAMODEL.get("mda_candidates", [])
    if not candidates:
        raise ValueError("metamodel.mda_candidates is empty in Samoa.json")
    return int(candidates[-1])


def forcing_columns() -> dict[str, str]:
    """
    Map MDA input names to SnapWave case NetCDF forcing variables.

    Returns
    -------
    dict[str, str]
        ``{mda_input: f"{name}_forcing"}``.
    """
    return {name: f"{name}_forcing" for name in METAMODEL["mda_inputs"]}


def get_offshore_wave(name: str | None = None) -> dict[str, Any]:
    """
    Return one ``offshore_waves`` entry from ``Samoa.json``.

    Parameters
    ----------
    name
        Catalog name (case-insensitive). First entry if omitted.

    Returns
    -------
    dict
        Offshore-waves configuration block.

    Raises
    ------
    ValueError, KeyError
        If the catalog list is empty or *name* is unknown.
    """
    if not OFFSHORE_WAVES:
        raise ValueError("offshore_waves is empty in Samoa.json")
    if name is None:
        return OFFSHORE_WAVES[0]
    key = name.casefold()
    for entry in OFFSHORE_WAVES:
        if entry.get("name", "").casefold() == key:
            return entry
    names = [e.get("name") for e in OFFSHORE_WAVES]
    raise KeyError(f"Unknown offshore wave {name!r}; available: {names}")


def offshore_catalog_key(entry: dict[str, Any]) -> str:
    """
    Goals hindcast key for a processed offshore-waves entry.

    Parameters
    ----------
    entry
        One ``offshore_waves`` block from ``Samoa.json``.

    Returns
    -------
    str
        Lowercase catalog name (e.g. ``whacs``).
    """
    return str(entry["name"]).casefold()


def open_offshore_waves(
    name: str | None = None,
    *,
    entry: dict[str, Any] | None = None,
) -> xr.Dataset:
    """
    Open the processed offshore-waves zarr for *name* or *entry*.

    Parameters
    ----------
    name
        Offshore catalog name in ``Samoa.json``.
    entry
        Pre-resolved offshore block (overrides *name*).

    Returns
    -------
    xarray.Dataset
        Lazy zarr dataset.

    Raises
    ------
    ValueError
        If the entry has no ``data_path``.
    """
    spec = entry or get_offshore_wave(name)
    path = spec.get("data_path")
    if not path:
        raise ValueError(f"offshore wave {spec.get('name')!r} has no data_path")
    return xr.open_zarr(resolve(path))


def reload() -> dict[str, Any]:
    """
    Re-read ``Samoa.json`` and refresh module globals.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    global \
        CONFIG, \
        PATHS, \
        OFFSHORE_WAVES, \
        ONSHORE_WAVES, \
        OUTPUT_SITES, \
        SIMULATIONS, \
        OUTPUT_SITES_CONFIG, \
        METAMODEL

    CONFIG = _load()
    PATHS = CONFIG["paths"]
    OFFSHORE_WAVES = CONFIG.get("offshore_waves", [])
    ONSHORE_WAVES = CONFIG["onshore_waves"]
    OUTPUT_SITES = CONFIG.get("output_sites", {})
    SIMULATIONS = CONFIG.get("simulations", {})
    OUTPUT_SITES_CONFIG = SIMULATIONS.get("output_sites_config", {})
    METAMODEL = CONFIG.get("metamodel", {})

    return CONFIG
