"""Load ``Samoa.json`` and expose its sections plus :func:`resolve`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UTILS_DIR = Path(__file__).resolve().parent
CASE_DIR = UTILS_DIR.parent
CASE_CONFIG_PATH = CASE_DIR / "Samoa.json"


def resolve(path: str | Path) -> Path:
    """Absolute path; relative inputs are resolved against ``CASE_DIR``."""

    p = Path(path)
    return p if p.is_absolute() else CASE_DIR / p


def _load() -> dict[str, Any]:
    with CASE_CONFIG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


CONFIG = _load()

PATHS: dict[str, Any] = CONFIG["paths"]
OFFSHORE_WAVES: list[dict] = CONFIG.get("offshore_waves", [])
ONSHORE_WAVES: str = CONFIG["onshore_waves"]
OUTPUT_SITES: dict[str, Any] = CONFIG.get(
    "output_sites", {}
)  # legacy; unused — see OUTPUT_SITES_CONFIG
SIMULATIONS: dict[str, Any] = CONFIG.get("simulations", {})
OUTPUT_SITES_CONFIG: dict[str, Any] = SIMULATIONS.get("output_sites_config", {})
METAMODEL: dict[str, Any] = CONFIG.get("metamodel", {})


def reload() -> dict[str, Any]:
    """Re-read ``Samoa.json`` and refresh module globals."""

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


def load_goals(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the ``goals`` mapping from ``simulations.goals_file``."""

    goals_path = resolve(path or SIMULATIONS["goals_file"])
    with goals_path.open(encoding="utf-8") as f:
        return json.load(f)["goals"]


def goals_for_snapwave(
    goals: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """``{id: {lon, lat, angles}}`` for :class:`SnapWaveMetaModelWrapper`."""

    return {
        key: {
            "lon": goal["target"]["longitude"],
            "lat": goal["target"]["latitude"],
            "angles": goal["angles"],
        }
        for key, goal in goals.items()
    }
