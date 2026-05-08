"""Case configuration loader + metamodel configuration validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

UTILS_DIR: Path = Path(__file__).resolve().parent
CASE_DIR: Path = UTILS_DIR.parent
REPO_ROOT: Path = CASE_DIR.parent

CASE_CONFIG_PATH: Path = CASE_DIR / f"{CASE_DIR.name.lower()}.json"
GLOBAL_CONFIG_PATH: Path = REPO_ROOT / "config.json"

_SECTIONS: tuple[str, ...] = (
    "offshore_waves",
    "offshore_sealevel",
    "onshore_waves",
)
_PATH_KEYWORDS: tuple[str, ...] = (
    "path",
    "paths",
    "dir",
    "dirs",
    "coordinates",
    "template",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_probable_path_key(key: str) -> bool:
    key_lower = key.lower()
    return any(token in key_lower for token in _PATH_KEYWORDS)


def _resolve_case_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = CASE_DIR / p
    return p


def _resolve_path_like_string(value: str) -> str:
    # Keep templates unresolved.
    if "{" in value and "}" in value:
        return value
    return str(_resolve_case_path(value))


def _resolve_paths_recursive(payload: Any) -> Any:
    """Resolve relative path-like fields against ``CASE_DIR``."""
    if isinstance(payload, dict):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str) and value and _is_probable_path_key(key):
                out[key] = _resolve_path_like_string(value)
            else:
                out[key] = _resolve_paths_recursive(value)
        return out
    if isinstance(payload, list):
        return [_resolve_paths_recursive(item) for item in payload]
    return payload


def _merge_section(
    case_entries: list[dict[str, Any]],
    global_entries: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge case + global entries for one section, keyed by source name."""

    global_by_name = {e["name"]: e for e in global_entries if "name" in e}
    merged: dict[str, dict[str, Any]] = {}
    for entry in case_entries:
        name = entry.get("name")
        if name is None:
            continue
        base = global_by_name.get(name, {})
        merged[name] = _resolve_paths_recursive({**base, **entry})
    return merged


def _validate_and_prepare_metamodel(metamodel_cfg: dict[str, Any]) -> dict[str, Any]:
    required = ("num_mda_points", "n_goals", "paths")
    missing = [key for key in required if key not in metamodel_cfg]
    if missing:
        raise KeyError(
            f"Missing required 'metamodel' keys in {CASE_CONFIG_PATH}: {missing}"
        )

    cfg = _resolve_paths_recursive(metamodel_cfg)
    num_mda_points = int(cfg["num_mda_points"])
    n_goals = int(cfg["n_goals"])
    cfg["goals"] = list(range(1, n_goals + 1))
    cfg.setdefault("partitions", ["bulk", "sea", "swell1", "swell2", "swell3"])
    cfg.setdefault("tp_factors", {partition: 1.0 for partition in cfg["partitions"]})

    paths = cfg["paths"]
    required_paths = (
        "mda_model_template",
        "pca_model_template",
        "gp_model_template",
        "metamodel_cases_template",
        "codec_dir",
        "codec_coordinates",
        "reconstruction_output_dir",
    )
    missing_paths = [key for key in required_paths if key not in paths]
    if missing_paths:
        raise KeyError(
            f"Missing required metamodel path keys in {CASE_CONFIG_PATH}: {missing_paths}"
        )

    # Ensure required directories exist before notebook execution.
    to_create = [
        _resolve_case_path(paths["codec_dir"]),
        _resolve_case_path(paths["reconstruction_output_dir"]),
    ]
    if "models_root" in paths:
        to_create.append(_resolve_case_path(paths["models_root"]))
    for model_dir_template in (
        "mda_dir_template",
        "pca_dir_template",
        "gp_dir_template",
    ):
        template = paths.get(model_dir_template)
        if isinstance(template, str):
            to_create.append(
                _resolve_case_path(template.format(num_mda_points=num_mda_points))
            )
    for directory in to_create:
        directory.mkdir(parents=True, exist_ok=True)

    # Validate required inputs exist.
    expected_files = [
        _resolve_case_path(paths["codec_coordinates"]),
        _resolve_case_path(paths["metamodel_cases_template"].format(goal=1)),
    ]
    for expected_path in expected_files:
        if not expected_path.exists():
            raise FileNotFoundError(f"Configured path does not exist: {expected_path}")

    return cfg


def _load_all() -> dict[str, Any]:
    case_cfg = _load_json(CASE_CONFIG_PATH)
    global_cfg = _load_json(GLOBAL_CONFIG_PATH)
    state: dict[str, Any] = {
        "NAME": case_cfg.get("name"),
        "DESCRIPTION": case_cfg.get("description"),
        "METAMODEL": _validate_and_prepare_metamodel(case_cfg.get("metamodel", {})),
        "_CASE_CFG": case_cfg,
        "_GLOBAL_CFG": global_cfg,
    }
    for section in _SECTIONS:
        state[section.upper()] = _merge_section(
            case_cfg.get(section, []),
            global_cfg.get(section, []),
        )
    return state


_state: dict[str, Any] = _load_all()

NAME: str | None = _state["NAME"]
DESCRIPTION: str | None = _state["DESCRIPTION"]
OFFSHORE_WAVES: dict[str, dict[str, Any]] = _state["OFFSHORE_WAVES"]
OFFSHORE_SEALEVEL: dict[str, dict[str, Any]] = _state["OFFSHORE_SEALEVEL"]
ONSHORE_WAVES: dict[str, dict[str, Any]] = _state["ONSHORE_WAVES"]
METAMODEL: dict[str, Any] = _state["METAMODEL"]
CASE_CONFIG: dict[str, Any] = _state["_CASE_CFG"]
GLOBAL_CONFIG: dict[str, Any] = _state["_GLOBAL_CFG"]


def get_source(category: str, name: str) -> dict[str, Any]:
    """Return the merged config dict for a given ``(category, name)``.

    Parameters
    ----------
    category : str
        One of ``"offshore_waves"``, ``"offshore_sealevel"``, ``"onshore_waves"``.
    name : str
        Source name as listed in the case JSON (e.g. ``"WHACS"``, ``"ERA5"``).
    """

    if category not in _SECTIONS:
        raise KeyError(f"Unknown category {category!r}; expected one of {_SECTIONS}")
    section = {
        "offshore_waves": OFFSHORE_WAVES,
        "offshore_sealevel": OFFSHORE_SEALEVEL,
        "onshore_waves": ONSHORE_WAVES,
    }[category]
    if name not in section:
        raise KeyError(
            f"Source {name!r} not configured under {category!r}. "
            f"Available: {list(section)}"
        )
    return section[name]


def reload() -> dict[str, Any]:
    """Re-read both JSON files from disk and refresh module globals.

    Useful inside an already-running notebook kernel after editing the JSONs
    without restarting.
    """

    global NAME, DESCRIPTION
    global OFFSHORE_WAVES, OFFSHORE_SEALEVEL, ONSHORE_WAVES, METAMODEL
    global CASE_CONFIG, GLOBAL_CONFIG, _state

    _state = _load_all()
    NAME = _state["NAME"]
    DESCRIPTION = _state["DESCRIPTION"]
    OFFSHORE_WAVES = _state["OFFSHORE_WAVES"]
    OFFSHORE_SEALEVEL = _state["OFFSHORE_SEALEVEL"]
    ONSHORE_WAVES = _state["ONSHORE_WAVES"]
    METAMODEL = _state["METAMODEL"]
    CASE_CONFIG = _state["_CASE_CFG"]
    GLOBAL_CONFIG = _state["_GLOBAL_CFG"]
    return _state


__all__ = [
    "UTILS_DIR",
    "CASE_DIR",
    "REPO_ROOT",
    "CASE_CONFIG_PATH",
    "GLOBAL_CONFIG_PATH",
    "NAME",
    "DESCRIPTION",
    "OFFSHORE_WAVES",
    "OFFSHORE_SEALEVEL",
    "ONSHORE_WAVES",
    "METAMODEL",
    "CASE_CONFIG",
    "GLOBAL_CONFIG",
    "get_source",
    "reload",
]
