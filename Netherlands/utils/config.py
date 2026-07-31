"""Load the case config JSON and expose its sections plus :func:`resolve`."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import xarray as xr

_HYGLOW_LOGGERS = ("hyglow.metamodel", "hyglow.validation")
_QUIET_LOGGER_PREFIXES = ("hyglow.", "bluemath", "BlueMath")

UTILS_DIR = Path(__file__).resolve().parent
CASE_DIR = UTILS_DIR.parent
CASE_CONFIG_PATH = CASE_DIR / f"{CASE_DIR.name}.json"


def resolve(path: str | Path) -> Path:
    """
    Return an absolute path under the current case directory.

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
    """Read and parse the case config JSON."""
    try:
        with CASE_CONFIG_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Case config not found. Expected "
            f"{CASE_CONFIG_PATH}. Ensure the file name matches the case folder "
            "(e.g. ``Samoa/Samoa.json``)."
        ) from exc


def _refresh_sections(config: dict[str, Any]) -> None:
    """Refresh module-level section shortcuts from *config*."""
    global PATHS, OFFSHORE_WAVES, ONSHORE_WAVES
    global OUTPUT_SITES, SIMULATIONS, OUTPUT_SITES_CONFIG, METAMODEL

    PATHS = config["paths"]
    OFFSHORE_WAVES = config.get("offshore_waves", [])
    ONSHORE_WAVES = config["onshore_waves"]
    OUTPUT_SITES = config.get("output_sites", {})
    SIMULATIONS = config.get("simulations", {})
    OUTPUT_SITES_CONFIG = SIMULATIONS.get("output_sites_config", {})
    METAMODEL = config.get("metamodel", {})


CONFIG = _load()
PATHS: dict[str, Any]
OFFSHORE_WAVES: list[dict]
ONSHORE_WAVES: str
OUTPUT_SITES: dict[str, Any]
SIMULATIONS: dict[str, Any]
OUTPUT_SITES_CONFIG: dict[str, Any]
METAMODEL: dict[str, Any]
_refresh_sections(CONFIG)


def case_name(default: str = "Case") -> str:
    """Display name for the current case (from config ``name``)."""
    return str(CONFIG.get("name") or default)


def case_description(default: str = "") -> str:
    """Long description for the current case (from config ``description``)."""
    return str(CONFIG.get("description") or default)


def display_title(
    base: str,
    *,
    include_case: bool = True,
    separator: str = " — ",
) -> str:
    """Build a plot/report title with optional case-name prefix.

    Parameters
    ----------
    base
        Core title text (e.g. ``"Goals + AoI"``).
    include_case
        Prefix with :func:`case_name` when ``True``.
    separator
        Separator between case name and base text.

    Returns
    -------
    str
        Composed title string.
    """
    text = str(base).strip()
    if not include_case:
        return text
    name = case_name().strip()
    if not name:
        return text
    return f"{name}{separator}{text}" if text else name


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
        raise ValueError("metamodel.mda_candidates is empty in case config")
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
    Return one ``offshore_waves`` entry from the case config.

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
        raise ValueError("offshore_waves is empty in case config")
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
        One ``offshore_waves`` block from the case config.

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
        Offshore catalog name in the case config.
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


def validate_metamodel_config(config: dict[str, Any] | None = None) -> None:
    """Validate required ``metamodel`` keys in the case config JSON.

    Raises
    ------
    ValueError
        If a required section or key is missing or malformed.
    TypeError
        If a value has the wrong type.
    """
    cfg = CONFIG if config is None else config
    mm = cfg.get("metamodel")
    if not isinstance(mm, dict):
        raise ValueError("config missing 'metamodel' section")

    candidates = mm.get("mda_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("metamodel.mda_candidates must be a non-empty list")
    for n in candidates:
        if not isinstance(n, int) or n <= 0:
            raise ValueError(
                f"metamodel.mda_candidates entry must be positive int, got {n!r}"
            )

    inputs = mm.get("mda_inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("metamodel.mda_inputs must be a non-empty list")
    if not all(isinstance(c, str) for c in inputs):
        raise TypeError("metamodel.mda_inputs must be a list of strings")

    directional = mm.get("mda_directional_variables", [])
    if not isinstance(directional, list):
        raise TypeError("metamodel.mda_directional_variables must be a list")
    unknown_dir = set(directional) - set(inputs)
    if unknown_dir:
        raise ValueError(
            f"metamodel.mda_directional_variables not in mda_inputs: {sorted(unknown_dir)}"
        )

    vars_ = mm.get("vars_to_predict")
    if not isinstance(vars_, dict) or not vars_:
        raise ValueError("metamodel.vars_to_predict must be a non-empty dict")
    for name, spec in vars_.items():
        if spec == "raw":
            continue
        if not isinstance(spec, dict):
            raise ValueError(
                f"metamodel.vars_to_predict[{name!r}] must be 'raw' or a dict"
            )
        stack = spec.get("vars_to_stack")
        if not isinstance(stack, list) or not stack:
            raise ValueError(
                f"metamodel.vars_to_predict[{name!r}].vars_to_stack must be a list"
            )
        variance = spec.get("pca_variance")
        if not isinstance(variance, (int, float)) or not 0 < float(variance) <= 1:
            raise ValueError(
                f"metamodel.vars_to_predict[{name!r}].pca_variance must be in (0, 1]"
            )

    site_filter = mm.get("site_filter", {})
    if not isinstance(site_filter, dict):
        raise TypeError("metamodel.site_filter must be a dict")
    for key in site_filter:
        if not (key.endswith("_max") or key.endswith("_min")):
            raise ValueError(
                f"metamodel.site_filter key {key!r} must end with '_max' or '_min'"
            )

    pca_pp = mm.get("pca_preprocessing", {})
    if pca_pp:
        if not isinstance(pca_pp, dict):
            raise TypeError("metamodel.pca_preprocessing must be a dict")
        for section in ("nan_threshold_to_drop", "value_to_replace_nans"):
            block = pca_pp.get(section, {})
            if block and not isinstance(block, dict):
                raise TypeError(f"metamodel.pca_preprocessing.{section} must be a dict")

    validation = mm.get("validation", {})
    metrics = validation.get("metrics_variables")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(
            "metamodel.validation.metrics_variables must be a non-empty list"
        )
    predict_names = set(vars_)
    missing_metrics = [v for v in metrics if v not in predict_names]
    if missing_metrics:
        raise ValueError(
            "metamodel.validation.metrics_variables not in vars_to_predict: "
            f"{missing_metrics}"
        )

    recon = mm.get("reconstruction")
    if not isinstance(recon, dict):
        raise ValueError("metamodel.reconstruction must be a dict")

    from wavespectra.construct import STATS as WAVESPECTRA_STATS

    if "variables" not in recon:
        raise ValueError("metamodel.reconstruction.variables is required")
    var_map = recon["variables"]
    if not isinstance(var_map, dict) or not var_map:
        raise ValueError(
            "metamodel.reconstruction.variables must be a non-empty dict "
            "(HyGloW name → wavespectra stat)"
        )
    unknown_stat = [s for s in var_map.values() if s not in WAVESPECTRA_STATS]
    if unknown_stat:
        raise ValueError(
            "metamodel.reconstruction.variables uses unsupported wavespectra stats: "
            f"{unknown_stat}; allowed: {WAVESPECTRA_STATS}"
        )
    missing_vars = [v for v in var_map if v not in predict_names]
    if missing_vars:
        raise ValueError(
            "metamodel.reconstruction.variables keys not in vars_to_predict: "
            f"{missing_vars}"
        )
    if "save_spectra" not in recon:
        raise ValueError("metamodel.reconstruction.save_spectra is required")
    if not isinstance(recon["save_spectra"], bool):
        raise TypeError("metamodel.reconstruction.save_spectra must be a bool")

    paths = mm.get("paths", {})
    for key in ("pca_dir", "gp_dir"):
        if key not in paths or not isinstance(paths[key], str):
            raise ValueError(f"metamodel.paths.{key} must be a template string")


def reload() -> dict[str, Any]:
    """
    Re-read the case config JSON and refresh module globals.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    global CONFIG
    CONFIG = _load()
    _refresh_sections(CONFIG)
    if _running_in_notebook():
        configure_notebook_logging()
    return CONFIG


def _normalize_log_level(level: int | str) -> int:
    if isinstance(level, str):
        return logging.getLevelName(level.upper())
    return int(level)


def _is_console_handler(handler: logging.Handler) -> bool:
    return isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, logging.FileHandler
    )


def configure_notebook_logging(
    *,
    console_level: int | str = "WARNING",
    file_level: int | str | None = None,
) -> None:
    """
    Show WARNING+ in the notebook; keep INFO in ``logs/*.log`` files.

    Applies to HyGloW loggers (``hyglow.metamodel``, ``hyglow.validation``),
    ``bluemath_tk`` loggers, and any root StreamHandler attached by Jupyter.
    Called automatically when :mod:`utils.config` is imported inside a notebook.
    """
    console_lv = _normalize_log_level(console_level)
    file_lv = _normalize_log_level(file_level) if file_level is not None else None

    # Re-apply HyGloW loggers so bluemath get_file_logger updates handlers
    # (fixes stale console_level when the logger was created earlier in the kernel).
    from bluemath_tk.core.logging import get_file_logger

    get_file_logger(
        "hyglow.metamodel",
        logs_path=str(resolve("logs/metamodel.log")),
        level="INFO",
        console=True,
        console_level=console_lv,
    )
    get_file_logger(
        "hyglow.validation",
        logs_path=str(resolve("logs/validation.log")),
        level="INFO",
        console=True,
        console_level=console_lv,
    )

    for name, obj in logging.Logger.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue
        if name not in _HYGLOW_LOGGERS and not any(
            name.startswith(prefix) for prefix in _QUIET_LOGGER_PREFIXES
        ):
            continue
        for handler in obj.handlers:
            if _is_console_handler(handler):
                handler.setLevel(console_lv)
            elif file_lv is not None and isinstance(handler, logging.FileHandler):
                handler.setLevel(file_lv)

    root = logging.getLogger()
    for handler in root.handlers:
        if _is_console_handler(handler) and handler.stream in (sys.stdout, sys.stderr):
            handler.setLevel(console_lv)


def _running_in_notebook() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return "IPKernelApp" in getattr(ip, "config", {})
    except ImportError:
        return False


if _running_in_notebook():
    configure_notebook_logging()


__all__ = [
    "CASE_CONFIG_PATH",
    "CASE_DIR",
    "CONFIG",
    "METAMODEL",
    "OFFSHORE_WAVES",
    "ONSHORE_WAVES",
    "OUTPUT_SITES",
    "OUTPUT_SITES_CONFIG",
    "PATHS",
    "SIMULATIONS",
    "case_description",
    "case_name",
    "configure_notebook_logging",
    "display_title",
    "forcing_columns",
    "get_offshore_wave",
    "goals_for_snapwave",
    "load_goals",
    "mda_max_points",
    "offshore_catalog_key",
    "open_offshore_waves",
    "reload",
    "resolve",
    "validate_metamodel_config",
]
