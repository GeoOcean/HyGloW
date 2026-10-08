"""Load the case config JSON and expose its sections, path resolution, and helpers.

Reads ``<CaseDir>/<CaseDir>.json`` once at import time into module-level
globals (:data:`PATHS`, :data:`OFFSHORE_WAVES`, :data:`ONSHORE_WAVES`,
:data:`OUTPUT_SITES`, :data:`SIMULATIONS`, :data:`OUTPUT_SITES_CONFIG`,
:data:`METAMODEL`); call :func:`reload` to re-read after editing the file.
Also provides case-relative path resolution (:func:`resolve`), goals-file
and offshore-waves accessors, metamodel config validation
(:func:`validate_metamodel_config`), and notebook logging setup
(:func:`configure_notebook_logging`, applied automatically on import when
running inside a Jupyter kernel).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import xarray as xr

_HYWAVES_LOGGERS = ("hywaves.metamodel", "hywaves.validation")
_HYWAVES_LOGGER_PREFIX = "hywaves."
_BLUEMATH_LOGGER_PREFIXES = ("bluemath", "BlueMath")

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
        clockwise ``[left, right]`` (see :mod:`bluemath_tk.core.operations`).
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
                f"metamodel.site_filter key {key!r} must end with '_max', "
                "'_ratio_max' or '_min'"
            )

    sets = mm.get("sets")
    if not isinstance(sets, dict) or not sets:
        raise ValueError(
            "metamodel.sets must be a non-empty dict "
            '(name → {"sites": selector}), e.g. {"all": {"sites": null}}'
        )
    for name, spec in sets.items():
        if not isinstance(spec, dict) or "sites" not in spec:
            raise ValueError(f"metamodel.sets[{name!r}] must be a dict with 'sites'")
        selector = spec["sites"]
        if selector is not None and not isinstance(selector, (str, list)):
            raise TypeError(
                f"metamodel.sets[{name!r}].sites must be null, a string or a list"
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
        if "{metamodel}" not in paths[key]:
            raise ValueError(
                f"metamodel.paths.{key} must contain '{{metamodel}}' (the set name)"
            )


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
    """Convert a level name (e.g. ``"WARNING"``) or int to a numeric logging level."""
    if isinstance(level, str):
        return logging.getLevelName(level.upper())
    return int(level)


def _is_console_handler(handler: logging.Handler) -> bool:
    """Return ``True`` for a stream handler that is not a file handler."""
    return isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, logging.FileHandler
    )


def _bluemath_model_logger_names() -> set[str]:
    """Every ``BlueMathModel`` subclass currently imported (PCA, MDA, RBF, ...).

    ``BlueMathModel.logger`` names its logger after the class itself
    (``get_file_logger(name=self.__class__.__name__)``), not with a
    ``"bluemath_tk...."`` prefix, so these loggers can't be matched by
    :data:`_BLUEMATH_LOGGER_PREFIXES`. Used as a one-off safety net for
    loggers created before :func:`_quiet_bluemath_model_console` was installed.
    """
    try:
        from bluemath_tk.core.models import BlueMathModel
    except ImportError:
        return set()

    names: set[str] = set()
    stack = [BlueMathModel]
    while stack:
        cls = stack.pop()
        names.add(cls.__name__)
        stack.extend(cls.__subclasses__())
    return names


def _quiet_bluemath_model_console(console_level: int) -> None:
    """Default *future* bluemath_tk model loggers (PCA, MDA, RBF, GP, ...) to a quiet console.

    Two separate leaks, both from the same root cause -- ``BlueMathModel``
    loggers are never durably configured once and left alone:

    1. **Fresh instantiation.** Every ``BlueMathModel.__init__`` calls
       ``self.set_logger_name(...)``, which calls bluemath_tk's
       ``get_file_logger`` without a ``console_level`` — so it always resets
       to bluemath_tk's own default (``WARNING``) *on every single call*.
       A loop that fits a PCA/GP per goal re-triggers the noisy default
       every time; a one-off level change doesn't survive the next fit.
       Fixed by patching the factory itself (``get_file_logger`` in
       :mod:`bluemath_tk.core.models`, the binding ``BlueMathModel`` calls)
       to default ``console_level`` to *console_level* unless the caller
       passes one explicitly.

    2. **Models loaded from disk** (``load_model`` / ``load_models`` —
       what a reconstruction/validation notebook actually does most of the
       time, as opposed to fitting). ``logging.Logger`` pickles by *name
       reference* (``Logger.__reduce__`` → ``logging.getLogger(name)``), not
       by value — so an unpickled model's ``_logger`` is whatever the
       process's logger for that name currently is. If nothing has called
       ``get_file_logger("PCA", ...)`` yet in this kernel, that's a bare
       logger with **no handlers and ``propagate=True``**, so its records
       (including routine WARNINGs) fall through straight to the root
       logger — which Jupyter/IPython usually prints. This bypasses fix #1
       entirely, since ``BlueMathModel.logger`` only calls ``get_file_logger``
       when ``self._logger is None``, and an unpickled logger reference is
       never ``None``. Fixed by also patching the ``.logger`` property to
       repair a handler-less/propagating logger the moment anything reads it.

    File logging is untouched in both cases — full detail still lands in
    ``logs/*.log``. Idempotent: safe to call every time
    :func:`configure_notebook_logging` runs.
    """
    import bluemath_tk.core.models as bluemath_models

    if getattr(bluemath_models.BlueMathModel, "_hywaves_quieted", False):
        return

    original_get_file_logger = bluemath_models.get_file_logger

    def quiet_get_file_logger(name, *args, **kwargs):
        kwargs.setdefault("console_level", console_level)
        return original_get_file_logger(name, *args, **kwargs)

    bluemath_models.get_file_logger = quiet_get_file_logger

    original_logger_fget = bluemath_models.BlueMathModel.logger.fget

    def quiet_logger_fget(self):
        lg = original_logger_fget(self)
        if not lg.handlers or lg.propagate:
            lg = quiet_get_file_logger(lg.name, level="INFO", console=True)
            self._logger = lg
        return lg

    bluemath_models.BlueMathModel.logger = property(
        quiet_logger_fget, bluemath_models.BlueMathModel.logger.fset
    )
    bluemath_models.BlueMathModel._hywaves_quieted = True


def configure_notebook_logging(
    *,
    console_level: int | str = "WARNING",
    bluemath_console_level: int | str = "ERROR",
    file_level: int | str | None = None,
) -> None:
    """
    Show WARNING+ in the notebook; keep INFO in ``logs/*.log`` files.

    Applies to HyGloW loggers (``hywaves.metamodel``, ``hywaves.validation``)
    at *console_level*, and to bluemath_tk model loggers (``PCA``, ``MDA``,
    ``RBF``, GP interpolation, ...) at the stricter *bluemath_console_level*
    — their routine "Using PCA"-style banners are logged at ``WARNING`` on
    every fit, which floods the notebook when fitting per goal/candidate in
    a loop. File logging (``logs/*.log``) always keeps the full ``INFO``
    detail for both. Called automatically when :mod:`utils.config` is
    imported inside a notebook.
    """
    console_lv = _normalize_log_level(console_level)
    bluemath_lv = _normalize_log_level(bluemath_console_level)
    file_lv = _normalize_log_level(file_level) if file_level is not None else None

    # Re-apply HyGloW loggers so bluemath get_file_logger updates handlers
    # (fixes stale console_level when the logger was created earlier in the kernel).
    from bluemath_tk.core.logging import get_file_logger

    get_file_logger(
        "hywaves.metamodel",
        logs_path=str(resolve("logs/metamodel.log")),
        level="INFO",
        console=True,
        console_level=console_lv,
    )
    get_file_logger(
        "hywaves.validation",
        logs_path=str(resolve("logs/validation.log")),
        level="INFO",
        console=True,
        console_level=console_lv,
    )

    # Future PCA()/MDA()/RBF()/... instantiations default to a quiet console.
    _quiet_bluemath_model_console(bluemath_lv)

    # Safety net for bluemath model loggers already created this kernel
    # (e.g. a model built before utils.config was first imported).
    bluemath_model_names = _bluemath_model_logger_names()

    for name, obj in logging.Logger.manager.loggerDict.items():
        if not isinstance(obj, logging.Logger):
            continue
        if name in _HYWAVES_LOGGERS or name.startswith(_HYWAVES_LOGGER_PREFIX):
            level_for_console = console_lv
        elif name in bluemath_model_names or name.startswith(_BLUEMATH_LOGGER_PREFIXES):
            level_for_console = bluemath_lv
        else:
            continue
        for handler in obj.handlers:
            if _is_console_handler(handler):
                handler.setLevel(level_for_console)
            elif file_lv is not None and isinstance(handler, logging.FileHandler):
                handler.setLevel(file_lv)

    root = logging.getLogger()
    for handler in root.handlers:
        if _is_console_handler(handler) and handler.stream in (sys.stdout, sys.stderr):
            handler.setLevel(console_lv)


def _running_in_notebook() -> bool:
    """Return ``True`` when executing inside a Jupyter/IPython kernel."""
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
