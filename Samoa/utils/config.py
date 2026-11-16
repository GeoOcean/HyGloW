"""Load the case config JSON and expose its sections, path resolution, and helpers.

Reads ``<CaseDir>/<CaseDir>.json`` once at import time into module-level
constants (:data:`PATHS`, :data:`OFFSHORE_WAVES`, :data:`ONSHORE_WAVES`,
:data:`SIMULATIONS`, :data:`OUTPUT_SITES_CONFIG`, :data:`METAMODEL`,
:data:`FORCING`, :data:`SATELLITE`); restart the kernel after editing the file.
Also provides case-relative path resolution (:func:`resolve`), goals-file
and offshore-waves accessors (the config itself is checked by
``python -m hywaves validate``), and notebook logging setup
(:func:`configure_notebook_logging`, applied automatically on import when
running inside a Jupyter kernel).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr


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


CONFIG: dict[str, Any] = _load()
PATHS: dict[str, Any] = CONFIG["paths"]
OFFSHORE_WAVES: list[dict] = CONFIG.get("offshore_waves", [])
ONSHORE_WAVES: str = CONFIG["onshore_waves"]
SIMULATIONS: dict[str, Any] = CONFIG.get("simulations", {})
OUTPUT_SITES_CONFIG: dict[str, Any] = SIMULATIONS.get("output_sites_config", {})
METAMODEL: dict[str, Any] = CONFIG.get("metamodel", {})
FORCING: dict[str, Any] = CONFIG.get("forcing", {})
SATELLITE: dict[str, Any] = CONFIG.get("satellite", {})


def case_name(default: str = "Case") -> str:
    """Display name for the current case (from config ``name``)."""
    return str(CONFIG.get("name") or default)


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


#: Numerical models a case can run its SnapWave-like cases with (``simulations.model``).
SIMULATION_MODELS = ("snapwave", "swan")


def simulation_model(model: str | None = None) -> str:
    """
    Numerical model of the case cases: *model*, else ``simulations.model``.

    Parameters
    ----------
    model
        Explicit model name (checked), or None for the case JSON choice.

    Returns
    -------
    str
        One of :data:`SIMULATION_MODELS`.

    Raises
    ------
    ValueError
        If the model is unknown or ``simulations.model`` is missing.
    """
    name = model if model is not None else SIMULATIONS.get("model")
    if name not in SIMULATION_MODELS:
        raise ValueError(
            f"simulations.model must be one of {SIMULATION_MODELS}, got {name!r}"
        )
    return name


#: Display names of the numerical models, for plot labels.
MODEL_LABELS: dict[str, str] = {"snapwave": "SnapWave", "swan": "SWAN"}


def model_label(model: str | None = None) -> str:
    """Display name of *model* (default ``simulations.model``), e.g. ``"SWAN"``."""
    return MODEL_LABELS[simulation_model(model)]


def model_config(model: str | None = None) -> dict[str, Any]:
    """
    The ``simulations.<model>`` block: grid, templates, cases folders, SLURM.

    Parameters
    ----------
    model
        Model name; default ``simulations.model``.

    Returns
    -------
    dict
        ``grid_file``, ``metamodel_templates``, ``dynamic_templates``,
        ``metamodel_cases``, ``dynamic_cases`` and optional ``slurm``.

    Raises
    ------
    KeyError
        If the case JSON has no block for the model.
    """
    name = simulation_model(model)
    if name not in SIMULATIONS:
        raise KeyError(f"simulations.{name} is not configured in the case JSON")
    return SIMULATIONS[name]


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

    The entry's ``hindcast`` (e.g. ``"WHACS"`` for a calibrated ``WHACS_cal``,
    whose series sit on the same nodes), else its ``name``.

    Parameters
    ----------
    entry
        One ``offshore_waves`` block from the case config.

    Returns
    -------
    str
        Lowercase catalog name (e.g. ``whacs``).
    """
    return str(entry.get("hindcast") or entry["name"]).casefold()


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


def _forcing_entry(kind: str) -> dict[str, Any]:
    """The ``forcing.<kind>`` block, with its ``data_path``."""
    spec = FORCING.get(kind)
    if not isinstance(spec, dict) or not spec.get("data_path"):
        raise KeyError(f"forcing.{kind}.data_path is not configured in the case JSON")
    return spec


def open_wind(time: str | slice | None = None) -> xr.Dataset:
    """
    Open the 10 m wind of ``forcing.wind`` (e.g. ERA5) over the case area.

    Parameters
    ----------
    time
        Optional time selection (cover the cases' times with a margin: the
        wrappers interpolate linearly in time).

    Returns
    -------
    xarray.Dataset
        ``u10`` and ``v10`` (m/s) on ``(time, latitude, longitude)``,
        latitude increasing.
    """
    spec = _forcing_entry("wind")
    ds = xr.open_zarr(resolve(spec["data_path"]))
    if "valid_time" in ds.dims:
        ds = ds.rename(valid_time="time")
    ds = ds[["u10", "v10"]].sortby("latitude")
    return ds if time is None else ds.sel(time=time)


def open_satellite() -> xr.Dataset:
    """
    Open the altimeter store of the ``satellite`` block (local zarr or URL).

    Returns
    -------
    xarray.Dataset
        Cell-indexed along-track observations (see
        :func:`bluemath_tk.waves.altimetry.open_altimeter`).
    """
    from bluemath_tk.waves.altimetry import open_altimeter

    path = SATELLITE.get("data_path")
    if not path:
        raise KeyError("satellite.data_path is not configured in the case JSON")
    return open_altimeter(path if "://" in str(path) else str(resolve(path)))


def open_water_level() -> xr.DataArray:
    """
    Open the water level of ``forcing.water_level`` (e.g. CODEC) at the goals.

    Returns
    -------
    xarray.DataArray
        Water level (m) on ``(time, goal)``, named ``wl``; the variable is
        ``forcing.water_level.variable`` (default ``"waterlevel"``).
    """
    spec = _forcing_entry("water_level")
    ds = xr.open_zarr(resolve(spec["data_path"]))
    return ds[spec.get("variable", "waterlevel")].rename("wl")


def goal_water_level(goal: str | int, times: Any) -> pd.Series:
    """
    Water level at one goal on *times*; 0 where the series has no value.

    Parameters
    ----------
    goal
        Goal id.
    times
        Times to return (e.g. an offshore forcing index).

    Returns
    -------
    pandas.Series
        Water level (m) indexed by *times*.
    """
    index = pd.DatetimeIndex(times)
    wl = open_water_level().sel(goal=int(goal)).reindex(time=index).load()
    series = pd.Series(wl.values, index=index, name="wl")
    missing = int(series.isna().sum())
    if missing:
        logging.getLogger("hywaves.forcing").warning(
            "Goal %s: no water level for %d of %d time steps (set to 0)",
            goal,
            missing,
            len(series),
        )
    return series.fillna(0.0)


def configure_notebook_logging(models_console_level: int | str = "ERROR") -> None:
    """
    Keep the bluemath_tk model messages (PCA, MDA, GP, ...) out of the notebook.

    They log routine messages on every fit; the notebook shows *models_console_level*
    and above, ``logs/*.log`` keep the detail. Called on import inside a notebook.
    """
    from bluemath_tk.core.models import set_models_console_level

    set_models_console_level(models_console_level)


def _running_in_notebook() -> bool:
    """Return ``True`` when executing inside a Jupyter/IPython kernel."""
    from IPython import get_ipython

    ip = get_ipython()
    return ip is not None and "IPKernelApp" in getattr(ip, "config", {})


if _running_in_notebook():
    configure_notebook_logging()


__all__ = [
    "CASE_CONFIG_PATH",
    "CASE_DIR",
    "CONFIG",
    "FORCING",
    "METAMODEL",
    "OFFSHORE_WAVES",
    "ONSHORE_WAVES",
    "OUTPUT_SITES_CONFIG",
    "PATHS",
    "SATELLITE",
    "SIMULATIONS",
    "SIMULATION_MODELS",
    "case_name",
    "configure_notebook_logging",
    "display_title",
    "forcing_columns",
    "get_offshore_wave",
    "goal_water_level",
    "goals_for_snapwave",
    "load_goals",
    "model_config",
    "model_label",
    "offshore_catalog_key",
    "open_offshore_waves",
    "open_satellite",
    "open_water_level",
    "open_wind",
    "resolve",
    "simulation_model",
]
