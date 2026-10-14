"""Numerical cases of a HyGloW case: LHS sampling and the wrapper setup.

The cases run with SnapWave or SWAN (``simulations.model``); their wrappers
(case building, templates rendering, SLURM arrays, postprocessing) live in
:mod:`bluemath_tk.wrappers.snapwave` and :mod:`bluemath_tk.wrappers.swan`.
Both write the same per-goal output (``output_sites_{goal}.nc`` on
``(case_num, sites)``), so everything downstream is model-agnostic. This
module turns the case JSON into their inputs:

- :func:`generate_cases` — Latin Hypercube per goal within its sector
  (``simulations.bounds``, ``num_samples_per_goal``), keeping plausible sea
  states (``simulations.max_steepness``);
- :func:`templates_dir`, :func:`cases_dir`, :func:`grid_file`,
  :func:`slurm_kwargs` — the ``simulations.<model>`` block;
- :class:`HyWavesWrapperSetup` — goals as boundary nodes (goal id = 1-based
  ``active_node``), the model boundary (SnapWave's AoI enclosure or SWAN's
  open boundary from the grid), the output sites labelled with the case site
  metadata (``sites``, ``site_source``, ...), and the model's wrappers.

Typical use (``05_Metamodel_Cases_Generation.ipynb``)::

    setup = sim.HyWavesWrapperSetup.from_case(goals)
    wrapper = setup.metamodel_wrapper(sim.generate_cases(goals))
    wrapper.build_cases(...)
    wrapper.run_cases_slurm(**sim.slurm_kwargs())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.waves.steepness import filter_by_steepness
from bluemath_tk.wrappers.snapwave import (
    BOUNDARY_VARS,
    SnapWaveDynamicModelWrapper,
    SnapWaveMetaModelWrapper,
    boundary_time_series,
    build_enclosure_polygon,
)
from bluemath_tk.wrappers.swan.swan_utils import read_adcirc_grid, swan_boundary_side
from bluemath_tk.wrappers.swan.swan_wrapper import (
    SwanDynamicModelWrapper,
    SwanMetaModelWrapper,
)
from matplotlib.figure import Figure
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from utils import viz
from utils.config import (
    PATHS,
    SIMULATIONS,
    goals_for_snapwave,
    load_goals,
    model_config,
    open_water_level,
    resolve,
    simulation_model,
)
from utils.geometry import min_depth, output_sites

#: Metamodel LHS parameters, stored as ``{name}_forcing`` on postprocessed output.
FORCING_VARS = ("hs", "tp", "dir", "spr", "wl")

#: Wrapper classes per model and kind of case.
WRAPPERS = {
    "snapwave": {
        "metamodel": SnapWaveMetaModelWrapper,
        "dynamic": SnapWaveDynamicModelWrapper,
    },
    "swan": {"metamodel": SwanMetaModelWrapper, "dynamic": SwanDynamicModelWrapper},
}

CASE_KINDS = ("metamodel", "dynamic")


def generate_cases(goals: Mapping[str, Mapping]) -> pd.DataFrame:
    """Generate the metamodel cases of each goal: LHS, then plausible sea states.

    ``num_samples_per_goal`` Latin Hypercube samples of ``hs``, ``tp``,
    ``dir``, ``spr`` and ``wl`` within ``simulations.bounds`` are drawn per
    goal; samples steeper than ``simulations.max_steepness`` (``hs / L0``, see
    :func:`bluemath_tk.waves.steepness.filter_by_steepness`) are dropped, so
    each goal keeps fewer cases than requested.

    Offshore wave direction ``dir`` is sampled in nautical degrees (wave-from,
    clockwise from North) within each goal's ``angles`` sector. Bounds are
    stored as ``[left, right]`` with ``left < right``; negative ``left`` values
    denote sectors that cross North (see :mod:`bluemath_tk.core.operations`).

    Parameters
    ----------
    goals
        Goals mapping from :func:`utils.config.load_goals`.

    Returns
    -------
    pandas.DataFrame
        Columns ``goal``, ``active_node``, ``hs``, ``tp``, ``dir``, ``spr``,
        ``wl``, one row per kept sample. ``active_node`` (= goal id) is the
        1-based boundary node where ``hs`` is applied. ``dir`` is in
        ``[0, 360)`` after normalization.
    """
    bounds = SIMULATIONS["bounds"]
    n = SIMULATIONS["num_samples_per_goal"]
    names = list(FORCING_VARS)

    frames: list[pd.DataFrame] = []
    for _goal_id, goal in goals.items():
        sampled = {**bounds, "dir": goal["angles"]}
        lhs_df = LHS(num_dimensions=len(names)).generate(
            dimensions_names=names,
            lower_bounds=[sampled[v][0] for v in names],
            upper_bounds=[sampled[v][1] for v in names],
            num_samples=n,
        )
        lhs_df["dir"] = np.mod(lhs_df["dir"], 360.0)
        lhs_df = filter_by_steepness(lhs_df, SIMULATIONS["max_steepness"])
        frames.append(lhs_df.reset_index(drop=True))

    cases = (
        pd.concat(frames, keys=list(goals.keys()))
        .reset_index()
        .rename(columns={"level_0": "goal"})
        .drop(columns=["level_1"])
    )
    cases.insert(1, "active_node", cases["goal"].astype(int))
    return cases


#: HyGloW site coordinate -> output-sites table column.
SITE_COORDS = {
    "sites": "source_id",
    "site_source": "source",
    "site_label": "source_label",
    "site_mesh_depth": "mesh_depth",
    "site_nominal_depth": "nominal_depth",
    "site_longitude": "lon",
    "site_latitude": "lat",
}

#: Output-points file written into every case (SnapWave ``obsfile``, SWAN ``POINTS``).
SITES_FILENAME = "output_sites.txt"

DEFAULT_ENCLOSURE_BUFFER_M = 10_000.0


def _check_kind(kind: str) -> str:
    if kind not in CASE_KINDS:
        raise ValueError(f"kind must be one of {CASE_KINDS}, got {kind!r}")
    return kind


def templates_dir(kind: str, model: str | None = None) -> str:
    """
    Templates folder of the model for ``"metamodel"`` or ``"dynamic"`` cases.

    From ``simulations.<model>.metamodel_templates`` / ``dynamic_templates``.
    SnapWave templates receive the context of
    :class:`bluemath_tk.wrappers.snapwave.SnapWaveModelWrapper` (``n_nodes``,
    ``active_node``, ``hs_nodes``, ...); SWAN's ``INPUT`` the context of
    :class:`bluemath_tk.wrappers.swan.swan_wrapper.SwanBoundaryModelWrapper`
    (``boundary_side``, ``boundary_par``, ``points_file``, ...).

    Parameters
    ----------
    kind
        ``"metamodel"`` or ``"dynamic"``.
    model
        Model name; default ``simulations.model``.

    Returns
    -------
    str
        Templates folder.
    """
    return str(resolve(model_config(model)[f"{_check_kind(kind)}_templates"]))


def cases_dir(kind: str, model: str | None = None) -> Path:
    """
    Cases folder of the model (``simulations.<model>.metamodel_cases`` / ``dynamic_cases``).

    Parameters
    ----------
    kind
        ``"metamodel"`` or ``"dynamic"``.
    model
        Model name; default ``simulations.model``.

    Returns
    -------
    Path
        Cases folder.
    """
    return resolve(model_config(model)[f"{_check_kind(kind)}_cases"])


def grid_file(model: str | None = None) -> Path:
    """
    Computational grid of the model (``simulations.<model>.grid_file``).

    SnapWave reads a UGRID NetCDF mesh (``gridfile``); SWAN an ADCIRC grid
    with counterclockwise triangles, linked as ``fort.14`` into each case.

    Parameters
    ----------
    model
        Model name; default ``simulations.model``.

    Returns
    -------
    Path
        Grid file.
    """
    return resolve(model_config(model)["grid_file"])


def slurm_kwargs(model: str | None = None, **overrides: Any) -> dict[str, Any]:
    """
    Keyword arguments for the wrapper's ``run_cases_slurm`` / ``write_slurm_array``.

    From ``simulations.<model>.slurm`` (e.g. ``cpus_per_task`` and ``mem`` for
    SnapWave, ``mem`` and ``extra_sbatch`` for SWAN, ``setup`` for the module
    lines), updated with *overrides*.

    Parameters
    ----------
    model
        Model name; default ``simulations.model``.
    **overrides
        Values replacing the configured ones (e.g. ``max_parallel=200``).

    Returns
    -------
    dict
        Keyword arguments.
    """
    return {**model_config(model).get("slurm", {}), **overrides}


def _forcing_name(kind: str) -> str | None:
    """``forcing.<kind>.name`` of the case JSON, if any."""
    from utils.config import FORCING

    return (FORCING.get(kind) or {}).get("name")


def dynamic_parameters(
    offshore: xr.Dataset, *, water_level: bool = True
) -> dict[str, list]:
    """
    Per-time-step boundary forcing of every goal, for the dynamic wrappers.

    Parameters
    ----------
    offshore
        Offshore hindcast at the goals on ``(time, goal)`` (``hs``, ``tp``,
        ``dir``, ``spr``), already selected on the period.
    water_level
        Add ``wl_nodes`` from ``forcing.water_level`` (0 where it has no
        value). False leaves the templates' water level at 0.

    Returns
    -------
    dict[str, list]
        ``tref`` and ``{var}_nodes`` lists (see
        :func:`bluemath_tk.wrappers.snapwave.boundary_time_series`).
    """
    forcing = offshore[list(BOUNDARY_VARS)]
    variables = BOUNDARY_VARS
    if water_level:
        wl = open_water_level().sel(goal=offshore["goal"].values)
        wl = wl.reindex(time=offshore["time"].values).load()
        missing = int(wl.isnull().any("goal").sum())
        if missing:
            logging.getLogger("hywaves.forcing").warning(
                "no water level for %d of %d time steps (set to 0)",
                missing,
                wl.sizes["time"],
            )
        forcing = forcing.assign(wl=wl.fillna(0.0).drop_vars(
            [c for c in wl.coords if c not in ("time", "goal")]
        ))
        variables = (*BOUNDARY_VARS, "wl")
    return boundary_time_series(forcing, node_dim="goal", variables=variables)


@dataclass
class HyWavesWrapperSetup:
    """Case geometry for the numerical cases: AoI, enclosure, goal nodes, output sites.

    The same geometry serves SnapWave and SWAN: goals are the boundary nodes
    (in goal-id order), the output sites the observation points. SnapWave
    also needs the enclosure (which mesh edge nodes are open boundary); SWAN
    finds its open boundary in the grid file.
    """

    aoi_gdf: gpd.GeoDataFrame
    enclosure_polygon: Polygon
    enclosure_buffer_m: float
    goals_config: dict[str, dict[str, Any]]
    output_sites_df: pd.DataFrame
    model: str = "snapwave"
    mesh_file: str | None = None

    @classmethod
    def from_case(
        cls,
        goals: dict[str, dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        enclosure_buffer_m: float | None = None,
        output_sites_df: pd.DataFrame | None = None,
    ) -> HyWavesWrapperSetup:
        """
        Build the setup from the case config (AoI, goals, mesh, output sites).

        Parameters
        ----------
        goals
            Goals mapping (default :func:`utils.config.load_goals`).
        model
            ``"snapwave"`` or ``"swan"`` (default ``simulations.model``).
        enclosure_buffer_m
            Outward AoI buffer in metres (default
            ``simulations.enclosure_buffer_m``, else 10 km).
        output_sites_df
            Output sites (default the saved table of the model, see
            :func:`utils.geometry.output_sites`).

        Returns
        -------
        HyWavesWrapperSetup
        """
        goals_config = goals_for_snapwave(goals or load_goals())
        ids = [int(k) for k in goals_config]
        if ids != list(range(1, len(ids) + 1)):
            raise ValueError(
                f"Goal ids must be 1..N in order (boundary node = goal); got {ids}"
            )
        if enclosure_buffer_m is None:
            enclosure_buffer_m = float(
                SIMULATIONS.get("enclosure_buffer_m", DEFAULT_ENCLOSURE_BUFFER_M)
            )
        aoi_gdf = gpd.read_file(resolve(PATHS["aoi_file"]))
        aoi_polygon = aoi_gdf.geometry.iloc[0]
        model = simulation_model(model)
        if output_sites_df is None:
            output_sites_df = output_sites(model)
        mesh = PATHS.get("mesh_file")
        return cls(
            aoi_gdf=aoi_gdf,
            enclosure_polygon=build_enclosure_polygon(aoi_polygon, enclosure_buffer_m),
            enclosure_buffer_m=float(enclosure_buffer_m),
            goals_config=goals_config,
            output_sites_df=output_sites_df.reset_index(drop=True),
            model=model,
            mesh_file=str(resolve(mesh)) if mesh else None,
        )

    @cached_property
    def model_boundary(self) -> tuple[BaseGeometry, str]:
        """Where the model is forced, and its label for the configuration maps.

        SnapWave: the enclosure polygon (mesh edge nodes inside it are open
        boundary). SWAN: the open boundary of the grid file, as SWAN walks it
        (``BOUNDSPEC SIDE``); the enclosure plays no part.
        """
        if self.model == "swan":
            side, _ = swan_boundary_side(read_adcirc_grid(str(grid_file(self.model))))
            return LineString(side), "SWAN open boundary (grid)"
        return self.enclosure_polygon, "SnapWave enclosure"

    @property
    def boundary_nodes(self) -> list[tuple[float, float]]:
        """Goal ``(lon, lat)`` in goal-id order — the boundary nodes."""
        return [(g["lon"], g["lat"]) for g in self.goals_config.values()]

    def wrapper_kwargs(self) -> dict[str, Any]:
        """Geometry and output-labelling arguments for the model's wrappers."""
        common = {
            "boundary_nodes": self.boundary_nodes,
            "output_points": self.output_sites_df,
            "point_dim": "sites",
            "point_coords": SITE_COORDS,
        }
        if self.model == "swan":
            return {
                **common,
                "grid_file": str(grid_file(self.model)),
                "points_filename": SITES_FILENAME,
                "table_filename": str(Path(SITES_FILENAME).with_suffix(".tab")),
            }
        return {
            **common,
            "enclosure_polygon": self.enclosure_polygon,
            "obs_filename": SITES_FILENAME,
            "his_filename": str(Path(SITES_FILENAME).with_suffix(".nc")),
        }

    def metamodel_kwargs(self) -> dict[str, Any]:
        """:meth:`wrapper_kwargs` plus metamodel output: forcing stored, one file per goal.

        Joined output is written as ``output_sites_{goal}.nc`` (read by
        :func:`utils.metamodel.cases_path`).
        """
        return {
            **self.wrapper_kwargs(),
            "store_parameters": FORCING_VARS,
            "split_by": "goal",
            "split_filename": Path(SITES_FILENAME).stem + "_{}.nc",
        }

    def fixed_parameters(self) -> dict[str, Any]:
        """Parameters shared by every case: ``min_depth`` and SnapWave's ``gridfile`` (SWAN links its grid)."""
        fixed: dict[str, Any] = {"min_depth": min_depth()}
        if self.model == "snapwave":
            fixed["gridfile"] = str(grid_file(self.model))
        return fixed

    def _wrapper(
        self,
        kind: str,
        parameters: pd.DataFrame | Mapping[str, list],
        output_dir: str | Path | None,
        templates: str | Path | None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.to_dict(orient="list")
        parameters = dict(parameters)
        geometry = (
            self.metamodel_kwargs() if kind == "metamodel" else self.wrapper_kwargs()
        )
        return WRAPPERS[self.model][kind](
            templates_dir=str(templates or templates_dir(kind, self.model)),
            metamodel_parameters=parameters,
            fixed_parameters=self.fixed_parameters(),
            output_dir=str(output_dir or cases_dir(kind, self.model)),
            **{**geometry, **kwargs},
        )

    def metamodel_wrapper(
        self,
        cases: pd.DataFrame | Mapping[str, list],
        *,
        output_dir: str | Path | None = None,
        templates: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        The model's metamodel wrapper for *cases* (e.g. :func:`generate_cases`).

        Parameters
        ----------
        cases
            One row (or list entry) per case: ``goal``, ``active_node``, ``hs``,
            ``tp``, ``dir``, ``spr``, ``wl`` (and optional ``save_map``: grid map output).
        output_dir, templates
            Override ``simulations.<model>.metamodel_cases`` / ``metamodel_templates``.
        **kwargs
            Further wrapper arguments (e.g. ``debug=False``).

        Returns
        -------
        SnapWaveMetaModelWrapper or SwanMetaModelWrapper
        """
        return self._wrapper("metamodel", cases, output_dir, templates, **kwargs)

    def dynamic_wrapper(
        self,
        parameters: pd.DataFrame | Mapping[str, list],
        *,
        output_dir: str | Path | None = None,
        templates: str | Path | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        The model's dynamic wrapper: one case per time step, every goal forced.

        Parameters
        ----------
        parameters
            Per-case ``tref`` and per-node ``hs_nodes``, ``tp_nodes``, ... (see
            :func:`dynamic_parameters`); optional uniform ``u10`` / ``u10dir``.
        output_dir, templates
            Override ``simulations.<model>.dynamic_cases`` / ``dynamic_templates``.
        **kwargs
            Further wrapper arguments, e.g. ``wind=open_wind(period)`` (see
            :func:`utils.config.open_wind`): the field at each case time.

        Returns
        -------
        SnapWaveDynamicModelWrapper or SwanDynamicModelWrapper
        """
        return self._wrapper("dynamic", parameters, output_dir, templates, **kwargs)

    # ------------------------------------------------------------------
    # Pre-run diagnostics (HyGloW goal/sector plots, utils.viz)
    # ------------------------------------------------------------------

    def _plot_common(self, **kwargs: Any) -> dict[str, Any]:
        boundary, label = self.model_boundary
        return {
            "model_boundary": boundary,
            "model_boundary_label": label,
            "goals_config": self.goals_config,
            "output_sites_df": self.output_sites_df,
            **kwargs,
        }

    @property
    def model_enclosure_buffer_m(self) -> float | None:
        """The enclosure buffer (m) when the model uses the enclosure (SnapWave), else None."""
        return self.enclosure_buffer_m if self.model == "snapwave" else None

    def plot_metamodel_configuration(
        self, cases_df: pd.DataFrame, **kwargs: Any
    ) -> Figure:
        """Metamodel pre-run diagnostic (see :func:`utils.viz.plot_metamodel_configuration`)."""
        return viz.plot_metamodel_configuration(
            **self._plot_common(
                aoi_gdf=self.aoi_gdf,
                cases_df=cases_df,
                bounds=SIMULATIONS.get("bounds"),
                num_samples_per_goal=SIMULATIONS.get("num_samples_per_goal"),
                max_steepness=SIMULATIONS.get("max_steepness"),
                mesh_file=str(grid_file(self.model)),
                enclosure_buffer_m=self.model_enclosure_buffer_m,
                title=f"Metamodel pre-run diagnostic ({self.model})",
                **kwargs,
            )
        )

    def plot_metamodel_configuration_map(
        self, cases_df: pd.DataFrame, **kwargs: Any
    ) -> Figure:
        """Large map-only view (see :func:`utils.viz.plot_metamodel_configuration_map`)."""
        kwargs.setdefault("title", "Metamodel configuration map")
        return viz.plot_metamodel_configuration_map(
            **self._plot_common(cases_df=cases_df, **kwargs)
        )

    def plot_dynamic_configuration(
        self,
        offshore: xr.Dataset,
        *,
        offshore_name: str | None = None,
        save_map: list[bool] | None = None,
        wind: xr.Dataset | None = None,
        water_level: xr.DataArray | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Dynamic pre-run diagnostic (see :func:`utils.viz.plot_dynamic_configuration`).

        *wind* and *water_level* (as given to the cases) add their coverage
        and ranges to the summary; see :meth:`plot_dynamic_forcing` for maps.
        """
        return viz.plot_dynamic_configuration(
            **self._plot_common(
                aoi_gdf=self.aoi_gdf,
                offshore=offshore,
                offshore_name=offshore_name,
                mesh_file=str(grid_file(self.model)),
                enclosure_buffer_m=self.model_enclosure_buffer_m,
                title=f"Dynamic pre-run diagnostic ({self.model})",
                maps_to_save=int(sum(save_map)) if save_map else None,
                wind=wind,
                water_level=water_level,
                wind_name=_forcing_name("wind"),
                water_level_name=_forcing_name("water_level"),
                **kwargs,
            )
        )

    def plot_dynamic_forcing(
        self,
        offshore: xr.Dataset,
        *,
        wind: xr.Dataset | None = None,
        water_level: xr.DataArray | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Wind and water level of the dynamic cases (see :func:`utils.viz.plot_dynamic_forcing`).

        The wind-wave angle panel shades the wind correction's training range
        (``metamodel.wind_correction.wind_wave_angle``).
        """
        from utils.wind import config as wind_config

        return viz.plot_dynamic_forcing(
            aoi_gdf=self.aoi_gdf,
            goals_config=self.goals_config,
            offshore=offshore,
            wind=wind,
            water_level=water_level,
            angle_range=tuple(wind_config()["wind_wave_angle"]),
            wind_name=_forcing_name("wind"),
            water_level_name=_forcing_name("water_level"),
            title=f"Dynamic met-ocean forcing ({self.model})",
            **kwargs,
        )

    def plot_dynamic_configuration_map(
        self, offshore: xr.Dataset, **kwargs: Any
    ) -> Figure:
        """Large map-only view (see :func:`utils.viz.plot_dynamic_configuration_map`)."""
        kwargs.setdefault("title", "Dynamic configuration map")
        return viz.plot_dynamic_configuration_map(
            **self._plot_common(offshore=offshore, **kwargs)
        )


__all__ = [
    "CASE_KINDS",
    "DEFAULT_ENCLOSURE_BUFFER_M",
    "FORCING_VARS",
    "HyWavesWrapperSetup",
    "SITES_FILENAME",
    "SITE_COORDS",
    "WRAPPERS",
    "cases_dir",
    "dynamic_parameters",
    "generate_cases",
    "grid_file",
    "slurm_kwargs",
    "templates_dir",
]
