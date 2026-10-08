"""SnapWave cases of a HyGloW case: LHS sampling and the case setup.

The SnapWave wrappers (case building, templates rendering, launchers and
SLURM arrays, postprocessing, case plots) are :mod:`bluemath_tk.wrappers.snapwave`;
notebooks use them directly. This module turns the case JSON into their inputs:

- :func:`generate_cases` — Latin Hypercube per goal within its sector
  (``simulations.bounds``, ``num_samples_per_goal``), keeping plausible sea
  states (``simulations.max_steepness``);
- :func:`templates_dir` — ``simulations.metamodel_templates`` /
  ``dynamic_templates``;
- :class:`SnapWaveCaseSetup` — goals as boundary nodes (goal id = 1-based
  ``active_node``), the AoI enclosure, and the output sites labelled with the
  case site metadata (``sites``, ``site_source``, ...).

Typical use (``05_Metamodel_Cases_Generation.ipynb``)::

    setup = sim.SnapWaveCaseSetup.from_case(goals)
    wrapper = SnapWaveMetaModelWrapper(
        templates_dir=sim.templates_dir("metamodel"),
        metamodel_parameters=sim.generate_cases(goals).to_dict(orient="list"),
        fixed_parameters={"gridfile": str(mesh_file)},
        output_dir=str(cases_dir),
        **setup.metamodel_kwargs(),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.waves.steepness import filter_by_steepness
from bluemath_tk.wrappers.snapwave import build_enclosure_polygon
from matplotlib.figure import Figure
from shapely.geometry import Polygon

from utils import viz
from utils.config import PATHS, SIMULATIONS, goals_for_snapwave, load_goals, resolve
from utils.geometry import compute_output_sites

#: Metamodel LHS parameters, stored as ``{name}_forcing`` on postprocessed output.
FORCING_VARS = ("hs", "tp", "dir", "spr", "wl")


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
        1-based SnapWave boundary node where ``hs`` is applied. ``dir`` is in
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
    "site_depth": "depth",
    "site_longitude": "lon",
    "site_latitude": "lat",
}

#: Observation file (``obsfile``); SnapWave writes its ``.nc`` sibling.
SITES_FILENAME = "output_sites.txt"

DEFAULT_ENCLOSURE_BUFFER_M = 10_000.0


def templates_dir(kind: str) -> str:
    """
    SnapWave templates folder for ``"metamodel"`` or ``"dynamic"`` cases.

    From the case JSON: ``simulations.metamodel_templates`` /
    ``simulations.dynamic_templates`` (``templates-metamodel/`` and
    ``templates-dynamic/`` in a scaffolded case). They receive the context of
    :mod:`bluemath_tk.wrappers.snapwave`: ``n_nodes``, ``active_node``,
    ``hs_nodes``, ``tp_nodes``, ``dir_nodes``, ``spr_nodes``, ``tref``, ...

    Parameters
    ----------
    kind
        ``"metamodel"`` or ``"dynamic"``.

    Returns
    -------
    str
        Templates folder.
    """
    if kind not in ("metamodel", "dynamic"):
        raise ValueError(f"kind must be 'metamodel' or 'dynamic', got {kind!r}")
    return str(resolve(SIMULATIONS[f"{kind}_templates"]))


@dataclass
class SnapWaveCaseSetup:
    """Case geometry for SnapWave: AoI enclosure, goal nodes, output sites."""

    aoi_gdf: gpd.GeoDataFrame
    enclosure_polygon: Polygon
    enclosure_buffer_m: float
    goals_config: dict[str, dict[str, Any]]
    output_sites_df: pd.DataFrame
    mesh_file: str | None = None

    @classmethod
    def from_case(
        cls,
        goals: dict[str, dict[str, Any]] | None = None,
        *,
        enclosure_buffer_m: float | None = None,
        output_sites_df: pd.DataFrame | None = None,
    ) -> SnapWaveCaseSetup:
        """
        Build the setup from the case config (AoI, goals, mesh, output sites).

        Parameters
        ----------
        goals
            Goals mapping (default :func:`utils.config.load_goals`).
        enclosure_buffer_m
            Outward AoI buffer in metres (default
            ``simulations.enclosure_buffer_m``, else 10 km).
        output_sites_df
            Pre-built output sites (default computed from the AoI, see
            :func:`utils.geometry.compute_output_sites`).

        Returns
        -------
        SnapWaveCaseSetup
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
        if output_sites_df is None:
            output_sites_df = compute_output_sites(aoi_polygon=aoi_polygon)
        mesh = PATHS.get("mesh_file")
        return cls(
            aoi_gdf=aoi_gdf,
            enclosure_polygon=build_enclosure_polygon(aoi_polygon, enclosure_buffer_m),
            enclosure_buffer_m=float(enclosure_buffer_m),
            goals_config=goals_config,
            output_sites_df=output_sites_df.reset_index(drop=True),
            mesh_file=str(resolve(mesh)) if mesh else None,
        )

    @property
    def boundary_nodes(self) -> list[tuple[float, float]]:
        """Goal ``(lon, lat)`` in goal-id order — the SnapWave boundary nodes."""
        return [(g["lon"], g["lat"]) for g in self.goals_config.values()]

    def wrapper_kwargs(self) -> dict[str, Any]:
        """Geometry and output-labelling arguments for any bluemath SnapWave wrapper."""
        return {
            "boundary_nodes": self.boundary_nodes,
            "enclosure_polygon": self.enclosure_polygon,
            "output_points": self.output_sites_df,
            "obs_filename": SITES_FILENAME,
            "his_filename": str(Path(SITES_FILENAME).with_suffix(".nc")),
            "point_dim": "sites",
            "point_coords": SITE_COORDS,
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

    # ------------------------------------------------------------------
    # Pre-run diagnostics (HyGloW goal/sector plots, utils.viz)
    # ------------------------------------------------------------------

    def _plot_common(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "enclosure_polygon": self.enclosure_polygon,
            "goals_config": self.goals_config,
            "output_sites_df": self.output_sites_df,
            **kwargs,
        }

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
                mesh_file=self.mesh_file,
                enclosure_buffer_m=self.enclosure_buffer_m,
                title="Metamodel pre-run diagnostic",
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
        **kwargs: Any,
    ) -> Figure:
        """Dynamic pre-run diagnostic (see :func:`utils.viz.plot_dynamic_configuration`)."""
        return viz.plot_dynamic_configuration(
            **self._plot_common(
                aoi_gdf=self.aoi_gdf,
                offshore=offshore,
                offshore_name=offshore_name,
                mesh_file=self.mesh_file,
                enclosure_buffer_m=self.enclosure_buffer_m,
                title="Dynamic pre-run diagnostic",
                maps_to_save=int(sum(save_map)) if save_map else None,
                **kwargs,
            )
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
    "DEFAULT_ENCLOSURE_BUFFER_M",
    "FORCING_VARS",
    "SITES_FILENAME",
    "SITE_COORDS",
    "SnapWaveCaseSetup",
    "generate_cases",
    "templates_dir",
]
