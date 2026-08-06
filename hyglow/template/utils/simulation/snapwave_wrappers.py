"""SnapWave case wrappers for a HyGloW case.

:class:`SnapWaveMetaModelWrapper` — LHS propagation cases (``case_num`` dim,
one NetCDF per goal).

:class:`SnapWaveDynamicModelWrapper` — offshore-waves forcing (``time`` dim,
concatenated time series).

Both produce site-labelled outputs ``hs``, ``tp``, ``dir`` (and ``spr`` when
present in SnapWave postprocess) on ``sites``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper
from bluemath_tk.wrappers._utils_wrappers import write_array_in_file
from shapely.geometry import MultiPolygon, Polygon

from utils.config import SIMULATIONS, display_title
from utils.directions import nautical_sector_to_wedge_angles
from utils.geometry.output_sites import compute_output_sites, write_output_sites
from utils.viz import case_plots as plot

if TYPE_CHECKING:
    from utils.viz.snapwave_case_plots import CasePlotStyle

# SnapWave history NetCDF → HyGloW names
POINT_VARS = {
    "point_hm0": "hs",
    "point_tp": "tp",
    "point_wavdir": "dir",
    "point_dirspr": "spr",
}

# Metamodel LHS forcing stored as ``{name}_forcing`` on postprocessed output
FORCING_VARS = ("tp", "dir", "spr", "wl")

SITE_COORDS = {
    "sites": "source_id",
    "site_source": "source",
    "site_label": "source_label",
    "site_depth": "depth",
    "site_longitude": "lon",
    "site_latitude": "lat",
}

_STATION_RENAMES = {
    "station_source": "site_source",
    "station_label": "site_label",
    "station_depth": "site_depth",
}
_DROP_NATIVE = ("station_x", "station_y", "station_name", "point_zb", "goal")


def read_polygon_from_shp_file(
    shp_path: str | Path,
) -> tuple[gpd.GeoDataFrame, Polygon | MultiPolygon]:
    """
    Read a shapefile and return its GeoDataFrame and first geometry.

    Parameters
    ----------
    shp_path
        Path to a polygon shapefile.

    Returns
    -------
    tuple[GeoDataFrame, Polygon | MultiPolygon]
        Full frame and primary geometry.
    """
    gdf = gpd.read_file(shp_path)
    return gdf, gdf.geometry.iloc[0]


def _single_exterior_polygon(polygon: Polygon | MultiPolygon) -> Polygon:
    """Return one polygon built from the exterior ring only (no holes)."""
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)
    if not isinstance(polygon, Polygon):
        raise TypeError(f"Expected Polygon/MultiPolygon, got {polygon.geom_type}")
    return Polygon(polygon.exterior.coords)


def buffer_polygon(
    polygon: Polygon | MultiPolygon,
    buffer_m: float,
    *,
    geo_crs: str = "EPSG:4326",
) -> Polygon | MultiPolygon:
    """
    Buffer a lon/lat polygon by *buffer_m* metres in a local projected CRS.

    Parameters
    ----------
    polygon
        Input domain polygon.
    buffer_m
        Outward buffer distance in metres. ``0`` returns *polygon* unchanged.
    geo_crs
        CRS of input/output coordinates.

    Returns
    -------
    Polygon | MultiPolygon
        Buffered geometry in *geo_crs*.
    """
    if buffer_m <= 0:
        return polygon

    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=geo_crs)
    work_crs = gdf.estimate_utm_crs()
    if work_crs is None:
        raise ValueError("Could not estimate UTM CRS for AoI polygon.")
    buffered = gdf.to_crs(work_crs).buffer(buffer_m).to_crs(geo_crs)
    return buffered.iloc[0]


def build_enclosure_polygon(
    polygon: Polygon | MultiPolygon,
    buffer_m: float,
    *,
    geo_crs: str = "EPSG:4326",
    n_vertices: int = 4,
) -> Polygon:
    """
    Build a simple SnapWave enclosure from an AoI polygon.

    Steps: exterior ring only -> outward buffer -> reduce to a quadrilateral.

    Parameters
    ----------
    polygon
        AoI polygon or multipolygon.
    buffer_m
        Outward buffer distance in metres.
    geo_crs
        CRS of input/output coordinates.
    n_vertices
        Target number of enclosure vertices. Currently only ``4`` is supported
        (minimum rotated rectangle).

    Returns
    -------
    Polygon
        Simple enclosure quadrilateral.
    """
    if n_vertices != 4:
        raise ValueError("Only n_vertices=4 is supported for SnapWave enclosures.")

    exterior = _single_exterior_polygon(polygon)
    buffered = buffer_polygon(exterior, buffer_m, geo_crs=geo_crs)
    if isinstance(buffered, MultiPolygon):
        buffered = max(buffered.geoms, key=lambda g: g.area)
    if not isinstance(buffered, Polygon):
        raise TypeError(
            f"Buffered geometry must be a Polygon, got {buffered.geom_type}"
        )

    return buffered.minimum_rotated_rectangle


def write_polygon_vertices_to_txt(
    polygon: Polygon | MultiPolygon,
    out_path: str | Path,
    *,
    drop_closing_vertex: bool = True,
    fmt: str = "{x}, {y}",
) -> Path:
    """
    Write polygon exterior vertices to a SnapWave enclosure text file.

    Parameters
    ----------
    polygon
        Domain polygon or multipolygon.
    out_path
        Output text file path.
    drop_closing_vertex
        Omit repeated closing coordinate when ``True``.
    fmt
        Per-vertex format string with ``x`` and ``y`` placeholders.

    Returns
    -------
    Path
        Written file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(polygon, Polygon):
        rings = [list(polygon.exterior.coords)]
    elif isinstance(polygon, MultiPolygon):
        rings = [list(poly.exterior.coords) for poly in polygon.geoms]
    else:
        raise TypeError(f"Expected Polygon/MultiPolygon, got {polygon.geom_type}")

    lines: list[str] = []
    for ring in rings:
        coords = (
            ring[:-1]
            if drop_closing_vertex and len(ring) > 1 and ring[0] == ring[-1]
            else ring
        )
        lines.extend(fmt.format(x=x, y=y) for x, y in coords)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def _site_coord_values(sites_df: pd.DataFrame, col: str) -> np.ndarray:
    """Extract one output-sites column as a numpy array."""
    if col == "depth":
        return pd.to_numeric(sites_df[col], errors="coerce").to_numpy(float)
    return sites_df[col].to_numpy()


def _label_sites(ds: xr.Dataset, sites_df: pd.DataFrame) -> xr.Dataset:
    """Attach HyGloW site metadata coordinates to a SnapWave output dataset."""
    dim = "stations" if "stations" in ds.dims else "sites"
    if ds.sizes[dim] != len(sites_df):
        raise ValueError(
            f"size mismatch: dataset has {ds.sizes[dim]} {dim!r} "
            f"but sites_df has {len(sites_df)} rows"
        )

    if dim != "sites":
        ds = ds.rename({dim: "sites"})
    ds = ds.rename({k: v for k, v in _STATION_RENAMES.items() if k in ds.coords})

    coords = {
        name: (("sites",), _site_coord_values(sites_df, col))
        for name, col in SITE_COORDS.items()
    }
    return ds.assign_coords(coords).drop_vars(list(_DROP_NATIVE), errors="ignore")


def _standardize_postprocessed(
    ds: xr.Dataset,
    case_context: dict,
    sites_df: pd.DataFrame,
    *,
    row_dim: str,
) -> xr.Dataset:
    """
    Standardise postprocessed SnapWave output for HyGloW.

    Renames point variables, attaches site labels, and stores scalar forcing.
    """
    ds = _label_sites(ds, sites_df)
    ds = ds.rename({k: v for k, v in POINT_VARS.items() if k in ds.data_vars})

    keep_vars = set(POINT_VARS.values()) | {f"{k}_forcing" for k in FORCING_VARS}
    ds = ds.drop_vars([v for v in ds.data_vars if v not in keep_vars], errors="ignore")

    for key in FORCING_VARS:
        if key in case_context:
            ds[f"{key}_forcing"] = ((row_dim,), [float(case_context[key])])

    keep_coords = set(SITE_COORDS) | {row_dim}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep_coords], errors="ignore")
    return ds


def _read_sites_nc(case_dir: str, sites_nc: str) -> xr.Dataset:
    """Open a SnapWave sites NetCDF from a case directory."""
    with xr.open_dataset(os.path.join(case_dir, sites_nc)) as ds:
        return (
            ds.drop_dims(["runtime"])
            .drop_vars(["crs", "station_id"], errors="ignore")
            .load()
        )


class SnapWaveModelWrapper(BaseModelWrapper):
    """Shared SnapWave setup: AoI, goals, output sites, case I/O."""

    default_parameters: dict = {}
    available_launchers = {
        "default": "/home/tausiaj/GitHub-GeoOcean/SnapWave/SnapWave/lnx64/bin/snapwave",
    }
    output_variables: dict = {}

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        aoi_file: str | Path,
        goals_config: dict,
        *,
        templates_name: dict = "all",
        debug: bool = True,
        output_sites_df: pd.DataFrame | None = None,
        sites_filename: str = "output_sites.txt",
        enclosure_buffer_m: float = 10_000.0,
        enclosure_vertices: int = 4,
    ) -> None:
        """
        Initialise shared SnapWave wrapper state.

        Parameters
        ----------
        templates_dir, output_dir, aoi_file
            SnapWave template and output paths.
        metamodel_parameters, fixed_parameters
            Case parameter dicts for the base wrapper.
        goals_config
            ``{id: {lon, lat, angles}}`` from :func:`utils.config.goals_for_snapwave`.
        templates_name
            Template subset passed to the base wrapper (``"all"`` by default).
        debug
            Set the wrapper logger to ``DEBUG`` when ``True``, else ``INFO``.
        output_sites_df
            Pre-built output sites table; computed from AoI when omitted.
        sites_filename
            Observation file name written into each case directory.
        enclosure_buffer_m
            Outward buffer (metres) applied to the AoI exterior when writing
            ``enclosure.txt``. Set to ``0`` to skip buffering.
        enclosure_vertices
            Number of enclosure corners (currently only ``4``).
        """
        super().__init__(
            templates_dir=templates_dir,
            metamodel_parameters=metamodel_parameters,
            fixed_parameters=fixed_parameters,
            output_dir=output_dir,
            templates_name=templates_name,
            default_parameters=self.default_parameters,
        )
        self.set_logger_name(
            name=self.__class__.__name__, level="DEBUG" if debug else "INFO"
        )

        self.aoi_gdf, self.aoi_polygon = read_polygon_from_shp_file(aoi_file)
        self.enclosure_buffer_m = float(enclosure_buffer_m)
        self.enclosure_vertices = int(enclosure_vertices)
        self.enclosure_polygon = build_enclosure_polygon(
            self.aoi_polygon,
            self.enclosure_buffer_m,
            n_vertices=self.enclosure_vertices,
        )
        self.goals_config = dict(goals_config)
        self.goals_nodes_array = np.column_stack(
            (
                [g["lon"] for g in self.goals_config.values()],
                [g["lat"] for g in self.goals_config.values()],
            )
        )

        if output_sites_df is None:
            output_sites_df = compute_output_sites(aoi_polygon=self.aoi_polygon)
        self.output_sites_df = output_sites_df.reset_index(drop=True)
        self._sites_txt = sites_filename

        self.sbatch_file_example = """
#!/bin/bash
#SBATCH --job-name=snapwave
#SBATCH --partition=geocean
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2        # cores per case -- also change OMP_NUM_THREADS export below stays in sync automatically
#SBATCH --mem=4gb
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --error=slurm_logs/%A_%a.err

SNAPWAVE_BIN=/home/tausiaj/GitHub-GeoOcean/SnapWave/SnapWave/lnx64/bin/snapwave

# SnapWave has no thread-count logic of its own (no omp_set_num_threads,
# no .inp parameter) -- it just inherits libgomp's default, which grabs
# every online CPU unless told otherwise. This is the only knob.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

case_dir=$(sed -n "${SLURM_ARRAY_TASK_ID}p" case_dirs.txt)
cd "$case_dir"
$SNAPWAVE_BIN > wrapper_out.log 2> wrapper_error.log
rm snapwave.upw

# --ntasks=1 pinned explicitly here too: with CR_Core, --cpus-per-task=1
# gets rounded up to a full physical core (2 hw threads on hyperthreaded
# nodes), and srun defaults its task count to allocated_cpus/cpus-per-task
# when -n isn't given -- without this it silently launches snapwave TWICE.
# srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK "$SNAPWAVE_BIN" > "wrapper_out.log" 2> "wrapper_error.log"
"""

    @property
    def _sites_nc(self) -> str:
        """NetCDF filename paired with the sites text file."""
        return str(Path(self._sites_txt).with_suffix(".nc"))

    def plot_configuration(self, radius_deg: float = 0.3) -> None:
        """
        Plot goals, direction wedges, and the AoI boundary.

        Goal ``angles`` are nautical clockwise ``[left, right]``; wedges are
        drawn via an internal matplotlib conversion (see :mod:`utils.directions`).

        Parameters
        ----------
        radius_deg
            Wedge radius in degrees.
        """
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
        self.aoi_gdf.plot(ax=ax, facecolor="none", edgecolor="black", label="AoI")
        gpd.GeoDataFrame(geometry=[self.enclosure_polygon], crs="EPSG:4326").plot(
            ax=ax,
            facecolor="none",
            edgecolor="red",
            linestyle="--",
            linewidth=1.5,
            label="enclosure",
        )

        for key, vals in self.goals_config.items():
            lon, lat = vals["lon"], vals["lat"]
            left, right = vals["angles"]
            theta1, theta2 = nautical_sector_to_wedge_angles(left, right)
            arc = np.linspace(np.deg2rad(theta1), np.deg2rad(theta2), 100)
            ax.plot(lon, lat, "ko")
            ax.text(lon, lat, str(key), fontsize=9, ha="right")
            ax.fill(
                np.concatenate([[lon], lon + radius_deg * np.cos(arc)]),
                np.concatenate([[lat], lat + radius_deg * np.sin(arc)]),
                alpha=0.4,
            )

        ax.set_title(display_title("Goals + AoI"))
        plt.show()

    def build_case(self, case_context: dict, case_dir: str) -> None:
        """
        Write boundary nodes and output sites into a SnapWave case directory.

        Parameters
        ----------
        case_context
            Per-case parameters (mutated with ``ngoals``).
        case_dir
            Case working directory.
        """
        case_context["ngoals"] = len(self.goals_config)
        write_array_in_file(
            self.goals_nodes_array, os.path.join(case_dir, "boundary.txt")
        )
        write_polygon_vertices_to_txt(
            self.enclosure_polygon, os.path.join(case_dir, "enclosure.txt")
        )
        write_output_sites(self.output_sites_df, case_dir, filename=self._sites_txt)
        if case_context["save_map"]:
            case_context["map_file"] = "output_map.nc"
            case_context["map_dirspr"] = 1
        else:
            case_context["map_file"] = ""
            case_context["map_dirspr"] = 0

    def run_case(
        self,
        case_num: int,
        case_dir: str,
        case_context: dict,
        launcher: str,
        output_log_file: str = "wrapper_out.log",
        error_log_file: str = "wrapper_error.log",
    ) -> None:
        """
        Run one SnapWave case and remove the ``snapwave.upw`` scratch file.

        Parameters
        ----------
        case_num, case_dir, case_context, launcher
            Passed to the base wrapper ``run_case``.
        output_log_file, error_log_file
            Log file names inside *case_dir*.
        """
        super().run_case(
            case_num=case_num,
            case_dir=case_dir,
            case_context=case_context,
            launcher=launcher,
            output_log_file=output_log_file,
            error_log_file=error_log_file,
        )
        upw = os.path.join(case_dir, "snapwave.upw")
        if os.path.exists(upw):
            os.remove(upw)

    def monitor_cases(self, value_counts: str | None = None):
        """
        Report case status from wrapper log file sizes.

        Parameters
        ----------
        value_counts
            Optional status filter passed to the base monitor.

        Returns
        -------
        Any
            Base wrapper monitor result.
        """
        cases_status: dict[str, str] = {}
        for case_dir in self.cases_dirs:
            log = os.path.join(case_dir, "wrapper_out.log")
            if not os.path.exists(log):
                cases_status[os.path.basename(case_dir)] = "NOT STARTED"
            elif os.path.getsize(log) == 0:
                cases_status[os.path.basename(case_dir)] = "RUNNING"
            else:
                cases_status[os.path.basename(case_dir)] = "FINISHED"

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )


class SnapWaveMetaModelWrapper(SnapWaveModelWrapper):
    """LHS SnapWave cases — postprocess to ``case_num``, one NetCDF per goal."""

    def plot_configuration(
        self,
        *,
        cases_df: pd.DataFrame | None = None,
        sector_radius_deg: float = 0.35,
        arrow_scale_deg: float = 0.15,
        max_arrows_per_goal: int = 25,
        show_topobathy: bool = True,
    ) -> plt.Figure:
        """
        Pre-run diagnostic for metamodel case generation.

        Shows the domain, propagation sectors, sampled wave-from directions,
        output sites, per-goal direction-range bars, and a text summary.

        Parameters
        ----------
        cases_df
            Optional LHS table override. Defaults to ``metamodel_parameters``.
        sector_radius_deg, arrow_scale_deg, max_arrows_per_goal
            Map styling controls for sectors and direction arrows.
        show_topobathy
            Draw bathymetry under the map panel.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = (
            cases_df
            if cases_df is not None
            else pd.DataFrame(self.metamodel_parameters)
        )
        fig = plot.plot_metamodel_configuration(
            aoi_gdf=self.aoi_gdf,
            enclosure_polygon=self.enclosure_polygon,
            goals_config=self.goals_config,
            cases_df=df,
            output_sites_df=self.output_sites_df,
            bounds=SIMULATIONS.get("bounds"),
            num_samples_per_goal=SIMULATIONS.get("num_samples_per_goal"),
            mesh_file=self.fixed_parameters.get("gridfile"),
            enclosure_buffer_m=self.enclosure_buffer_m,
            sector_radius_deg=sector_radius_deg,
            arrow_scale_deg=arrow_scale_deg,
            max_arrows_per_goal=max_arrows_per_goal,
            show_topobathy=show_topobathy,
            title="Metamodel pre-run diagnostic",
        )
        plt.show()
        return fig

    def plot_configuration_map(
        self,
        *,
        cases_df: pd.DataFrame | None = None,
        sector_radius_deg: float = 0.35,
        arrow_scale_deg: float = 0.15,
        max_arrows_per_goal: int = 25,
        show_topobathy: bool = True,
        figsize: tuple[float, float] = (12, 11),
        show: bool = True,
    ) -> plt.Figure:
        """
        Bigger, standalone map-only view of :meth:`plot_configuration`.

        Same map as the combined diagnostic, just on its own larger axes —
        use this to actually read goal labels on a case with many goals,
        instead of the map being squeezed into the left column next to the
        bar and info panels.

        Parameters
        ----------
        cases_df
            Optional LHS table override. Defaults to ``metamodel_parameters``.
        sector_radius_deg, arrow_scale_deg, max_arrows_per_goal
            Map styling controls for sectors and direction arrows.
        show_topobathy
            Draw bathymetry under the map panel.
        figsize
            Figure size in inches.
        show
            Call ``plt.show()`` when ``True``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = (
            cases_df
            if cases_df is not None
            else pd.DataFrame(self.metamodel_parameters)
        )
        fig = plot.plot_metamodel_configuration_map(
            enclosure_polygon=self.enclosure_polygon,
            goals_config=self.goals_config,
            cases_df=df,
            output_sites_df=self.output_sites_df,
            sector_radius_deg=sector_radius_deg,
            arrow_scale_deg=arrow_scale_deg,
            max_arrows_per_goal=max_arrows_per_goal,
            show_topobathy=show_topobathy,
            title="Metamodel configuration map",
            figsize=figsize,
        )
        if show:
            plt.show()
        return fig

    def plot_case(
        self,
        goal: int,
        case_num: int,
        *,
        time_index: int = 0,
        show_map: bool = True,
        map_variable: str = "hs",
        style: CasePlotStyle | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot one finished metamodel case folder (``goal/case_num``)."""
        from utils.viz.snapwave_case_plots import plot_metamodel_case

        case_dir = Path(self.output_dir) / f"{int(goal):02}" / f"{int(case_num):04}"
        return plot_metamodel_case(
            case_dir,
            time_index=time_index,
            show_map=show_map,
            map_variable=map_variable,
            style=style,
            show=show,
        )

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        """
        Postprocess one LHS case to ``(case_num, sites)`` HyGloW format.
        """
        self.logger.info(f"[{case_num}]: Postprocessing case {case_num} in {case_dir}.")
        ds = _read_sites_nc(case_dir, self._sites_nc).rename({"time": "case_num"})
        ds["case_num"] = [case_num]
        return _standardize_postprocessed(
            ds, case_context, self.output_sites_df, row_dim="case_num"
        )

    def join_postprocessed_files(self, postprocessed_files: list[xr.Dataset]) -> str:
        """
        Merge postprocessed cases into one NetCDF per goal.

        Returns
        -------
        str
            Status message ``"Done"``.
        """
        self.logger.info("Joining postprocessed files per goal.")

        n_goals = len(self.goals_config)
        n_per_goal = len(postprocessed_files) // n_goals
        out_stem = Path(self._sites_txt).stem

        for goal in range(n_goals):
            sl = slice(goal * n_per_goal, (goal + 1) * n_per_goal)
            xr.merge(postprocessed_files[sl]).to_netcdf(
                os.path.join(self.output_dir, f"{out_stem}_{goal + 1}.nc")
            )

        return "Done"


class SnapWaveDynamicModelWrapper(SnapWaveModelWrapper):
    """Offshore-forced SnapWave cases — postprocess with ``time``, concat on time."""

    def plot_configuration(
        self,
        *,
        offshore: xr.Dataset,
        offshore_name: str | None = None,
        sector_radius_deg: float = 0.35,
        arrow_scale_deg: float = 0.15,
        max_arrows_per_goal: int = 25,
        max_dir_marks: int = 200,
        show_topobathy: bool = True,
        show: bool = True,
    ) -> plt.Figure:
        """
        Pre-run diagnostic for dynamic case generation.

        Shows propagation sectors, observed reanalysis bulk forcing ranges,
        subsampled wave-from directions, output sites, and a text summary.

        Parameters
        ----------
        offshore
            Processed offshore dataset with dims ``(time, goal)`` and bulk
            variables ``hs``, ``tp``, ``dir``, ``spr``.
        offshore_name
            Optional catalog label (e.g. ``"WHACS"``) for the summary panel.
        sector_radius_deg, arrow_scale_deg, max_arrows_per_goal, max_dir_marks
            Map and bar-panel styling controls.
        show_topobathy
            Draw bathymetry under the map panel.
        show
            Call ``plt.show()`` when ``True``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plot.plot_dynamic_configuration(
            aoi_gdf=self.aoi_gdf,
            enclosure_polygon=self.enclosure_polygon,
            goals_config=self.goals_config,
            offshore=offshore,
            output_sites_df=self.output_sites_df,
            offshore_name=offshore_name,
            mesh_file=self.fixed_parameters.get("gridfile"),
            enclosure_buffer_m=self.enclosure_buffer_m,
            sector_radius_deg=sector_radius_deg,
            arrow_scale_deg=arrow_scale_deg,
            max_arrows_per_goal=max_arrows_per_goal,
            max_dir_marks=max_dir_marks,
            show_topobathy=show_topobathy,
            title="Dynamic pre-run diagnostic",
            maps_to_save=(
                int(sum(self.metamodel_parameters["save_map"]))
                if self.metamodel_parameters.get("save_map")
                else None
            ),
        )
        if show:
            plt.show()
        return fig

    def plot_configuration_map(
        self,
        *,
        offshore: xr.Dataset,
        sector_radius_deg: float = 0.35,
        arrow_scale_deg: float = 0.15,
        max_arrows_per_goal: int = 25,
        show_topobathy: bool = True,
        figsize: tuple[float, float] = (12, 11),
        show: bool = True,
    ) -> plt.Figure:
        """
        Bigger, standalone map-only view of :meth:`plot_configuration`.

        Same map as the combined diagnostic, just on its own larger axes —
        use this to actually read goal labels on a case with many goals,
        instead of the map being squeezed into the left column next to the
        bar and info panels.

        Parameters
        ----------
        offshore
            Processed offshore dataset with dims ``(time, goal)`` and bulk
            variables ``hs``, ``tp``, ``dir``, ``spr``.
        sector_radius_deg, arrow_scale_deg, max_arrows_per_goal
            Map styling controls for sectors and direction arrows.
        show_topobathy
            Draw bathymetry under the map panel.
        figsize
            Figure size in inches.
        show
            Call ``plt.show()`` when ``True``.

        Returns
        -------
        matplotlib.figure.Figure
        """
        fig = plot.plot_dynamic_configuration_map(
            enclosure_polygon=self.enclosure_polygon,
            goals_config=self.goals_config,
            offshore=offshore,
            output_sites_df=self.output_sites_df,
            sector_radius_deg=sector_radius_deg,
            arrow_scale_deg=arrow_scale_deg,
            max_arrows_per_goal=max_arrows_per_goal,
            show_topobathy=show_topobathy,
            title="Dynamic configuration map",
            figsize=figsize,
        )
        if show:
            plt.show()
        return fig

    def plot_case(
        self,
        case_dir: str | Path,
        *,
        time_index: int = 0,
        show_map: bool = True,
        map_variable: str = "hs",
        style: CasePlotStyle | None = None,
        show: bool = True,
    ) -> plt.Figure:
        """Plot one finished dynamic case folder."""
        from utils.viz.snapwave_case_plots import plot_dynamic_case

        return plot_dynamic_case(
            case_dir,
            time_index=time_index,
            show_map=show_map,
            map_variable=map_variable,
            style=style,
            show=show,
        )

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        """
        Postprocess one dynamic case to ``(time, sites)`` HyGloW format.
        """
        ds = _read_sites_nc(case_dir, self._sites_nc)
        return _standardize_postprocessed(
            ds, case_context, self.output_sites_df, row_dim="time"
        )

    def join_postprocessed_files(
        self, postprocessed_files: list[xr.Dataset]
    ) -> xr.Dataset:
        """
        Concatenate postprocessed dynamic cases along ``time``.

        Returns
        -------
        xarray.Dataset
            Combined time series for all cases.
        """
        self.logger.info("Joining postprocessed files along time.")
        return xr.concat(postprocessed_files, dim="time")


__all__ = [
    "FORCING_VARS",
    "POINT_VARS",
    "SITE_COORDS",
    "SnapWaveDynamicModelWrapper",
    "SnapWaveMetaModelWrapper",
    "SnapWaveModelWrapper",
    "buffer_polygon",
    "build_enclosure_polygon",
    "read_polygon_from_shp_file",
    "write_polygon_vertices_to_txt",
]
