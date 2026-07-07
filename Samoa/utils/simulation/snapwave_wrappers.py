"""SnapWave case wrappers for Samoa.

:class:`SnapWaveMetaModelWrapper` — LHS propagation cases (``case_num`` dim,
one NetCDF per goal).

:class:`SnapWaveDynamicModelWrapper` — offshore-waves forcing (``time`` dim,
concatenated time series).

Both produce site-labelled outputs ``hs``, ``tp``, ``dir`` on ``sites``.
"""

from __future__ import annotations

import os
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper
from bluemath_tk.wrappers._utils_wrappers import write_array_in_file
from shapely.geometry import MultiPolygon, Polygon
from utils.geometry.output_sites import compute_output_sites, write_output_sites
from utils.directions import nautical_sector_to_wedge_angles

# SnapWave history NetCDF → HyGloW names
POINT_VARS = {
    "point_hm0": "hs",
    "point_tp": "tp",
    "point_wavdir": "dir",
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


def write_polygon_vertices_to_txt(
    polygon: Polygon | MultiPolygon,
    out_path: str | Path,
    *,
    drop_closing_vertex: bool = True,
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
        lines.extend(f"{x} {y}" for x, y in coords)

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
    return (
        xr.open_dataset(os.path.join(case_dir, sites_nc))
        .drop_dims(["runtime"])
        .drop_vars(["crs", "station_id"], errors="ignore")
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
        output_sites_df
            Pre-built output sites table; computed from AoI when omitted.
        sites_filename
            Observation file name written into each case directory.
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
        self.aoi_gdf.plot(ax=ax, facecolor="none", edgecolor="black")

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

        ax.set_title("Samoa goals + AoI")
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
        # write_polygon_vertices_to_txt(
        #     self.aoi_polygon, os.path.join(case_dir, "enclosure.txt")
        # )
        write_output_sites(self.output_sites_df, case_dir, filename=self._sites_txt)

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
    "read_polygon_from_shp_file",
    "write_polygon_vertices_to_txt",
]
