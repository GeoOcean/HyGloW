import os
from pathlib import Path
from typing import Optional, Union

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import regionmask
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from .._base_wrappers import BaseModelWrapper
from .._utils_wrappers import write_array_in_file


def load_polygon_coords(path: str) -> Polygon:
    """
    Load lon, lat pairs from a text file; returns (N, 2) array.
    """

    coords = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            lon, lat = float(parts[0]), float(parts[1])
            coords.append([lon, lat])
    pts = np.array(coords)

    return Polygon(pts)


def read_polygon_vertices(
    shp_path: str | Path,
    *,
    feature_index: int = 0,
    include_holes: bool = False,
    target_crs: str | None = "EPSG:4326",
) -> list[list[tuple[float, float]]]:
    """
    Return polygon vertex coordinates from a shapefile.

    Parameters
    ----------
    shp_path
        Path to the ``.shp`` file (the sibling ``.shx``/``.dbf``/``.prj``
        must live next to it).
    feature_index
        Row index of the feature to extract.
    include_holes
        If ``True``, interior rings are returned after the exterior ring.
    target_crs
        Reproject to this CRS before extracting coordinates. Set to
        ``None`` to keep the file's native CRS.

    Returns
    -------
    list of rings, each a list of ``(x, y)`` tuples. The first/last
    vertex of each ring is identical (closed ring).
    """

    gdf = gpd.read_file(shp_path)
    if target_crs is not None and gdf.crs is not None:
        gdf = gdf.to_crs(target_crs)

    geom = gdf.geometry.iloc[feature_index]

    polygons: list[Polygon]
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        raise TypeError(f"Expected Polygon/MultiPolygon, got {geom.geom_type}")

    rings: list[list[tuple[float, float]]] = []
    for poly in polygons:
        rings.append(list(poly.exterior.coords))
        if include_holes:
            rings.extend(list(interior.coords) for interior in poly.interiors)

    return rings


def write_vertices_txt(
    rings: list[list[tuple[float, float]]],
    out_path: str | Path,
    *,
    drop_closing_vertex: bool = True,
    fmt: str = "{x} {y}",
    ring_separator: str | None = None,
) -> Path:
    """
    Write polygon vertices to a plain-text file.

    Parameters
    ----------
    rings
        Output of :func:`read_polygon_vertices`.
    out_path
        Destination ``.txt`` path.
    drop_closing_vertex
        If ``True`` (default), drop the last point of each ring when it
        equals the first one. SnapWave/Delft3D boundary files in this
        repo follow this convention (see ``bndenc_nl.txt``).
    fmt
        Format string applied to each vertex; defaults to
        ``"{x} {y}"`` (space-separated, full float precision).
    ring_separator
        Optional line written between rings (e.g. ``""`` for a blank
        line). Ignored when there is only one ring.
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for i, ring in enumerate(rings):
        coords = (
            ring[:-1]
            if drop_closing_vertex and len(ring) > 1 and ring[0] == ring[-1]
            else ring
        )
        if i > 0 and ring_separator is not None:
            lines.append(ring_separator)
        lines.extend(fmt.format(x=x, y=y) for x, y in coords)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out_path


class SnapWaveModelWrapper(BaseModelWrapper):
    """
    Wrapper for the SnapWave model.
    https://gmd.copernicus.org/articles/18/9469/2025/

    Attributes
    ----------
    default_parameters : dict
        The default parameters type for the wrapper.
    available_launchers : dict
        The available launchers for the wrapper.
    output_variables : dict
        The output variables for the wrapper.
    """

    default_parameters = {}

    available_launchers = {
        "default": "/home/tausiaj/GitHub-GeoOcean/SnapWave/SnapWave/lnx64/bin/snapwave",
    }

    output_variables = {}

    def __init__(
        self,
        templates_dir: str,
        metamodel_parameters: dict,
        fixed_parameters: dict,
        output_dir: str,
        region_of_interest: Optional[regionmask.Regions] = None,
        goals_config: Optional[dict] = None,
        templates_name: dict = "all",
        debug: bool = True,
    ) -> None:
        """
        Initialize the SWAN model wrapper.
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

        # Load boundary conditions polygon
        self.boundary_conditions_polygon = None
        for template_name in self.templates_name:
            if "enc" in template_name:
                self.boundary_conditions_polygon = load_polygon_coords(
                    os.path.join(self.templates_dir, template_name)
                )
                break

        # Set region of interest
        if region_of_interest is not None:
            self.region_of_interest = region_of_interest
        else:
            self.region_of_interest = None

        # Set goals config
        if goals_config is not None:
            self.goals_config = goals_config
        else:
            self.goals_config = None

    def calculate_output_grid_parameters(
        self,
        output_grid_resolution: float,
    ) -> None:
        """
        Calculate some mesh and output grid parameters.
        """

        # Store goals lon and lat values
        goals_lons = [goal["lon"] for goal in self.goals_config.values()]
        goals_lats = [goal["lat"] for goal in self.goals_config.values()]
        self.goals_nodes_array = np.column_stack((goals_lons, goals_lats))

        # Calculate the output grid parameters
        self.output_lons = np.arange(
            np.min(goals_lons),
            np.max(goals_lons),
            output_grid_resolution,
        )
        self.output_lats = np.arange(
            np.min(goals_lats),
            np.max(goals_lats),
            output_grid_resolution,
        )
        self.output_mesh_lons, self.output_mesh_lats = np.meshgrid(
            self.output_lons,
            self.output_lats,
        )

    def plot_configuration(
        self,
        radius_deg: float = 0.3,
    ) -> None:
        """
        Plot the configuration of the wrapper.

        Parameters
        ----------
        radius_deg : float, optional
            Radius of sector in degrees. Default is 0.3.
        """

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

        # -------------------------
        # 1️⃣ Plot region of interest
        # -------------------------
        if self.region_of_interest is not None:
            self.region_of_interest.plot(ax=ax)

        # -------------------------
        # 2️⃣ Plot boundary conditions polygon
        # -------------------------
        if self.boundary_conditions_polygon is not None:
            ax.plot(
                self.boundary_conditions_polygon.exterior.xy[0],
                self.boundary_conditions_polygon.exterior.xy[1],
                "k-",
            )

        # -------------------------
        # 3️⃣ Plot goal nodes and angular sectors
        # -------------------------
        if self.goals_config is not None:
            for goal_key, goal_values in self.goals_config.items():
                lon = goal_values["lon"]
                lat = goal_values["lat"]
                angle1, angle2 = goal_values["angles"]

                angle1 = np.deg2rad(angle1 % 360)
                angle2 = np.deg2rad(angle2 % 360)

                if angle2 < angle1:
                    theta = np.linspace(angle1, angle2 + 2 * np.pi, 100)
                else:
                    theta = np.linspace(angle1, angle2, 100)

                arc_lon = lon + radius_deg * np.cos(theta)
                arc_lat = lat + radius_deg * np.sin(theta)

                ax.plot(lon, lat, "ko")
                ax.text(lon, lat, str(goal_key), fontsize=9, ha="right")

                ax.fill(
                    np.concatenate([[lon], arc_lon]),
                    np.concatenate([[lat], arc_lat]),
                    alpha=0.4,
                )

        # -------------------------
        # Formatting
        # -------------------------
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            "Goal Nodes + Region of Interest + Boundary Conditions Polygon (Lon/Lat)"
        )
        plt.show()

    def build_case(self, case_context: dict, case_dir: str) -> None:
        """
        Build the case.
        """

        case_context["ngoals"] = len(self.goals_config)
        write_array_in_file(
            self.goals_nodes_array,
            os.path.join(case_dir, "boundary.txt"),
        )

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
        Run the case based on the launcher specified.

        Parameters
        ----------
        case_dir : str
            The case directory.
        launcher : str
            The launcher to run the case.
        output_log_file : str, optional
            The name of the output log file. Default is "wrapper_out.log".
        error_log_file : str, optional
            The name of the error log file. Default is "wrapper_error.log".
        """

        super().run_case(
            case_num=case_num,
            case_dir=case_dir,
            case_context=case_context,
            launcher=launcher,
            output_log_file=output_log_file,
            error_log_file=error_log_file,
        )
        os.remove(os.path.join(case_dir, "snapwave.upw"))

    def monitor_cases(self, value_counts: str = None) -> Union[pd.DataFrame, dict]:
        """
        Monitor the cases based on the wrapper_out.log file.
        """

        cases_status = {}

        for case_dir in self.cases_dirs:
            output_log_file = os.path.join(case_dir, "wrapper_out.log")
            if not os.path.exists(output_log_file):
                progress = "NOT STARTED"
            elif os.path.getsize(output_log_file) == 0:
                progress = "RUNNING"
            else:
                progress = "FINISHED"
            cases_status[os.path.basename(case_dir)] = progress

        return super().monitor_cases(
            cases_status=cases_status, value_counts=value_counts
        )

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        """
        Postprocess the case.
        """

        self.logger.info(f"[{case_num}]: Postprocessing case {case_num} in {case_dir}.")

        output_nc_path = os.path.join(case_dir, "10mDepthAndBuoysPoints.nc")
        output_nc = (
            xr.open_dataset(output_nc_path)
            .drop_dims(["runtime"])
            .drop_vars(["crs", "station_id"])
            .rename({"time": "case_num"})
        )
        output_nc["case_num"] = [case_num]
        output_nc = output_nc.assign_coords(
            {
                "goal": (("case_num"), [int(case_context["goal"])]),
                "station_name": (("stations"), output_nc.station_name.values),
                "station_x": (("stations"), output_nc.station_x.values),
                "station_y": (("stations"), output_nc.station_y.values),
                "point_zb": (("stations"), output_nc.point_zb.values),
            }
        )

        return output_nc

    def join_postprocessed_files(self, postprocessed_files: list[xr.Dataset]) -> str:
        """
        Join the postprocessed files.
        """

        self.logger.info("Joining postprocessed files.")

        for goal in range(10):
            postprocessed_goal = xr.merge(
                postprocessed_files[goal * 2000 : (goal + 1) * 2000]
            )
            postprocessed_goal = postprocessed_goal.assign_coords(
                {
                    "tp": (
                        ("case_num"),
                        self.metamodel_parameters["tp"][
                            goal * 2000 : (goal + 1) * 2000
                        ],
                    ),
                    "wd": (
                        ("case_num"),
                        self.metamodel_parameters["wd"][
                            goal * 2000 : (goal + 1) * 2000
                        ],
                    ),
                    "ds": (
                        ("case_num"),
                        self.metamodel_parameters["ds"][
                            goal * 2000 : (goal + 1) * 2000
                        ],
                    ),
                    "zs": (
                        ("case_num"),
                        self.metamodel_parameters["zs"][
                            goal * 2000 : (goal + 1) * 2000
                        ],
                    ),
                }
            )
            postprocessed_goal.to_netcdf(
                os.path.join(self.output_dir, f"10mDepthAndBuoysPoints_{goal + 1}.nc")
            )

        return "Olrait!"
