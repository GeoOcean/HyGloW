"""SnapWave wrapper for the North Carolina case (and drop-in replacement for
``bluemath_tk.wrappers.snapwave.snapwave_wrapper``).

The file bundles three things:

1. Small **I/O helpers** (``read_polygon_from_shp_file``,
   ``write_polygon_vertices_to_txt``).
2. **Output-sites helpers** (``compute_output_sites``, ``write_output_sites``,
   ``relabel_output_stations``) that turn the ``output_sites`` section of
   ``NorthCarolina.json`` into a SnapWave ``obsfile`` and tag the resulting
   stations as ``10m_0``, ``20m_65``, ``buoy_41013``, etc.
3. The actual **SnapWave wrappers** —
   :class:`SnapWaveModelWrapper`, :class:`SnapWaveMetaModelWrapper` and
   :class:`SnapWaveDynamicModelWrapper` — inheriting directly from
   :class:`bluemath_tk.wrappers._base_wrappers.BaseModelWrapper`. Each
   ``build_case`` writes ``boundary.txt``, ``enclosure.txt`` and
   ``output_sites.txt``; each ``postprocess_case`` opens
   ``output_sites.nc`` and re-labels the stations via
   :func:`relabel_output_stations`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Union

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.wrappers._base_wrappers import BaseModelWrapper
from bluemath_tk.wrappers._utils_wrappers import write_array_in_file
from shapely.geometry import MultiPolygon, Polygon

from . import config as cfg
from .contours import get_isobath_points

# ---------------------------------------------------------------------------
# Geometry I/O helpers (carried over from the toolkit version)
# ---------------------------------------------------------------------------


def read_polygon_from_shp_file(
    shp_path: str | Path,
) -> tuple[gpd.GeoDataFrame, Polygon | MultiPolygon]:
    """Read a polygon/multi-polygon from a shapefile."""

    gdf = gpd.read_file(shp_path)
    return gdf, gdf.loc[0, "geometry"]


def write_polygon_vertices_to_txt(
    polygon: Polygon | MultiPolygon,
    out_path: str | Path,
    *,
    drop_closing_vertex: bool = True,
    fmt: str = "{x} {y}",
    ring_separator: str | None = None,
) -> Path:
    """Write polygon/multi-polygon vertices to a SnapWave/Delft3D-style ``.txt``."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(polygon, Polygon):
        rings = [list(polygon.exterior.coords)]
    elif isinstance(polygon, MultiPolygon):
        rings = [list(poly.exterior.coords) for poly in polygon.geoms]
    else:
        raise TypeError(f"Expected Polygon/MultiPolygon, got {polygon.geom_type}")

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


# ---------------------------------------------------------------------------
# Output-sites helpers
# ---------------------------------------------------------------------------


def _isobath_rows(da: xr.DataArray, specs: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for spec in specs:
        name = spec.get("name") or f"{abs(spec['level']):g}m"
        pts = get_isobath_points(
            da,
            level=float(spec["level"]),
            spacing=spec.get("spacing"),
            longest_only=bool(spec.get("longest_only", False)),
        )
        for i, (lon, lat) in enumerate(pts):
            rows.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "source": "isobath",
                    "source_id": f"{name}_{i}",
                    "source_label": name,
                    "depth": float(spec["level"]),
                    "name": None,
                }
            )
    return rows


def _buoy_rows(role: str | None = None) -> list[dict]:
    rows: list[dict] = []
    for iid, b in cfg.get_buoy_coordinates(role=role).items():
        rows.append(
            {
                "lon": float(b["lon"]),
                "lat": float(b["lat"]),
                "source": "buoy",
                "source_id": f"buoy_{iid}",
                "source_label": b.get("role") or "buoy",
                "depth": b.get("depth"),
                "name": b.get("name") or b.get("institution"),
            }
        )
    return rows


def compute_output_sites(
    config: Mapping[str, Any] | None = None,
    topobathy: xr.DataArray | None = None,
) -> pd.DataFrame:
    """Build the SnapWave output-sites DataFrame from the ``output_sites`` config.

    Returns columns ``[lon, lat, source, source_id, source_label, depth, name]``.
    Isobath samples first (grouped by isobath, in raster order), buoys after.
    """

    cfg_dict = dict(config or cfg.OUTPUT_SITES or {})
    if not cfg_dict:
        raise ValueError("Missing `output_sites` section in NorthCarolina.json.")

    rows: list[dict] = []
    iso_specs = cfg_dict.get("isobaths") or []
    if iso_specs:
        if topobathy is None:
            import rioxarray  # noqa: F401  (registers the .rio accessor)

            topobathy = xr.open_dataarray(
                cfg.resolve(cfg_dict["topobathy_file"]), engine="rasterio"
            )
            if "band" in topobathy.dims and topobathy.sizes["band"] == 1:
                topobathy = topobathy.squeeze("band", drop=True)
        rows.extend(_isobath_rows(topobathy, list(iso_specs)))

    if cfg_dict.get("include_buoys", True):
        rows.extend(_buoy_rows(role=cfg_dict.get("buoy_role")))

    if not rows:
        raise ValueError("No output-site points produced.")
    return pd.DataFrame(rows).reset_index(drop=True)


def write_output_sites(
    df: pd.DataFrame,
    out_dir: str | Path,
    filename: str = "output_sites.txt",
) -> Path:
    """Write ``df['lon', 'lat']`` to a SnapWave ``lon lat`` ``.txt`` file."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / filename
    np.savetxt(txt_path, df[["lon", "lat"]].to_numpy(float), fmt="%.8f")
    return txt_path


def relabel_output_stations(
    ds: xr.Dataset,
    sites_df: pd.DataFrame,
    dim: str = "stations",
) -> xr.Dataset:
    """Override ``station_name`` and add provenance coords on a SnapWave history.

    Row order in ``sites_df`` must match the SnapWave ``stations`` dimension.
    Adds/overwrites:

    * ``station_name``   — ``source_id`` (e.g. ``10m_0``, ``buoy_41013``)
    * ``station_source`` — ``"isobath"`` or ``"buoy"``
    * ``station_label``  — ``source_label`` (isobath name or buoy role)
    * ``station_depth``  — ``depth`` (NaN where missing)
    """

    n_ds = ds.sizes.get(dim)
    if n_ds != len(sites_df):
        raise ValueError(
            f"size mismatch: dataset has {n_ds} {dim!r} but sites_df has "
            f"{len(sites_df)} rows"
        )
    return ds.assign_coords(
        {
            "station_name": ((dim,), sites_df["source_id"].to_numpy()),
            "station_source": ((dim,), sites_df["source"].to_numpy()),
            "station_label": ((dim,), sites_df["source_label"].to_numpy()),
            "station_depth": (
                (dim,),
                pd.to_numeric(sites_df["depth"], errors="coerce").to_numpy(float),
            ),
        }
    )


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class SnapWaveModelWrapper(BaseModelWrapper):
    """Base SnapWave wrapper (https://gmd.copernicus.org/articles/18/9469/2025/).

    Pre-computes the output-sites DataFrame from
    :data:`utils.config.OUTPUT_SITES` (or the ``output_sites_df`` keyword
    override) and stores it on ``self.output_sites_df``. Each ``build_case``
    drops ``boundary.txt``, ``enclosure.txt`` and ``output_sites.txt`` into
    the case directory; postprocess subclasses then read
    ``output_sites.nc`` (filename derived from the configured ``output_sites.filename``
    with the extension swapped to ``.nc``) and relabel its stations.
    """

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
        templates_name: dict = "all",
        debug: bool = True,
        output_sites_df: pd.DataFrame | None = None,
    ) -> None:
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
            output_sites_df = compute_output_sites()
        self.output_sites_df = output_sites_df.reset_index(drop=True)
        self._sites_txt = (cfg.OUTPUT_SITES or {}).get("filename") or "output_sites.txt"

    @property
    def _sites_nc(self) -> str:
        return str(Path(self._sites_txt).with_suffix(".nc"))

    def plot_configuration(self, radius_deg: float = 0.3) -> None:
        """Plot the AoI polygon and the per-goal angular sectors."""

        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

        if self.aoi_gdf is not None:
            self.aoi_gdf.plot(ax=ax, facecolor="none", edgecolor="black")

        for key, vals in self.goals_config.items():
            lon, lat = vals["lon"], vals["lat"]
            a1 = np.deg2rad(vals["angles"][0] % 360)
            a2 = np.deg2rad(vals["angles"][1] % 360)
            theta = (
                np.linspace(a1, a2 + 2 * np.pi, 100)
                if a2 < a1
                else np.linspace(a1, a2, 100)
            )
            arc_lon = lon + radius_deg * np.cos(theta)
            arc_lat = lat + radius_deg * np.sin(theta)
            ax.plot(lon, lat, "ko")
            ax.text(lon, lat, str(key), fontsize=9, ha="right")
            ax.fill(
                np.concatenate([[lon], arc_lon]),
                np.concatenate([[lat], arc_lat]),
                alpha=0.4,
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Goal Nodes + AoI Polygon (Lon/Lat)")
        plt.show()

    def build_case(self, case_context: dict, case_dir: str) -> None:
        case_context["ngoals"] = len(self.goals_config)
        write_array_in_file(
            self.goals_nodes_array, os.path.join(case_dir, "boundary.txt")
        )
        write_polygon_vertices_to_txt(
            self.aoi_polygon, os.path.join(case_dir, "enclosure.txt")
        )
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

    def monitor_cases(self, value_counts: str = None) -> Union[pd.DataFrame, dict]:
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
    """SnapWave wrapper for the **metamodel** (LHS-driven) cases."""

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        self.logger.info(f"[{case_num}]: Postprocessing case {case_num} in {case_dir}.")
        ds = (
            xr.open_dataset(os.path.join(case_dir, self._sites_nc))
            .drop_dims(["runtime"])
            .drop_vars(["crs", "station_id"], errors="ignore")
            .rename({"time": "case_num"})
        )
        ds["case_num"] = [case_num]
        ds = ds.assign_coords(
            {
                "goal": (("case_num",), [int(case_context["goal"])]),
                "station_x": (("stations",), ds.station_x.values),
                "station_y": (("stations",), ds.station_y.values),
                "point_zb": (("stations",), ds.point_zb.values),
            }
        )
        return relabel_output_stations(ds, self.output_sites_df)

    def join_postprocessed_files(self, postprocessed_files: list[xr.Dataset]) -> str:
        """Merge cases per goal and write one NetCDF per goal to ``output_dir``."""

        self.logger.info("Joining postprocessed files.")

        n_goals = len(self.goals_config)
        n_per_goal = len(postprocessed_files) // n_goals
        params = self.metamodel_parameters
        out_stem = Path(self._sites_txt).stem

        for goal in range(n_goals):
            sl = slice(goal * n_per_goal, (goal + 1) * n_per_goal)
            merged = xr.merge(postprocessed_files[sl]).assign_coords(
                {
                    name: (("case_num",), params[name][sl])
                    for name in ("tp", "wd", "ds", "zs")
                    if name in params
                }
            )
            merged.to_netcdf(os.path.join(self.output_dir, f"{out_stem}_{goal + 1}.nc"))

        return "Olrait!"


class SnapWaveDynamicModelWrapper(SnapWaveModelWrapper):
    """SnapWave wrapper for the **dynamic** cases (per-time-step forcing)."""

    def postprocess_case(
        self, case_num: int, case_dir: str, case_context: dict
    ) -> xr.Dataset:
        self.logger.info(f"[{case_num}]: Postprocessing case {case_num} in {case_dir}.")
        ds = (
            xr.open_dataset(os.path.join(case_dir, self._sites_nc))
            .drop_dims(["runtime"])
            .drop_vars(["crs", "station_id"], errors="ignore")
        )
        ds = ds.assign_coords(
            {
                "station_x": (("stations",), ds.station_x.values),
                "station_y": (("stations",), ds.station_y.values),
                "point_zb": (("stations",), ds.point_zb.values),
            }
        )
        return relabel_output_stations(ds, self.output_sites_df)

    def join_postprocessed_files(
        self, postprocessed_files: list[xr.Dataset]
    ) -> xr.Dataset:
        self.logger.info("Joining postprocessed files.")
        return xr.concat(postprocessed_files, dim="time")
