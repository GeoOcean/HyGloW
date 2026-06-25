"""SnapWave metamodel wrapper for the Samoa case."""

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
from utils.contours import compute_output_sites, write_output_sites

# SnapWave outputs → HyGloW variable names
POINT_VARS = {
    "point_hm0": "hs",
    "point_tp": "tp",
    "point_wavdir": "dir",
}

# LHS parameters (see utils.simulations) → stored as ``{name}_forcing``
FORCING_VARS = ("tp", "dir", "spr", "wl")

# sites_df column → coordinate on the ``sites`` dimension
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
    gdf = gpd.read_file(shp_path)
    return gdf, gdf.geometry.iloc[0]


def write_polygon_vertices_to_txt(
    polygon: Polygon | MultiPolygon,
    out_path: str | Path,
    *,
    drop_closing_vertex: bool = True,
) -> Path:
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
    if col == "depth":
        return pd.to_numeric(sites_df[col], errors="coerce").to_numpy(float)
    return sites_df[col].to_numpy()


def _label_sites(ds: xr.Dataset, sites_df: pd.DataFrame) -> xr.Dataset:
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
) -> xr.Dataset:
    """Outputs, site metadata, and LHS forcing only."""

    ds = _label_sites(ds, sites_df)
    ds = ds.rename({k: v for k, v in POINT_VARS.items() if k in ds.data_vars})

    keep_vars = set(POINT_VARS.values()) | {f"{k}_forcing" for k in FORCING_VARS}
    drop_vars = [v for v in ds.data_vars if v not in keep_vars]
    ds = ds.drop_vars(drop_vars, errors="ignore")

    for key in FORCING_VARS:
        if key in case_context:
            ds[f"{key}_forcing"] = (("case_num",), [float(case_context[key])])

    keep_coords = set(SITE_COORDS) | {"case_num"}
    ds = ds.drop_vars([c for c in ds.coords if c not in keep_coords], errors="ignore")
    return ds


class SnapWaveMetaModelWrapper(BaseModelWrapper):
    """LHS-driven SnapWave metamodel cases."""

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
        return str(Path(self._sites_txt).with_suffix(".nc"))

    def plot_configuration(self, radius_deg: float = 0.3) -> None:
        fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
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
            ax.plot(lon, lat, "ko")
            ax.text(lon, lat, str(key), fontsize=9, ha="right")
            ax.fill(
                np.concatenate([[lon], lon + radius_deg * np.cos(theta)]),
                np.concatenate([[lat], lat + radius_deg * np.sin(theta)]),
                alpha=0.4,
            )

        ax.set_title("Samoa goals + AoI")
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

    def monitor_cases(self, value_counts: str | None = None):
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
        return _standardize_postprocessed(ds, case_context, self.output_sites_df)

    def join_postprocessed_files(self, postprocessed_files: list[xr.Dataset]) -> str:
        self.logger.info("Joining postprocessed files.")

        n_goals = len(self.goals_config)
        n_per_goal = len(postprocessed_files) // n_goals
        out_stem = Path(self._sites_txt).stem

        for goal in range(n_goals):
            sl = slice(goal * n_per_goal, (goal + 1) * n_per_goal)
            xr.merge(postprocessed_files[sl]).to_netcdf(
                os.path.join(self.output_dir, f"{out_stem}_{goal + 1}.nc")
            )

        return "Done"
