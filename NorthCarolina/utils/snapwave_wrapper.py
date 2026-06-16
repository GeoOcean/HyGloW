"""SnapWave helpers for the North Carolina case.

Standalone callables (usable from any notebook):

* :func:`compute_output_sites` — build the (lon, lat, source, source_id, ...)
  DataFrame from the ``output_sites`` section of ``NorthCarolina.json``.
* :func:`write_output_sites` — write that DataFrame's ``lon`` / ``lat``
  columns to ``output_sites.txt`` (the file SnapWave reads through
  ``obsfile``). Metadata is kept in memory on the wrapper and consumed
  later by :func:`relabel_output_stations`.
* :func:`relabel_output_stations` — override ``station_name`` and add
  ``station_source`` / ``station_label`` / ``station_depth`` on a SnapWave
  history dataset using the metadata DataFrame, so coordinates read e.g.
  ``10m_0``, ``20m_65``, ``buoy_41013``, ``buoy_jprn7``.

Wrapper subclasses that plug the above into the bluemath_tk pipeline:

* :class:`NCSnapWaveMetaModelWrapper` and
  :class:`NCSnapWaveDynamicModelWrapper` — drop ``output_sites.txt`` into
  each case directory in ``build_case`` and, in ``postprocess_case``, read
  the SnapWave history NetCDF and relabel its stations using the in-memory
  ``output_sites_df`` so the saved ``.nc`` carries ``station_name``,
  ``station_source``, ``station_label`` and ``station_depth`` coordinates.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.wrappers.snapwave.snapwave_wrapper import (
    SnapWaveDynamicModelWrapper,
    SnapWaveMetaModelWrapper,
)

from . import config as cfg
from .contours import get_isobath_points


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


class _NCOutputSitesMixin:
    """Pre-compute the output-sites DataFrame at ``__init__`` (kept on
    ``self.output_sites_df`` for ``postprocess_case``) and drop
    ``output_sites.txt`` into every case directory inside ``build_case``."""

    def __init__(
        self,
        *args,
        output_sites_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if output_sites_df is None:
            output_sites_df = compute_output_sites()
        self.output_sites_df = output_sites_df.reset_index(drop=True)
        self._sites_txt = (cfg.OUTPUT_SITES or {}).get("filename") or "output_sites.txt"

    @property
    def _sites_nc(self) -> str:
        return str(Path(self._sites_txt).with_suffix(".nc"))

    def build_case(self, case_context: dict, case_dir: str) -> None:
        super().build_case(case_context, case_dir)
        write_output_sites(self.output_sites_df, case_dir, filename=self._sites_txt)


class NCSnapWaveMetaModelWrapper(_NCOutputSitesMixin, SnapWaveMetaModelWrapper):
    """Metamodel SnapWave wrapper with auto-managed output sites + relabeling."""

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


class NCSnapWaveDynamicModelWrapper(_NCOutputSitesMixin, SnapWaveDynamicModelWrapper):
    """Dynamic SnapWave wrapper with auto-managed output sites + relabeling."""

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
