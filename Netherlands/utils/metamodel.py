from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from .config import CASE_DIR, METAMODEL, OFFSHORE_WAVES

PARTITION_VARIABLES: dict[str, dict[str, str]] = {
    "bulk": {"hs": "hs", "tp": "tp", "wd": "dp", "ds": "spr"},
    "sea": {"hs": "phs0", "tp": "ptp0", "wd": "pdp0", "ds": "pspr0"},
    "swell1": {"hs": "phs1", "tp": "ptp1", "wd": "pdp1", "ds": "pspr1"},
    "swell2": {"hs": "phs2", "tp": "ptp2", "wd": "pdp2", "ds": "pspr2"},
    "swell3": {"hs": "phs3", "tp": "ptp3", "wd": "pdp3", "ds": "pspr3"},
}


def _safe_int(value: object) -> int:
    return int(float(value))


def _resolve_path(path_like: str | Path) -> Path:
    p = Path(path_like)
    if not p.is_absolute():
        p = CASE_DIR / p
    return p


def _meta_path(path_key: str, **kwargs: object) -> Path:
    template = METAMODEL["paths"][path_key]
    if isinstance(template, str):
        return _resolve_path(template.format(**kwargs))
    raise TypeError(f"Config value METAMODEL.paths.{path_key} must be a string.")


def _load_goal_mapping(coords_csv: str | Path) -> pd.DataFrame:
    coords = pd.read_csv(_resolve_path(coords_csv))
    return coords


def _load_codec_station_mapping() -> dict[int, int]:
    codec_coords = pd.read_csv(_meta_path("codec_coordinates"))
    return {
        _safe_int(row.goal): _safe_int(row.stations)
        for row in codec_coords.itertuples(index=False)
    }


def _open_whacs_zarr(whacs_zarr_path: str | Path) -> xr.Dataset:
    path = _resolve_path(whacs_zarr_path)
    try:
        return xr.open_dataset(path, engine="zarr")
    except Exception:
        os.environ.setdefault("ZARR_V3_EXPERIMENTAL_API", "1")
        group = zarr.open_group(str(path), mode="r")
        time_values = np.asarray(group["time"][:], dtype=float)
        units = dict(getattr(group["time"], "attrs", {})).get("units", "days since 1990-01-01")
        origin = units.split("since", maxsplit=1)[-1].strip()
        time = pd.to_datetime(origin) + pd.to_timedelta(time_values, unit="D")
        seapoint = np.asarray(group["seapoint"][:], dtype=int)
        vars_needed = sorted(
            {v for m in PARTITION_VARIABLES.values() for v in m.values()} | {"longitude", "latitude"}
        )
        data_vars = {
            name: (("time", "seapoint"), np.asarray(group[name][:]))
            for name in vars_needed
            if name in group
        }
        return xr.Dataset(data_vars=data_vars, coords={"time": time, "seapoint": seapoint})


def fit(goals: Iterable[int] | None = None, *, force_retrain: bool = False) -> dict[str, dict[int, object]]:
    from bluemath_tk.core.io import load_model
    from bluemath_tk.core.operations import get_uv_components
    from bluemath_tk.datamining.mda import MDA
    from bluemath_tk.datamining.pca import PCA
    from bluemath_tk.interpolation.gps import ExactGPInterpolation

    num_mda_points = int(METAMODEL["num_mda_points"])
    active_goals = list(goals or METAMODEL["goals"])

    models: dict[str, dict[int, object]] = {"mda": {}, "pca": {}, "gp": {}}
    for goal in active_goals:
        mda_model_path = _meta_path("mda_model_template", num_mda_points=num_mda_points, goal=goal)
        pca_model_path = _meta_path("pca_model_template", num_mda_points=num_mda_points, goal=goal)
        gp_model_path = _meta_path("gp_model_template", num_mda_points=num_mda_points, goal=goal)

        if not force_retrain and mda_model_path.exists() and pca_model_path.exists() and gp_model_path.exists():
            models["mda"][goal] = load_model(str(mda_model_path))
            models["pca"][goal] = load_model(str(pca_model_path))
            models["gp"][goal] = load_model(str(gp_model_path))
            continue

        cases_path = _meta_path("metamodel_cases_template", goal=goal)
        metamodel_cases_goal = xr.open_dataset(cases_path)
        wavdir_u, wavdir_v = get_uv_components(metamodel_cases_goal.point_wavdir)
        metamodel_cases_goal["point_wavdir_u"] = wavdir_u
        metamodel_cases_goal["point_wavdir_v"] = wavdir_v
        metamodel_cases_goal["stations"] = range(metamodel_cases_goal.sizes["stations"])
        cases_df = metamodel_cases_goal[["tp", "wd", "ds", "zs"]].to_dataframe()

        mda = MDA(num_centers=num_mda_points)
        mda.fit(cases_df, directional_variables=["wd"])
        pca = PCA(n_components=0.95)
        pca.fit_transform(
            data=metamodel_cases_goal.isel(case_num=mda.centroid_real_indices),
            vars_to_stack=["point_hm0", "point_tp", "point_wavdir_u", "point_wavdir_v"],
            coords_to_stack=["stations"],
            pca_dim_for_rows="case_num",
        )
        gp = ExactGPInterpolation()
        gp.fit(
            subset_data=mda.centroids[["tp", "wd", "ds", "zs"]],
            subset_directional_variables=["wd"],
            target_data=pca.pcs_df,
        )
        mda.save_model(str(mda_model_path))
        pca.save_model(str(pca_model_path))
        gp.save_model(str(gp_model_path))
        models["mda"][goal] = mda
        models["pca"][goal] = pca
        models["gp"][goal] = gp

    return models


def reconstruct(
    target_coordinates_csv: str | Path,
    *,
    whacs_zarr_path: str | Path | None = None,
    time_selector: str | slice | None = None,
) -> xr.Dataset:
    from bluemath_tk.core.io import load_model
    from bluemath_tk.core.operations import get_degrees_from_uv

    whacs_path = whacs_zarr_path or OFFSHORE_WAVES["WHACS"]["local_path"]
    whacs_coords = OFFSHORE_WAVES["WHACS"]["coordinates"]
    requested_points = pd.read_csv(_resolve_path(target_coordinates_csv))
    requested_goals = sorted({_safe_int(g) for g in requested_points["goal"]})
    goal_mapping_df = _load_goal_mapping(whacs_coords)
    goal_to_seapoint = (
        goal_mapping_df.drop_duplicates(subset=["goal"]).set_index("goal")["seapoint"].map(_safe_int).to_dict()
    )
    codec_station_by_goal = _load_codec_station_mapping()
    num_mda_points = int(METAMODEL["num_mda_points"])
    tp_factors = METAMODEL["tp_factors"]

    pca_models = {
        goal: load_model(str(_meta_path("pca_model_template", num_mda_points=num_mda_points, goal=goal)))
        for goal in requested_goals
    }
    gp_models = {
        goal: load_model(str(_meta_path("gp_model_template", num_mda_points=num_mda_points, goal=goal)))
        for goal in requested_goals
    }

    whacs_ds = _open_whacs_zarr(whacs_path)
    if time_selector is not None:
        whacs_ds = whacs_ds.sel(time=time_selector)

    out: list[xr.Dataset] = []
    codec_dir = _meta_path("codec_dir")
    partitions = METAMODEL["partitions"]
    for partition in partitions:
        variable_map = PARTITION_VARIABLES[partition]
        tp_factor = float(tp_factors.get(partition, 1.0))
        for goal in requested_goals:
            seapoint = goal_to_seapoint[goal]
            forcing_ds = whacs_ds.sel(seapoint=seapoint)[list(variable_map.values())].rename(
                {source_name: target_name for target_name, source_name in variable_map.items()}
            )
            forcing_df = forcing_ds[["hs", "tp", "wd", "ds"]].to_dataframe()
            forcing_df["tp"] = forcing_df["tp"] * tp_factor
            station_id = codec_station_by_goal[goal]
            codec_ds = xr.open_dataset(codec_dir / f"CODEC_{station_id}.nc").sel(time=forcing_df.index)
            forcing_df["zs"] = codec_ds["waterlevel"].values
            forcing_df = forcing_df.dropna()

            pcs = gp_models[goal].predict(dataset=forcing_df).values
            point_ds = pca_models[goal].inverse_transform(
                xr.Dataset({"PCs": (("case_num", "n_component"), pcs)}, coords={"case_num": forcing_df.index})
            )
            n_stations = point_ds.sizes["stations"]
            ds = xr.Dataset(
                {
                    "hs": (("time", "stations"), point_ds.point_hm0.values),
                    "tp": (("time", "stations"), point_ds.point_tp.values),
                    "dp": (
                        ("time", "stations"),
                        get_degrees_from_uv(point_ds.point_wavdir_u.values, point_ds.point_wavdir_v.values),
                    ),
                    "spr": (
                        ("time", "stations"),
                        np.repeat(forcing_df["ds"].to_numpy()[:, np.newaxis], n_stations, axis=1),
                    ),
                },
                coords={"time": forcing_df.index},
            ).expand_dims({"goal": [goal], "partition": [partition]})
            ds["hs"] = ds["hs"] * forcing_ds["hs"].sel(time=forcing_df.index)
            out.append(ds)
    return xr.combine_by_coords(out, combine_attrs="drop_conflicts").sortby("goal")


__all__ = ["PARTITION_VARIABLES", "fit", "reconstruct"]
