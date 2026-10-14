"""Case output sites: the model output-site table and site groups.

Sites are where nearshore waves are computed and validated. They come from
``simulations.output_sites_config`` in the case JSON (:func:`compute_output_sites`):

1. isobath points: smoothed depth contours of ``paths.topobathy_file``, with
   the raster blanked outside the model mesh (no tiff: :func:`topobathy_from_mesh`),
2. model-mesh nodes inside named polygons (e.g. Apia), one site per node,
3. buoys from the onshore-waves GeoJSON.

Every site then gets its depth from the model mesh (``mesh_depth``, m
positive down: the depth the models see and the one used from here on; the
contour level or the buoy ``measured_depth`` stays as ``nominal_depth``) and a
``status``: ``ok`` only if it is inside the mesh and wet at the lowest
metamodel water level (``simulations.bounds.wl[0]``, ``simulations.min_depth``).
The table, dropped sites included, is saved per model as
``outputs/output_sites_{model}.csv`` (:func:`output_sites`, built in notebook
03 with :func:`sites_summary` / :func:`buoy_depth_check`) and every step reads
its ``ok`` sites from there.

The point extraction itself is :mod:`bluemath_tk.waves.hywaves.output_points`.
Once the model has run, every site carries metadata coordinates (``SITE_META``:
source, label, depth, lon/lat) used to group sites for metrics and plots
(:func:`metric_groups`, :func:`find_group`, :func:`resolve_sites`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import shapely
import xarray as xr
from bluemath_tk.topo_bathy.mesh import (
    depth_at_points,
    mask_raster_to_mesh,
    mesh_to_raster,
    read_mesh,
)
from bluemath_tk.waves.hywaves.output_points import (
    SITE_CAN_DRY,
    SITE_OK,
    SITE_OUTSIDE,
    get_isobath_points,
    site_status,
    spatial_subsample,
)
from shapely.geometry import MultiPolygon, Polygon

from utils.config import (
    OUTPUT_SITES_CONFIG,
    PATHS,
    SIMULATIONS,
    model_config,
    resolve,
    simulation_model,
)
from utils import onshore_sites as obs

logger = logging.getLogger("hywaves.geometry")

#: Default minimum depth the models compute (SnapWave ``hmin``, SWAN ``DEPMIN``).
DEFAULT_MIN_DEPTH = 0.1

# ---------------------------------------------------------------------------
# Output-site table (SnapWave observation points)
# ---------------------------------------------------------------------------

def min_depth() -> float:
    """``simulations.min_depth`` (m): the shallowest depth the models compute."""
    return float(SIMULATIONS.get("min_depth", DEFAULT_MIN_DEPTH))


def lowest_water_level() -> float:
    """Lowest water level of the metamodel cases (``simulations.bounds.wl[0]``)."""
    return float(SIMULATIONS["bounds"]["wl"][0])


def model_grid(model: str | None = None):
    """The model mesh (:class:`bluemath_tk.topo_bathy.mesh.Mesh`) of ``simulations.<model>.grid_file``."""
    return read_mesh(resolve(model_config(model)["grid_file"]))


def topobathy_from_mesh(
    model: str | None = None,
    resolution: float | None = None,
    path: str | Path | None = None,
) -> xr.DataArray:
    """
    Write ``paths.topobathy_file`` from the model mesh, for cases without a tiff.

    Parameters
    ----------
    model
        Model whose grid is rasterised; default ``simulations.model``.
    resolution
        Cell size in the mesh coordinates; default the median mesh edge.
    path
        Output GeoTIFF; default ``paths.topobathy_file``.

    Returns
    -------
    xarray.DataArray
        Elevation (m, positive up), NaN outside the mesh.
    """
    raster = mesh_to_raster(model_grid(model), resolution=resolution)
    out = resolve(path or PATHS["topobathy_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    raster.rio.to_raster(out)
    logger.info("Topobathy %s from the %s mesh: %s cells", out, simulation_model(model), raster.shape)
    return raster


def compute_output_sites(
    *,
    model: str | None = None,
    config: Mapping[str, Any] | None = None,
    topobathy: xr.DataArray | None = None,
    aoi_polygon: Polygon | MultiPolygon | None = None,
) -> pd.DataFrame:
    """
    Build the output-sites table on the model mesh from config and static inputs.

    Isobaths are traced on ``paths.topobathy_file`` blanked outside the mesh;
    every site then gets its ``mesh_depth`` and ``status``
    (:func:`check_model_sites`). Dropped sites stay in the table.

    Parameters
    ----------
    config
        Override or extend ``simulations.output_sites_config`` from
        the case config JSON. Each ``isobaths`` entry is ``{"name", "level"}``
        plus optional ``smooth_m`` (Gaussian smoothing of the bathymetry
        before contouring), ``min_length_m`` (drop shorter contour pieces) and
        ``spacing_m`` (points every that many metres along the smoothed
        contour); see :func:`bluemath_tk.waves.hywaves.output_points.get_isobath_points`.
        Each ``polygons`` entry is ``{"name", "file"}`` (a polygon file): every
        node of the model mesh inside it becomes a site ``{name}_{node}``.
        Optional ``fraction`` (in ``(0, 1]``) keeps about that share of the
        always-wet nodes, one random node per equal-area cell so the subset
        covers the whole polygon (``seed``, default 0; see
        :func:`bluemath_tk.waves.hywaves.output_points.spatial_subsample`).
    topobathy
        Bathymetry raster; opened from ``paths.topobathy_file`` when omitted.
    aoi_polygon
        AoI geometry for clipping; read from ``paths.aoi_file`` when omitted
        and ``clip_to_aoi`` is enabled.
    model
        Model whose mesh the sites must lie in; default ``simulations.model``.

    Returns
    -------
    pandas.DataFrame
        Columns ``lon``, ``lat``, ``source``, ``source_id``, ``source_label``,
        ``nominal_depth`` (contour level, polygon node depth or buoy
        ``measured_depth``, m positive down), ``mesh_depth`` (m positive down,
        NaN outside the mesh), ``name``, ``status``. Row order: isobaths →
        mesh polygons → buoys.

    Raises
    ------
    ValueError
        If no sites are produced or all are clipped by the AoI.
    """
    cfg = {**OUTPUT_SITES_CONFIG, **dict(config or {})}
    rows: list[dict[str, Any]] = []
    grid = model_grid(model)

    clip_to_aoi = bool(cfg.get("clip_to_aoi", True))
    if clip_to_aoi and aoi_polygon is None:
        aoi_polygon = gpd.read_file(resolve(PATHS["aoi_file"])).geometry.iloc[0]

    iso_specs = cfg.get("isobaths") or []
    if iso_specs:
        if topobathy is None:
            topobathy = rioxarray.open_rasterio(
                resolve(PATHS["topobathy_file"])
            ).squeeze("band", drop=True)
        topobathy = mask_raster_to_mesh(topobathy, grid)
        for spec in iso_specs:
            name = spec.get("name") or f"{abs(spec['level']):g}m"
            pts = get_isobath_points(
                topobathy,
                level=float(spec["level"]),
                aoi_polygon=aoi_polygon,
                spacing_m=spec.get("spacing_m"),
                smooth_m=spec.get("smooth_m"),
                min_length_m=spec.get("min_length_m"),
            )
            for i, (lon, lat) in enumerate(pts):
                rows.append(
                    {
                        "lon": float(lon),
                        "lat": float(lat),
                        "source": "isobath",
                        "source_id": f"{name}_{i}",
                        "source_label": name,
                        "nominal_depth": -float(spec["level"]),
                        "name": None,
                    }
                )

    for spec in cfg.get("polygons") or []:
        name = spec.get("name")
        if not name:
            raise ValueError("Each entry in `polygons` must have a `name`.")
        if not spec.get("file"):
            raise ValueError(f"Polygon {name!r} is missing `file` in output_sites_config.")
        polygon = gpd.read_file(resolve(spec["file"]))
        if grid.geographic and polygon.crs is not None:
            polygon = polygon.to_crs("EPSG:4326")
        nodes = np.flatnonzero(shapely.contains_xy(polygon.geometry.union_all(), grid.x, grid.y))
        if spec.get("fraction") is not None:
            wet = site_status(grid.depth[nodes], lowest_water_level(), min_depth()) == SITE_OK
            pick = spatial_subsample(
                grid.x[nodes[wet]],
                grid.y[nodes[wet]],
                float(spec["fraction"]),
                seed=int(spec.get("seed", 0)),
                geographic=grid.geographic,
            )
            logger.info(
                "Polygon %s: %d nodes, %d always wet -> %d sites (fraction %g)",
                name, len(nodes), int(wet.sum()), len(pick), float(spec["fraction"]),
            )
            nodes = nodes[wet][pick]
        for i in nodes:
            rows.append(
                {
                    "lon": float(grid.x[i]),
                    "lat": float(grid.y[i]),
                    "source": "mesh",
                    "source_id": f"{name}_{i}",
                    "source_label": name,
                    "nominal_depth": float(grid.depth[i]),
                    "name": None,
                }
            )

    if cfg.get("include_buoys", True):
        for iid, site in obs.get_coordinates(role=cfg.get("buoy_role")).items():
            rows.append(
                {
                    "lon": float(site["lon"]),
                    "lat": float(site["lat"]),
                    "source": "buoy",
                    "source_id": f"buoy_{iid}",
                    "source_label": site.get("type") or "buoy",
                    "nominal_depth": site.get("measured_depth"),
                    "name": site.get("name") or site.get("institution"),
                }
            )

    if not rows:
        raise ValueError("No output-site points produced.")

    df = pd.DataFrame(rows).reset_index(drop=True)

    if clip_to_aoi:
        mask = shapely.contains_xy(
            aoi_polygon, df["lon"].to_numpy(float), df["lat"].to_numpy(float)
        )
        df = df.loc[mask].reset_index(drop=True)
        if df.empty:
            raise ValueError("All output-site points were clipped by the AoI.")

    return check_model_sites(df, grid)


def check_model_sites(df: pd.DataFrame, mesh=None, model: str | None = None) -> pd.DataFrame:
    """
    Mesh depth and status of each site: inside the mesh and always wet, or not.

    Parameters
    ----------
    df
        Sites with ``lon``, ``lat``, ``source_label``.
    mesh
        Model mesh; default :func:`model_grid` of *model*.
    model
        Model name; default ``simulations.model``.

    Returns
    -------
    pandas.DataFrame
        *df* with ``mesh_depth`` (m positive down, NaN outside the mesh) and
        ``status`` (``ok``, ``outside_mesh`` or ``can_dry`` at
        ``simulations.bounds.wl[0]`` with ``simulations.min_depth``).

    Raises
    ------
    ValueError
        If no site is ``ok``.
    """
    mesh = mesh if mesh is not None else model_grid(model)
    out = df.copy()
    out["mesh_depth"] = depth_at_points(mesh, out["lon"].to_numpy(float), out["lat"].to_numpy(float))
    out["status"] = site_status(
        out["mesh_depth"].to_numpy(), wl_min=lowest_water_level(), min_depth=min_depth()
    )
    dropped = out.loc[out["status"] != SITE_OK]
    if not dropped.empty:
        counts = dropped.groupby(["source_label", "status"]).size()
        logger.info(
            "Dropped %d/%d sites (outside the mesh, or dry at wl %.2f m with min depth %.2f m): %s",
            len(dropped),
            len(out),
            lowest_water_level(),
            min_depth(),
            "; ".join(f"{label} {why} {n}" for (label, why), n in counts.items()),
        )
    if not (out["status"] == SITE_OK).any():
        raise ValueError("No output site is inside the mesh and always wet.")
    return out


def site_group(df: pd.DataFrame) -> pd.Series:
    """Summary group of each site: the isobath or polygon label, or ``buoy``."""
    return df["source_label"].where(df["source"] != "buoy", "buoy").astype(str)


def sites_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sites per group: how many are kept or dropped, and the mesh depth of the kept ones.

    Parameters
    ----------
    df
        Output-sites table with every site (``output_sites(all_sites=True)``).

    Returns
    -------
    pandas.DataFrame
        One row per group (isobaths, polygons, ``buoy``) and a ``total`` row:
        ``sites``, ``ok``, ``outside_mesh``, ``can_dry``, and ``mesh_depth``
        ``min`` / ``p10`` / ``median`` / ``p90`` / ``max`` of the ``ok`` sites.
    """
    group = site_group(df)
    counts = pd.crosstab(group, df["status"]).reindex(
        columns=[SITE_OK, SITE_OUTSIDE, SITE_CAN_DRY], fill_value=0
    )
    counts.insert(0, "sites", counts.sum(axis=1))
    ok = df["status"] == SITE_OK
    depth = df.loc[ok, "mesh_depth"].groupby(group[ok]).quantile([0, 0.1, 0.5, 0.9, 1]).unstack()
    depth.columns = ["min", "p10", "median", "p90", "max"]
    out = counts.join(depth.round(1))
    out.loc["total", counts.columns] = counts.sum()
    order = list(dict.fromkeys(group))
    return out.reindex(order + ["total"]).rename_axis("group")


def buoy_depth_check(
    df: pd.DataFrame, tolerance_m: float = 3.0, tolerance_ratio: float = 0.3
) -> pd.DataFrame:
    """
    Buoys: measured depth (metadata) against the mesh depth the models see.

    A large difference usually means wrong coordinates, a buoy on a bank or
    channel the mesh smooths out, or a different vertical datum.

    Parameters
    ----------
    df
        Output-sites table with every site (``output_sites(all_sites=True)``).
    tolerance_m, tolerance_ratio
        ``flag`` is set where ``|mesh - measured|`` exceeds both *tolerance_m*
        and *tolerance_ratio* times the measured depth.

    Returns
    -------
    pandas.DataFrame
        Per buoy: ``source_id``, ``name``, ``measured_depth``, ``mesh_depth``,
        ``difference`` (mesh - measured), ``status``, ``flag``.
    """
    buoys = df.loc[df["source"] == "buoy"]
    out = pd.DataFrame(
        {
            "source_id": buoys["source_id"],
            "name": buoys["name"],
            "measured_depth": buoys["nominal_depth"].astype(float),
            "mesh_depth": buoys["mesh_depth"].round(1),
            "status": buoys["status"],
        }
    )
    out["difference"] = (out["mesh_depth"] - out["measured_depth"]).round(1)
    out["flag"] = (out["difference"].abs() > tolerance_m) & (
        out["difference"].abs() > tolerance_ratio * out["measured_depth"]
    )
    return out[
        ["source_id", "name", "measured_depth", "mesh_depth", "difference", "status", "flag"]
    ].reset_index(drop=True)


def output_sites_path(model: str | None = None) -> Path:
    """``outputs/output_sites_{model}.csv``."""
    return resolve(f"outputs/output_sites_{simulation_model(model)}.csv")


def output_sites(
    model: str | None = None, recompute: bool = False, all_sites: bool = False
) -> pd.DataFrame:
    """
    The output sites of a model, built once (notebook 03) and saved.

    The first line of the CSV (``# grid: ...``) records the mesh; a table
    built on another grid is refused (rebuild with ``recompute=True``, which
    also invalidates the model runs made with the old sites).

    Parameters
    ----------
    model
        Model name; default ``simulations.model``.
    recompute
        Rebuild with :func:`compute_output_sites` and overwrite the file.
    all_sites
        Also return the dropped sites (``status`` other than ``ok``).

    Returns
    -------
    pandas.DataFrame
        See :func:`compute_output_sites`; only ``ok`` sites unless *all_sites*.
    """
    path = output_sites_path(model)
    grid = str(model_config(model)["grid_file"])
    if path.is_file() and not recompute:
        with open(path) as f:
            saved = f.readline().removeprefix("# grid:").strip()
        if saved != grid:
            raise ValueError(
                f"{path.name} was built on {saved!r}, the model grid is {grid!r}; "
                "rebuild it with output_sites(recompute=True)."
            )
        df = pd.read_csv(path, comment="#", dtype={"source_id": str, "source_label": str})
    else:
        df = compute_output_sites(model=model)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(f"# grid: {grid}\n")
            df.to_csv(f, index=False)
        logger.info(
            "Output sites (%s): %d of %d ok -> %s",
            simulation_model(model),
            int((df["status"] == SITE_OK).sum()),
            len(df),
            path,
        )
    if all_sites:
        return df
    return df.loc[df["status"] == SITE_OK].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Site metadata, groups and selection
# ---------------------------------------------------------------------------

SITE_META = (
    "site_source",
    "site_label",
    "site_mesh_depth",
    "site_nominal_depth",
    "site_longitude",
    "site_latitude",
)


def per_site_coord(ds: xr.Dataset, name: str, default: str = "") -> dict[str, str]:
    """
    Map each ``sites`` label to a string coordinate value.

    Parameters
    ----------
    ds
        Dataset with a ``sites`` dimension.
    name
        Coordinate name to read (e.g. ``site_source``).
    default
        Value used when *name* is missing from *ds*.

    Returns
    -------
    dict[str, str]
        ``{site_id: value}`` for every site in *ds*.
    """
    if name not in ds.coords:
        return {str(s): default for s in ds.sites.values}
    values = ds[name].values
    labels = [str(s) for s in ds.sites.values]
    if values.ndim == 0:
        return {s: str(values) for s in labels}
    return {s: str(v) for s, v in zip(labels, values, strict=True)}


def _isobath_sort_key(label: str) -> float:
    """Sort isobath labels by numeric depth (``10m`` → 10)."""
    try:
        return float(str(label).rstrip("m"))
    except ValueError:
        return float("inf")


def metric_groups(ds: xr.Dataset) -> list[tuple[str, str, list[str]]]:
    """
    Validation groups as ``(name, site_source, site_ids)``.

    Parameters
    ----------
    ds
        Dataset with ``sites`` and site metadata coordinates.

    Returns
    -------
    list[tuple[str, str, list[str]]]
        One entry per isobath depth, mesh polygon label, or buoy site.
    """
    labels = [str(s) for s in ds.sites.values]
    source_by_site = per_site_coord(ds, "site_source", "")
    label_by_site = per_site_coord(ds, "site_label", "")

    groups: list[tuple[str, str, list[str]]] = []

    isobath_sites = [s for s in labels if source_by_site.get(s) == "isobath"]
    for iso_label in sorted(
        {label_by_site[s] for s in isobath_sites},
        key=_isobath_sort_key,
    ):
        site_ids = [s for s in isobath_sites if label_by_site[s] == iso_label]
        if site_ids:
            groups.append((iso_label, "isobath", site_ids))

    mesh_sites = [s for s in labels if source_by_site.get(s) == "mesh"]
    for mesh_label in sorted({label_by_site[s] for s in mesh_sites}):
        site_ids = [s for s in mesh_sites if label_by_site[s] == mesh_label]
        if site_ids:
            groups.append((mesh_label, "mesh", site_ids))

    for site_id in labels:
        if source_by_site.get(site_id) == "buoy" or str(site_id).startswith("buoy_"):
            groups.append((site_id, "buoy", [site_id]))

    return groups


def find_group(
    ds: xr.Dataset,
    group: str,
) -> tuple[str, str, list[str]]:
    """
    Resolve a group name to ``(name, site_source, site_ids)``.

    Parameters
    ----------
    ds
        Reference dataset for site metadata.
    group
        Group name (e.g. ``10m``, ``Apia``, ``buoy_Apolima_Buoy``).

    Returns
    -------
    tuple[str, str, list[str]]
        Group name, site source, and member site ids.

    Raises
    ------
    KeyError
        If *group* is not found.
    """
    for name, source, site_ids in metric_groups(ds):
        if name == group:
            return name, source, site_ids
    available = [g[0] for g in metric_groups(ds)]
    raise KeyError(f"Unknown group {group!r}. Available: {available}")


def resolve_sites(
    ds: xr.Dataset,
    sites: str | Iterable[str] | None = None,
) -> list[str]:
    """
    Resolve a site selector to concrete ``sites`` coordinate labels.

    Parameters
    ----------
    ds
        Dataset with a ``sites`` dimension.
    sites
        ``None`` — all sites in *ds*; a group name (``10m``, ``Apia``, …);
        an umbrella source (``buoy``, ``isobath``, ``mesh``); a site id; or a
        list mixing those forms.

    Returns
    -------
    list[str]
        Unique site labels in selection order.

    Raises
    ------
    ValueError
        If a selector string does not match any group or site id.
    """
    labels = [str(s) for s in ds.sites.values]
    if sites is None:
        return labels

    def from_selector(selector: str) -> list[str]:
        if selector in ("buoy", "isobath", "mesh"):
            return [
                sid
                for _name, source, ids in metric_groups(ds)
                if source == selector
                for sid in ids
            ]
        for name, _source, ids in metric_groups(ds):
            if name == selector:
                return list(ids)
        if selector in labels:
            return [selector]
        raise ValueError(
            f"Unknown site selector {selector!r}. "
            f"Use a group name ({[g[0] for g in metric_groups(ds)]}), "
            f"umbrella ('buoy', 'isobath', 'mesh'), or a site id."
        )

    if isinstance(sites, str):
        return from_selector(sites)

    out: list[str] = []
    seen: set[str] = set()
    for item in sites:
        for sid in resolve_sites(ds, item):
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def attach_site_meta(ds: xr.Dataset, template: xr.Dataset) -> xr.Dataset:
    """
    Copy SnapWave site metadata coordinates from *template* onto *ds*.

    Parameters
    ----------
    ds
        Dataset to annotate (e.g. offshore hindcast at buoys).
    template
        Reference dataset with ``SITE_META`` coordinates (typically LHS cases).

    Returns
    -------
    xarray.Dataset
        *ds* with shared metadata coords when sites overlap *template*.
    """
    site_ids = [str(s) for s in ds.sites.values]
    shared = [s for s in site_ids if s in template.sites.values]
    if not shared:
        return ds
    ref = template.sel(sites=site_ids)
    meta = {name: ref[name] for name in SITE_META if name in ref.coords}
    return ds.assign_coords(meta)


__all__ = [
    "SITE_META",
    "attach_site_meta",
    "compute_output_sites",
    "find_group",
    "metric_groups",
    "per_site_coord",
    "resolve_sites",
]
