"""Case output sites: the SnapWave output-site table and site groups.

Sites are where nearshore waves are computed and validated. They come from
``simulations.output_sites_config`` in the case JSON (:func:`compute_output_sites`):

1. isobath points (depth contours of ``paths.topobathy_file``),
2. mesh nodes inside named polygons (e.g. Apia),
3. buoys from the onshore-waves GeoJSON.

The point extraction itself is :mod:`bluemath_tk.waves.hywaves.output_points`.
Once SnapWave has run, every site carries metadata coordinates (``SITE_META``:
source, label, depth, lon/lat) used to group sites for metrics and plots
(:func:`metric_groups`, :func:`find_group`, :func:`resolve_sites`).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import geopandas as gpd
import pandas as pd
import rioxarray
import shapely
import xarray as xr
from bluemath_tk.waves.hywaves.output_points import (
    get_isobath_points,
    get_mesh_nodes_in_polygon,
)
from shapely.geometry import MultiPolygon, Polygon

from utils.config import OUTPUT_SITES_CONFIG, PATHS, resolve
from utils import onshore_sites as obs

# ---------------------------------------------------------------------------
# Output-site table (SnapWave observation points)
# ---------------------------------------------------------------------------

def compute_output_sites(
    *,
    config: Mapping[str, Any] | None = None,
    topobathy: xr.DataArray | None = None,
    aoi_polygon: Polygon | MultiPolygon | None = None,
) -> pd.DataFrame:
    """
    Build the SnapWave output-sites table from config and static inputs.

    Parameters
    ----------
    config
        Override or extend ``simulations.output_sites_config`` from
        the case config JSON.
    topobathy
        Bathymetry raster; opened from ``paths.topobathy_file`` when omitted.
    aoi_polygon
        AoI geometry for clipping; read from ``paths.aoi_file`` when omitted
        and ``clip_to_aoi`` is enabled.

    Returns
    -------
    pandas.DataFrame
        Columns ``lon``, ``lat``, ``source``, ``source_id``, ``source_label``,
        ``depth``, ``name``. Row order: isobaths → mesh polygons → buoys.

    Raises
    ------
    ValueError
        If no sites are produced or all are clipped by the AoI.
    """
    cfg = {**OUTPUT_SITES_CONFIG, **dict(config or {})}
    rows: list[dict[str, Any]] = []

    clip_to_aoi = bool(cfg.get("clip_to_aoi", True))
    if clip_to_aoi and aoi_polygon is None:
        aoi_polygon = gpd.read_file(resolve(PATHS["aoi_file"])).geometry.iloc[0]

    iso_specs = cfg.get("isobaths") or []
    if iso_specs:
        if topobathy is None:
            topobathy = rioxarray.open_rasterio(
                resolve(PATHS["topobathy_file"])
            ).squeeze("band", drop=True)
        for spec in iso_specs:
            name = spec.get("name") or f"{abs(spec['level']):g}m"
            pts = get_isobath_points(
                topobathy,
                level=float(spec["level"]),
                spacing=spec.get("spacing"),
                aoi_polygon=aoi_polygon,
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

    poly_specs = cfg.get("polygons") or []
    if poly_specs:
        mesh = xr.open_dataset(resolve(PATHS["mesh_file"]))
        try:
            for spec in poly_specs:
                name = spec.get("name")
                if not name:
                    raise ValueError("Each entry in `polygons` must have a `name`.")
                poly_path = spec.get("file")
                if not poly_path:
                    raise ValueError(
                        f"Polygon {name!r} is missing `file` in output_sites_config."
                    )
                polygon = gpd.read_file(resolve(poly_path)).geometry.iloc[0]
                nodes = get_mesh_nodes_in_polygon(
                    polygon,
                    mesh,
                    wet_only=bool(spec.get("wet_only", False)),
                )
                for node in nodes:
                    i = node["node_index"]
                    rows.append(
                        {
                            "lon": node["lon"],
                            "lat": node["lat"],
                            "source": "mesh",
                            "source_id": f"{name}_{i}",
                            "source_label": name,
                            "depth": node["depth"],
                            "name": None,
                        }
                    )
        finally:
            mesh.close()

    if cfg.get("include_buoys", True):
        for iid, site in obs.get_coordinates(role=cfg.get("buoy_role")).items():
            rows.append(
                {
                    "lon": float(site["lon"]),
                    "lat": float(site["lat"]),
                    "source": "buoy",
                    "source_id": f"buoy_{iid}",
                    "source_label": site.get("type") or "buoy",
                    "depth": site.get("depth"),
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

    return df


# ---------------------------------------------------------------------------
# Site metadata, groups and selection
# ---------------------------------------------------------------------------

SITE_META = (
    "site_source",
    "site_label",
    "site_depth",
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
