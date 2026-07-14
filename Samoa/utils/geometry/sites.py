"""SnapWave output-site metadata, grouping, and selection.

Site groups (isobath depth, mesh polygon, buoy id) are shared by
:mod:`utils.model.metamodel`, :mod:`utils.evaluation.validation`, and the
validation notebooks.
"""

from __future__ import annotations

from typing import Iterable

import xarray as xr

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
    "find_group",
    "metric_groups",
    "per_site_coord",
    "resolve_sites",
]
