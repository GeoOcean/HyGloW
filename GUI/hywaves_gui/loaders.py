from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .paths import CATALOG_PATH, case_root, portal_root

#: Separates a catalog case id from a snapshot tag: ``netherlands@hs1``.
VERSION_SEP = "@"


def load_catalog() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["cases"]


def case_versions(case: dict) -> list[dict]:
    """Snapshot versions of a catalog case (``{Case}/portal_<tag>/``, newest first).

    Each is the catalog entry with ``id`` = ``<case id>@<tag>``, ``portal`` =
    the snapshot folder and ``version`` = ``{"tag", "note", "date"}`` from the
    snapshot's ``manifest.json`` (see ``python -m hywaves snapshot``).
    """
    versions = []
    for folder in case_root(case).glob("portal_*"):
        manifest = folder / "manifest.json"
        if not manifest.is_file():
            continue
        info = json.loads(manifest.read_text(encoding="utf-8")).get("snapshot", {})
        tag = info.get("tag") or folder.name.removeprefix("portal_")
        versions.append(
            {
                **case,
                "id": f"{case['id']}{VERSION_SEP}{tag}",
                "portal": folder.name,
                "version": {
                    "tag": tag,
                    "note": info.get("note", ""),
                    "date": str(info.get("created", ""))[:10],
                },
            }
        )
    return sorted(versions, key=lambda v: v["version"]["date"], reverse=True)


def load_case(case_id: str) -> dict:
    """Catalog case by id, or one of its versions (``<case id>@<tag>``)."""
    base_id = case_id.split(VERSION_SEP, 1)[0]
    for case in load_catalog():
        if case["id"] != base_id:
            continue
        if base_id == case_id:
            return case
        for version in case_versions(case):
            if version["id"] == case_id:
                return version
    raise KeyError(f"Unknown case: {case_id}")


def case_options(cases: list[dict]) -> list[dict]:
    """Dropdown options: each case (latest portal), followed by its versions.

    Version labels carry the tag, note and date, so the searchable dropdown
    finds them by any of those (e.g. typing "hs1" or "unit").
    """
    options = []
    for case in cases:
        name = case.get("case_dir", case["id"])
        options.append({"label": name, "value": case["id"]})
        for v in case_versions(case):
            info = v["version"]
            note = f" — {info['note']}" if info["note"] else ""
            date = f" ({info['date']})" if info["date"] else ""
            options.append(
                {"label": f"{name} · {info['tag']}{note}{date}", "value": v["id"]}
            )
    return options


def catalog_by_continent(catalog: list[dict]) -> dict[str, list[dict]]:
    """Catalog cases grouped by ``continent`` (catalog order preserved within each group)."""
    groups: dict[str, list[dict]] = {}
    for case in catalog:
        groups.setdefault(case.get("continent", "Other"), []).append(case)
    return groups


def _load_sandbox_layer(filename: str) -> list:
    """A pre-simplified static sandbox-mode reference layer from ``catalog/``.

    Every one of these is derived offline from the full QGIS/HyWorld source
    data (too heavy to ship as-is) by ``catalog/build_sandbox_layers.py``.
    Returns ``[]`` if that step was never run for this file — the sandbox
    map still works, just without that particular reference layer.
    """
    path = CATALOG_PATH.parent / filename
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_world_coastline() -> list[list[tuple[float, float]]]:
    """Coarse world coastline rings for the sandbox-mode world view."""
    return _load_sandbox_layer("world_coastline.json")


def load_depth200_world() -> list[list[tuple[float, float]]]:
    """Simplified global 200m isobath line segments."""
    return _load_sandbox_layer("depth200_world.json")


def load_coastal_buffer_fill() -> list[list[tuple[float, float]]]:
    """Simplified 50km-from-coast buffer polygon rings."""
    return _load_sandbox_layer("coastal_buffer_fill.json")


def load_coastal_buffer_line() -> list[list[tuple[float, float]]]:
    """Simplified 50km-or-200m boundary line segments."""
    return _load_sandbox_layer("coastal_buffer_line.json")


# ----------------------------------------------------------------------
# Case data — read only from each case's published bundle, ``{Case}/portal/``
# (built by ``python -m hywaves publish``; see hywaves/contract/CONTRACT.md §6).
# Every loader degrades to an empty result when its file is absent — a
# not-yet-published case, or a layer the publish step skipped — so the GUI
# shows a "not generated yet" message instead of failing.
# ----------------------------------------------------------------------


def _portal_json(case: dict, name: str, default=None):
    path = portal_root(case) / name
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def portal_manifest(case: dict) -> dict | None:
    """The bundle's ``manifest.json`` (publish date, hywaves version, skipped layers) — ``None`` if unpublished."""
    return _portal_json(case, "manifest.json")


#: Display names of the numerical models in the portal manifest (``model``).
MODEL_LABELS = {"snapwave": "SnapWave", "swan": "SWAN"}


def case_model_label(case: dict) -> str:
    """Display name of the model that ran the case (manifest ``model``; SnapWave before 0.11.0)."""
    model = (portal_manifest(case) or {}).get("model") or "snapwave"
    return MODEL_LABELS.get(model, model)


def load_case_config(case: dict) -> dict:
    """The case config as published (``portal/case.json``).

    An unpublished case gets a stub with just its catalog id as the name,
    so the panels still render (with "not published yet" data messages).
    """
    return _portal_json(case, "case.json") or {"name": case.get("case_dir", case["id"])}


def _goal_lon_lat(goal: dict) -> tuple[float, float]:
    if "target" in goal:
        return goal["target"]["longitude"], goal["target"]["latitude"]
    return goal["lon"], goal["lat"]


def load_goals(case: dict, config: dict | None = None) -> pd.DataFrame:
    data = _portal_json(case, "goals.json", {})
    goals = data.get("goals", data)
    rows = [{"goal": str(gid), "lon": lon, "lat": lat} for gid, g in goals.items() for lon, lat in [_goal_lon_lat(g)]]
    return pd.DataFrame(rows, columns=["goal", "lon", "lat"])


def load_observations(case: dict, config: dict | None = None) -> pd.DataFrame:
    columns = ["name", "instrument_id", "lon", "lat", "institution", "recording"]
    data = _portal_json(case, "observations.json", {})
    if data.get("type") != "FeatureCollection":
        return pd.DataFrame(columns=columns)

    rows = []
    for feat in data["features"]:
        props = feat.get("properties", {})
        lon, lat = feat["geometry"]["coordinates"][:2]
        rows.append(
            {
                "name": props.get("name") or props.get("instrument_id", "site"),
                "instrument_id": props.get("instrument_id"),
                "lon": lon,
                "lat": lat,
                "institution": props.get("institution"),
                "recording": props.get("recording"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def load_aoi(case: dict, config: dict | None = None) -> list[list[tuple[float, float]]]:
    """Rings of the case AoI polygon(s) from ``portal/aoi.geojson`` (EPSG:4326).

    Every ring (exterior or hole) is returned flat — the map just outlines
    and fills them. ``[]`` when the bundle has no AoI layer.
    """
    data = _portal_json(case, "aoi.geojson", {})
    rings: list[list[tuple[float, float]]] = []
    for feat in data.get("features", []):
        geometry = feat.get("geometry") or {}
        polygons = [geometry["coordinates"]] if geometry.get("type") == "Polygon" else geometry.get("coordinates", [])
        for polygon in polygons:
            for ring in polygon:
                if len(ring) >= 3:
                    rings.append([(float(x), float(y)) for x, y, *_ in ring])
    return rings


def all_case_aois(catalog: list[dict]) -> list[dict]:
    """Every catalog case's real AoI, for the landing/sandbox world view.

    Cases with no published AoI (``load_aoi`` returns ``[]`` — including a
    case that hasn't been published at all) are skipped: they're still
    reachable via the "Browse cases" modal, just not drawn on the map.
    ``center`` is a plain arithmetic-mean centroid of the ring points (not
    area-weighted) — good enough for placing a click marker, and avoids
    pulling in shapely as a GUI runtime dependency just for this.
    """
    results = []
    for case in catalog:
        rings = load_aoi(case)
        if not rings:
            continue
        config = load_case_config(case)
        points = [p for ring in rings for p in ring]
        center = (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
        results.append(
            {
                "case_id": case["id"],
                "name": config.get("name", case["id"]),
                "continent": case.get("continent"),
                "rings": rings,
                "center": center,
            }
        )
    return results


def load_isobaths(case: dict, config: dict | None = None) -> list[dict]:
    """Depth-contour lines pre-computed from the topobathy raster at publish time.

    One entry per isobath: ``{"name", "level", "segments"}`` where
    ``segments`` is a list of ``[(lon, lat), ...]`` lines. ``[]`` when the
    bundle has no isobaths layer.
    """
    return [
        {
            "name": iso["name"],
            "level": iso["level"],
            "segments": [[(x, y) for x, y in seg] for seg in iso["segments"]],
        }
        for iso in _portal_json(case, "isobaths.json", [])
    ]


def load_output_sites(case: dict, config: dict | None = None) -> pd.DataFrame:
    """Isobath/mesh output-site coordinates (buoy sites excluded — see :func:`load_observations`)."""
    data = _portal_json(case, "output_sites.json")
    if not data:
        return pd.DataFrame(columns=["lon", "lat", "source", "label"])
    return pd.DataFrame(data, columns=["lon", "lat", "source", "label"])


def load_all_metrics(case: dict) -> pd.DataFrame:
    """Holdout metrics for every goal (``portal/validation_metrics.csv``)."""
    path = portal_root(case) / "validation_metrics.csv"
    if not path.is_file():
        return pd.DataFrame()
    out = pd.read_csv(path)
    out["goal"] = out["goal"].astype(str)
    return out


def load_metamodel_validation(case: dict) -> pd.DataFrame:
    """Linear-summation-vs-dynamic metrics (``portal/metamodel_validation/*.csv``).

    Written by ``09_Validate_Linear_Summation.ipynb`` — one row per site
    group and variable, domain-wide (no per-goal breakdown).
    """
    frames = [pd.read_csv(path) for path in sorted((portal_root(case) / "metamodel_validation").glob("*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_buoy_validation(case: dict) -> pd.DataFrame:
    """Buoy validation summaries (``portal/buoy_validation/*_summary_*.csv``).

    Written by the case's ``10*_Validate_*_Buoys.ipynb`` notebook(s) — a case
    with one buoy network only has one such file (e.g. the template's
    ``10a``/``10b`` split collapsed to a single ``10_...`` notebook), a case
    with two writes both. Adds a ``buoy_type`` column ("offshore"/"onshore")
    from each file's name so any subsets present can be shown together in one
    table.
    """
    frames = []
    for path in sorted((portal_root(case) / "buoy_validation").glob("*_summary_*.csv")):
        df = pd.read_csv(path)
        df["buoy_type"] = path.name.split("_summary_")[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def buoy_figures_dir(case: dict) -> Path:
    return portal_root(case) / "buoy_validation" / "figures"


def gui_figures_dir(case: dict) -> Path:
    return portal_root(case) / "gui_figures"


def book_dir(case: dict) -> Path:
    return portal_root(case) / "book"


def available_buoy_figures(case: dict) -> set[str]:
    """Site ids with a published summary PNG (``portal/buoy_validation/figures/<site>.png``).

    Written by the ``val.plot_buoy_summary`` save cell in ``10a``/``10b``.
    """
    return {p.stem for p in buoy_figures_dir(case).glob("*.png")}


def available_gui_figures(case: dict) -> set[str]:
    """Figure names with a published PNG (``portal/gui_figures/<name>.png``).

    Written by a case's own ``12_Final_Plots.ipynb`` — a case that hasn't run
    it yet just has none, so the "Final Plots" tab can degrade to a "not
    generated yet" message per figure instead of a broken ``<img>``.
    """
    return {p.stem for p in gui_figures_dir(case).glob("*.png")}


def case_mda_candidate(case: dict) -> int | None:
    """Fixed MDA train size for the validation tabs, from the catalog's ``metamodel.mda_candidate``.

    Falls back to ``None`` for cases that haven't pinned one yet (e.g. still
    exploring several candidates) — callers should fall back to
    :func:`production_mda` in that case.
    """
    value = case.get("metamodel", {}).get("mda_candidate")
    return int(value) if value is not None else None


def book_available(case: dict) -> bool:
    """Whether the bundle carries a built Jupyter Book (``portal/book/``, deployed by hand — not in git)."""
    return (book_dir(case) / "index.html").is_file()


def region_bounds(
    case: dict,
    config: dict,
    goals: pd.DataFrame,
    observations: pd.DataFrame,
    aoi: list[list[tuple[float, float]]] | None = None,
) -> dict[str, list[float]]:
    lons = list(goals["lon"]) + list(observations.get("lon", []))
    lats = list(goals["lat"]) + list(observations.get("lat", []))
    for ring in aoi or []:
        lons += [p[0] for p in ring]
        lats += [p[1] for p in ring]
    if not lons:
        return {"lon": [-180, 180], "lat": [-90, 90]}
    pad_lon = max(0.15, (max(lons) - min(lons)) * 0.08)
    pad_lat = max(0.15, (max(lats) - min(lats)) * 0.08)
    return {
        "lon": [min(lons) - pad_lon, max(lons) + pad_lon],
        "lat": [min(lats) - pad_lat, max(lats) + pad_lat],
    }


def mda_candidates(case: dict, config: dict, metrics: pd.DataFrame) -> list[int]:
    if not metrics.empty and "mda_candidate" in metrics.columns:
        return sorted(int(v) for v in metrics["mda_candidate"].unique())
    meta = config.get("metamodel", {})
    return [int(v) for v in meta.get("mda_candidates", [])]


def production_mda(candidates: list[int]) -> int | None:
    return candidates[-1] if candidates else None


def offshore_names(config: dict) -> list[str]:
    sources = config.get("offshore_waves", [])
    if isinstance(sources, dict):
        sources = [sources]
    return [str(s.get("name", "offshore")) for s in sources]


def metrics_variables(config: dict) -> list[str]:
    return list(config.get("metamodel", {}).get("validation", {}).get("metrics_variables", ["hs", "tp", "dir", "spr"]))
