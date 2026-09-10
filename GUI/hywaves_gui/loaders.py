from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .paths import CATALOG_PATH, case_root, resolve_case


def load_catalog() -> list[dict]:
    with CATALOG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["cases"]


def load_case(case_id: str) -> dict:
    for case in load_catalog():
        if case["id"] == case_id:
            return case
    raise KeyError(f"Unknown case: {case_id}")


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


def find_case_config(case: dict) -> Path:
    """Locate ``{Case}/{Case}.json`` (case-insensitive fallback)."""
    root = case_root(case)
    name = root.name
    for candidate in (root / f"{name}.json", root / f"{name.lower()}.json"):
        if candidate.is_file():
            return candidate
    configs = sorted(p for p in root.glob("*.json") if "Goals" not in p.name and "observations" not in p.name.lower())
    if len(configs) == 1:
        return configs[0]
    raise FileNotFoundError(f"No case config JSON found under {root}")


def load_case_config(case: dict) -> dict:
    return json.loads(find_case_config(case).read_text(encoding="utf-8"))


def _goal_lon_lat(goal: dict) -> tuple[float, float]:
    if "target" in goal:
        return goal["target"]["longitude"], goal["target"]["latitude"]
    return goal["lon"], goal["lat"]


def load_goals(case: dict, config: dict | None = None) -> pd.DataFrame:
    config = config or load_case_config(case)
    sim = config.get("simulations", {})
    if goals_file := sim.get("goals_file"):
        data = json.loads(resolve_case(case, goals_file).read_text(encoding="utf-8"))
        goals = data.get("goals", data)
    else:
        goals = config.get("goals", {})

    rows = [{"goal": str(gid), "lon": lon, "lat": lat} for gid, g in goals.items() for lon, lat in [_goal_lon_lat(g)]]
    return pd.DataFrame(rows)


def load_observations(case: dict, config: dict | None = None) -> pd.DataFrame:
    config = config or load_case_config(case)
    onshore = config.get("onshore_waves")
    if not isinstance(onshore, str):
        return pd.DataFrame(columns=["name", "instrument_id", "lon", "lat", "institution", "recording"])

    path = resolve_case(case, onshore)
    if not path.is_file():
        return pd.DataFrame(columns=["name", "instrument_id", "lon", "lat", "institution", "recording"])

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        return pd.DataFrame(columns=["name", "instrument_id", "lon", "lat", "institution", "recording"])

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
    return pd.DataFrame(rows)


def load_aoi(case: dict, config: dict | None = None) -> list[list[tuple[float, float]]]:
    """Rings of the case AoI polygon(s) from ``paths.aoi_file`` (EPSG:4326).

    Returns an empty list when the AoI is not declared, missing, or pyshp
    is unavailable — the map simply omits the layer.
    """
    config = config or load_case_config(case)
    aoi_rel = config.get("paths", {}).get("aoi_file")
    if not isinstance(aoi_rel, str) or not aoi_rel:
        return []
    path = resolve_case(case, aoi_rel)
    if not path.is_file():
        return []
    try:
        import shapefile
    except ImportError:
        return []

    rings: list[list[tuple[float, float]]] = []
    for shape in shapefile.Reader(str(path)).shapes():
        breaks = list(shape.parts) + [len(shape.points)]
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            ring = [(float(x), float(y)) for x, y in shape.points[lo:hi]]
            if len(ring) >= 3:
                rings.append(ring)
    return rings


_RASTER_X_NAMES = ("x", "lon", "longitude")
_RASTER_Y_NAMES = ("y", "lat", "latitude")

# Display-only resampling for isobath lines: independent of the case config's
# per-isobath `spacing` (that one spaces discrete SnapWave output sites along
# the contour; this one just keeps the drawn line smooth and light).
ISOBATH_DISPLAY_SPACING_DEG = 0.02


def _resample_polyline(points, spacing: float):
    """Resample an (N, 2) polyline at approximately uniform arc length."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 0.0:
        return pts
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    ds = np.arange(0.0, total, float(spacing))
    if ds.size == 0 or (total - ds[-1]) > 0.25 * spacing:
        ds = np.append(ds, total)
    out = np.empty((ds.size, 2), dtype=float)
    for k, d in enumerate(ds):
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = max(0, min(i, len(seg_len) - 1))
        t = 0.0 if seg_len[i] == 0 else (d - cum[i]) / seg_len[i]
        out[k] = pts[i] * (1.0 - t) + pts[i + 1] * t
    return out


def load_isobaths(case: dict, config: dict | None = None) -> list[dict]:
    """Depth-contour lines extracted directly from the topobathy raster.

    One entry per ``simulations.output_sites_config.isobaths`` spec:
    ``{"name", "level", "segments"}`` where ``segments`` is a list of
    ``[(lon, lat), ...]`` rings (a level can have several disjoint contours,
    e.g. one per island). Returns ``[]`` when the raster/config is missing
    or rioxarray/contourpy are unavailable — the map simply omits the layer.
    """
    config = config or load_case_config(case)
    topobathy_rel = config.get("paths", {}).get("topobathy_file")
    specs = config.get("simulations", {}).get("output_sites_config", {}).get("isobaths") or []
    if not isinstance(topobathy_rel, str) or not specs:
        return []
    path = resolve_case(case, topobathy_rel)
    if not path.is_file():
        return []
    try:
        import contourpy
        import numpy as np
        import rioxarray
    except ImportError:
        return []

    da = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    xname = next((n for n in _RASTER_X_NAMES if n in da.coords), None)
    yname = next((n for n in _RASTER_Y_NAMES if n in da.coords), None)
    if xname is None or yname is None:
        return []

    cg = contourpy.contour_generator(
        x=da[xname].values,
        y=da[yname].values,
        z=np.asarray(da.values, dtype=float),
        line_type=contourpy.LineType.Separate,
    )

    isobaths: list[dict] = []
    for spec in specs:
        level = spec.get("level")
        if level is None:
            continue
        name = str(spec.get("name") or f"{abs(level):g}m")
        segments = [s for s in cg.lines(float(level)) if len(s) >= 2]
        segments = [_resample_polyline(s, ISOBATH_DISPLAY_SPACING_DEG) for s in segments]
        isobaths.append(
            {
                "name": name,
                "level": float(level),
                "segments": [[(float(x), float(y)) for x, y in seg] for seg in segments],
            }
        )
    return isobaths


def load_output_sites(case: dict, config: dict | None = None) -> pd.DataFrame:
    """Isobath/mesh output-site coordinates from one metamodel LHS case file.

    Site layout (isobaths, mesh polygons) is shared across every goal, so
    any single ``output_sites_*.nc`` gives the full case-wide table without
    opening one per goal. Buoy sites are excluded — already shown via
    :func:`load_observations`. Returns an empty frame when the metamodel
    cases folder, xarray, or a matching file is unavailable.
    """
    empty = pd.DataFrame(columns=["lon", "lat", "source", "label"])
    config = config or load_case_config(case)
    cases_rel = config.get("simulations", {}).get("metamodel_cases")
    if not isinstance(cases_rel, str) or not cases_rel:
        return empty
    cases_dir = resolve_case(case, cases_rel)
    if not cases_dir.is_dir():
        return empty
    nc_files = sorted(cases_dir.glob("output_sites_*.nc"))
    if not nc_files:
        return empty
    try:
        import xarray as xr
    except ImportError:
        return empty

    with xr.open_dataset(nc_files[0]) as ds:
        required = {"site_longitude", "site_latitude", "site_source"}
        if not required.issubset(ds.coords.keys() | ds.data_vars.keys()):
            return empty
        df = pd.DataFrame(
            {
                "lon": ds["site_longitude"].values,
                "lat": ds["site_latitude"].values,
                "source": ds["site_source"].values.astype(str),
                "label": ds["site_label"].values.astype(str)
                if "site_label" in ds.variables
                else "",
            }
        )
    return df[df["source"] != "buoy"].reset_index(drop=True)


def load_all_metrics(case: dict) -> pd.DataFrame:
    outputs = case_root(case) / "outputs"
    frames = [pd.read_csv(path) for path in sorted(outputs.glob("goal_*/validation_metrics.csv"))]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["goal"] = out["goal"].astype(str)
    return out


def load_metamodel_validation(case: dict) -> pd.DataFrame:
    """Linear-summation-vs-dynamic metrics (``outputs/metamodel_validation/*.csv``).

    Written by ``09_Validate_Linear_Summation.ipynb`` — one row per site
    group and variable, domain-wide (no per-goal breakdown).
    """
    outputs = case_root(case) / "outputs" / "metamodel_validation"
    frames = [pd.read_csv(path) for path in sorted(outputs.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_buoy_validation(case: dict) -> pd.DataFrame:
    """Buoy validation summaries (``outputs/buoy_validation/*_summary_*.csv``).

    Written by ``10a_Validate_Offshore_Buoys.ipynb`` / ``10b_Validate_Onshore_Buoys.ipynb``.
    Adds a ``buoy_type`` column ("offshore"/"onshore") from each file's name
    so both subsets can be shown together in one table.
    """
    outputs = case_root(case) / "outputs" / "buoy_validation"
    frames = []
    for path in sorted(outputs.glob("*_summary_*.csv")):
        df = pd.read_csv(path)
        df["buoy_type"] = path.name.split("_summary_")[0]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def available_buoy_figures(case: dict) -> set[str]:
    """Site ids with a saved summary PNG (``outputs/buoy_validation/figures/<site>.png``).

    Written by the ``val.plot_buoy_summary`` save cell in ``10a``/``10b``.
    """
    fig_dir = case_root(case) / "outputs" / "buoy_validation" / "figures"
    if not fig_dir.is_dir():
        return set()
    return {p.stem for p in fig_dir.glob("*.png")}


def case_mda_candidate(case: dict) -> int | None:
    """Fixed MDA train size for the validation tabs, from the catalog's ``metamodel.mda_candidate``.

    Falls back to ``None`` for cases that haven't pinned one yet (e.g. still
    exploring several candidates) — callers should fall back to
    :func:`production_mda` in that case.
    """
    value = case.get("metamodel", {}).get("mda_candidate")
    return int(value) if value is not None else None


def book_available(case: dict) -> bool:
    """Whether a built Jupyter Book (see ``book/``) exists for this case."""
    return (case_root(case) / "book" / "_build" / "html" / "index.html").is_file()


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
