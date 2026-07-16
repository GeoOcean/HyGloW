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


def load_all_metrics(case: dict) -> pd.DataFrame:
    outputs = case_root(case) / "outputs"
    frames = [pd.read_csv(path) for path in sorted(outputs.glob("goal_*/validation_metrics.csv"))]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["goal"] = out["goal"].astype(str)
    return out


def list_figures(case: dict) -> list[Path]:
    fig_dir = case_root(case) / "outputs" / "course_figures"
    if not fig_dir.is_dir():
        return []
    return sorted(fig_dir.glob("fig*.png"))


def region_bounds(case: dict, config: dict, goals: pd.DataFrame, observations: pd.DataFrame) -> dict[str, list[float]]:
    lons = list(goals["lon"]) + list(observations.get("lon", []))
    lats = list(goals["lat"]) + list(observations.get("lat", []))
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
