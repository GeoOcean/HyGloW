"""Regenerate presentation/data/data.js from the cases' published portal bundles.

Reads ONLY each case's ``{Case}/portal/`` (the same contract files the GUI
reads) plus ``GUI/catalog/cases.yaml`` for the case list, so the deck's
numbers move whenever a case is re-published with ``python -m hywaves publish``.

Standard library only:

    python3 presentation/data/build_data.py
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "data.js"

# Deck display order + the human-friendly region label for each case.
REGIONS = {
    "NewSouthWales": "New South Wales, Australia",
    "Netherlands": "Dutch coast, North Sea",
    "Samoa": "Samoan archipelago, Pacific",
}

HINDCAST_MODEL = "WHACS (closest goal)"
HYGLOW_MODEL = "Spectral metamodel"


def catalog_cases() -> list[dict]:
    """Minimal parse of GUI/catalog/cases.yaml: case_dir, continent, mda_candidate."""
    text = (REPO / "GUI" / "catalog" / "cases.yaml").read_text()
    cases, cur = [], None
    for line in text.splitlines():
        if m := re.match(r"\s*-\s+id:\s*(\S+)", line):
            cur = {"id": m.group(1)}
            cases.append(cur)
        elif cur is not None and (m := re.match(r"\s+(case_dir|continent|mda_candidate):\s*(\S+)", line)):
            cur[m.group(1)] = m.group(2).strip('"')
    return cases


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def convergence(portal: Path) -> dict:
    """Median holdout RMSE (metamodel vs SnapWave) per MDA size, isobath sites."""
    rows = [r for r in read_csv(portal / "validation_metrics.csv") if r["site_source"] == "isobath"]
    sizes = sorted({int(r["mda_candidate"]) for r in rows})
    out = {"mda": sizes, "n_goals": len({r["goal"] for r in rows})}
    for var in ("hs", "dir"):
        out[var] = [
            median([fnum(r["rmse"]) for r in rows if r["variable"] == var and int(r["mda_candidate"]) == n])
            for n in sizes
        ]
    return out


def buoy_summary(portal: Path) -> dict | None:
    """Median RMSE across buoys, hindcast vs HyWaves (spectral metamodel). Prefers nearshore buoys."""
    folder = portal / "buoy_validation"
    files = sorted(folder.glob("onshore_summary_*.csv")) or sorted(folder.glob("offshore_summary_*.csv"))
    if not files:
        return None
    rows = read_csv(files[0])
    kind = "nearshore" if files[0].name.startswith("onshore") else "offshore"

    def stat(model: str, var: str) -> float | None:
        return median([fnum(r["rmse"]) for r in rows if r["model"] == model and r["variable"] == var])

    per_buoy = {}
    for r in rows:
        if r["variable"] not in ("hs", "dir") or r["model"] not in (HINDCAST_MODEL, HYGLOW_MODEL):
            continue
        entry = per_buoy.setdefault(r["group"], {})
        for metric in ("rmse", "r2"):
            v = fnum(r[metric])
            entry[(r["variable"], r["model"], metric)] = round(v, 3) if v is not None else None

    def figure(group: str) -> str | None:
        fig = folder / "figures" / f"{group}.png"
        return f"../{portal.parent.name}/portal/buoy_validation/figures/{fig.name}" if fig.exists() else None

    buoys = [
        {
            "buoy": g.removeprefix("buoy_").removesuffix("_Buoy"),
            "hindcast": v.get(("hs", HINDCAST_MODEL, "rmse")),
            "hyglow": v.get(("hs", HYGLOW_MODEL, "rmse")),
            "hindcast_r2": v.get(("hs", HINDCAST_MODEL, "r2")),
            "hyglow_r2": v.get(("hs", HYGLOW_MODEL, "r2")),
            "dir_hindcast": v.get(("dir", HINDCAST_MODEL, "rmse")),
            "dir_hyglow": v.get(("dir", HYGLOW_MODEL, "rmse")),
            "figure": figure(g),
        }
        for g, v in sorted(per_buoy.items())
    ]

    return {
        "kind": kind,
        "period": rows[0]["period"],
        # buoys with an Hs comparison on both sides -- what the medians cover
        "n_buoys": sum(b["hindcast"] is not None and b["hyglow"] is not None for b in buoys),
        "hs": {"hindcast": stat(HINDCAST_MODEL, "hs"), "hyglow": stat(HYGLOW_MODEL, "hs")},
        "dir": {"hindcast": stat(HINDCAST_MODEL, "dir"), "hyglow": stat(HYGLOW_MODEL, "dir")},
        "per_buoy": buoys,
    }


def linear_summation(portal: Path) -> dict | None:
    """Domain-wide reconstruction vs dynamic SnapWave: Hs R2 per depth contour."""
    files = sorted((portal / "metamodel_validation").glob("linear_summation_*.csv"))
    if not files:
        return None
    rows = [r for r in read_csv(files[0]) if r["site_source"] == "isobath"]
    hs = [r for r in rows if r["variable"] == "hs"]
    return {
        "period": hs[0]["period"] if hs else None,
        "n_sites": int(hs[0]["n_sites"]) if hs else None,
        "hs_r2": {r["group"]: round(fnum(r["r2"]), 3) for r in hs},
        # {var: {contour: {r2, rmse}}} for hs / dir
        "pooled": {
            var: {
                r["group"]: {"r2": round(fnum(r["r2"]), 3), "rmse": round(fnum(r["rmse"]), 3)}
                for r in rows
                if r["variable"] == var
            }
            for var in ("hs", "dir")
        },
    }


def quantile(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def per_site(case_dir: str) -> dict | None:
    """Per-site RMSE spread for hs / dir (metamodel vs dynamic SnapWave), from the
    hand-run snapshot in presentation/data/persite/ (see compute_persite.py there)."""
    path = HERE / "persite" / f"{case_dir}_linear_summation.csv"
    if not path.exists():
        return None
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in read_csv(path):
        if r["variable"] in ("hs", "dir"):
            groups.setdefault((r["variable"], r["group"]), []).append(r)
    out: dict[str, dict] = {}
    for (var, g), rows in sorted(groups.items()):
        err = sorted(float(r["rmse"]) for r in rows)
        cut = quantile(err, 0.75)
        worst = [float(r["bias"]) for r in rows if float(r["rmse"]) >= cut]
        out.setdefault(var, {})[g] = {
            "n": len(rows),
            "q": [round(quantile(err, q), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)],
            "best": round(err[0], 3),
            # worst quarter of sites by RMSE: how many, and their median bias
            "n_worst": len(worst),
            "bias_worst": median(worst),
            "bias_rest": median([float(r["bias"]) for r in rows if float(r["rmse"]) < cut]),
        }
    return out


def per_site_vs_pooled(case_dir: str, portal: Path) -> dict | None:
    """per_site() plus, per contour, the share of sites beating the pooled RMSE."""
    ps = per_site(case_dir)
    pooled = (linear_summation(portal) or {}).get("pooled", {})
    if not ps:
        return ps
    rows = read_csv(HERE / "persite" / f"{case_dir}_linear_summation.csv")
    for var, groups in ps.items():
        for g, entry in groups.items():
            ref = pooled.get(var, {}).get(g)
            if ref is None:
                continue
            err = [float(r["rmse"]) for r in rows if r["variable"] == var and r["group"] == g]
            entry["frac_below_pooled"] = round(sum(e < ref["rmse"] for e in err) / len(err), 3)
    return ps


def case_entry(case_dir: str, continent: str) -> dict:
    portal = REPO / case_dir / "portal"
    goals = json.loads((portal / "goals.json").read_text())["goals"]
    obs = json.loads((portal / "observations.json").read_text())
    meta = json.loads((portal / "case.json").read_text())
    samples = meta.get("simulations", {}).get("num_samples_per_goal")
    return {
        "id": case_dir,
        "region": REGIONS.get(case_dir, case_dir),
        "continent": continent,
        "n_goals": len(goals),
        "n_observations": len(obs.get("features", [])),
        "samples_per_goal": samples,
        "n_training_runs": samples * len(goals) if samples else None,
        "figures": {
            p.stem: f"../{case_dir}/portal/gui_figures/{p.name}"
            for p in sorted((portal / "gui_figures").glob("*.png"))
        },
        "convergence": convergence(portal),
        "buoys": buoy_summary(portal),
        "linear_summation": linear_summation(portal),
        "per_site": per_site_vs_pooled(case_dir, portal),
    }


def main() -> None:
    cases = []
    for c in catalog_cases():
        case_dir = c.get("case_dir")
        if case_dir and (REPO / case_dir / "portal" / "goals.json").exists():
            cases.append(case_entry(case_dir, c.get("continent", "")))
    order = list(REGIONS)
    cases.sort(key=lambda c: order.index(c["id"]) if c["id"] in order else len(order))

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cases": cases,
        "totals": {
            "cases": len(cases),
            "goals": sum(c["n_goals"] for c in cases),
            "observations": sum(c["n_observations"] for c in cases),
            "training_runs": sum(c["n_training_runs"] or 0 for c in cases),
        },
    }
    OUT.write_text(
        "// GENERATED by presentation/data/build_data.py -- do not hand-edit.\n"
        f"window.DECK_DATA = {json.dumps(data, indent=2)};\n"
    )
    print(f"wrote {OUT.relative_to(REPO)}: {len(cases)} cases")


if __name__ == "__main__":
    main()
