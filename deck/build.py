"""Build the HyGloW deck into ``deck/_site/``, a static site for GitHub Pages.

Numbers and case figures come from each case's published ``{Case}/portal/``
(the files the GUI reads) and ``GUI/catalog/cases.yaml`` (the case list);
per-audience text from ``deck/config.json``. Only the images the deck shows
are copied, as WebP no wider than ``MAX_PX``, so the site stays small.

Needs Pillow:

    python deck/build.py           # then open deck/_site/index.html
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SITE = HERE / "_site"
MAX_PX = 2000
WEBP_QUALITY = 82

# Deck display order + the human-friendly region label for each case.
REGIONS = {
    "NewSouthWales": "New South Wales, Australia",
    "Netherlands": "Belgian and Dutch coasts, North Sea",
    "Samoa": "Samoan archipelago, Pacific",
}
HYGLOW_MODEL = "Spectral metamodel"


def image(src: Path, dest: str) -> str:
    """Write ``src`` as ``_site/<dest>.webp`` (downscaled to ``MAX_PX``) and return its site path."""
    out = (SITE / dest).with_suffix(".webp")
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.thumbnail((MAX_PX, MAX_PX))
        im.save(out, "WEBP", quality=WEBP_QUALITY, method=6)
    return out.relative_to(SITE).as_posix()


def catalog_cases() -> list[dict]:
    """Minimal parse of GUI/catalog/cases.yaml: case_dir, continent."""
    text = (REPO / "GUI" / "catalog" / "cases.yaml").read_text()
    cases, cur = [], None
    for line in text.splitlines():
        if m := re.match(r"\s*-\s+id:\s*(\S+)", line):
            cur = {"id": m.group(1)}
            cases.append(cur)
        elif cur is not None and (m := re.match(r"\s+(case_dir|continent):\s*(\S+)", line)):
            cur[m.group(1)] = m.group(2).strip('"')
    return cases


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fnum(v: str | None) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pretty_period(period: str) -> str:
    """'2013-01-01-2013-12-31' -> '2013'; whole years only, anything else unchanged."""
    if m := re.fullmatch(r"(\d{4})-01-01-(\d{4})-12-31", period):
        return m[1] if m[1] == m[2] else f"{m[1]}–{m[2]}"
    return period


def median(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def convergence(portal: Path) -> dict:
    """Median holdout RMSE (metamodel vs the physics model) per MDA size, isobath sites."""
    rows = [r for r in read_csv(portal / "validation_metrics.csv") if r["site_source"] == "isobath"]
    sizes = sorted({int(r["mda_candidate"]) for r in rows})
    out = {"mda": sizes, "n_goals": len({r["goal"] for r in rows})}
    for var in ("hs", "dir"):
        out[var] = [
            median([fnum(r["rmse"]) for r in rows if r["variable"] == var and int(r["mda_candidate"]) == n])
            for n in sizes
        ]
    return out


def buoy_summary(portal: Path, case_dir: str, featured: str | None) -> dict | None:
    """Median RMSE across buoys: the offshore hindcast that forced the metamodel vs HyWaves.

    Prefers nearshore buoys. Only the ``featured`` buoy's figure is copied.
    """
    folder = portal / "buoy_validation"
    files = sorted(folder.glob("onshore_summary_*.csv")) or sorted(folder.glob("offshore_summary_*.csv"))
    if not files:
        return None
    rows = read_csv(files[0])
    hindcast_model = f"{rows[0]['offshore_name']} (closest goal)"
    models = (hindcast_model, HYGLOW_MODEL)

    per_buoy: dict[str, dict] = {}
    for r in rows:
        if r["variable"] in ("hs", "dir") and r["model"] in models:
            per_buoy.setdefault(r["group"], {})[(r["variable"], r["model"])] = fnum(r["rmse"])

    def rnd(v: float | None) -> float | None:
        return round(v, 3) if v is not None else None

    buoys = []
    for group, v in sorted(per_buoy.items()):
        name = group.removeprefix("buoy_").removesuffix("_Buoy")
        fig = folder / "figures" / f"{group}.png"
        buoys.append({
            "buoy": name,
            "hindcast": rnd(v.get(("hs", hindcast_model))),
            "hyglow": rnd(v.get(("hs", HYGLOW_MODEL))),
            "dir_hindcast": rnd(v.get(("dir", hindcast_model))),
            "dir_hyglow": rnd(v.get(("dir", HYGLOW_MODEL))),
            "figure": image(fig, f"cases/{case_dir}/buoy_{name}") if name == featured and fig.exists() else None,
        })

    def stat(var: str, model: str) -> float | None:
        return median([fnum(r["rmse"]) for r in rows if r["model"] == model and r["variable"] == var])

    spectral = [r for r in rows if r["model"] == HYGLOW_MODEL]
    return {
        "kind": "nearshore" if files[0].name.startswith("onshore") else "offshore",
        "period": pretty_period(rows[0]["period"]),
        "hindcast_name": rows[0]["offshore_name"],
        "mda": int(rows[0]["mda_candidate"]),
        # buoys with an Hs comparison on both sides -- what the medians cover
        "n_buoys": sum(b["hindcast"] is not None and b["hyglow"] is not None for b in buoys),
        "reconstruct_seconds": rnd(fnum(spectral[0].get("reconstruct_seconds"))) if spectral else None,
        "hs": {"hindcast": stat("hs", hindcast_model), "hyglow": stat("hs", HYGLOW_MODEL)},
        "dir": {"hindcast": stat("dir", hindcast_model), "hyglow": stat("dir", HYGLOW_MODEL)},
        "per_buoy": buoys,
    }


def case_entry(case_dir: str, continent: str, config: dict) -> dict:
    portal = REPO / case_dir / "portal"
    goals = json.loads((portal / "goals.json").read_text())["goals"]
    obs = json.loads((portal / "observations.json").read_text())
    sites = json.loads((portal / "output_sites.json").read_text())
    featured = config["featuredBuoy"] if case_dir == config["mainCase"] else None
    buoys = buoy_summary(portal, case_dir, featured)
    mda = buoys["mda"] if buoys else None
    gui = HERE / "assets" / "gui" / case_dir if case_dir == config["mainCase"] else None
    return {
        "id": case_dir,
        "region": REGIONS.get(case_dir, case_dir),
        "continent": continent,
        "n_goals": len(goals),
        "n_observations": len(obs["features"]),
        "n_sites": len(sites["lon"]),
        # runs per goal behind the published results (the MDA size used)
        "runs_per_goal": mda,
        "n_training_runs": mda * len(goals) if mda else None,
        "figures": {
            p.stem: image(p, f"cases/{case_dir}/{p.stem}") for p in sorted((portal / "gui_figures").glob("*.png"))
        },
        "gui": {p.stem: image(p, f"assets/gui/{case_dir}/{p.stem}") for p in sorted(gui.glob("*.png"))} if gui else {},
        "convergence": convergence(portal),
        "buoys": buoys,
    }


def build_data(config: dict) -> dict:
    cases = [
        case_entry(c["case_dir"], c.get("continent", ""), config)
        for c in catalog_cases()
        if (REPO / c["case_dir"] / "portal" / "goals.json").exists()
    ]
    order = list(REGIONS)
    cases.sort(key=lambda c: order.index(c["id"]) if c["id"] in order else len(order))
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "cases": cases,
        "totals": {
            "cases": len(cases),
            "goals": sum(c["n_goals"] for c in cases),
            # buoys in the published Hs comparison, not every observation site
            "buoys": sum(c["buoys"]["n_buoys"] for c in cases if c["buoys"]),
        },
    }


def build_html() -> None:
    """Copy index.html with its assets as WebP and every image lazy-loaded."""
    html = (HERE / "index.html").read_text()
    html = re.sub(
        r'src="(assets/[^"]+\.(?:png|jpe?g))"',
        lambda m: f'src="{image(HERE / m.group(1), str(Path(m.group(1)).with_suffix("")))}"',
        html,
    )
    html = html.replace("<img ", '<img loading="lazy" decoding="async" ')
    (SITE / "index.html").write_text(html)


def main() -> None:
    shutil.rmtree(SITE, ignore_errors=True)
    SITE.mkdir()
    config = json.loads((HERE / "config.json").read_text())
    data = build_data(config)
    if not any(c["id"] == config["mainCase"] for c in data["cases"]):
        raise SystemExit(f"mainCase {config['mainCase']!r} has no published portal/")
    (SITE / "data.js").write_text(
        "// GENERATED by deck/build.py -- do not hand-edit.\n"
        f"window.DECK_CONFIG = {json.dumps(config, ensure_ascii=False)};\n"
        f"window.DECK_DATA = {json.dumps(data, ensure_ascii=False)};\n"
    )
    for name in ("style.css", "deck.js"):
        shutil.copy2(HERE / name, SITE / name)
    build_html()
    (SITE / ".nojekyll").touch()
    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file())
    print(f"wrote {SITE.relative_to(REPO)}: {len(data['cases'])} cases, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
