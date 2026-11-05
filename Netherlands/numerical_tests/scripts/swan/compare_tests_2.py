"""Score the second round of SWAN tests against round 1. Run from Netherlands/:
python SWAN_test/compare_tests_2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "SWAN_test")
sys.path.insert(0, ".")
from compare_tests import ROOT, convergence, minutes, read_hs, score  # noqa: E402
from utils import model_comparison as mc  # noqa: E402
from utils import validation as val  # noqa: E402
from utils.config import load_goals  # noqa: E402

KOMEN = ROOT / "cases" / "komen"
VARIANTS = {"komen": KOMEN, **{v: ROOT / "cases_2" / v for v in ("era5cal", "era5cal_fric", "whacs_map", "whacs_map_fric")}}
CLASSES = [0, 1, 2, 3, 4, 5, 6, 12]


def load_hours(hours: pd.DataFrame, folders: dict[str, Path]) -> dict[str, np.ndarray]:
    names = [t.strftime("%Y%m%dT%H") for t in hours["time"]]
    return {
        label: np.stack([read_hs(folder / name) for name in names]) for label, folder in folders.items()
    }


def report_dynamic() -> None:
    hours = pd.read_csv(ROOT / "dynamic_hours.csv", parse_dates=["time"]).sort_values("time")
    snapwave = mc.load_snapwave_dynamic("Dynamic_Cases_WHACS_cal_2013.nc").sel(time=hours["time"].values)
    buoy = val.load_buoy_observations(slice(hours["time"].min(), hours["time"].max()), variables=["hs"])
    sites = list(buoy.sites.values)
    index = [list(snapwave.sites.values).index(s) for s in sites]
    obs = buoy["hs"].reindex(time=hours["time"].values).sel(sites=sites).transpose("time", "sites").values
    models = {"SnapWave": snapwave["hs"].sel(sites=sites).transpose("time", "sites").values}
    models |= {k: v[:, index] for k, v in load_hours(hours, VARIANTS).items()}
    depth = snapwave["site_mesh_depth"].sel(sites=sites).values
    group = hours["group"].values

    print(f"Hs vs buoys, bias/rmse (n), {len(sites)} buoys")
    print(" " * 16 + "".join(f"{name:>22}" for name in models))

    def line(label: str, mask: np.ndarray) -> None:
        print(f"{label:16s}" + "".join(f"{score(obs, v, mask):>22}" for v in models.values()))

    for name in ("calm", "storm", "xaver"):
        line(name, np.repeat((group == name)[:, None], len(sites), axis=1))
    print()
    for low, high in zip(CLASSES[:-1], CLASSES[1:]):
        line(f"obs {low}-{high} m", (obs >= low) & (obs < high))
    print()
    for low, high in [(0, 15), (15, 25), (25, 60)]:
        shallow = (depth >= low)[None, :] & (depth < high)[None, :]
        line(f"obs<3, d {low}-{high}", (obs < 3) & shallow)
        line(f"obs>=3, d {low}-{high}", (obs >= 3) & shallow)

    for label, folder in VARIANTS.items():
        table = pd.DataFrame(
            [(*convergence(folder / t.strftime("%Y%m%dT%H")), minutes(folder / t.strftime("%Y%m%dT%H"))) for t in hours["time"]],
            columns=["iterations", "accuracy", "minutes"],
        )
        print(f"{label}: hit limit {(table.iterations >= 50).sum()}/{len(table)}, "
              f"min accuracy {table.accuracy.min():.1f} %, median {table.minutes.median():.0f} min")

    calm = hours.query("group == 'calm'")
    names = [t.strftime("%Y%m%dT%H") for t in calm["time"]]
    print("\ncalm hours, 50 vs 100 iterations:")
    rows = [(*convergence(KOMEN / n), *convergence(ROOT / "cases_2" / "iter100" / n), minutes(ROOT / "cases_2" / "iter100" / n)) for n in names]
    print(pd.DataFrame(rows, columns=["it50", "acc50", "it100", "acc100", "min100"], index=names).round(2).to_string())
    a = np.stack([read_hs(KOMEN / n) for n in names])
    b = np.stack([read_hs(ROOT / "cases_2" / "iter100" / n) for n in names])
    print(f"Hs 100 vs 50 iterations at all sites: mean diff {np.nanmean(b - a):+.4f} m, "
          f"p99 |diff| {np.nanpercentile(abs(b - a), 99):.3f} m")


def report_metamodel() -> None:
    cases = pd.read_csv(ROOT / "metamodel_cases.csv").sort_values("case")
    snapwave = mc.load_snapwave_metamodel(load_goals()).sel(case_num=cases["case"].values)
    base = mc.load_swan(snapwave)
    variants = {
        "SWAN": base["hs"],
        **{
            label: base["hs"].copy(data=np.stack([read_hs(folder / f"{c:04}") for c in cases["case"]]))
            for label, folder in (
                ("no_wcap", ROOT / "cases" / "no_wcap"),
                ("+low fric", ROOT / "cases_2" / "friction"),
                ("+no triad", ROOT / "cases_2" / "no_triad"),
            )
        },
    }
    energy = snapwave["hs"] >= 0.25 * snapwave["hs_forcing"]
    deep = snapwave["site_mesh_depth"] >= 18
    print("\nHs ratio to SnapWave (variants on top of OFF WCAP)")
    print(f"{'steepness':14s}{'where':>7s}" + "".join(f"{k:>11s}" for k in variants))
    for low, high in [(0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.07)]:
        rows = cases["steepness"].between(low, high).values
        for label, where in (("deep", deep), ("<18m", ~deep)):
            ref = snapwave["hs"].where(energy & where).isel(case_num=rows)
            ratios = [float(mc.hs_ratio(ref, v.isel(case_num=rows))) for v in variants.values()]
            print(f"{low:.2f}-{high:.2f}     {label:>7s}" + "".join(f"{r:11.3f}" for r in ratios))


if __name__ == "__main__":
    report_dynamic()
    report_metamodel()
