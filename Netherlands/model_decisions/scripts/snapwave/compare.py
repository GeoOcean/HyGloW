"""Score the SnapWave tests. Run from Netherlands/: python SWAN_test/snapwave/compare.py"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "SWAN_test")
sys.path.insert(0, ".")
from compare_tests import ROOT as SWAN_ROOT, read_hs, score  # noqa: E402
from utils import model_comparison as mc  # noqa: E402
from utils import validation as val  # noqa: E402
from utils.config import load_goals  # noqa: E402

ROOT = Path("SWAN_test/snapwave")
CLASSES = [0, 1, 2, 3, 4, 5, 6, 12]
UNSTABLE_ERROR = 0.01
VARIANTS = ("base", "enc", "enc_upw", "enc_upw_fw02", "enc_upw_fw04", "enc_upw_nowind")


def hm0(folder: Path) -> np.ndarray:
    with xr.open_dataset(folder / "output_sites.nc") as ds:
        return ds["point_hm0"].values.ravel().astype(float)


def last_iteration(folder: Path) -> tuple[int, float, float, float]:
    text = (folder / "wrapper_out.log").read_text()
    rows = re.findall(r"iteration\s+(\d+) error =\s+([\d.Ee+-]+)\s+%ok =\s+([\d.]+)", text)
    seconds = re.findall(r"Computation:\s+([\d.]+)", text)
    iteration, error, ok = rows[-1]
    return int(iteration), float(error), float(ok), float(seconds[-1]) if seconds else np.nan


def report_dynamic() -> None:
    hours = pd.read_csv("SWAN_test/dynamic_hours.csv", parse_dates=["time"]).sort_values("time")
    names = [t.strftime("%Y%m%dT%H") for t in hours["time"]]
    production = mc.load_snapwave_dynamic("Dynamic_Cases_WHACS_cal_2013.nc").sel(time=hours["time"].values)
    buoy = val.load_buoy_observations(slice(hours["time"].min(), hours["time"].max()), variables=["hs"])
    sites = list(buoy.sites.values)
    index = [list(production.sites.values).index(s) for s in sites]
    obs = buoy["hs"].reindex(time=hours["time"].values).sel(sites=sites).transpose("time", "sites").values
    models = {
        v: np.stack([hm0(ROOT / "dyn" / v / n) for n in names])[:, index]
        for v in VARIANTS
    }
    for v in ("whacs_map", "whacs_map_fric"):
        models[v] = np.stack([read_hs(SWAN_ROOT / "cases_2" / v / n) for n in names])[:, index]
    depth = production["site_mesh_depth"].sel(sites=sites).values
    group = hours["group"].values

    print(f"Hs vs buoys, new WHACS_cal with wind, bias/rmse (n), {len(sites)} buoys")
    print(" " * 16 + "".join(f"{k:>22}" for k in models))

    def line(label: str, mask: np.ndarray) -> None:
        print(f"{label:16s}" + "".join(f"{score(obs, v, mask):>22}" for v in models.values()))

    for name in ("calm", "storm", "xaver"):
        line(name, np.repeat((group == name)[:, None], len(sites), axis=1))
    for low, high in zip(CLASSES[:-1], CLASSES[1:]):
        line(f"obs {low}-{high} m", (obs >= low) & (obs < high))
    for low, high in [(0, 15), (15, 25), (25, 60)]:
        where = (depth >= low)[None, :] & (depth < high)[None, :]
        line(f"d {low}-{high} m", np.repeat(where, len(hours), axis=0))
    print("\nconvergence (39 hours):")
    for v in VARIANTS:
        table = pd.DataFrame([last_iteration(ROOT / "dyn" / v / n) for n in names], columns=["it", "err", "ok", "s"])
        print(f"{v:10s} not converged {(table.err > UNSTABLE_ERROR).sum()}/{len(table)}, max err {table.err.max():.3f}, "
              f"min %ok {table.ok.min():.1f}, median {table.s.median():.0f} s")


def report_stability() -> None:
    names = [t.strftime("%Y%m%dT%H") for t in pd.read_csv(ROOT / "unstable_hours.csv", parse_dates=["time"])["time"]]
    print(f"\nunstable hours ({len(names)} of the 160), new WHACS_cal with wind")
    print(f"{'variant':12s}{'unstable':>9s}{'med err':>9s}{'med %ok':>9s}{'med Hsmax':>10s}{'max Hsmax':>10s}{'med n>3m':>9s}{'med s':>7s}")
    for variant in sorted(p.name for p in (ROOT / "stab").iterdir()):
        table = pd.DataFrame(
            [last_iteration(ROOT / "stab" / variant / n) for n in names], columns=["it", "err", "ok", "s"]
        )
        hs = np.stack([hm0(ROOT / "stab" / variant / n) for n in names])
        hs = np.where(hs <= 15, hs, np.nan)
        peak = np.nanmax(hs, axis=1)
        print(f"{variant:12s}{(table.err > UNSTABLE_ERROR).sum():>9d}{table.err.median():>9.4f}{table.ok.median():>9.1f}"
              f"{np.median(peak):>10.2f}{peak.max():>10.2f}{np.median((hs > 3).sum(1)):>9.0f}{table.s.median():>7.0f}")


def report_metamodel() -> None:
    cases = pd.read_csv("SWAN_test/metamodel_cases.csv").sort_values("case")
    snapwave = mc.load_snapwave_metamodel(load_goals()).sel(case_num=cases["case"].values)
    labels = [f"{c:04}" for c in cases["case"]]
    swan = snapwave["hs"].copy(data=np.stack([read_hs(SWAN_ROOT / "cases" / "no_wcap" / n) for n in labels]))
    variants = {
        "production": snapwave["hs"],
        **{
            v: snapwave["hs"].copy(data=np.stack([hm0(ROOT / "mm" / v / n) for n in labels]))
            for v in ("enc", "enc_n200", "enc_upw", "enc_360")
        },
    }
    energy = snapwave["hs"] >= 0.25 * snapwave["hs_forcing"]
    deep = snapwave["site_mesh_depth"] >= 18
    print("\nSnapWave Hs ratio to SWAN (OFF WCAP), metamodel cases")
    print(f"{'steepness':14s}{'where':>7s}" + "".join(f"{k:>12s}" for k in variants))
    for low, high in [(0, 0.02), (0.02, 0.04), (0.04, 0.07)]:
        rows = cases["steepness"].between(low, high).values
        for label, where in (("deep", deep), ("<18m", ~deep)):
            ref = swan.where(energy & where).isel(case_num=rows)
            ratios = [float(mc.hs_ratio(ref, v.isel(case_num=rows))) for v in variants.values()]
            print(f"{low:.2f}-{high:.2f}     {label:>7s}" + "".join(f"{r:12.3f}" for r in ratios))
    base = variants["production"].values
    print("\neffect at all sites, Hs vs production:")
    for name, v in variants.items():
        diff = v.values - base
        print(f"{name:12s} mean {np.nanmean(diff):+.4f}  p1 {np.nanpercentile(diff, 1):+.3f}  p99 {np.nanpercentile(diff, 99):+.3f} m")
    converged = variants["enc_n200"].values
    diff = variants["enc"].values - converged
    print(f"niter 40 vs 200: mean {np.nanmean(diff):+.4f}, p99 |diff| {np.nanpercentile(abs(diff), 99):.3f} m")
    for v in ("enc", "enc_n200", "enc_upw", "enc_360"):
        table = pd.DataFrame([last_iteration(ROOT / "mm" / v / n) for n in labels], columns=["it", "err", "ok", "s"])
        print(f"{v:10s} max err {table.err.max():.5f}, min %ok {table.ok.min():.1f}, median it {table['it'].median():.0f}, median {table.s.median():.1f} s")


if __name__ == "__main__":
    report_dynamic()
    report_stability()
    report_metamodel()
