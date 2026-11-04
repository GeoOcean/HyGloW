"""Score the SWAN physics tests (see README). Run from Netherlands/:
python SWAN_test/compare_tests.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, ".")
from bluemath_tk.wrappers.swan.swan_utils import read_swan_table  # noqa: E402
from utils import model_comparison as mc  # noqa: E402
from utils import validation as val  # noqa: E402
from utils.config import get_offshore_wave, load_goals, open_offshore_waves  # noqa: E402

ROOT = Path("SWAN_test")
BASE = "base"
HS_CLASSES = [0, 1, 2, 3, 4, 5, 6, 12]


def read_hs(folder: Path) -> np.ndarray | None:
    if not (folder / "norm_end").exists():
        return None
    return read_swan_table(folder / "output_sites.tab", ["HSIGN", "TPS", "DIR", "DSPR"])["hs"].to_numpy()


def convergence(folder: Path) -> tuple[int, float]:
    text = (folder / "PRINT").read_text(errors="ignore")
    return (
        int(re.findall(r"iteration\s+(\d+)", text)[-1]),
        float(re.findall(r"accuracy OK in\s+([\d.]+)\s*%", text)[-1]),
    )


def minutes(folder: Path) -> float:
    return ((folder / "norm_end").stat().st_mtime - (folder / "swaninit").stat().st_mtime) / 60


def dynamic_hs(hours: pd.DataFrame, snapwave: xr.Dataset) -> dict[str, xr.DataArray]:
    """Hs on (time, sites) per model; only hours that finished in every variant."""
    folders = mc.case_folders(snapwave.sel(time=hours["time"].values))
    models: dict[str, dict] = {BASE: {}, "komen": {}, "st6": {}}
    for time, folder in folders.items():
        name = pd.Timestamp(time).strftime("%Y%m%dT%H")
        models[BASE][time] = read_hs(ROOT / "cases" / BASE / name if (ROOT / "cases" / BASE / name).exists() else folder)
        for variant in ("komen", "st6"):
            models[variant][time] = read_hs(ROOT / "cases" / variant / name)
    done = [t for t in folders if all(m[t] is not None for m in models.values())]
    print(f"dynamic hours finished in every variant: {len(done)} of {len(folders)}")
    out = {
        name: xr.DataArray(
            np.stack([m[t] for t in done]), dims=("time", "sites"),
            coords={"time": done, "sites": snapwave.sites},
        )
        for name, m in models.items()
    }
    return {"SnapWave": snapwave["hs"].sel(time=done), **out}


def score(buoy: np.ndarray, model: np.ndarray, mask: np.ndarray) -> str:
    use = mask & np.isfinite(buoy) & np.isfinite(model) & (model <= mc.HS_MAX)
    if use.sum() < 5:
        return f"{'':>16}"
    error = model[use] - buoy[use]
    return f"{error.mean():+6.2f}/{np.sqrt((error ** 2).mean()):5.2f} ({use.sum():5d})"


def report_dynamic() -> None:
    hours = pd.read_csv(ROOT / "dynamic_hours.csv", parse_dates=["time"])
    snapwave = mc.load_snapwave_dynamic("Dynamic_Cases_WHACS_cal_2013.nc")
    models = dynamic_hs(hours, snapwave)
    times = models[BASE].time
    offshore = open_offshore_waves(entry=get_offshore_wave("WHACS_cal"))["hs"].sel(time=times).load()
    suspect = mc.suspect_hours(models["SnapWave"], offshore)
    print("suspect SnapWave hours among them:", int(suspect.sum()))

    buoy_ds = val.load_buoy_observations(
        slice(times.values.min(), times.values.max()), variables=["hs"]
    )
    sites = list(buoy_ds.sites.values)
    obs = buoy_ds["hs"].reindex(time=times).sel(sites=sites).transpose("time", "sites").values
    values = {k: v.sel(sites=sites).transpose("time", "sites").values for k, v in models.items()}
    values["SnapWave"] = np.where(suspect.values[:, None], np.nan, values["SnapWave"])
    group = hours.set_index("time")["group"].reindex(times.values).values
    depth = snapwave["site_mesh_depth"].sel(sites=sites).values

    print(f"\nHs vs buoys, bias/rmse (n), {len(sites)} buoys")
    print(" " * 16 + "".join(f"{name:>22}" for name in values))
    def line(label: str, mask: np.ndarray) -> None:
        print(f"{label:16s}" + "".join(f"{score(obs, v, mask):>22}" for v in values.values()))

    for name in ("calm", "storm", "xaver"):
        line(name, np.repeat((group == name)[:, None], len(sites), axis=1))
    print()
    for low, high in zip(HS_CLASSES[:-1], HS_CLASSES[1:]):
        line(f"obs {low}-{high} m", (obs >= low) & (obs < high))
    print()
    for low, high in [(0, 15), (15, 25), (25, 60)]:
        line(f"obs>=3, d {low}-{high}", (obs >= 3) & (depth >= low)[None, :] & (depth < high)[None, :])

    for variant in ("komen", "st6"):
        folders = [ROOT / "cases" / variant / pd.Timestamp(t).strftime("%Y%m%dT%H") for t in times.values]
        table = pd.DataFrame([(*convergence(f), minutes(f)) for f in folders], columns=["iterations", "accuracy", "minutes"])
        print(f"\n{variant}: median iterations {table.iterations.median():.0f}, hit 50: "
              f"{(table.iterations >= 50).sum()}/{len(table)}, median accuracy {table.accuracy.median():.2f} %, "
              f"median {table.minutes.median():.0f} min, max {table.minutes.max():.0f} min")


def report_metamodel() -> None:
    cases = pd.read_csv(ROOT / "metamodel_cases.csv")
    snapwave = mc.load_snapwave_metamodel(load_goals()).sel(case_num=cases["case"].values)
    base = mc.load_swan(snapwave)
    tests = [read_hs(ROOT / "cases" / "no_wcap" / f"{c:04}") for c in cases["case"]]
    done = np.array([t is not None for t in tests])
    print(f"\nno-whitecapping metamodel cases finished: {done.sum()} of {len(cases)}")
    no_wcap = base["hs"].isel(case_num=done).copy(data=np.stack([t for t in tests if t is not None]))
    snapwave, base, cases = snapwave.isel(case_num=done), base.isel(case_num=done), cases[done]
    energy = snapwave["hs"] >= 0.25 * snapwave["hs_forcing"]
    deep = snapwave["site_mesh_depth"] >= 18
    print("Hs ratio to SnapWave, sites with >= 25 % of the offshore Hs (dir 250-340)")
    print(f"{'steepness':14s}{'cases':>6s}{'SWAN deep':>11s}{'no WCAP':>9s}{'SWAN <18m':>11s}{'no WCAP':>9s}")
    for low, high in [(0, 0.01), (0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.07)]:
        rows = cases["steepness"].between(low, high).values
        ref = snapwave["hs"].where(energy).isel(case_num=rows)
        print(
            f"{low:.2f}-{high:.2f}     {rows.sum():6d}"
            f"{float(mc.hs_ratio(ref.where(deep), base['hs'].isel(case_num=rows))):11.3f}"
            f"{float(mc.hs_ratio(ref.where(deep), no_wcap.isel(case_num=rows))):9.3f}"
            f"{float(mc.hs_ratio(ref.where(~deep), base['hs'].isel(case_num=rows))):11.3f}"
            f"{float(mc.hs_ratio(ref.where(~deep), no_wcap.isel(case_num=rows))):9.3f}"
        )


if __name__ == "__main__":
    report_dynamic()
    report_metamodel()
