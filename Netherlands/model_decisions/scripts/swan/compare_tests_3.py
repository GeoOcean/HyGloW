"""Wind vs no wind at the buoys: 39 test hours, then the 500 no-wind hours.
Run from Netherlands/: python SWAN_test/compare_tests_3.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "SWAN_test")
sys.path.insert(0, ".")
from compare_tests import ROOT, convergence, minutes, read_hs, score  # noqa: E402
from compare_tests_2 import CLASSES  # noqa: E402
from utils import model_comparison as mc  # noqa: E402
from utils import validation as val  # noqa: E402

NO_WIND = ROOT / "cases_4" / "no_wind"
WIND = ROOT / "cases_2" / "whacs_map"
WIND_FRIC = ROOT / "cases_2" / "whacs_map_fric"


def stack(folder, hours: pd.Series) -> np.ndarray:
    return np.stack([read_hs(folder / t.strftime("%Y%m%dT%H")) for t in hours])


def main() -> None:
    tested = pd.read_csv(ROOT / "dynamic_hours.csv", parse_dates=["time"]).sort_values("time")
    hours = pd.read_csv(ROOT / "no_wind_hours.csv", parse_dates=["time"]).sort_values("time")["time"]
    snapwave = mc.load_snapwave_dynamic("Dynamic_Cases_WHACS_cal_2013.nc").sel(time=hours.values)
    buoy = val.load_buoy_observations(slice(hours.min(), hours.max()), variables=["hs"])
    sites = list(buoy.sites.values)
    index = [list(snapwave.sites.values).index(s) for s in sites]
    obs_all = buoy["hs"].reindex(time=hours.values).sel(sites=sites).transpose("time", "sites").values
    depth = snapwave["site_mesh_depth"].sel(sites=sites).values
    no_wind = stack(NO_WIND, hours)[:, index]
    snap = snapwave["hs"].sel(sites=sites).transpose("time", "sites").values

    is_test = hours.isin(tested["time"]).values
    models = {
        "no wind": no_wind[is_test],
        "wind": stack(WIND, tested["time"])[:, index],
        "wind+fric": stack(WIND_FRIC, tested["time"])[:, index],
    }
    obs = obs_all[is_test]
    print(f"39 hours, new WHACS_cal, Komen with wind vs no wind, {len(sites)} buoys, bias/rmse (n)")
    print(" " * 16 + "".join(f"{k:>22}" for k in models))
    for name in ("calm", "storm", "xaver"):
        mask = np.repeat((tested["group"].values == name)[:, None], len(sites), axis=1)
        print(f"{name:16s}" + "".join(f"{score(obs, v, mask):>22}" for v in models.values()))
    for low, high in zip(CLASSES[:-1], CLASSES[1:]):
        mask = (obs >= low) & (obs < high)
        print(f"{f'obs {low}-{high} m':16s}" + "".join(f"{score(obs, v, mask):>22}" for v in models.values()))
    gain = models["wind"] / models["no wind"]
    print("median Hs wind / no wind by group:", {
        name: round(float(np.nanmedian(gain[tested["group"].values == name])), 3)
        for name in ("calm", "storm", "xaver")
    })

    print(f"\n{len(hours)} hours of 2013, no wind, vs buoys (SnapWave: with wind, old WHACS_cal)")
    print(" " * 16 + f"{'no wind':>22}{'SnapWave':>22}")
    print(f"{'all':16s}{score(obs_all, no_wind, np.isfinite(obs_all)):>22}{score(obs_all, snap, np.isfinite(obs_all)):>22}")
    for low, high in zip(CLASSES[:-1], CLASSES[1:]):
        mask = (obs_all >= low) & (obs_all < high)
        print(f"{f'obs {low}-{high} m':16s}{score(obs_all, no_wind, mask):>22}{score(obs_all, snap, mask):>22}")
    for low, high in [(0, 15), (15, 25), (25, 60)]:
        where = (depth >= low)[None, :] & (depth < high)[None, :]
        print(f"{f'd {low}-{high} m':16s}{score(obs_all, no_wind, where):>22}{score(obs_all, snap, where):>22}")

    table = pd.DataFrame(
        [(*convergence(NO_WIND / t.strftime("%Y%m%dT%H")), minutes(NO_WIND / t.strftime("%Y%m%dT%H"))) for t in hours],
        columns=["iterations", "accuracy", "minutes"],
    )
    print(f"\nno wind: hit limit {(table.iterations >= 50).sum()}/{len(table)}, "
          f"min accuracy {table.accuracy.min():.1f} %, median {table.minutes.median():.0f} min, "
          f"p90 {table.minutes.quantile(.9):.0f} min")


if __name__ == "__main__":
    main()
