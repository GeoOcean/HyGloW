"""Score the 120-hour sample at the buoys. Run from Netherlands/: python SWAN_test/compare_tests_6.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bluemath_tk.wrappers.swan.swan_utils import read_swan_table

sys.path.insert(0, "SWAN_test")
sys.path.insert(0, ".")
from compare_tests import score  # noqa: E402
from utils import model_comparison as mc  # noqa: E402
from utils import validation as val  # noqa: E402

ROOT = Path("SWAN_test/cases_6")
CLASSES = [0, 1, 2, 3, 4, 5, 6, 12]


def read_hs(folder: Path) -> np.ndarray:
    if not (folder / "norm_end").exists():
        return np.full(5156, np.nan)
    return read_swan_table(folder / "output_sites.tab", mc.TABLE_QUANTITIES)["hs"].to_numpy()


def main() -> None:
    hours = pd.read_csv("SWAN_test/hours_6.csv", parse_dates=["time"])["time"]
    names = [t.strftime("%Y%m%dT%H") for t in hours]
    production = mc.load_snapwave_dynamic("Dynamic_Cases_WHACS_cal_2013.nc")
    buoy = val.load_buoy_observations(slice(hours.min(), hours.max()), variables=["hs"])
    sites = list(buoy.sites.values)
    index = [list(production.sites.values).index(s) for s in sites]
    obs = buoy["hs"].reindex(time=hours.values).sel(sites=sites).transpose("time", "sites").values
    models = {v: np.stack([read_hs(ROOT / v / n) for n in names])[:, index] for v in ("whacs_038", "whacs", "era5")}
    print({v: int(np.isfinite(m).any(1).sum()) for v, m in models.items()}, "hours finished")
    print(f"Hs vs buoys, bias/rmse (n), {len(sites)} buoys; whacs and era5 = friction 0.019")
    print(" " * 16 + "".join(f"{k:>22}" for k in models))
    line = lambda label, mask: print(f"{label:16s}" + "".join(f"{score(obs, v, mask):>22}" for v in models.values()))
    line("all", np.isfinite(obs))
    for low, high in zip(CLASSES[:-1], CLASSES[1:]):
        line(f"obs {low}-{high} m", (obs >= low) & (obs < high))


if __name__ == "__main__":
    main()
