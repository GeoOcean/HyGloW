"""SWAN on a Samoa storm: the reference for the SnapWave wind2 build (per-point boundary directions).

Same forcing as the 2018 dynamic run (WHACS_cal bulk at every goal, goal-mean CODEC level, no
wind), over a window around Cyclone Gita. SWAN builds one JONSWAP x cos^m spectrum per goal and
interpolates the spectra (energy) along the boundary: each boundary point keeps its own direction,
which is what wind2 fixed in SnapWave. The same hours are also run with SnapWave wind1 (the old
build, everything else as wind2) to see which build SWAN agrees with.

Run from the case folder (PYTHONPATH=.):

    python numerical_tests/scripts/06_swan_storm.py grid          # inputs/GRID/Samoa_mesh.grd + SWAN sites table
    python numerical_tests/scripts/06_swan_storm.py submit        # SWAN cases of the window (SLURM)
    python numerical_tests/scripts/06_swan_storm.py wind1         # the same hours with SnapWave wind1 (SLURM)
    python numerical_tests/scripts/06_swan_storm.py postprocess   # -> <cases>/Dynamic_Cases_<tag>.nc (SWAN, wind1)

The SWAN grid: the mesh nodes and triangles (counter-clockwise), the outer outline as one open
boundary that starts at goal 1, runs in goal order and closes back on goal 1 (a ring), and the
islands as land boundaries. The last PAR row (the ring closing on its start) carries goal 1, so
the stretch between goal 16 and goal 1 is interpolated like every other one.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.topo_bathy.mesh import outline_rings
from bluemath_tk.wrappers.swan import swan_wrapper

from utils import simulation as sim
from utils.config import get_offshore_wave, goals_for_snapwave, load_goals, open_offshore_waves, resolve
from utils.geometry import model_grid, output_sites

OFFSHORE = "WHACS_cal"
WINDOW = slice("2018-02-08 12:00", "2018-02-11 12:00")  # Cyclone Gita (peak offshore 2018-02-09 19:00)
TAG = "gita2018"
MAP_HOURS = ["2018-02-09 21:00"]  # the hour whose SnapWave map 08b draws
WIND1_MODULE = "module use /nfs/software/geocean/modulefiles\nmodule load snapwave/b663c81-wind1"


def _ring_area(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def grid() -> None:
    mesh = model_grid("snapwave")
    rings = outline_rings(mesh)
    outer = max(rings, key=lambda r: abs(_ring_area(mesh.x[r], mesh.y[r])))
    goals = goals_for_snapwave(load_goals())
    goal_xy = np.array([[g["lon"], g["lat"]] for g in goals.values()])
    snapped = [int(np.argmin(np.hypot(mesh.x[outer] - gx, mesh.y[outer] - gy))) for gx, gy in goal_xy]
    ring = np.roll(outer, -snapped[0])
    pos = [int(np.argmin(np.hypot(mesh.x[ring] - gx, mesh.y[ring] - gy))) for gx, gy in goal_xy]
    if pos[1] > len(ring) // 2:  # goal 2 lies behind goal 1: walk the other way
        ring = np.concatenate([ring[:1], ring[1:][::-1]])
        pos = [int(np.argmin(np.hypot(mesh.x[ring] - gx, mesh.y[ring] - gy))) for gx, gy in goal_xy]
    if np.any(np.diff(pos) <= 0):
        raise ValueError(f"goals are not in order along the outer ring: {pos}")
    open_string = np.append(ring, ring[0])  # closed: back to goal 1
    islands = [r for r in rings if r is not outer]

    tri = mesh.triangles.copy()
    x, y = mesh.x, mesh.y
    cross = (x[tri[:, 1]] - x[tri[:, 0]]) * (y[tri[:, 2]] - y[tri[:, 0]]) - (y[tri[:, 1]] - y[tri[:, 0]]) * (
        x[tri[:, 2]] - x[tri[:, 0]])
    tri[cross < 0] = tri[cross < 0][:, [0, 2, 1]]

    path = resolve("inputs/GRID/Samoa_mesh.grd")
    with open(path, "w") as f:
        f.write("Samoa mesh for SWAN (from Samoa_mesh.nc; open boundary = outer ring from goal 1)\n")
        f.write(f"{len(tri)} {len(x)}\n")
        depth = mesh.depth
        f.writelines(f"{i + 1} {x[i]:.8f} {y[i]:.8f} {depth[i]:.4f}\n" for i in range(len(x)))
        f.writelines(f"{k + 1} 3 {a + 1} {b + 1} {c + 1}\n" for k, (a, b, c) in enumerate(tri))
        f.write("1 = Number of open boundaries\n")
        f.write(f"{len(open_string)} = Total number of open boundary nodes\n")
        f.write(f"{len(open_string)} = Number of nodes for open boundary 1\n")
        f.writelines(f"{n + 1}\n" for n in open_string)
        f.write(f"{len(islands)} = Number of land boundaries\n")
        f.write(f"{sum(len(r) for r in islands)} = Total number of land boundary nodes\n")
        for k, r in enumerate(islands):
            f.write(f"{len(r)} 1 = Number of nodes for land boundary {k + 1}\n")
            f.writelines(f"{n + 1}\n" for n in r)
    print(f"wrote {path}: {len(x)} nodes, {len(tri)} triangles, open ring {len(ring)} nodes, "
          f"{len(islands)} islands; goals at ring positions {pos}")
    sites = output_sites("swan", recompute=True)
    print(f"SWAN output sites: {int((sites['status'] == 'ok').sum())} ok of {len(sites)}")


class RingSwanDynamic(swan_wrapper.SwanDynamicModelWrapper):
    """The case wrapper with the last PAR row (the ring closing on goal 1) set to goal 1."""

    def build_case(self, case_context: dict, case_dir: str) -> None:
        super().build_case(case_context, case_dir)
        first = case_context["boundary_par"][0].split()
        last = case_context["boundary_par"][-1].split()
        case_context["boundary_par"][-1] = " ".join([last[0], *first[1:]])


def _parameters() -> dict:
    offshore = open_offshore_waves(entry=get_offshore_wave(OFFSHORE)).sel(time=WINDOW).load()
    params = sim.dynamic_parameters(offshore, water_level=True)
    params["save_map"] = [str(pd.Timestamp(t)) in [str(pd.Timestamp(h)) for h in MAP_HOURS] for t in params["tref"]]
    return params


def _swan_wrapper():
    setup = sim.HyWavesWrapperSetup.from_case(load_goals(), model="swan")
    w = setup.dynamic_wrapper(_parameters(), offshore=OFFSHORE, wind=None)
    w.__class__ = RingSwanDynamic
    return w


def _name(ctx: dict) -> str:
    return f"{ctx.get('tref')}"


def submit() -> None:
    w = _swan_wrapper()
    print(f"SWAN cases: {w.output_dir}")
    w.build_cases(cases_name_format=_name)
    w.run_cases_slurm(**sim.slurm_kwargs(model="swan", max_parallel=100, extra_sbatch=["--time=04:00:00"]))


def _wind1_dir() -> Path:
    return sim.cases_dir("dynamic", offshore=OFFSHORE, wind=False).with_name(
        f"SnapWave_CASES_Dynamic_{OFFSHORE}_wind-off_wind1-{TAG}")


def wind1() -> None:
    """Copy the wind2 run's case folders of the window and rerun them with the wind1 build."""
    src = sim.cases_dir("dynamic", offshore=OFFSHORE, wind=False)
    dst = _wind1_dir()
    dst.mkdir(exist_ok=True)
    hours = pd.date_range(WINDOW.start, WINDOW.stop, freq="h")
    dirs = []
    for t in hours:
        name = f"{t:%Y%m%d %H%M%S}"
        case = dst / name
        if case.exists():
            shutil.rmtree(case)
        shutil.copytree(src / name, case, ignore=shutil.ignore_patterns("output_*.nc", "wrapper_*.log"))
        dirs.append(str(case))
    (dst / "case_dirs.txt").write_text("\n".join(dirs) + "\n")
    script = dst / "snapwave_array.sh"
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name=sw-wind1
#SBATCH --partition=geocean
#SBATCH --array=1-{len(dirs)}%100
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --output={dst}/slurm_logs/%A_%a.out
{WIND1_MODULE}
CASE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {dst}/case_dirs.txt)
cd "$CASE" && snapwave > wrapper_out.log 2> wrapper_error.log
""")
    (dst / "slurm_logs").mkdir(exist_ok=True)
    print(subprocess.run(["sbatch", str(script)], capture_output=True, text=True, check=True).stdout.strip())


def postprocess() -> None:
    w = _swan_wrapper()
    w.load_cases(cases_name_format=_name)
    out = Path(w.output_dir) / f"Dynamic_Cases_{TAG}.nc"
    w.postprocess_cases().to_netcdf(out)
    print(f"wrote {out}")
    # wind1: the SnapWave wrapper of the 2018 run, pointed at the copied folders
    setup = sim.HyWavesWrapperSetup.from_case(load_goals())
    sw = setup.dynamic_wrapper(_parameters(), offshore=OFFSHORE, wind=None, output_dir=_wind1_dir())
    sw.load_cases(cases_name_format=_name)
    out = _wind1_dir() / f"Dynamic_Cases_{TAG}.nc"
    sw.postprocess_cases().to_netcdf(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    {"grid": grid, "submit": submit, "wind1": wind1, "postprocess": postprocess}[sys.argv[1]]()
