"""Old and new SnapWave boundary nodes on the Netherlands mesh -> results/figures/snapwave_enclosure.png.

Run from ``Netherlands/`` with GDAL_DATA and PROJ_DATA set:
python model_decisions/scripts/snapwave/plot_enclosure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, ".")
from bluemath_tk.topo_bathy.mesh import open_boundary_nodes, outline_rings  # noqa: E402
from utils.config import load_goals  # noqa: E402
from utils.geometry import model_grid  # noqa: E402
from utils.simulation import goals_for_snapwave  # noqa: E402

OUT = Path("model_decisions/results/figures/snapwave_enclosure.png")


def main() -> None:
    mesh = model_grid("snapwave")
    goals = goals_for_snapwave(load_goals())
    goal_xy = np.array([[g["lon"], g["lat"]] for g in goals.values()])
    outline = np.concatenate(outline_rings(mesh))
    open_nodes = open_boundary_nodes(mesh, goal_xy)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.scatter(mesh.x[outline], mesh.y[outline], s=2, c="tab:red",
               label=f"forced with the AoI rectangle: {len(outline)} outline nodes (coast included)")
    ax.scatter(mesh.x[open_nodes], mesh.y[open_nodes], s=4, c="tab:blue",
               label=f"forced with the open-boundary enclosure: {len(open_nodes)} nodes")
    ax.scatter(*goal_xy.T, marker="x", c="k", s=40, label="goals")
    ax.set_aspect(1 / np.cos(np.radians(mesh.y.mean())))
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("SnapWave boundary nodes (msk = 2), Netherlands mesh")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(OUT, len(outline), len(open_nodes))


if __name__ == "__main__":
    main()
