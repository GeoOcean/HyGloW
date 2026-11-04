"""Thin SnapWave enclosure around the grid's open boundary (same nodes as SWAN)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from bluemath_tk.wrappers.snapwave.snapwave_utils import write_polygon_vertices_to_txt
from bluemath_tk.wrappers.swan.swan_utils import read_adcirc_grid
from shapely.geometry import LineString, Polygon

BUFFER_DEG = 0.005


def open_boundary_enclosure(grid_file: str = "inputs/GRID/Netherlands_mesh.grd") -> Polygon:
    grid = read_adcirc_grid(grid_file)
    nodes = np.asarray(grid["vertices"])[np.asarray(grid["open_boundaries"][0])][:, :2]
    return LineString(nodes).buffer(BUFFER_DEG, cap_style="flat", join_style="mitre")


def write_enclosure(path: Path, polygon: Polygon) -> None:
    write_polygon_vertices_to_txt(polygon, path)
