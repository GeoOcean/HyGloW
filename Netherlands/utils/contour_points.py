import os
from typing import Dict, Iterable, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from shapely import contains_xy

from example_pymesh2d.pymesh2d.geom_util.getiso import getiso


def _deg_to_m(
    x_deg: np.ndarray, y_deg: np.ndarray, lon0: float, lat0: float
) -> Tuple[np.ndarray, np.ndarray]:
    lat0_rad = np.deg2rad(lat0)
    x_m = (x_deg - lon0) * np.cos(lat0_rad) * 111320.0
    y_m = (y_deg - lat0) * 110540.0
    return x_m, y_m


def _edges_to_paths(edge: np.ndarray, n_nodes: int) -> Iterable[np.ndarray]:
    """Reconstruct ordered node paths from PSLG edges."""
    adj = [[] for _ in range(n_nodes)]
    for a, b in np.asarray(edge, dtype=int):
        adj[a].append(b)
        adj[b].append(a)

    visited = set()
    paths = []

    for start in range(n_nodes):
        if len(adj[start]) != 1:
            continue

        current = start
        prev = None
        path = [current]

        while True:
            neighs = adj[current]
            candidates = neighs if prev is None else [n for n in neighs if n != prev]
            if not candidates:
                break

            unvisited = [
                n for n in candidates if tuple(sorted((current, n))) not in visited
            ]
            nxt = unvisited[0] if unvisited else candidates[0]

            visited.add(tuple(sorted((current, nxt))))
            prev, current = current, nxt
            path.append(current)

            if len(adj[current]) == 1 and current != start:
                break

            if len(path) > len(edge) + 5:
                break

        if len(path) >= 2:
            paths.append(np.array(path, dtype=int))

    for a, b in np.asarray(edge, dtype=int):
        ekey = tuple(sorted((int(a), int(b))))
        if ekey in visited:
            continue

        start = int(a)
        current = start
        prev = None
        path = [current]

        while True:
            neighs = adj[current]
            candidates = neighs if prev is None else [n for n in neighs if n != prev]
            if not candidates:
                break

            unvisited = [
                n for n in candidates if tuple(sorted((current, n))) not in visited
            ]
            nxt = unvisited[0] if unvisited else candidates[0]

            visited.add(tuple(sorted((current, nxt))))
            prev, current = current, int(nxt)
            path.append(current)

            if current == start:
                break

            if len(path) > len(edge) + 5:
                break

        if len(path) >= 2:
            paths.append(np.array(path, dtype=int))

    return paths


def _sample_along_polyline(
    path_xy_deg: np.ndarray, path_xy_m: np.ndarray, spacing_m: float
) -> np.ndarray:
    if path_xy_deg.shape[0] < 2:
        return np.empty((0, 2), dtype=float)

    seg = np.diff(path_xy_m, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    if total <= 0:
        return np.empty((0, 2), dtype=float)

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    ds = np.arange(0.0, total, spacing_m)
    if ds.size == 0 or (total - ds[-1]) > 0.25 * spacing_m:
        ds = np.append(ds, total)

    out = []
    for d in ds:
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = max(0, min(i, len(seg_len) - 1))
        t = 0.0 if seg_len[i] == 0 else (d - cum[i]) / seg_len[i]
        p = path_xy_deg[i] * (1.0 - t) + path_xy_deg[i + 1] * t
        out.append(p)

    return np.asarray(out, dtype=float)


def generate_contour_point_files(
    mesh_nc,
    alpha_shape,
    depths_m: Iterable[float],
    spacing_m: float,
    output_dir: str = "outputs/contours_txts",
    step: int = 5,
    grid_res: float = 0.02,
    dedup_decimals: int = 4,
) -> Tuple[Dict[float, Dict[str, np.ndarray]], Dict[float, Dict[str, np.ndarray]]]:
    """
    Extract depth contours, sample points every `spacing_m` inside alpha_shape,
    and save one txt file per depth in `output_dir`.
    """
    depths = np.asarray(list(depths_m), dtype=float)

    x = mesh_nc.mesh2d_node_x.values
    y = mesh_nc.mesh2d_node_y.values
    z = mesh_nc.mesh2d_node_z.values

    x_s = x[::step]
    y_s = y[::step]
    z_s = z[::step]

    if np.nanmedian(z_s) < 0:
        contour_levels_z = -depths
    else:
        contour_levels_z = depths

    xi = np.arange(np.nanmin(x_s), np.nanmax(x_s), grid_res)
    yi = np.arange(np.nanmin(y_s), np.nanmax(y_s), grid_res)
    X, Y = np.meshgrid(xi, yi)

    grid_z = griddata(
        points=np.column_stack([x_s, y_s]),
        values=z_s,
        xi=(X, Y),
        method="linear",
    )
    grid_z = np.nan_to_num(grid_z, nan=np.nanmedian(z_s))

    iso_contours = {}
    for depth_m, lev_z in zip(depths, contour_levels_z):
        node, edge = getiso(X, Y, grid_z, ilev=float(lev_z), filt=0.0)

        if node.size == 0:
            node = np.empty((0, 2), dtype=float)
            edge = np.empty((0, 2), dtype=int)
        else:
            node = np.asarray(node, dtype=float)
            edge = np.asarray(edge, dtype=int)
            edge = (
                np.empty((0, 2), dtype=int) if edge.size == 0 else edge.reshape(-1, 2)
            )

        iso_contours[float(depth_m)] = {
            "level_z": float(lev_z),
            "node": node,
            "edge": edge,
        }

    print(
        "iso_contours node/edge counts:",
        {
            d: (iso_contours[d]["node"].shape[0], iso_contours[d]["edge"].shape[0])
            for d in iso_contours
        },
    )

    lon0 = float(np.nanmean(x))
    lat0 = float(np.nanmean(y))

    result = {}
    for depth_m in depths:
        depth_m = float(depth_m)
        data = iso_contours[depth_m]
        node_xy = np.asarray(data["node"], dtype=float)
        edge = np.asarray(data["edge"], dtype=int)

        if node_xy.size == 0 or edge.size == 0:
            result[depth_m] = {
                "x": np.empty((0,), dtype=float),
                "y": np.empty((0,), dtype=float),
                "z_level": np.full(
                    (0,), float(data.get("level_z", depth_m)), dtype=float
                ),
            }
            print(f"depth {depth_m} m: no contour segments")
            continue

        paths = _edges_to_paths(edge, n_nodes=node_xy.shape[0])
        node_x_m, node_y_m = _deg_to_m(node_xy[:, 0], node_xy[:, 1], lon0, lat0)

        all_pts = []
        for pidx in paths:
            pidx = np.asarray(pidx, dtype=int)
            if pidx.size < 2:
                continue

            path_deg = node_xy[pidx]
            path_m = np.column_stack([node_x_m[pidx], node_y_m[pidx]])
            sampled = _sample_along_polyline(path_deg, path_m, spacing_m=spacing_m)
            if sampled.shape[0] == 0:
                continue

            mask = contains_xy(alpha_shape, sampled[:, 0], sampled[:, 1])
            sampled = sampled[mask]
            if sampled.shape[0] > 0:
                all_pts.append(sampled)

        if not all_pts:
            all_pts = np.empty((0, 2), dtype=float)
        else:
            all_pts = np.vstack(all_pts)
            rounded = np.round(all_pts, decimals=dedup_decimals)
            _, unique_idx = np.unique(rounded, axis=0, return_index=True)
            all_pts = all_pts[unique_idx]

        z_level = float(data.get("level_z", depth_m))
        result[depth_m] = {
            "x": all_pts[:, 0],
            "y": all_pts[:, 1],
            "z_level": np.full((all_pts.shape[0],), z_level, dtype=float),
        }
        print(
            f"depth {depth_m} m: selected {all_pts.shape[0]:,} points (~{spacing_m / 1000:.1f} km spacing)"
        )

    os.makedirs(output_dir, exist_ok=True)
    for depth_m in depths:
        depth_m = float(depth_m)
        out_txt = os.path.join(output_dir, f"{int(round(depth_m))}mpoints.txt")
        x_pts = result[depth_m]["x"]
        y_pts = result[depth_m]["y"]

        if x_pts.size > 0:
            arr = np.column_stack([x_pts, y_pts])
            np.savetxt(out_txt, arr, fmt="%.8f")
        else:
            open(out_txt, "w").close()

        print(f"Saved: {out_txt} ({x_pts.size:,} points)")

    return result, iso_contours


def plot_contour_points(
    result: Dict[float, Dict[str, np.ndarray]],
    alpha_shape=None,
    figsize: Tuple[float, float] = (8, 6),
):
    fig, ax = plt.subplots(figsize=figsize)

    for depth_m in sorted(result.keys()):
        x = result[depth_m]["x"]
        y = result[depth_m]["y"]
        if x.size == 0:
            continue
        ax.scatter(x, y, s=8, alpha=0.8, label=f"{int(round(depth_m))} m")

    if alpha_shape is not None and hasattr(alpha_shape, "exterior"):
        bx, by = alpha_shape.exterior.xy
        ax.plot(bx, by, "k-", linewidth=1.5, label="alpha_shape")

    ax.set_title("Sampled contour points inside alpha_shape")
    ax.set_xlabel("mesh2d_node_x")
    ax.set_ylabel("mesh2d_node_y")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    return fig, ax
