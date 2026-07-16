"""Mesh sizing functions: gradient-limited bathymetry grids for bluemesh2d.

Used by ``02_Create_Mesh_By_bluemesh2d.ipynb`` to build a smoothed element-size
field ``hfuns`` with bounded ``|∇h|`` in the model projected CRS.
"""

from __future__ import annotations

import numpy as np
import pyproj
from scipy import ndimage


def extent_lonlat_bbox(lon_1d, lat_1d, crs_from, crs_to):
    """
    Bounding box of a lon/lat grid corners projected to *crs_to*.

    Parameters
    ----------
    lon_1d, lat_1d
        1D longitude and latitude arrays.
    crs_from, crs_to
        pyproj-compatible CRS strings or objects.

    Returns
    -------
    tuple[float, float, float, float]
        ``(xmin, xmax, ymin, ymax)`` in *crs_to* coordinates.
    """
    lon_1d, lat_1d = np.asarray(lon_1d), np.asarray(lat_1d)
    lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)
    tr = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)
    x, y = tr.transform(lon_grid, lat_grid)
    return (
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmin(y)),
        float(np.nanmax(y)),
    )


def _shift_neighbor(arr, di, dj, fill=np.inf):
    """Shift a 2D array by (di, dj) cells, filling exposed cells with *fill*."""
    out = np.full_like(arr, fill, dtype=float)
    ni, nj = arr.shape
    si0, si1 = max(0, di), ni + min(0, di)
    di0, di1 = max(0, -di), ni - max(0, di)
    sj0, sj1 = max(0, dj), nj + min(0, dj)
    dj0, dj1 = max(0, -dj), nj - max(0, dj)
    if si0 < si1 and sj0 < sj1:
        out[di0:di1, dj0:dj1] = arr[si0:si1, sj0:sj1]
    return out


def _fill_nan_nearest(arr):
    """Fill NaN cells with the value of the nearest finite neighbour."""
    arr = np.asarray(arr, dtype=float)
    out = arr.copy()
    bad = ~np.isfinite(out)
    if not np.any(bad):
        return out
    if not np.any(np.isfinite(out)):
        raise ValueError("No finite values available to fill NaNs")
    _, idx = ndimage.distance_transform_edt(bad, return_indices=True)
    out[bad] = out[tuple(idx[:, bad])]
    return out


def _clip_grad(h, xpts, ypts, grad_max=0.10, n_iter=None, tol=1e-6):
    """Enforce ``|∇h| <= grad_max`` on a regular grid by neighbour relaxation."""
    h = np.asarray(h, dtype=float).copy()
    if not np.any(np.isfinite(h)):
        raise ValueError("Grid h has no finite values")
    h[~np.isfinite(h)] = np.inf
    h0 = h.copy()
    dx = float(np.mean(np.diff(xpts)))
    dy = float(np.mean(np.diff(ypts)))
    neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    if n_iter is None:
        n_iter = int(2 * max(h.shape))
    for _ in range(n_iter):
        prev = h.copy()
        for di, dj in neigh:
            d = np.hypot(di * dy, dj * dx)
            if d == 0:
                continue
            h = np.minimum(h, _shift_neighbor(h, di, dj) + float(grad_max) * d)
        h = np.minimum(h, h0)
        if np.max(np.abs(h - prev)) < tol:
            break
    return h


def _grid_size(x0, x1, y0, y1, grid_res, grad, nmin, nmax):
    """Choose lookup-grid resolution from domain span and gradient limit."""
    if grid_res != "auto":
        return int(grid_res)
    span = max(float(x1 - x0), float(y1 - y0))
    step = np.clip(800.0 * float(grad), 300.0, 2500.0)
    n = int(np.ceil(span / step)) + 1
    return int(np.clip(n, int(nmin), int(nmax)))


def precalc_hfuns(
    hfun,
    *,
    work_crs,
    extent_xy=None,
    lon=None,
    lat=None,
    geo_crs=None,
    crs_geographic=None,
    name="default",
    cache_id=None,
    max_gradient=0.20,
    gradient_mesh=None,
    min_cells=120,
    max_cells=600,
):
    """
    Build a gradient-limited sizing-function callback for mesh generation.

    Parameters
    ----------
    hfun
        Callable ``hfun(xy)`` with ``xy`` shape ``(N, 2)`` in metres (*work_crs*).
    work_crs
        Projected CRS for the sizing grid.
    extent_xy
        Domain ``(xmin, xmax, ymin, ymax)`` in *work_crs* metres.
    lon, lat, geo_crs
        Alternative domain specification via a lon/lat grid.
    name, cache_id
        Cache key label for precomputed grids.
    max_gradient, gradient_mesh
        Maximum allowed ``|∇h|`` (m per m).
    min_cells, max_cells
        Bounds on the internal lookup-grid resolution.

    Returns
    -------
    callable
        ``lookup(xy)`` with ``.precompute()`` and ``.build()`` helpers.
    """
    if cache_id is not None:
        name = str(cache_id)
    if geo_crs is None and crs_geographic is not None:
        geo_crs = crs_geographic
    g = float(gradient_mesh) if gradient_mesh is not None else float(max_gradient)

    if extent_xy is not None:
        x0, x1 = float(extent_xy[0]), float(extent_xy[1])
        y0, y1 = float(extent_xy[2]), float(extent_xy[3])
    elif lon is not None and lat is not None and geo_crs is not None:
        x0, x1, y0, y1 = extent_lonlat_bbox(lon, lat, geo_crs, work_crs)
    else:
        raise ValueError(
            "precalc_hfuns: provide extent_xy=(xmin,xmax,ymin,ymax) "
            "or lon, lat, geo_crs (with work_crs)."
        )

    ext_key = (round(x0, 3), round(x1, 3), round(y0, 3), round(y1, 3))
    caches = {}
    state = {"cache": None, "params": None}

    def build(grid_res="auto", max_gradient=None):
        """Build or return a cached gradient-limited h grid."""
        mg = float(g if max_gradient is None else max_gradient)
        n = _grid_size(x0, x1, y0, y1, grid_res, mg, min_cells, max_cells)
        key = (name, ext_key, n, mg)
        if key in caches:
            return caches[key]
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        xq, yq = np.meshgrid(xs, ys)
        pts = np.column_stack([xq.ravel(), yq.ravel()])
        raw = np.asarray(hfun(pts), dtype=float).reshape(n, n)
        raw = _fill_nan_nearest(raw)
        lim = _clip_grad(raw, xs, ys, grad_max=mg)
        dx = float(xs[1] - xs[0]) if n > 1 else 1.0
        dy_sp = float(ys[1] - ys[0]) if n > 1 else 1.0
        entry = {
            "h_grid": lim,
            "x_min": x0,
            "x_max": x1,
            "y_min": y0,
            "y_max": y1,
            "dx": dx,
            "dy": dy_sp,
            "grid_res": n,
            "work_crs": work_crs,
            "extent_xy": (x0, x1, y0, y1),
        }
        caches[key] = entry
        return entry

    def precompute(grid_res="auto", max_gradient=None):
        """Precompute the lookup grid and store it in the callback state."""
        mg = float(g if max_gradient is None else max_gradient)
        cached = build(grid_res=grid_res, max_gradient=mg)
        state["cache"] = cached
        state["params"] = (grid_res, mg)
        return cached

    def lookup(xy, return_grid_res=False, max_gradient=None, grid_res="auto"):
        """Sample the gradient-limited h grid at points *xy*."""
        mg = float(g if max_gradient is None else max_gradient)
        par = (grid_res, mg)
        if state["cache"] is None or state["params"] != par:
            precompute(grid_res=grid_res, max_gradient=mg)
        c = state["cache"]
        hg = c["h_grid"]
        x_a, y_a = c["x_min"], c["y_min"]
        dx_, dy_ = c["dx"], c["dy"]
        xy = np.asarray(xy, dtype=float)
        one = xy.ndim == 1
        if one:
            xy = xy.reshape(1, -1)
        ix = np.rint((xy[:, 0] - x_a) / dx_).astype(np.int64)
        iy = np.rint((xy[:, 1] - y_a) / dy_).astype(np.int64)
        ix = np.clip(ix, 0, hg.shape[1] - 1)
        iy = np.clip(iy, 0, hg.shape[0] - 1)
        r = hg[iy, ix]
        out = float(r[0]) if one else r
        if return_grid_res:
            return out, c["grid_res"]
        return out

    lookup.precompute = precompute
    lookup.build = build
    lookup.extent_xy = (x0, x1, y0, y1)
    lookup.work_crs = work_crs
    lookup.name = name
    return lookup


def plot_hfun(
    hfuns_cb, hfun, *, figsize=(16, 5), show=True, grid_res="auto", max_gradient=None
):
    """
    Compare raw bathymetry-based h and the gradient-limited lookup grid.

    Parameters
    ----------
    hfuns_cb
        Callback returned by :func:`precalc_hfuns`.
    hfun
        Original sizing function ``hfun(xy)`` in *work_crs* metres.
    figsize
        Matplotlib figure size.
    show
        Call ``plt.show()`` when ``True``.
    grid_res, max_gradient
        Passed to ``hfuns_cb.precompute``.

    Returns
    -------
    tuple[Figure, ndarray]
        Figure and axes array.
    """
    if not callable(getattr(hfuns_cb, "precompute", None)):
        raise TypeError(
            "First argument must be the callback returned by precalc_hfuns(...)."
        )
    cache = hfuns_cb.precompute(grid_res=grid_res, max_gradient=max_gradient)
    hs = cache["h_grid"]
    ny, nx = hs.shape
    xs = np.linspace(cache["x_min"], cache["x_max"], nx)
    ys = np.linspace(cache["y_min"], cache["y_max"], ny)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    h0 = np.asarray(hfun(pts), dtype=float).reshape(ny, nx)
    reduc = np.maximum(h0 - hs, 0.0)
    extent = [cache["x_min"], cache["x_max"], cache["y_min"], cache["y_max"]]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, data, title, cmap, lab in (
        (axes[0], h0, "raw hfun", "viridis", "h (m)"),
        (axes[1], hs, "hfuns (grid)", "viridis", "h (m)"),
        (axes[2], reduc, "max(raw h − hfuns, 0)", "RdYlBu_r", "m"),
    ):
        im = ax.imshow(
            data,
            extent=extent,
            origin="lower",
            cmap=cmap,
            aspect="equal",
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(im, ax=ax, label=lab)
    wc = cache.get("work_crs", "")
    print(f"work_crs: {wc}  |  grid_res: {cache['grid_res']}")
    print(
        f"raw: [{np.nanmin(h0):.1f}, {np.nanmax(h0):.1f}]  |  "
        f"hfuns: [{np.nanmin(hs):.1f}, {np.nanmax(hs):.1f}]"
    )
    print(f"max reduction: {np.nanmax(reduc):.1f}")
    if show:
        plt.show()
    return fig, axes


__all__ = [
    "extent_lonlat_bbox",
    "plot_hfun",
    "precalc_hfuns",
]
