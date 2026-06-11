"""Extract and export points along an isobath from a 2D xarray DataArray."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_X_NAMES = ("x", "lon", "longitude")
_Y_NAMES = ("y", "lat", "latitude")


def _xy_names(da: xr.DataArray) -> tuple[str, str]:
    x = next((n for n in _X_NAMES if n in da.coords), None)
    y = next((n for n in _Y_NAMES if n in da.coords), None)
    if x is None or y is None:
        raise KeyError(
            f"Could not find x/y coords in {tuple(da.coords)}; "
            f"expected one of {_X_NAMES} and {_Y_NAMES}."
        )
    return x, y


def get_isobath_points(
    da: xr.DataArray,
    level: float,
    *,
    spacing: float | None = None,
    longest_only: bool = False,
) -> np.ndarray:
    """Return ``(N, 2)`` array of ``(x, y)`` points along an isobath.

    Parameters
    ----------
    da
        2D ``xr.DataArray`` of elevation/depth with coordinates named
        ``x/y``, ``lon/lat`` or ``longitude/latitude``.
    level
        Isobath value in the same units/sign convention as ``da``
        (e.g. ``-10`` for the 10 m isobath when depths are negative).
    spacing
        If given, resample the contour at this constant spacing
        (in coordinate units).
    longest_only
        If ``True``, keep only the longest contour segment.
    """
    xname, yname = _xy_names(da)
    x = da[xname].values
    y = da[yname].values
    z = np.asarray(da.values, dtype=float)
    if z.shape != (y.size, x.size):
        raise ValueError(
            f"Expected da.shape == (len({yname}), len({xname})); "
            f"got {z.shape} vs ({y.size}, {x.size}). "
            f"Try da.transpose({yname!r}, {xname!r})."
        )

    fig, ax = plt.subplots()
    cs = ax.contour(x, y, z, levels=[float(level)])
    segments = [np.asarray(p) for p in cs.allsegs[0] if len(p) >= 2]
    plt.close(fig)

    if not segments:
        return np.empty((0, 2), dtype=float)

    if longest_only:
        segments = [max(segments, key=lambda s: _polyline_length(s))]

    if spacing is not None and spacing > 0:
        segments = [_resample_polyline(s, spacing) for s in segments]
        segments = [s for s in segments if s.shape[0] > 0]

    return np.vstack(segments) if segments else np.empty((0, 2), dtype=float)


def write_points_txt(
    points: np.ndarray,
    path: str | Path,
    *,
    fmt: str = "%.8f",
) -> Path:
    """Write ``(N, 2)`` points to a space-separated ``x y`` text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(points, dtype=float), fmt=fmt)
    return path


def _polyline_length(pts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _resample_polyline(pts: np.ndarray, spacing: float) -> np.ndarray:
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 0.0:
        return np.empty((0, 2), dtype=float)

    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    ds = np.arange(0.0, total, float(spacing))
    if ds.size == 0 or (total - ds[-1]) > 0.25 * spacing:
        ds = np.append(ds, total)

    out = np.empty((ds.size, 2), dtype=float)
    for k, d in enumerate(ds):
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = max(0, min(i, len(seg_len) - 1))
        t = 0.0 if seg_len[i] == 0 else (d - cum[i]) / seg_len[i]
        out[k] = pts[i] * (1.0 - t) + pts[i + 1] * t
    return out
