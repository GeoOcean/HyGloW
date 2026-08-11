"""Nautical wave-direction conventions for a HyGloW case pipeline.

Convention (used everywhere except hidden matplotlib geometry helpers)
-----------------------------------------------------------------------
All **stored** and **user-facing** directions are **nautical**:

- **Wave-from direction** (same as WHACS ``wave_from_direction`` / CF).
- **Clockwise from North**: 0° = North, 90° = East, 180° = South, 270° = West.

Goal propagation sectors in the case goals JSON use::

    "angles": [left, right]

- **left** — clockwise sector start (left boundary when looking from above).
- **right** — clockwise sector end (right boundary).
- Always **left < right** as plain numbers (required for LHS bounds).
- If the sector crosses North, **left** may be negative (e.g. ``-10`` means
  350°). After sampling or filtering, apply :func:`normalize_nautical`.

Examples
--------
- ``[100, 200]`` — directions from 100° to 200° clockwise (no wrap).
- ``[-10, 50]`` — from 350° through North to 50° clockwise.

Internal geometry in :mod:`utils.geometry.sectors` still uses arctan2 math
while building wedges; conversion to nautical happens once at export via
:func:`math_sector_to_nautical_clockwise`. Matplotlib wedges use
:func:`nautical_sector_to_wedge_angles` (plotting only).
"""

from __future__ import annotations

import numpy as np

NAUTICAL_CONVENTION_DOC = (
    "Nautical wave-from direction, clockwise from North (0°=N, 90°=E)."
)


def normalize_nautical(deg: float | np.ndarray) -> float | np.ndarray:
    """
    Map direction(s) to ``[0, 360)`` degrees.

    Parameters
    ----------
    deg
        Nautical direction(s); may be negative before normalization.

    Returns
    -------
    float or numpy.ndarray
        Direction(s) in ``[0, 360)``.
    """

    return np.mod(deg, 360.0)


def math_to_nautical(deg: float) -> float:
    """
    Convert internal math angle (CCW from East) to nautical degrees.

    Used only when translating geometry from :mod:`utils.geometry.sectors`.
    Not part of the user-facing API.

    Parameters
    ----------
    deg
        Math angle in degrees (CCW from East).

    Returns
    -------
    float
        Nautical direction in ``[0, 360)``.
    """

    return float(normalize_nautical(90.0 - deg))


def math_sector_to_nautical_clockwise(low_math: float, high_math: float) -> list[float]:
    """
    Convert an internal math wedge to stored nautical [left, right] bounds.

    The open-sea wedge in :class:`~utils.geometry.sectors.AoIRingSectors` is built
    from ``low_math`` to ``high_math`` counter-clockwise from East. This
    function returns the equivalent nautical clockwise sector with ``left <
    right``.

    Parameters
    ----------
    low_math, high_math
        Sector bounds in math degrees (CCW from East).

    Returns
    -------
    list[float]
        ``[left, right]`` nautical clockwise bounds (``left < right``).
    """

    left_n = float(normalize_nautical(90.0 - high_math))
    right_n = float(normalize_nautical(90.0 - low_math))
    if left_n <= right_n:
        return [round(left_n, 1), round(right_n, 1)]

    return [round(left_n - 360.0, 1), round(right_n, 1)]


def in_nautical_sector(
    direction: float | np.ndarray,
    left: float,
    right: float,
) -> bool | np.ndarray:
    """
    Test whether direction(s) lie inside a nautical clockwise sector.

    Parameters
    ----------
    direction
        Nautical wave-from direction(s); normalized with :func:`normalize_nautical`.
    left, right
        Stored sector bounds (``left < right``; ``left`` may be negative).

    Returns
    -------
    bool or numpy.ndarray
        ``True`` where *direction* is inside the sector.
    """

    left_n = float(normalize_nautical(left))
    right_n = float(normalize_nautical(right))
    dir_n = normalize_nautical(direction)
    if left_n <= right_n:
        return (dir_n >= left_n) & (dir_n <= right_n)

    return (dir_n >= left_n) | (dir_n <= right_n)


def nautical_sector_to_wedge_angles(left: float, right: float) -> tuple[float, float]:
    """
    Map a nautical sector to matplotlib wedge angles (plotting only).

    Matplotlib ``Wedge`` uses degrees counter-clockwise from East. This helper
    is **not** for data storage — only for drawing maps.

    Parameters
    ----------
    left, right
        Stored nautical sector bounds.

    Returns
    -------
    theta1, theta2
        Wedge angles (CCW from East) with ``theta1 < theta2`` after adjustment.
    """

    right_n = float(normalize_nautical(right))
    left_n = float(normalize_nautical(left))
    theta1 = 90.0 - right_n
    theta2 = 90.0 - left_n
    if theta2 <= theta1:
        theta2 += 360.0

    return theta1, theta2


def nautical_sector_wedge_coords(
    lon: float,
    lat: float,
    left: float,
    right: float,
    radius_deg: float,
    *,
    n_pts: int | None = None,
) -> list[tuple[float, float]]:
    """
    Polygon ring for a nautical sector wedge (plotting / GPKG only).

    Parameters
    ----------
    lon, lat
        Wedge origin (degrees).
    left, right
        Stored nautical sector bounds.
    radius_deg
        Wedge radius in degrees.
    n_pts
        Arc point count; default from sector span.

    Returns
    -------
    list[tuple[float, float]]
        Closed polygon coordinates ``[(lon, lat), ..., (lon, lat)]``.
    """

    theta1, theta2 = nautical_sector_to_wedge_angles(left, right)
    if n_pts is None:
        n_pts = max(8, int(abs(theta2 - theta1) / 5))
    arc = np.linspace(theta1, theta2, n_pts)
    coords = [(lon, lat)]
    for angle in arc:
        r = np.radians(angle)
        coords.append((lon + radius_deg * np.cos(r), lat + radius_deg * np.sin(r)))
    coords.append((lon, lat))

    return coords


__all__ = [
    "NAUTICAL_CONVENTION_DOC",
    "in_nautical_sector",
    "math_sector_to_nautical_clockwise",
    "math_to_nautical",
    "normalize_nautical",
    "nautical_sector_to_wedge_angles",
    "nautical_sector_wedge_coords",
]
