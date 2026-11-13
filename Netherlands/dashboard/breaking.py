"""
SnapWave's depth-induced breaking (``snapwave_solver.f90``), in numpy, for the Breaking page and the store.

SnapWave works with Hrms = sqrt(8 E / rho g) and writes Hm0 = sqrt(2) Hrms. Per node it dissipates
(Baldock et al. 1998, ``opt = 1``)

    Dw = 0.28 alpha rho g / T * exp(-(Hmax / Hrms)^2) * (Hmax^2 + Hrms^2),   Hmax = gamma h,

with T the peak period and h = zs - zb, and clips the energy so that Hrms <= gammax h (``depthlimfac``).
``point_dw`` of the output is this Dw on the converged H and Tp (W/m²); ``point_df`` is the bottom
friction 0.28 rho fw u_orb^3.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

RHO = 1025.0  # snapwave_data.f90
G = 9.81
#: SnapWave's defaults (``snapwave_input.f90``) for the keys a template may leave out.
DEFAULTS = {"gamma": 0.7, "alpha": 1.0, "gammax": 0.6, "fw": 0.01, "hmin": 0.1}


def read_params(template: Path) -> dict[str, float]:
    """Breaking and friction parameters of a ``snapwave.inp`` template (SnapWave's defaults for missing keys)."""
    params = dict(DEFAULTS)
    for line in template.read_text().splitlines():
        m = re.match(r"\s*(\w+)\s*=\s*([-+0-9.eE]+)\s*$", line)
        if m and m.group(1) in params:
            params[m.group(1)] = float(m.group(2))
    return params


def depth(mesh_depth: np.ndarray, water_level: np.ndarray, hmin: float = DEFAULTS["hmin"]) -> np.ndarray:
    """Water depth h = mesh depth (positive down) + water level, at least *hmin* (as SnapWave)."""
    return np.maximum(mesh_depth + water_level, hmin)


def qb(hm0: np.ndarray, h: np.ndarray, gamma: float) -> np.ndarray:
    """Baldock's breaking factor exp(-(Hmax / Hrms)²): ~0 for small waves, -> 1 once Hrms reaches gamma h."""
    hrms = np.maximum(hm0 / np.sqrt(2), 1e-6)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(-((gamma * h) / hrms) ** 2)


def dissipation(hm0: np.ndarray, tp: np.ndarray, h: np.ndarray, gamma: float, alpha: float = 1.0) -> np.ndarray:
    """SnapWave's breaking dissipation Dw (W/m²) for Hm0 (m), Tp (s) and depth h (m)."""
    hrms = np.maximum(hm0 / np.sqrt(2), 1e-6)
    hmax = gamma * h
    with np.errstate(over="ignore", invalid="ignore"):
        return 0.28 * alpha * RHO * G / tp * np.exp(-(hmax / hrms) ** 2) * (hmax**2 + hrms**2)


def wavenumber(tp: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear-dispersion wavenumber (Fenton & McKee 1990 explicit form, < 1 % error)."""
    sigma = 2 * np.pi / tp
    k0 = sigma**2 / G
    return k0 / np.tanh((k0 * h) ** 0.75) ** (2 / 3)


def group_velocity(tp: np.ndarray, h: np.ndarray) -> np.ndarray:
    k = wavenumber(tp, h)
    kh = np.minimum(k * h, 50.0)
    return 0.5 * (1 + 2 * kh / np.sinh(2 * kh)) * (2 * np.pi / tp) / k


def decay_length(hm0: np.ndarray, tp: np.ndarray, h: np.ndarray, gamma: float, alpha: float = 1.0) -> np.ndarray:
    """
    Distance over which breaking alone would remove the whole energy flux at this rate, E cg / Dw (m).

    Along a ray d(E cg)/ds = -Dw: a 1 km decay length means ~10 % of the flux is lost every 100 m.
    """
    energy = RHO * G * hm0**2 / 16
    with np.errstate(divide="ignore", invalid="ignore"):
        return energy * group_velocity(tp, h) / dissipation(hm0, tp, h, gamma, alpha)
