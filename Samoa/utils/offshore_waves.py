"""Processed offshore-waves data for dynamic SnapWave forcing.

Processed zarrs (see ``07_Download_Process_WHACS.ipynb``) share one schema:

- dims: ``(time, goal)``
- bulk variables: ``hs``, ``tp``, ``dir``, ``spr`` (metamodel partition ``-1``)
- partition variables: ``phs0``…``phs3``, ``ptp0``…``ptp3``, … (partitions ``0``–``3``)
- ``goal`` ids ``1…N`` aligned with ``Samoa_goals.json`` (``seapoint`` is metadata only)

Open catalogs via :func:`utils.config.open_offshore_waves`.
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

GOAL_DIM = "goal"
BOUNDARY_VARS = ("hs", "tp", "dir", "spr")


def boundary_parameters(offshore: xr.Dataset) -> dict[str, list]:
    """Offshore time series → SnapWave dynamic template inputs.

    Returns lists keyed ``hs_goals``, ``tp_goals``, ``wd_goals``, ``ds_goals``,
    ``tref`` (one entry per timestep). ``dir``/``spr`` map to SnapWave ``wd``/``ds``.
    """

    if GOAL_DIM not in offshore.dims:
        raise ValueError(
            f"offshore waves must have a {GOAL_DIM!r} dimension; "
            f"got dims {dict(offshore.sizes)}"
        )
    missing = [v for v in BOUNDARY_VARS if v not in offshore]
    if missing:
        raise ValueError(f"offshore waves missing variables: {missing}")

    goal_ids = sorted(int(g) for g in offshore[GOAL_DIM].values)
    offshore = offshore.sel({GOAL_DIM: goal_ids})

    n_time = offshore.sizes["time"]
    return {
        "hs_goals": [offshore.hs.isel(time=i).values.tolist() for i in range(n_time)],
        "tp_goals": [offshore.tp.isel(time=i).values.tolist() for i in range(n_time)],
        "dir_goals": [offshore.dir.isel(time=i).values.tolist() for i in range(n_time)],
        "spr_goals": [offshore.spr.isel(time=i).values.tolist() for i in range(n_time)],
        "tref": pd.to_datetime(offshore.time.values).strftime("%Y%m%d %H%M%S").tolist(),
    }
