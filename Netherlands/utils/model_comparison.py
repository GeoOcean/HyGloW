"""
SnapWave against SWAN on the same cases (Netherlands, notebooks 05c and 08c).

Case-only code: end users compare the two models here to select one. SnapWave
is read from its postprocessed files; SWAN from the ``output_sites.tab`` of the
finished cases, so both notebooks work while the SLURM arrays are running.
SnapWave is the reference only in the sense of the metrics convention (bias =
SWAN - SnapWave): neither model is truth, the buoys in 08c decide that.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.core.metrics import compute_metrics
from bluemath_tk.wrappers.snapwave.snapwave_plots import map_triangulation
from bluemath_tk.wrappers.swan.swan_utils import (
    read_adcirc_grid,
    read_swan_map,
    read_swan_table,
)
from matplotlib.figure import Figure

from utils import simulation as sim
from utils.config import model_config, open_wind, resolve
from utils.validation import METRICS, PHYSICAL_BOUNDS

#: SWAN ``TABLE`` quantities of the case templates, in order.
TABLE_QUANTITIES = ["HSIGN", "TPS", "TMM10", "DIR", "DSPR"]
VARIABLES = ["hs", "tp", "dir", "spr"]
HS_MAX = PHYSICAL_BOUNDS["hs"][1]
#: Sites below this share of the offshore Hs carry no energy in a metamodel case.
MIN_HS_FRACTION = 0.1
#: Thresholds of :func:`suspect_hours`.
SUSPECT_HS = 3.0
SUSPECT_RATIO = 2.0
READ_THREADS = 16


def load_snapwave_metamodel(goals: dict) -> xr.Dataset:
    """SnapWave metamodel cases of every goal on ``(case_num, sites)``, with ``goal``."""
    folder = sim.cases_dir("metamodel", model="snapwave")
    parts = []
    for goal in goals:
        with xr.open_dataset(folder / f"output_sites_{goal}.nc") as ds:
            part = ds.load()
        parts.append(part.assign_coords(goal=("case_num", np.full(part.sizes["case_num"], int(goal)))))
    return xr.concat(parts, dim="case_num", data_vars="minimal", coords="minimal", compat="override")


def load_snapwave_dynamic(offshore: str, wind: bool, period: str) -> xr.Dataset:
    """
    SnapWave dynamic cases on ``(time, sites)`` of one run (notebook 08).

    Parameters
    ----------
    offshore, wind
        The run (:func:`utils.simulation.cases_dir`); kept in the attrs, so
        :func:`case_folders` finds the SWAN run with the same forcing.
    period
        ``Dynamic_Cases_{period}.nc`` of the run.

    Returns
    -------
    xarray.Dataset
    """
    folder = sim.cases_dir("dynamic", "snapwave", offshore=offshore, wind=wind)
    with xr.open_dataset(folder / f"Dynamic_Cases_{period}.nc") as ds:
        dynamic = ds[VARIABLES].load()
    return dynamic.assign_attrs(offshore=offshore, wind=int(wind))


def case_folders(reference: xr.Dataset) -> dict:
    """SWAN case folder of each row of *reference* (``case_num`` or ``time``)."""
    if "case_num" in reference.dims:
        root = sim.cases_dir("metamodel", model="swan")
        return {
            int(case): root / f"{int(goal):02}" / f"{int(case):04}"
            for case, goal in zip(reference.case_num.values, reference.goal.values)
        }
    run = {"offshore": reference.attrs["offshore"], "wind": bool(reference.attrs["wind"])}
    root = sim.cases_dir("dynamic", "swan", **run)
    return {
        time: root / pd.Timestamp(time).strftime("%Y%m%d %H%M%S")
        for time in reference.time.values
    }


def _read_table(folder: Path) -> np.ndarray | None:
    if not (folder / "norm_end").exists():
        return None
    return read_swan_table(folder / "output_sites.tab", TABLE_QUANTITIES)[VARIABLES].to_numpy(dtype="float32")


def load_swan(reference: xr.Dataset) -> xr.Dataset:
    """
    The finished SWAN cases of *reference*, on its rows and sites.

    Parameters
    ----------
    reference
        SnapWave cases (:func:`load_snapwave_metamodel` or
        :func:`load_snapwave_dynamic`); both models write the same ``ok`` sites
        in the same order.

    Returns
    -------
    xarray.Dataset
        ``hs``, ``tp``, ``dir``, ``spr`` with the site coordinates of *reference*.
    """
    folders = case_folders(reference)
    with ThreadPoolExecutor(READ_THREADS) as pool:
        tables = dict(zip(folders, pool.map(_read_table, folders.values())))
    rows = [row for row, table in tables.items() if table is not None]
    if not rows:
        raise ValueError("No finished SWAN case")
    values = np.stack([tables[row] for row in rows])
    dim = "case_num" if "case_num" in reference.dims else "time"
    site_coords = {k: v for k, v in reference.coords.items() if "sites" in v.dims}
    estimate = xr.Dataset(
        {var: ((dim, "sites"), values[..., i]) for i, var in enumerate(VARIABLES)},
        coords={dim: rows, **site_coords},
    )
    if dim == "case_num":
        estimate = estimate.assign_coords(goal=(dim, reference.goal.sel(case_num=rows).values))
    return estimate


def hs_ratio(reference: xr.DataArray, estimate: xr.DataArray, dim: str | list[str] | None = None) -> xr.DataArray:
    """
    Energy-based Hs ratio ``sqrt(sum(SWAN^2) / sum(SnapWave^2))`` over *dim*.

    A ratio of means of energy, so sites and cases without energy weigh
    nothing (a median of pointwise ratios is mostly noise there).
    """
    valid = reference.notnull() & estimate.notnull() & (reference <= HS_MAX) & (estimate <= HS_MAX)
    return np.sqrt(
        (estimate.where(valid) ** 2).sum(dim) / (reference.where(valid) ** 2).sum(dim)
    )


def with_energy(reference: xr.Dataset) -> xr.DataArray:
    """Sites of each metamodel case reached by at least :data:`MIN_HS_FRACTION` of its Hs."""
    return reference["hs"] >= MIN_HS_FRACTION * reference["hs_forcing"]


def energetic(ds: xr.Dataset, reference: xr.Dataset) -> xr.Dataset:
    """*ds* on the sites of each metamodel case that get energy (:func:`with_energy`)."""
    return ds[VARIABLES].where(with_energy(reference))


def goal_direction_ratio(
    reference: xr.Dataset, estimate: xr.Dataset, width: float = 45.0
) -> pd.DataFrame:
    """Hs ratio per goal (rows) and offshore direction bin (columns), metamodel cases."""
    reference = reference.sel(case_num=estimate.case_num)
    hs_ref = energetic(reference, reference)["hs"]
    table = pd.DataFrame(
        {
            "goal": reference["goal"].values,
            "bin": (reference["dir_forcing"].values // width * width).astype(int),
            "ratio": hs_ratio(hs_ref, estimate["hs"], "sites").values,
        }
    )
    return table.pivot_table(index="goal", columns="bin", values="ratio", aggfunc="median")


def plot_goal_direction(table: pd.DataFrame, ax: plt.Axes | None = None) -> Figure:
    """Heat map of :func:`goal_direction_ratio`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    mesh = ax.pcolormesh(table.to_numpy(), cmap="RdBu_r", vmin=0.0, vmax=2.0)
    ax.set_xticks(np.arange(table.shape[1]) + 0.5, table.columns)
    ax.set_yticks(np.arange(table.shape[0]) + 0.5, table.index)
    for (row, col), value in np.ndenumerate(table.to_numpy()):
        if np.isfinite(value):
            ax.text(col + 0.5, row + 0.5, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("Offshore direction, bin start (deg, coming from)")
    ax.set_ylabel("Goal")
    ax.figure.colorbar(mesh, ax=ax, label="Hs ratio SWAN / SnapWave (median over cases)")
    return ax.figure


def hs_by_class(
    reference: xr.DataArray, estimate: xr.DataArray, classes: xr.DataArray, edges: list[float]
) -> pd.DataFrame:
    """
    Hs metrics per class of *classes* (same or fewer dims than *reference*).

    Parameters
    ----------
    reference, estimate
        Aligned Hs of SnapWave and SWAN.
    classes
        Value that sets the class: Hs itself, or a value per case (wind, ...).
    edges
        Class edges.

    Returns
    -------
    pandas.DataFrame
        Hs ratio and metrics (SnapWave first) per class.
    """
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (classes >= low) & (classes < high)
        ref, est = reference.where(inside), estimate.where(inside)
        rows.append(
            {
                "class": f"{low:g}-{high:g}",
                "ratio": float(hs_ratio(ref, est)),
                **compute_metrics(ref.values, est.values, METRICS, bounds=PHYSICAL_BOUNDS["hs"]),
            }
        )
    return pd.DataFrame(rows).set_index("class")


def suspect_hours(hs: xr.DataArray, offshore_hs: xr.DataArray) -> xr.DataArray:
    """
    Hours whose largest Hs cannot come from the forcing: an unstable SnapWave run.

    The SnapWave dynamic runs have hours with Hs of 5-10 m at thousands of sites
    under 0.5 m offshore, alternating with sound hours. Flagged when the largest
    site Hs passes 3 m and twice the largest goal Hs (normal hours stay near 1).

    Parameters
    ----------
    hs
        SnapWave Hs on ``(time, sites)``.
    offshore_hs
        Offshore Hs on ``(goal, time)`` at the same hours.

    Returns
    -------
    xarray.DataArray
        True at the suspect hours.
    """
    largest = hs.where(hs <= HS_MAX).max("sites")
    return (largest > SUSPECT_HS) & (largest > SUSPECT_RATIO * offshore_hs.max("goal"))


def mean_wind_speed(times: xr.DataArray) -> xr.DataArray:
    """ERA5 10 m wind speed averaged over the case area, at *times*."""
    wind = open_wind(slice(times.values[0], times.values[-1]))
    speed = np.hypot(wind["u10"], wind["v10"]).mean(["latitude", "longitude"]).load()
    return speed.sel(time=times)


def plot_site_ratio(
    ratio: xr.DataArray, ax: plt.Axes | None = None, title: str = "Hs ratio SWAN / SnapWave"
) -> Figure:
    """Map of an Hs ratio per output site."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    points = ax.scatter(
        ratio["site_longitude"], ratio["site_latitude"], c=ratio, s=6,
        cmap="RdBu_r", vmin=0.5, vmax=1.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect(1.6)
    ax.figure.colorbar(points, ax=ax, extend="both", label="Hs ratio")
    return ax.figure


def map_rows(reference: xr.Dataset) -> list:
    """Rows of *reference* with a grid map in both models."""
    return [
        row
        for row, folder in case_folders(reference).items()
        if (folder / "output_map.mat").exists()
        and (_snapwave_folder(folder) / "output_map.nc").exists()
    ]


def _snapwave_folder(swan_folder: Path) -> Path:
    """The same case in SnapWave: the SWAN cases folder name with SnapWave's base name."""
    kind = "metamodel" if swan_folder.parent.name.isdigit() else "dynamic"
    swan_root = swan_folder.parents[1] if kind == "metamodel" else swan_folder.parent
    swan_base, snapwave_base = (
        resolve(model_config(model)[f"{kind}_cases"]) for model in ("swan", "snapwave")
    )
    snapwave_root = snapwave_base.parent / swan_root.name.replace(
        swan_base.name, snapwave_base.name, 1
    )
    return snapwave_root / swan_folder.relative_to(swan_root)


def plot_map_comparison(swan_folder: Path, vmax: float | None = None) -> Figure:
    """
    Hs of one case on the mesh: SnapWave, SWAN and their difference.

    Parameters
    ----------
    swan_folder
        SWAN case with a map (see :func:`map_rows`); both meshes share their nodes.
    vmax
        Upper Hs limit; default the largest SnapWave Hs below the physical bound.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with xr.open_dataset(_snapwave_folder(swan_folder) / "output_map.nc") as ds:
        triangulation = map_triangulation(ds)
        snapwave = ds["hm0"].isel(time=0).values.astype(float)
    swan = read_swan_map(swan_folder / "output_map.mat", read_adcirc_grid(sim.grid_file("swan")))
    swan = swan["hs"].values
    snapwave = np.where(snapwave > HS_MAX, np.nan, snapwave)
    vmax = vmax or float(np.nanmax(snapwave))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    panels = (
        ("SnapWave Hs", np.nan_to_num(snapwave), {"cmap": "viridis", "vmin": 0, "vmax": vmax}, "m"),
        ("SWAN Hs", np.nan_to_num(swan), {"cmap": "viridis", "vmin": 0, "vmax": vmax}, "m"),
        ("SWAN - SnapWave", np.nan_to_num(swan - snapwave), {"cmap": "RdBu_r", "vmin": -1, "vmax": 1}, "m"),
    )
    for ax, (title, values, style, units) in zip(axes, panels):
        mesh = ax.tripcolor(triangulation, values, shading="gouraud", **style)
        ax.set_title(title)
        ax.set_aspect(1.6)
        fig.colorbar(mesh, ax=ax, shrink=0.8, label=units)
    fig.suptitle(swan_folder.relative_to(swan_folder.parents[1]))
    fig.tight_layout()
    return fig


def _swan_minutes(folder: Path) -> float | None:
    start, end = folder / "swaninit", folder / "norm_end"
    if not end.exists():
        return None
    return (end.stat().st_mtime - start.stat().st_mtime) / 60.0


def _snapwave_minutes(folder: Path) -> float:
    start, end = folder / "wrapper_error.log", folder / "output_sites.nc"
    return (end.stat().st_mtime - start.stat().st_mtime) / 60.0


def runtimes(reference: xr.Dataset, n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Minutes per case of each model on *n* random finished cases (one CPU task)."""
    folders = list(case_folders(reference).values())
    rng = np.random.default_rng(seed)
    sample = [folders[i] for i in rng.permutation(len(folders))]
    swan = [m for m in map(_swan_minutes, sample[: 3 * n]) if m is not None][:n]
    snapwave = [_snapwave_minutes(_snapwave_folder(f)) for f in sample[:n]]
    return pd.DataFrame({"SnapWave": snapwave}).describe().join(
        pd.DataFrame({"SWAN": swan}).describe(), how="outer"
    ).loc[["count", "50%", "max"]]
