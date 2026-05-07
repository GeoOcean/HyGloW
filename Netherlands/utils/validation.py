import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from bluemath_tk.core.plotting.base_plotting import DefaultStaticPlotting
from bluemath_tk.core.plotting.old_scatter import validation_scatter
from matplotlib.gridspec import GridSpec

era5_lons = [7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
era5_lats = [54.0, 54.0, 54.0, 54.0, 54.0, 53.5, 53.0, 52.5, 52.0, 51.5]


def coincident_non_values(array1, array2):
    non_nan_values = np.where(~np.isnan(array1) & ~np.isnan(array2))[0]
    return array1[non_nan_values], array2[non_nan_values]


def _plot_validation_scatter(ax, x, y, xlabel, ylabel, title):
    x_vals, y_vals = coincident_non_values(x, y)
    if x_vals.size == 0:
        ax.set_title(f"{title}\n(no overlapping non-NaN values)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return

    validation_scatter(
        axs=ax,
        x=x_vals,
        y=y_vals,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        cmap="rainbow",
    )


def plot_buoy_validation_figure(
    observation_data,
    metamodel_data,
    dynamic_data,
    era5_data,
    buoy_idx,
    figsize=(10, 10),
):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=3, ncols=4, figure=fig)

    ax_ts = fig.add_subplot(gs[2, :])
    ax_map = fig.add_subplot(gs[:2, 2:], projection=ccrs.PlateCarree())
    scatter_axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    observation_data.isel(buoy=buoy_idx).Hm0.plot(
        ax=ax_ts, label="Buoy", color="darkblue"
    )
    metamodel_data.isel(stations=buoy_idx).bulk_hm0.plot(
        ax=ax_ts, label="Metamodel - Bulk", color="red", alpha=0.6
    )
    metamodel_data.isel(stations=buoy_idx).sum_hm0.plot(
        ax=ax_ts, label="Metamodel - Sum", color="crimson"
    )
    dynamic_data.isel(stations=buoy_idx).point_hm0.plot(
        ax=ax_ts, label="Dynamic", color="darkgreen", alpha=0.6
    )
    era5_data.sel(
        goal=observation_data.isel(buoy=buoy_idx).closest_goal.values
    ).swh.plot(ax=ax_ts, label="ERA5", color="orange", alpha=0.6)
    ax_ts.legend(loc="upper right")
    ax_ts.set_xlim(datetime.datetime(2011, 12, 1), datetime.datetime(2011, 12, 31))
    ax_ts.set_title("")
    ax_ts.set_xlabel("Time")
    ax_ts.set_ylabel("Hs [m]")

    lon, lat = (
        observation_data.isel(buoy=buoy_idx).lon.values,
        observation_data.isel(buoy=buoy_idx).lat.values,
    )
    if lon is not None and lat is not None:
        plot = DefaultStaticPlotting()
        plot.plot_satellite(ax=ax_map, area=[2.4, 7.1, 51.4, 54.1])
        ax_map.scatter(
            lon,
            lat,
            s=100,
            marker="*",
            color="red",
            label=f"Buoy {int(observation_data.buoy.isel(buoy=buoy_idx).values)}",
            zorder=10,
        )
        ax_map.text(
            lon,
            lat,
            f"Depth: {-observation_data.isel(buoy=buoy_idx).depth.values[0]} m",
            fontsize=12,
            color="black",
            bbox=dict(facecolor="white", alpha=0.8),
            position=(lon + 0.05, lat + 0.05),
        )
        ax_map.scatter(
            era5_lons,
            era5_lats,
            s=50,
            color="orange",
            alpha=0.6,
            label="ERA5",
            zorder=10,
        )
        for era5_lon, era5_lat, depth in zip(
            era5_lons, era5_lats, era5_data.isel(time=0).wmb.values
        ):
            ax_map.text(
                era5_lon,
                era5_lat,
                depth,
                fontsize=12,
                color="black",
                bbox=dict(facecolor="white", alpha=0.8),
                position=(era5_lon + 0.05, era5_lat + 0.05),
            )
        ax_map.scatter(
            era5_data.sel(
                goal=observation_data.isel(buoy=buoy_idx).closest_goal.values
            ).longitude.values,
            era5_data.sel(
                goal=observation_data.isel(buoy=buoy_idx).closest_goal.values
            ).latitude.values,
            s=100,
            color="purple",
            label="Closest ERA5 Point",
            zorder=11,
        )
        ax_map.legend(loc="lower right")
    else:
        ax_map.text(0.5, 0.5, "No lat/lon found", ha="center", va="center")

    _plot_validation_scatter(
        scatter_axes[0],
        observation_data.isel(buoy=buoy_idx).Hm0.values,
        metamodel_data.isel(stations=buoy_idx).bulk_hm0.values,
        "Observations",
        "Metamodel - Bulk",
        "",
    )
    _plot_validation_scatter(
        scatter_axes[1],
        observation_data.isel(buoy=buoy_idx).Hm0.values,
        metamodel_data.isel(stations=buoy_idx).sum_hm0.values,
        "Observations",
        "Metamodel - Sum",
        "",
    )
    try:
        _plot_validation_scatter(
            scatter_axes[2],
            observation_data.isel(buoy=buoy_idx).Hm0.values,
            dynamic_data.isel(stations=buoy_idx).point_hm0.values,
            "Observations",
            "Dynamic",
            "",
        )
    except Exception as e:
        print(e)
        pass
    _plot_validation_scatter(
        scatter_axes[3],
        observation_data.isel(buoy=buoy_idx).Hm0.values,
        era5_data.sel(
            goal=observation_data.isel(buoy=buoy_idx).closest_goal.values
        ).swh.values,
        "Observations",
        "ERA5",
        "",
    )

    return fig


def plot_all_buoy_validation_figures(
    observation_data,
    metamodel_data,
    dynamic_data,
    era5_data,
    figsize=(20, 12),
):
    n_buoys = observation_data.sizes["buoy"]
    figures = []
    for buoy_idx in range(n_buoys):
        fig = plot_buoy_validation_figure(
            observation_data=observation_data,
            metamodel_data=metamodel_data,
            dynamic_data=dynamic_data,
            era5_data=era5_data,
            buoy_idx=buoy_idx,
            figsize=figsize,
        )
        figures.append(fig)

    return figures


def _get_data_array(dataset, var_name, selector=None):
    """Return a 1D DataArray for plotting from a dataset and variable name."""
    data = dataset[var_name]
    if selector is not None:
        data = data.isel(**selector)
    return data


def _theta_to_360(values):
    """Normalize direction values to [0, 360)."""
    return np.mod(values, 360.0)


def plot_series_validation_hs_tp_thetam(
    dataset_a,
    dataset_b,
    var_map_a,
    var_map_b,
    selector_a=None,
    selector_b=None,
    label_a="Dataset A",
    label_b="Dataset B",
    time_dim="time",
    title=None,
    figsize=(14, 7),
):
    """
    Plot Hs, Tp and ThetaM time-series comparison for two datasets.

    Parameters
    ----------
    dataset_a, dataset_b : xarray.Dataset
        Datasets to compare.
    var_map_a, var_map_b : dict
        Variable mappings with keys: "Hs", "Tp", "ThetaM".
        Example: {"Hs": "Hm0", "Tp": "Tp", "ThetaM": "ThetaM"}.
    selector_a, selector_b : dict, optional
        xarray .isel selectors to pick one point/station per dataset.
    """
    required_vars = ("Hs", "Tp", "ThetaM")
    for key in required_vars:
        if key not in var_map_a or key not in var_map_b:
            raise ValueError(
                f"Missing mapping for '{key}'. Required keys: {required_vars}"
            )

    series_a = {
        key: _get_data_array(dataset_a, var_map_a[key], selector=selector_a)
        for key in required_vars
    }
    series_b = {
        key: _get_data_array(dataset_b, var_map_b[key], selector=selector_b)
        for key in required_vars
    }

    if time_dim in series_a["Hs"].dims and time_dim in series_b["Hs"].dims:
        aligned_a = xr.Dataset(series_a)
        aligned_b = xr.Dataset(series_b)
        aligned_a, aligned_b = xr.align(aligned_a, aligned_b, join="inner")
        x_values = aligned_a[time_dim].values
        values_a = {key: aligned_a[key].values for key in required_vars}
        values_b = {key: aligned_b[key].values for key in required_vars}
    else:
        x_values = np.arange(series_a["Hs"].size)
        values_a = {key: np.asarray(series_a[key].values) for key in required_vars}
        values_b = {key: np.asarray(series_b[key].values) for key in required_vars}

    values_a["ThetaM"] = _theta_to_360(values_a["ThetaM"])
    values_b["ThetaM"] = _theta_to_360(values_b["ThetaM"])

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=figsize)

    plot_config = [
        ("Hs", "Hs [m]"),
        ("Tp", "Tp [s]"),
        ("ThetaM", "ThetaM [deg]"),
    ]

    for ax, (var_key, ylabel) in zip(axes, plot_config):
        ax.plot(x_values, values_a[var_key], color="blue", lw=0.8, label=label_a)
        ax.plot(
            x_values, values_b[var_key], color="red", lw=0.8, label=label_b, alpha=0.8
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[2].set_xlabel("Time" if time_dim in series_a["Hs"].dims else "Index")
    axes[2].set_ylim(0, 360)
    axes[2].set_yticks([0, 90, 180, 270, 360])
    axes[2].set_yticklabels(["N", "E", "S", "W", "N"])
    axes[0].legend(loc="upper right")

    if title is not None:
        axes[0].set_title(title)

    fig.tight_layout()
    return fig


def plot_multi_point_series_validation(
    dataset_a,
    dataset_b,
    var_map_a,
    var_map_b,
    points_a,
    points_b=None,
    point_dim_a="point",
    point_dim_b="point",
    label_a="Dataset A",
    label_b="Dataset B",
    time_dim="time",
    figsize=(14, 7),
):
    """
    Iterate over several points and create Hs/Tp/ThetaM validation figures.

    Parameters
    ----------
    points_a : sequence of int
        Indices for dataset_a points/stations.
    points_b : sequence of int, optional
        Indices for dataset_b. If None, points_a is reused.
    """
    if points_b is None:
        points_b = points_a
    if len(points_a) != len(points_b):
        raise ValueError("points_a and points_b must have same length.")

    figures = []
    for idx_a, idx_b in zip(points_a, points_b):
        selector_a = {point_dim_a: idx_a}
        selector_b = {point_dim_b: idx_b}
        fig = plot_series_validation_hs_tp_thetam(
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            var_map_a=var_map_a,
            var_map_b=var_map_b,
            selector_a=selector_a,
            selector_b=selector_b,
            label_a=label_a,
            label_b=label_b,
            time_dim=time_dim,
            title=f"Validation point A:{idx_a} vs B:{idx_b}",
            figsize=figsize,
        )
        figures.append(fig)

    return figures
