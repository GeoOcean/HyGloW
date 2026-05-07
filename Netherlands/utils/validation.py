import datetime

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
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
