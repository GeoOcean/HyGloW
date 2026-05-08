import numpy as np
import pandas as pd
from bluemath_tk.datamining.lhs import LHS
from bluemath_tk.waves.series import waves_dispersion


def get_real_scenarios_from_dataset(
    lhs_dataset: pd.DataFrame, h0: float
) -> pd.DataFrame:
    """
    Function to filter the LHS dataset to obtain real scenarios.
    It filters out scenarios that are not physically realistic based on the wave dispersion relation.

    Parameters
    ----------
    lhs_dataset : pd.DataFrame
        DataFrame containing the Latin Hypercube Sampling data with columns:
        'hs', 'tp', 'wd', 'ds'
    h0 : float
        Water depth in meters.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only physically realistic scenarios.
    """

    # Calculation of the wave length and other parameters to avoid unphysical values
    df_centroids = lhs_dataset.copy()
    df_centroids["L"] = [waves_dispersion(i, h0)[0] for i in df_centroids["tp"]]
    df_centroids["h/L"] = h0 / df_centroids["L"]
    df_centroids["kh"] = (2 * np.pi / df_centroids["L"]) * h0
    df_centroids = df_centroids.loc[(df_centroids["kh"] < 1) & (df_centroids["tp"] > 7)]
    df_centroids = df_centroids.loc[(df_centroids["h/L"] < 0.5)]
    lhs_dataset = lhs_dataset.loc[df_centroids.index]

    return lhs_dataset


def generate_cases(goals_config: dict, num_samples_per_goal: int = 100) -> pd.DataFrame:
    """
    Function to generate cases for each goal node.
    """

    list_of_lhs_dfs = []

    for goal_node, goal_values in goals_config.items():
        variable_names = ["tp", "wd", "ds", "zs"]
        lower_bounds = [1, goal_values["angles"][0], 1, -3.5]
        upper_bounds = [15, goal_values["angles"][1], 45, 3.5]
        lhs = LHS(num_dimensions=4)
        lhs_df = lhs.generate(
            dimensions_names=variable_names,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            num_samples=num_samples_per_goal,
        )
        lhs_df["wd"] = (90 - lhs_df["wd"]) % 360
        list_of_lhs_dfs.append(lhs_df)

    return (
        pd.concat(list_of_lhs_dfs, keys=goals_config.keys())
        .reset_index()
        .rename(columns={"level_0": "goal"})
        .drop(columns=["level_1"])
    )
