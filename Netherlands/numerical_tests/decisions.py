"""Tables and plots for the model-decision notebooks, from ``results/*.csv``.

The CSVs are built by ``scripts/collect_results.py`` from the printed scores in
``results/raw/``; nothing here needs the case runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

RESULTS = Path(__file__).resolve().parent / "results"

#: Row order of the buoy tables.
GROUPS = [
    "calm", "storm", "xaver", "all",
    "obs 0-1 m", "obs 1-2 m", "obs 2-3 m", "obs 3-4 m", "obs 4-5 m", "obs 5-6 m", "obs 6-12 m",
    "obs<3, d 0-15", "obs>=3, d 0-15", "obs<3, d 15-25", "obs>=3, d 15-25", "obs<3, d 25-60", "obs>=3, d 25-60",
    "d 0-15 m", "d 15-25 m", "d 25-60 m",
    "obs>=3, d 0-15", "obs>=3, d 15-25", "obs>=3, d 25-60",
]
OBS_CLASSES = ["obs 0-1 m", "obs 1-2 m", "obs 2-3 m", "obs 3-4 m", "obs 4-5 m", "obs 5-6 m", "obs 6-12 m"]


def load(name: str) -> pd.DataFrame:
    """One results table, e.g. ``"buoy_scores"``."""
    return pd.read_csv(RESULTS / f"{name}.csv")


def buoy_table(experiment: str, labels: list[str] | None = None) -> pd.DataFrame:
    """
    Hs vs buoys as ``bias / rmse (n)`` text, groups as rows and variants as columns.

    Parameters
    ----------
    experiment
        ``experiment`` value of ``buoy_scores.csv``.
    labels
        Variants (``label``) to show, in order; default all.
    """
    scores = load("buoy_scores").query("experiment == @experiment")
    labels = labels or list(dict.fromkeys(scores["label"]))
    scores = scores[scores["label"].isin(labels)]
    text = scores.assign(
        cell=scores["bias"].map("{:+.2f}".format) + " / " + scores["rmse"].map("{:.2f}".format)
        + " (" + scores["n"].astype(str) + ")"
    )
    table = text.pivot(index="group", columns="label", values="cell")
    order = [g for g in dict.fromkeys(GROUPS) if g in table.index]
    return table.loc[order, labels]


def plot_by_class(experiment: str, labels: list[str], title: str | None = None) -> Figure:
    """
    Bias and RMSE (m) per observed-Hs class for some variants of an experiment.

    Parameters
    ----------
    experiment
        ``experiment`` value of ``buoy_scores.csv``.
    labels
        Variants (``label``) to plot.
    title
        Figure title; default the experiment name.
    """
    scores = load("buoy_scores").query("experiment == @experiment and group in @OBS_CLASSES")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    x = np.arange(len(OBS_CLASSES))
    width = 0.8 / len(labels)
    for k, label in enumerate(labels):
        rows = scores[scores["label"] == label].set_index("group").reindex(OBS_CLASSES)
        for ax, metric in zip(axes, ("bias", "rmse")):
            ax.bar(x + (k - (len(labels) - 1) / 2) * width, rows[metric], width, label=label)
    n = scores.drop_duplicates("group").set_index("group").reindex(OBS_CLASSES)["n"]
    for ax, metric in zip(axes, ("Hs bias (m)", "Hs RMSE (m)")):
        ax.set_xticks(x, [f"{g[4:]}\nn={int(c)}" for g, c in zip(OBS_CLASSES, n)])
        ax.set_ylabel(metric)
        ax.set_xlabel("observed Hs")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(title or experiment)
    return fig


def ratio_table(experiment: str) -> pd.DataFrame:
    """Hs ratios of the metamodel test cases: (steepness, where) rows, variants as columns."""
    ratios = load("metamodel_ratios").query("experiment == @experiment")
    order = list(dict.fromkeys(ratios["variant"]))
    return ratios.pivot_table(index=["steepness", "where"], columns="variant", values="ratio")[order]


__all__ = ["GROUPS", "OBS_CLASSES", "buoy_table", "load", "plot_by_class", "ratio_table"]
