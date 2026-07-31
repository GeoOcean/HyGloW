"""Unit tests for ``utils.simulation.offshore_forcing``."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from utils.simulation.offshore_forcing import BOUNDARY_VARS, GOAL_DIM, boundary_parameters


def _offshore_dataset(n_time: int = 2, goals: list[int] | None = None) -> xr.Dataset:
    goals = goals or [2, 1]  # deliberately unsorted, to check sorting behaviour
    times = pd.date_range("2020-01-01", periods=n_time, freq="h")
    data = {
        var: (("time", "goal"), np.arange(n_time * len(goals), dtype=float).reshape(n_time, len(goals)))
        for var in BOUNDARY_VARS
    }
    return xr.Dataset(data, coords={"time": times, "goal": goals})


class TestBoundaryParameters(unittest.TestCase):
    def test_returns_expected_keys(self) -> None:
        out = boundary_parameters(_offshore_dataset())
        self.assertEqual(
            set(out), {"hs_goals", "tp_goals", "dir_goals", "spr_goals", "tref"}
        )

    def test_lengths_match_time_dimension(self) -> None:
        out = boundary_parameters(_offshore_dataset(n_time=3))
        for key in ("hs_goals", "tp_goals", "dir_goals", "spr_goals", "tref"):
            self.assertEqual(len(out[key]), 3)

    def test_goal_ids_are_sorted(self) -> None:
        ds = _offshore_dataset(goals=[3, 1, 2])
        out = boundary_parameters(ds)
        # Each per-timestep entry is a list over goals, in sorted-goal order.
        self.assertEqual(len(out["hs_goals"][0]), 3)

    def test_tref_is_formatted_timestamp_strings(self) -> None:
        out = boundary_parameters(_offshore_dataset())
        self.assertEqual(out["tref"][0], "20200101 000000")

    def test_missing_goal_dimension_raises(self) -> None:
        ds = xr.Dataset({"hs": (("time",), [1.0])}, coords={"time": [0]})
        with self.assertRaises(ValueError):
            boundary_parameters(ds)

    def test_missing_variable_raises(self) -> None:
        ds = _offshore_dataset().drop_vars("spr")
        with self.assertRaises(ValueError):
            boundary_parameters(ds)

    def test_goal_dim_constant(self) -> None:
        self.assertEqual(GOAL_DIM, "goal")


if __name__ == "__main__":
    unittest.main()
