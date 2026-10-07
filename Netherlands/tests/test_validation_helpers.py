"""Minimal unit tests for validation helpers (no fitted models required)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from utils import validation as val


class TestValidationHelpers(unittest.TestCase):
    def test_scatter_grid_shape(self) -> None:
        self.assertEqual(val.scatter_grid_shape(1), (1, 1))
        self.assertEqual(val.scatter_grid_shape(2), (1, 2))
        self.assertEqual(val.scatter_grid_shape(4), (2, 2))
        self.assertGreaterEqual(val.scatter_grid_shape(5)[0], 2)

    def test_available_metrics_variables(self) -> None:
        est = xr.Dataset(
            {
                "hs": (("x",), [1.0, 2.0]),
                "tp": (("x",), [10.0, 11.0]),
            }
        )
        ref = xr.Dataset(
            {
                "hs": (("x",), [1.0, 2.0]),
                "tp": (("x",), [10.0, 11.0]),
                "dir": (("x",), [90.0, 180.0]),
            }
        )
        keys = val._available_metrics_variables(
            est, ref, variables=["hs", "tp", "dir", "spr"]
        )
        self.assertEqual(keys, ["hs", "tp"])

    def test_resolve_comparison_variables(self) -> None:
        buoy = xr.Dataset({"hs": (("x",), [1.0]), "tp": (("x",), [8.0])})
        model = xr.Dataset(
            {
                "hs": (("x",), [1.0]),
                "tp": (("x",), [8.0]),
                "dir": (("x",), [270.0]),
                "spr": (("x",), [30.0]),
            }
        )
        keys = val.resolve_comparison_variables(
            buoy, model, variables=["hs", "tp", "dir", "spr"]
        )
        self.assertEqual(keys, ["hs", "tp"])

    def test_metrics_table_circular_dir(self) -> None:
        n = 100
        ref_dir = np.linspace(0.0, 360.0, n, endpoint=False)
        est_dir = (ref_dir + 5.0) % 360.0
        m = val.compute_metrics(ref_dir, est_dir, circular=True)
        self.assertLess(abs(m["bias"] - 5.0), 0.5)
        self.assertGreater(m["r2"], 0.9)
        self.assertTrue(np.isnan(m["rrmse"]))

    def test_compute_metrics_reference_first(self) -> None:
        ref = np.array([1.0, 2.0, 3.0, 4.0])
        est = 1.1 * ref
        m = val.compute_metrics(ref, est)
        self.assertEqual(set(val.METRICS) | {"n"}, set(m))
        # positive bias = overestimation; R² normalised by the reference
        self.assertGreater(m["bias"], 0.0)
        expected_r2 = 1 - np.sum((est - ref) ** 2) / np.sum((ref - ref.mean()) ** 2)
        self.assertAlmostEqual(m["r2"], expected_r2)
        self.assertAlmostEqual(m["rrmse"], m["rmse"] / ref.mean())

    def test_physical_bounds_drop_corrupted_pairs(self) -> None:
        ref = np.array([1.0, 2.0, 3.0])
        est = np.array([1.0, 2.0, 8.8e13])
        m = val.compute_metrics(ref, est, variable="hs")
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["rmse"], 0.0)

    def test_weighted_stats_pools_rmse(self) -> None:
        group = pd.DataFrame(
            {"n": [1, 3], "bias": [0.0, 1.0], "rmse": [1.0, 3.0], "r2": [0.5, 0.9]}
        )
        out = val._weighted_stats(group)
        self.assertAlmostEqual(out["bias"], 0.75)
        self.assertAlmostEqual(out["rmse"], np.sqrt((1 * 1 + 3 * 9) / 4))
        self.assertTrue(np.isnan(out["rrmse"]))  # older CSV without the column
        self.assertEqual(out["n"], 4)

    def test_haversine_matches_known_distance(self) -> None:
        # one degree of latitude ~ 111.19 km on the 6371 km sphere
        self.assertAlmostEqual(val._haversine_m(0.0, 0.0, 0.0, 1.0) / 1e3, 111.19, 1)


if __name__ == "__main__":
    unittest.main()
