"""Minimal unit tests for validation helpers (no fitted models required)."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from utils.evaluation import validation as val


class TestValidationHelpers(unittest.TestCase):
    def test_scatter_grid_shape(self) -> None:
        self.assertEqual(val._scatter_grid_shape(1), (1, 1))
        self.assertEqual(val._scatter_grid_shape(2), (1, 2))
        self.assertEqual(val._scatter_grid_shape(4), (2, 2))
        self.assertGreaterEqual(val._scatter_grid_shape(5)[0], 2)

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

    def test_metrics_table_circular_dir(self) -> None:
        n = 100
        ref_dir = np.linspace(0.0, 360.0, n, endpoint=False)
        est_dir = (ref_dir + 5.0) % 360.0
        m = val.compute_metrics(est_dir, ref_dir, circular=True)
        self.assertLess(abs(m["bias"] - 5.0), 0.5)
        self.assertGreater(m["r2"], 0.9)


if __name__ == "__main__":
    unittest.main()
