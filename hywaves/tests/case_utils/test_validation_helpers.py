"""Unit tests for utils.validation (metrics tables, comparisons, sweep pooling)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from utils import validation as val


def _sites_ds(values: dict[str, np.ndarray], sites: list[str], sources: list[str], labels: list[str]) -> xr.Dataset:
    return xr.Dataset(
        {name: (("time", "sites"), v) for name, v in values.items()},
        coords={
            "time": pd.date_range("2020-01-01", periods=next(iter(values.values())).shape[0], freq="h"),
            "sites": sites,
            "site_source": ("sites", sources),
            "site_label": ("sites", labels),
        },
    )


class TestValidationHelpers(unittest.TestCase):
    def test_resolve_comparison_variables(self) -> None:
        buoy = xr.Dataset({"hs": (("x",), [1.0]), "tp": (("x",), [8.0])})
        model = xr.Dataset({v: (("x",), [1.0]) for v in ("hs", "tp", "dir", "spr")})
        keys = val.resolve_comparison_variables(buoy, model, variables=["hs", "tp", "dir", "spr"])
        self.assertEqual(keys, ["hs", "tp"])
        with self.assertRaises(ValueError):
            val.resolve_comparison_variables(buoy, model, variables=["spr"])

    def test_compute_metrics_circular_dir(self) -> None:
        ref = np.linspace(0.0, 360.0, 100, endpoint=False)
        m = val.compute_metrics(ref, (ref + 5.0) % 360.0, "dir")
        self.assertLess(abs(m["bias"] - 5.0), 0.5)
        self.assertTrue(np.isnan(m["rrmse"]))

    def test_compute_metrics_reference_first(self) -> None:
        ref = np.array([1.0, 2.0, 3.0, 4.0])
        est = ref + 0.5
        m = val.compute_metrics(ref, est, "hs")
        self.assertEqual(set(val.METRICS) | {"n"}, set(m))
        self.assertGreater(m["bias"], 0.0)
        expected_r2 = 1 - np.sum((est - ref) ** 2) / np.sum((ref - ref.mean()) ** 2)
        self.assertAlmostEqual(m["r2"], expected_r2)
        self.assertAlmostEqual(m["rrmse"], m["rmse"] / ref.mean())

    def test_physical_bounds_drop_corrupted_pairs(self) -> None:
        m = val.compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 8.8e13]), "hs")
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["rmse"], 0.0)

    def test_compare_aligns_and_groups(self) -> None:
        sites = ["10m_0", "10m_1", "buoy_A"]
        meta = dict(sites=sites, sources=["isobath", "isobath", "buoy"], labels=["10m", "10m", "waves"])
        ref = _sites_ds({"hs": np.ones((4, 3))}, **meta)
        est = _sites_ds({"hs": np.full((4, 3), 1.5)}, **meta).isel(time=slice(1, 4))
        result = val.compare(ref, est, variables=["hs"], estimate_label="Model")
        table = result.metrics.set_index("group")
        self.assertEqual(list(table.index), ["10m", "buoy_A"])
        self.assertEqual(table.loc["10m", "n"], 6)
        self.assertAlmostEqual(table.loc["buoy_A", "bias"], 0.5)
        self.assertEqual(table.loc["buoy_A", "site_label"], "waves")
        self.assertEqual(result.estimate_label, "Model")
        self.assertEqual(result.groups, ["10m", "buoy_A"])

    def test_aggregate_sweep_keeps_sets_apart(self) -> None:
        metrics = pd.DataFrame(
            {
                "goal": ["1"] * 4,
                "metamodel": ["buoys", "buoys", "isobaths", "isobaths"],
                "mda_candidate": [50] * 4,
                "group": ["b1", "b2", "5m", "10m"],
                "site_source": ["buoy", "buoy", "isobath", "isobath"],
                "site_label": ["b1", "b2", "5m", "10m"],
                "variable": ["hs"] * 4,
                "n": [1, 3, 2, 2],
                "bias": [0.0, 1.0, 0.1, 0.3],
                "rmse": [1.0, 3.0, 0.2, 0.2],
                "rrmse": [0.1, 0.3, 0.1, 0.1],
                "r2": [0.5, 0.9, 0.8, 0.8],
            }
        )
        out = val.aggregate_sweep(metrics, by="bulk", variables=["hs"]).set_index("metamodel")
        self.assertAlmostEqual(out.loc["buoys", "bias"], 0.75)
        self.assertAlmostEqual(out.loc["buoys", "rmse"], np.sqrt((1 + 27) / 4))
        self.assertEqual(out.loc["isobaths", "n"], 4)
        buoys = val.aggregate_sweep(metrics, by="buoy", variables=["hs"])
        self.assertEqual(sorted(buoys["group"]), ["b1", "b2"])

    def test_closest_goal_distance(self) -> None:
        from bluemath_tk.core.geo import geodesic_distance

        self.assertAlmostEqual(np.radians(geodesic_distance(0.0, 0.0, 1.0, 0.0)) * 6371.0, 111.19, 1)


if __name__ == "__main__":
    unittest.main()
