"""Unit tests for utils.validation (metrics tables, comparisons, sweep pooling)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_load_buoy_observations(self) -> None:
        time = pd.date_range("2013-12-01", periods=48, freq="h")
        with tempfile.TemporaryDirectory() as tmp:
            hs = np.linspace(1.0, 3.0, 48)
            xr.Dataset({"hs": ("time", hs), "dspr": ("time", np.full(48, 30.0))},
                       coords={"time": time}).to_netcdf(Path(tmp) / "A.nc")
            xr.Dataset({"hs": ("time", np.full(48, np.nan))}, coords={"time": time}).to_netcdf(Path(tmp) / "B.nc")
            with mock.patch.object(val, "resolve", lambda _: Path(tmp)), \
                    mock.patch.object(val.obs, "instrument_id", lambda s: s.removeprefix("buoy_")), \
                    mock.patch.object(val, "attach_site_meta", lambda ds: ds):
                buoy = val.load_buoy_observations(slice("2013-12-01", "2013-12-01 23:00"), ["buoy_A", "buoy_B"])
                year = val.load_buoy_observations("2013", ["buoy_A"])
                with self.assertRaises(ValueError):  # a year outside every record: no buoy left
                    val.load_buoy_observations("2018", ["buoy_A"])
        self.assertEqual(year.sizes["time"], 48)
        self.assertEqual(list(buoy.sites.values), ["buoy_A"])
        self.assertEqual(set(buoy.data_vars), {"hs", "spr"})
        self.assertEqual(buoy.sizes["time"], 24)
        self.assertEqual(buoy["hs"].attrs["units"], "m")

    def test_buoy_spectra_path(self) -> None:
        with mock.patch.object(val, "resolve", lambda p: Path("/case") / p):
            path = val.buoy_spectra_path("buoy_Aunuu_1")
        self.assertEqual(path, Path("/case/inputs/Observations/processed/spectra/Aunuu_1.nc"))

    def test_event_peaks(self) -> None:
        time = pd.date_range("2013-12-05", periods=10, freq="h")
        ref = xr.Dataset({"hs": (("time", "sites"), np.r_[np.zeros(4), 5.0, np.zeros(5)][:, None])},
                         coords={"time": time, "sites": ["a"]})
        est = xr.Dataset({"hs": (("time", "sites"), np.r_[np.zeros(6), 4.0, np.zeros(3)][:, None])},
                         coords={"time": time, "sites": ["a"]})
        peaks = val.event_peaks(ref, {"model": est})
        self.assertEqual(list(peaks.index), [("a", "Buoy"), ("a", "model")])
        self.assertEqual(peaks.loc[("a", "Buoy"), "peak"], 5.0)
        self.assertEqual(peaks.loc[("a", "model"), "minus Buoy"], -1.0)
        self.assertEqual(peaks.loc[("a", "model"), "lag (h)"], 2.0)

    def test_on_sites_of_and_storm_peaks(self) -> None:
        like = _sites_ds({"hs": np.ones((3, 2))}, ["a", "b"], ["isobath", "buoy"], ["10m", "waves"])
        like = like.assign_coords(site_longitude=("sites", [1.0, 2.0]))
        run = _sites_ds({"hs": np.ones((3, 2))}, ["b", "c"], ["buoy", "isobath"], ["waves", "10m"])
        run = run.assign_coords(site_longitude=("sites", [2.00001, 9.0]))
        out = val.on_sites_of(run, like)
        self.assertEqual(list(out.sites.values), ["b"])
        self.assertEqual(float(out.site_longitude[0]), 2.0)

        time = pd.date_range("2018-01-01", periods=240, freq="h")
        hs = np.ones(240)
        hs[[20, 22, 100, 200]] = [5.0, 6.0, 4.0, 3.0]  # 20 and 22: one storm, peak at 22
        offshore = xr.Dataset({"hs": (("time", "goal"), np.column_stack([hs, hs]))}, coords={"time": time})
        storms = val.storm_peaks(offshore, 3, min_gap="2D")
        self.assertEqual(list(storms["peak"]), [time[22], time[100], time[200]])
        maps = pd.DatetimeIndex([time[23], time[103]])  # 100: map 3 h away
        storms = val.storm_peaks(offshore, 3, min_gap="2D", map_hours=maps, max_lag="2h")
        self.assertEqual(list(storms["map_hour"])[:1], [time[23]])
        self.assertNotIn(time[103], list(storms["map_hour"]))

    def test_common_hours_metrics(self) -> None:
        ref = _sites_ds({"hs": np.ones((4, 1))}, ["buoy_A"], ["buoy"], ["waves"])
        full = ref.copy(data={"hs": np.full((4, 1), 1.5)})
        part = ref.isel(time=[1, 2]).copy(data={"hs": np.full((2, 1), 2.0)})
        m = val.common_hours_metrics(ref, {"full": full, "part": part}, ["hs", "tp"]).set_index("estimate")
        self.assertEqual(list(m["n"]), [2, 2])  # the hours both have
        self.assertAlmostEqual(m.loc["full", "bias"], 0.5)
        self.assertEqual(set(m["variable"]), {"hs"})

    def test_buoy_timeseries_variable_in_one_estimate(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rng = np.random.default_rng(0)
        cols = {"hs": rng.random((6, 1)), "tp": 8 + rng.random((6, 1)), "tm01": 6 + rng.random((6, 1))}
        buoy = _sites_ds(cols, ["buoy_A"], ["buoy"], ["waves"])
        hindcast = buoy[["hs", "tp"]]
        fig = val.plot_buoy_timeseries(buoy, {"hindcast": hindcast, "metamodel": buoy}, "buoy_A",
                                       variables=["hs", "tp", "tm01"])
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(fig.axes[2].get_ylabel(), "Tm01 [s]")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
