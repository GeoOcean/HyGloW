"""Unit tests for utils.metamodel helpers (no model artefacts required)."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.waves.hywaves import metamodel as hw

from utils import metamodel as mm

META = xr.Dataset(
    coords={
        "sites": ["5m_0", "5m_1", "buoy_X"],
        "site_source": ("sites", ["isobath", "isobath", "buoy"]),
        "site_label": ("sites", ["5m", "5m", "waves"]),
    }
)


class TestSites(unittest.TestCase):
    def setUp(self) -> None:
        sets = {"buoys": {"sites": "buoy"}, "all": {"sites": None}}
        for patcher in (
            mock.patch.dict(mm.SETS, sets, clear=True),
            mock.patch("utils.metamodel.site_meta", return_value=META),
            mock.patch("utils.metamodel._goals", return_value=["1", "2"]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_set_sites(self) -> None:
        self.assertEqual(mm.set_sites("buoys"), ["buoy_X"])
        self.assertEqual(mm.set_sites("all"), ["5m_0", "5m_1", "buoy_X"])

    def test_reconstruction_sites_drop_removed_in_any_goal(self) -> None:
        removed = {"1": ["5m_1"], "2": []}
        with mock.patch("utils.metamodel.removed_sites", side_effect=lambda g: removed[g]):
            self.assertEqual(mm.reconstruction_sites("all"), ["5m_0", "buoy_X"])
            self.assertEqual(mm.reconstruction_sites("all", "5m"), ["5m_0"])
            with self.assertRaises(ValueError):
                mm.reconstruction_sites("buoys", "5m_1")


class TestConfigAndPaths(unittest.TestCase):
    def test_paths_include_set(self) -> None:
        set_name = next(iter(mm.SETS))
        self.assertIn(set_name, str(mm.gp_model_path(3, set_name, 450)))
        self.assertTrue(str(mm.pca_model_path(3, set_name, 450, "hs", 0.99)).endswith("pca_hs_0.99_model.pkl"))
        with self.assertRaises(ValueError):
            mm.gp_model_path(3, "nope", 450)

    def test_pca_preprocessing_defaults(self) -> None:
        for key in ("value_to_replace_nans", "nan_threshold_to_drop"):
            self.assertTrue(set(hw.DEFAULT_PCA_PREPROCESSING[key]) <= set(mm.PCA_PREPROCESSING[key]))


class TestReconstructionHelpers(unittest.TestCase):
    def test_sum_contributions(self) -> None:
        times = pd.date_range("2020-01-01", periods=2, freq="h")
        chunks = [
            xr.Dataset(
                {
                    name: (("goal", "partition", "time", "sites"), np.full((1, 1, 2, 1), value))
                    for name, value in (("hs", 1.0), ("tp", 8.0), ("dir", 270.0), ("spr", 30.0))
                },
                coords={"goal": [g], "partition": [0], "time": times, "sites": ["s"]},
            )
            for g in (1, 2)
        ]
        bulk, spec = mm.sum_contributions(
            chunks, ["s"], variables={"hs": "hs", "tp": "tp", "dir": "dm", "spr": "dspr"}, save_spectra=True
        )
        self.assertEqual(set(bulk.data_vars), {"hs", "tp", "dir", "spr"})
        self.assertAlmostEqual(float(bulk["hs"].isel(time=0, sites=0)), np.sqrt(2.0), places=1)
        self.assertEqual(set(spec.dims), {"time", "sites", "freq", "dir"})


class TestReconstructionStore(unittest.TestCase):
    def test_store_name(self) -> None:
        with mock.patch("utils.metamodel.simulation_model", return_value="snapwave"):
            self.assertEqual(
                mm._store_name(550, "buoys", False, 0.0, False, None, False),
                "buoys_mda550_snapwave_bulk_wl-0_wind-off",
            )
            self.assertEqual(
                mm._store_name(100, "isobaths", True, "forcing", True, ["1", "2"], True),
                "isobaths_mda100_snapwave_spectral_wl-forcing_wind-on_goals-1-2_spectra",
            )

    def test_fingerprint_follows_the_files(self) -> None:
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "gp_model.pkl"
            store = Path(tmp) / "offshore.zarr"
            model.write_text("a")
            store.mkdir()
            (store / "zarr.json").write_text("{}")
            first = mm._fingerprint([model, store, Path(tmp) / "missing.nc"])
            self.assertEqual(first, mm._fingerprint([model, store, Path(tmp) / "missing.nc"]))
            os.utime(store / "zarr.json", ns=(1, 1))
            self.assertNotEqual(first, mm._fingerprint([model, store, Path(tmp) / "missing.nc"]))

    def test_cache_report_and_clear(self) -> None:
        import tempfile
        from pathlib import Path

        from bluemath_tk.waves.hywaves.store import ReconstructionStore

        times = pd.date_range("2020-01-01", "2020-02-29 23:00", freq="h")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ReconstructionStore(
                root / "tmp/reconstruct/WHACS/buoys_mda100_x.zarr", times, ["a", "b"], {"hs": ()},
                key={"metamodel": "buoys", "n": 100},
            )
            jan = times[times.month == 1]
            store.write(
                xr.Dataset({"hs": (("time", "sites"), np.ones((len(jan), 1)))},
                           coords={"time": jan, "sites": ["a"]}),
                ["2020-01"],
            )
            with mock.patch("utils.metamodel.resolve", side_effect=lambda p: root / p):
                report = mm.reconstruction_cache()
                self.assertEqual(report.loc[0, "store"], "buoys_mda100_x")
                self.assertEqual((report.loc[0, "months"], report.loc[0, "site_months"]), (1, 1))
                self.assertEqual(len(mm.clear_reconstruction_cache("buoys_*")), 1)
                self.assertTrue(mm.reconstruction_cache().empty)


if __name__ == "__main__":
    unittest.main()
