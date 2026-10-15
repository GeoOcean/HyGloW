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
    def test_periods(self) -> None:
        offshore = xr.Dataset(coords={"time": pd.date_range("2013-11-30", "2013-12-02", freq="h")})
        self.assertEqual(list(mm._periods(slice("2013-11-30", "2013-12-01"), offshore)), ["2013-11", "2013-12"])
        self.assertEqual(mm._periods("2013-12-01", offshore), {"2013-12-01": "2013-12-01"})
        one = mm._periods(slice("2013-12-01T03:00", "2013-12-01T09:00"), offshore)
        self.assertEqual(list(one.values()), [slice("2013-12-01T03:00", "2013-12-01T09:00")])

    def test_cache_path_depends_on_every_key(self) -> None:
        key = {"metamodel": "buoys", "n": 100, "spectral": False, "sites": ["a"]}
        base = mm._cache_path(key, "2013-12")
        self.assertEqual(base, mm._cache_path(dict(key), "2013-12"))
        self.assertNotEqual(base, mm._cache_path({**key, "sites": ["b"]}, "2013-12"))
        self.assertTrue(base.name.startswith("buoys_n100_bulk_2013-12_"))
        self.assertTrue(mm._cache_path(key, "2013-12", spectra=True).name.endswith("_spectra.nc"))

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


if __name__ == "__main__":
    unittest.main()
