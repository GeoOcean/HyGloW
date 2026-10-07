"""Minimal unit tests for metamodel helpers (no SnapWave artefacts required)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from utils import metamodel as mm


class TestMetamodelHelpers(unittest.TestCase):
    def test_consolidated_pca_sites_intersection(self) -> None:
        kept = {"dir_u": ["a", "b", "c"], "dir_v": ["b", "c", "d"]}
        self.assertEqual(
            mm._consolidated_pca_sites(kept, ["dir_u", "dir_v"]), ["b", "c"]
        )

    def test_detect_removed_sites_hs_max(self) -> None:
        cases = xr.Dataset(
            {
                "hs": (("case_num", "sites"), np.array([[1.0, 11.0], [2.0, 3.0]])),
                "tp": (("case_num", "sites"), np.array([[5.0, 5.0], [5.0, 5.0]])),
            },
            coords={"case_num": [0, 1], "sites": ["ok", "bad"]},
        )
        removed = mm.detect_removed_sites(cases)
        self.assertIn("bad", removed)
        self.assertNotIn("ok", removed)

    def test_pca_preprocessing_from_config(self) -> None:
        replace, drop = mm._pca_preprocessing()
        self.assertEqual(replace["hs"], 0.0)
        self.assertEqual(drop["hs"], 0.99)

    def test_reconstruction_variables_from_config(self) -> None:
        self.assertEqual(
            mm._reconstruction_variables(),
            {"hs": "hs", "tp": "tp", "dir": "dm", "spr": "dspr"},
        )

    def test_reconstruction_save_spectra_from_config(self) -> None:
        self.assertTrue(mm._reconstruction_save_spectra())

    def test_pca_variance_label_includes_config_defaults(self) -> None:
        label = mm._pca_variance_label({"hs": 0.99, "dir": 0.99})
        self.assertIn("spr0.99", label)
        self.assertIn("hs0.99", label)
        self.assertIn("dir0.99", label)

        n_time, n_goals, n_parts = 2, 2, 1
        site = "test_site"
        times = pd.date_range("2020-01-01", periods=n_time, freq="h")
        cube = xr.Dataset(
            {
                "hs": (
                    ("time", "goal", "partition", "sites"),
                    np.full((n_time, n_goals, n_parts, 1), 1.0),
                ),
                "tp": (
                    ("time", "goal", "partition", "sites"),
                    np.full((n_time, n_goals, n_parts, 1), 8.0),
                ),
                "dir": (
                    ("time", "goal", "partition", "sites"),
                    np.full((n_time, n_goals, n_parts, 1), 270.0),
                ),
                "spr": (
                    ("time", "goal", "partition", "sites"),
                    np.full((n_time, n_goals, n_parts, 1), 30.0),
                ),
            },
            coords={
                "time": times,
                "goal": [1, 2],
                "partition": [0],
                "sites": [site],
            },
        )
        bulk, spec = mm._wavespectra_sum(
            cube,
            [site],
            variables={"hs": "hs", "tp": "tp", "dir": "dm", "spr": "dspr"},
            save_spectra=True,
        )
        self.assertEqual(set(bulk.data_vars), {"hs", "tp", "dir", "spr"})
        self.assertEqual(bulk.sizes["sites"], 1)
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertIn("efth", spec)
        self.assertEqual(set(spec.dims), {"time", "sites", "freq", "dir"})


if __name__ == "__main__":
    unittest.main()
