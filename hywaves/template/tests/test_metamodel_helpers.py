"""Minimal unit tests for metamodel helpers (no SnapWave artefacts required)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from unittest import mock

from bluemath_tk.waves.hywaves import metamodel as hw

from utils import metamodel as mm


class TestMetamodelHelpers(unittest.TestCase):
    def test_consolidated_pca_sites_intersection(self) -> None:
        kept = {"dir_u": ["a", "b", "c"], "dir_v": ["b", "c", "d"]}
        self.assertEqual(
            mm._consolidated_pca_sites(kept, ["dir_u", "dir_v"]), ["b", "c"]
        )

    def test_detect_removed_sites(self) -> None:
        """A site far above the case Hs (numerical anomaly) is removed."""
        cases = xr.Dataset(
            {
                "hs": (("case_num", "sites"), np.array([[1.0, 40.0], [3.0, 3.0]])),
                "tp": (("case_num", "sites"), np.array([[5.0, 5.0], [5.0, 5.0]])),
                "hs_forcing": (("case_num",), np.array([1.0, 4.0])),
            },
            coords={"case_num": [0, 1], "sites": ["ok", "bad"]},
        )
        with mock.patch.dict(mm.SITE_FILTER, {"hs_ratio_max": 3.0}, clear=True):
            removed = mm.detect_removed_sites(cases)
        self.assertEqual(removed, ["bad"])

    def test_detect_removed_sites_dry(self) -> None:
        """Sites dry (or outside the grid) in too many cases are removed."""
        hs = np.array([[1.0, 0.0, np.nan], [2.0, 1.0, np.nan], [3.0, 2.0, np.nan]])
        cases = xr.Dataset(
            {
                "hs": (("case_num", "sites"), hs),
                "tp": (("case_num", "sites"), np.full((3, 3), 5.0)),
                "hs_forcing": (("case_num",), np.array([2.0, 3.0, 4.0])),
                "tp_forcing": (("case_num",), np.array([8.0, 9.0, 10.0])),
            },
            coords={"case_num": [0, 1, 2], "sites": ["wet", "dry_once", "outside"]},
        )
        masked = hw.mask_dry_sites(cases)
        with mock.patch.dict(mm.SITE_FILTER, {"hs_nan_max": 0.5}, clear=True):
            self.assertEqual(mm.detect_removed_sites(masked), ["outside"])
        with mock.patch.dict(mm.SITE_FILTER, {"hs_nan_max": 0.0}, clear=True):
            self.assertEqual(mm.detect_removed_sites(masked), ["dry_once", "outside"])
        filled = hw.fill_dry_sites(mm.drop_sites(masked, ["outside"]))
        self.assertEqual(float(filled.hs.sel(sites="dry_once")[0]), 0.0)
        self.assertEqual(float(filled.tp.sel(sites="dry_once")[0]), 8.0)

    def test_set_sites(self) -> None:
        """A set's selector picks its sites; unknown sets are rejected."""
        cases = xr.Dataset(
            coords={
                "sites": ["iso_a", "iso_b", "buoy_X"],
                "site_source": ("sites", ["isobath", "isobath", "buoy"]),
                "site_label": ("sites", ["5m", "10m", "X"]),
            }
        )
        sets = {"buoys": {"sites": "buoy"}, "all": {"sites": None}}
        with mock.patch.dict(mm.SETS, sets, clear=True):
            self.assertEqual(mm.set_sites(cases, "buoys"), ["buoy_X"])
            self.assertEqual(len(mm.set_sites(cases, "all")), 3)
            with self.assertRaises(ValueError):
                mm.set_sites(cases, "nope")

    def test_paths_include_set(self) -> None:
        set_name = next(iter(mm.SETS))
        path = mm.gp_model_path(3, set_name, 450)
        self.assertEqual(path.parts[-4:], ("goal_03", set_name, "gp_450", "gp_model.pkl"))

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
        self.assertEqual(
            mm._reconstruction_save_spectra(),
            bool(mm.METAMODEL["reconstruction"]["save_spectra"]),
        )

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
