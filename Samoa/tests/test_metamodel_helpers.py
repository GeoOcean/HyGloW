"""Minimal unit tests for metamodel helpers (no SnapWave artefacts required)."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from utils.model import metamodel as mm


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


if __name__ == "__main__":
    unittest.main()
