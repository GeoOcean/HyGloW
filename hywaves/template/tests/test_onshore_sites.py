"""Unit tests for onshore observation site helpers."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from utils import config
from utils import onshore_sites as obs


class TestOnshoreSites(unittest.TestCase):
    def test_snapwave_site_id_roundtrip(self) -> None:
        self.assertEqual(obs.snapwave_site_id("Example_Buoy"), "buoy_Example_Buoy")
        self.assertEqual(obs.instrument_id("buoy_Example_Buoy"), "Example_Buoy")

    def test_attach_buoy_site_meta(self) -> None:
        if not config.resolve(config.ONSHORE_WAVES).is_file():
            self.skipTest("observations GeoJSON not generated yet (notebook 03)")
        coordinates = obs.get_coordinates()
        if not coordinates:
            self.skipTest("no instruments in the observations GeoJSON")
        instrument, meta = next(iter(coordinates.items()))
        site_id = obs.snapwave_site_id(instrument)
        ds = xr.Dataset(
            {"hs": (("time", "sites"), np.ones((3, 1)))},
            coords={"time": np.arange(3), "sites": [site_id]},
        )
        table = pd.DataFrame({"source_id": [site_id], "mesh_depth": [12.5]})
        with mock.patch("utils.geometry.output_sites", return_value=table):
            out = obs.attach_buoy_site_meta(ds, site_id)
        self.assertEqual(float(out["site_mesh_depth"].values[0]), 12.5)
        self.assertEqual(str(out["site_source"].values[0]), "buoy")
        self.assertEqual(str(out["site_label"].values[0]), instrument)
        self.assertAlmostEqual(float(out["site_longitude"].values[0]), meta["lon"])


if __name__ == "__main__":
    unittest.main()
