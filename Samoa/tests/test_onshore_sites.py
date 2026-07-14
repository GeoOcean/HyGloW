"""Unit tests for onshore observation site helpers."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from utils.observation import onshore_sites as obs


class TestOnshoreSites(unittest.TestCase):
    def test_snapwave_site_id_roundtrip(self) -> None:
        self.assertEqual(obs.snapwave_site_id("Aunuu_Buoy"), "buoy_Aunuu_Buoy")
        self.assertEqual(obs.instrument_id("buoy_Aunuu_Buoy"), "Aunuu_Buoy")

    def test_attach_buoy_site_meta(self) -> None:
        site_id = obs.snapwave_site_id("Aunuu_Buoy")
        ds = xr.Dataset(
            {"hs": (("time", "sites"), np.ones((3, 1)))},
            coords={"time": np.arange(3), "sites": [site_id]},
        )
        out = obs.attach_buoy_site_meta(ds, site_id)
        self.assertEqual(str(out["site_source"].values[0]), "buoy")
        self.assertEqual(str(out["site_label"].values[0]), "Aunuu_Buoy")
        self.assertAlmostEqual(float(out["site_longitude"].values[0]), -170.5005)


if __name__ == "__main__":
    unittest.main()
