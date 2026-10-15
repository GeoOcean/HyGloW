"""Unit tests for onshore observation site helpers."""

from __future__ import annotations

import unittest


from utils import onshore_sites as obs


class TestOnshoreSites(unittest.TestCase):
    def test_snapwave_site_id_roundtrip(self) -> None:
        self.assertEqual(obs.snapwave_site_id("Example_Buoy"), "buoy_Example_Buoy")
        self.assertEqual(obs.instrument_id("buoy_Example_Buoy"), "Example_Buoy")


if __name__ == "__main__":
    unittest.main()
