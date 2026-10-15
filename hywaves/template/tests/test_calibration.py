"""Case layer of the satellite calibration (utils.calibration)."""

from __future__ import annotations

import unittest
from unittest import mock

from utils import calibration as cal
from utils.config import offshore_catalog_key


class TestCalibrationConfig(unittest.TestCase):
    def test_config_over_defaults(self) -> None:
        """The satellite block overrides the defaults key by key."""
        with mock.patch.dict(cal.SATELLITE, {"radius_km": 30, "train_fraction": 0.5}, clear=True):
            cfg = cal.config()
        self.assertEqual(cfg["radius_km"], 30)
        self.assertEqual(cfg["train_fraction"], 0.5)
        self.assertEqual(cfg["min_depth"], cal.DEFAULTS["min_depth"])

    def test_calibrated_entry_reuses_hindcast_nodes(self) -> None:
        """A calibrated entry names the goals JSON hindcast of its source."""
        self.assertEqual(offshore_catalog_key({"name": "WHACS"}), "whacs")
        self.assertEqual(
            offshore_catalog_key({"name": "WHACS_cal", "hindcast": "WHACS"}), "whacs"
        )

    def test_goal_nodes_use_hindcast_snap(self) -> None:
        """Goal nodes come from the goals JSON snap of the hindcast."""
        goals = {
            "1": {
                "target": {"longitude": 2.0, "latitude": 51.0},
                "hindcast": {"whacs": {"longitude": 2.1, "latitude": 51.1}},
            },
            "2": {"target": {"longitude": 3.0, "latitude": 52.0}},
        }
        with mock.patch.object(cal, "load_goals", return_value=goals), mock.patch.object(
            cal, "get_offshore_wave", return_value={"name": "WHACS"}
        ):
            nodes = cal.goal_nodes("WHACS")
        self.assertEqual(nodes.loc["1", "lon"], 2.1)
        self.assertEqual(nodes.loc["2", "lat"], 52.0)  # no snap: the goal target


if __name__ == "__main__":
    unittest.main()
