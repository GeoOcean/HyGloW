"""Case layer of the satellite calibration (utils.calibration)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

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


def _hindcast(partitions: bool) -> xr.Dataset:
    """Two goals, 4 hours: goal 1 loses energy only when both partitions are full."""
    hs = np.array([[2.0, 1.0], [2.0, 1.0], [2.0, 1.0], [2.0, 1.0]])
    data = {"hs": (("time", "goal"), hs), "dir": (("time", "goal"), np.full((4, 2), 90.0))}
    if partitions:
        phs0 = np.array([[1.0, 0.6], [1.0, 0.6], [2.0, 0.6], [2.0, 0.6]])
        phs1 = np.array([[1.0, 0.8], [1.0, 0.8], [0.0, 0.8], [np.nan, 0.8]])
        data |= {"phs0": (("time", "goal"), phs0), "phs1": (("time", "goal"), phs1),
                 "pdir0": (("time", "goal"), np.full((4, 2), 90.0)), "pdir1": (("time", "goal"), np.full((4, 2), 200.0))}
    return xr.Dataset(data, coords={"time": pd.date_range("2020", periods=4, freq="h"), "goal": [1, 2]})


class TestPartitionEnergy(unittest.TestCase):
    def test_share_per_goal_and_closure(self) -> None:
        """The share splits by full slots; the closed hindcast adds up to the bulk."""
        with mock.patch.object(cal, "open_offshore_waves", return_value=_hindcast(True)):
            table = cal.partition_energy_share("H")
            closed = cal.closed_hindcast("H")
        self.assertEqual(table.loc["1", "partitions"], 2)
        self.assertAlmostEqual(table.loc["1", "energy_share"], (2 + 2 + 4 + 4) / 16)
        self.assertAlmostEqual(table.loc["1", "hours_last_full"], 0.5)
        self.assertAlmostEqual(table.loc["1", "energy_share_last_full"], 0.5)
        self.assertAlmostEqual(table.loc["1", "energy_share_others"], 1.0)
        self.assertAlmostEqual(table.loc["2", "energy_share"], 1.0)
        energy = closed["phs0"].fillna(0) ** 2 + closed["phs1"].fillna(0) ** 2
        np.testing.assert_allclose(energy.values, closed["hs"].values ** 2)
        np.testing.assert_allclose(closed["pdir1"], 200.0)

    def test_closure_is_optional(self) -> None:
        """close_partitions = false leaves the partitions as the hindcast gives them."""
        with mock.patch.object(cal, "open_offshore_waves", return_value=_hindcast(True)), mock.patch.dict(
            cal.SATELLITE, {"close_partitions": False}, clear=True
        ):
            xr.testing.assert_identical(cal.closed_hindcast("H"), _hindcast(True))

    def test_bulk_only_hindcast(self) -> None:
        """Without partitions there is nothing to measure or close."""
        with mock.patch.object(cal, "open_offshore_waves", return_value=_hindcast(False)):
            self.assertTrue(cal.partition_energy_share("H").empty)
            xr.testing.assert_identical(cal.closed_hindcast("H").drop_attrs(), _hindcast(False))


class TestSatellitePairsCache(unittest.TestCase):
    def test_cache_rebuilt_when_hindcast_is_newer(self) -> None:
        """pairs.csv is read only while it is newer than the offshore zarr."""
        with tempfile.TemporaryDirectory() as tmp:
            folder, source = Path(tmp), Path(tmp) / "w.zarr"
            pd.DataFrame({"goal": ["1"], "time": ["2020-01-01"], "sat_hs": [1.0]}).to_csv(
                folder / "pairs.csv", index=False
            )
            source.mkdir()
            os.utime(source, (10, 10))
            os.utime(folder / "pairs.csv", (1, 1))  # older than the zarr
            patches = [
                mock.patch.object(cal, "calibration_dir", return_value=folder),
                mock.patch.object(cal, "get_offshore_wave", return_value={"data_path": str(source)}),
                mock.patch.object(cal, "resolve", side_effect=Path),
                mock.patch.object(cal, "goal_nodes", side_effect=RuntimeError("rebuild")),
            ]
            for patch in patches:
                patch.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "rebuild"):
                    cal.satellite_pairs("WHACS")
                os.utime(folder / "pairs.csv", (20, 20))  # now newer
                self.assertEqual(len(cal.satellite_pairs("WHACS")), 1)
            finally:
                for patch in patches:
                    patch.stop()


class TestQuantileMappingPerGoal(unittest.TestCase):
    def test_one_map_per_goal(self) -> None:
        """Each goal gets its own map and tail quantile, fitted on its passes only."""
        rng = np.random.default_rng(0)
        chunks = []
        for gid, n, factor in (("1", 800, 1.1), ("2", 6000, 0.9)):
            hs = rng.weibull(1.3, n) * 1.5
            chunks.append(pd.DataFrame({"goal": gid, "sat_hs": hs * factor, "hs_raw": hs, "hs": hs}))
        pairs = pd.concat(chunks, ignore_index=True)
        pairs.loc[0, "hs"] = np.nan
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            cal, "calibration_dir", return_value=Path(tmp)
        ), mock.patch.object(cal, "quantile_mapping_pairs", return_value=pairs), mock.patch.dict(
            cal.SATELLITE, {}, clear=True
        ):
            fitted = cal.fit_quantile_mapping("WHACS")
            loaded = cal.load_quantile_mapping("WHACS")
            mapped = cal.mapped_pairs("WHACS")
            table = pd.read_csv(Path(tmp) / "quantile_mapping.csv", dtype={"goal": str})
        self.assertEqual(sorted(loaded), ["1", "2"])
        self.assertAlmostEqual(fitted["1"].tail_quantile, 1 - 5 / 799)
        self.assertEqual(fitted["2"].tail_quantile, 0.999)
        self.assertEqual(table.groupby("goal")["n"].first().to_dict(), {"1": 799, "2": 6000})
        for gid, factor in (("1", 1.1), ("2", 0.9)):
            rows = mapped[mapped["goal"] == gid].dropna()
            ratio = np.median(rows["hs_mapped"] / rows["hs"])
            self.assertAlmostEqual(ratio, factor, delta=0.03)
        fitted["1"].quantiles(raw=pairs.loc[pairs["goal"] == "1", "hs_raw"].to_numpy())


if __name__ == "__main__":
    unittest.main()
