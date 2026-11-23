"""WHACS at the goal nodes from a local archive (utils.whacs)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import netCDF4
import numpy as np

from utils import whacs

N_SEAPOINTS, CHUNK = 30, 10
LON = np.linspace(0.0, 29.0, N_SEAPOINTS)
LAT = np.full(N_SEAPOINTS, 40.0)


def _write_month(folder: Path, variable: str, month: str, start_day: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{variable}_WHACS_hindcast_1hr_{month}010000-{month}312300.nc"
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("time", 3)
        nc.createDimension("seapoint", N_SEAPOINTS)
        t = nc.createVariable("time", "f8", ("time",))
        t.units = "days since 1990-01-01"
        t[:] = start_day + np.arange(3) / 24 + 1e-7  # julian days carry float noise
        nc.createVariable("longitude", "f4", ("seapoint",))[:] = LON
        nc.createVariable("latitude", "f4", ("seapoint",))[:] = LAT
        var = nc.createVariable(variable, "f4", ("time", "seapoint"), chunksizes=(3, CHUNK))
        scale = 1.0 if variable == "hs" else 0.01
        var[:] = scale * (np.arange(3)[:, None] * 100 + np.arange(N_SEAPOINTS)[None, :] + start_day)


class TestWhacsArchive(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.archive = Path(self.tmp.name)
        for variable in ("hs", "fp"):
            _write_month(self.archive / variable, variable, "202301", 12053.0)
            _write_month(self.archive / variable, variable, "202302", 12084.0)
        goals = {
            "1": {"hindcast": {"whacs": {"longitude": 3.0, "latitude": 40.0}}},
            "2": {"hindcast": {"whacs": {"longitude": 25.0, "latitude": 40.0}}},
        }
        entry = {"name": "WHACS", "data_path": "w.zarr", "archive": str(self.archive)}
        self.patches = [
            mock.patch.object(whacs, "load_goals", return_value=goals),
            mock.patch.object(whacs, "get_offshore_wave", return_value=entry),
            mock.patch.object(whacs, "VARIABLES", ("hs", "fp")),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self.patches:
            patch.stop()
        self.tmp.cleanup()

    def test_goals_read_across_chunks(self) -> None:
        """Each goal gets its seapoint's series (two storage chunks), hours on the minute."""
        ds = whacs.whacs_at_goals(workers=1)
        self.assertEqual(ds.sizes, {"time": 6, "goal": 2})
        self.assertEqual(list(ds["seapoint"].values), [3, 25])
        np.testing.assert_allclose(ds["hs"].sel(goal=2).values[:3], [25 + 12053, 125 + 12053, 225 + 12053])
        np.testing.assert_allclose(ds["tp"], 1.0 / ds["fp"])
        self.assertEqual(str(ds["time"].values[1])[:19], "2023-01-01T01:00:00")

    def test_period_and_far_node(self) -> None:
        """The period keeps whole months; a node off the mesh is refused."""
        self.assertEqual(len(whacs.archive_files(self.archive, "hs", slice("2023-02", None))), 1)
        self.assertEqual(len(whacs.archive_files(self.archive, "hs", slice("2023", "2023"))), 2)
        self.assertEqual(len(whacs.archive_files(self.archive, "hs", slice(None, "2023-01"))), 1)
        far = {"1": {"hindcast": {"whacs": {"longitude": 3.5, "latitude": 40.0}}}}
        with mock.patch.object(whacs, "load_goals", return_value=far), self.assertRaises(ValueError):
            whacs.goal_seapoints(self.archive)


if __name__ == "__main__":
    unittest.main()
