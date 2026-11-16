"""Unit tests for utils.viz (storm figures)."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import xarray as xr

from utils import viz


def _sites(values: list[float], sites: list[str]) -> xr.DataArray:
    return xr.DataArray(values, dims="sites", coords={
        "sites": sites, "site_longitude": ("sites", np.linspace(0.0, 1.0, len(sites))),
        "site_latitude": ("sites", np.linspace(0.0, 1.0, len(sites)))})


class TestStormFields(unittest.TestCase):
    def setUp(self) -> None:
        self.tri = mtri.Triangulation([0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [[0, 1, 2], [1, 3, 2]])

    def tearDown(self) -> None:
        plt.close("all")

    def test_circular_difference_on_shared_sites(self) -> None:
        ref = _sites([350.0, 10.0, 180.0], ["a", "b", "c"])
        est = _sites([350.0, 10.0], ["b", "a"])  # b: 340 -> -20, a: -340 -> +20
        fig = viz.plot_storm_fields({"SWAN": est}, self.tri, "dir", reference=ref, reference_label="buoy")
        diff = np.asarray(fig.axes[0].collections[1].get_array())
        np.testing.assert_allclose(diff, [-20.0, 20.0])
        self.assertEqual(fig.axes[0].get_title(), "SWAN - buoy")

    def test_mesh_and_sites_panels(self) -> None:
        node = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="node")
        fig = viz.plot_storm_fields({"map": node, "sites": _sites([1.0, 2.0], ["a", "b"])}, self.tri, "hs",
                                    observed=_sites([2.5], ["a"]), ncols=3)
        self.assertFalse(fig.axes[2].get_visible())
        self.assertEqual(fig.axes[1].get_title(), "sites")


if __name__ == "__main__":
    unittest.main()
