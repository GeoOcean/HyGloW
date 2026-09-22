"""Unit tests for the pure geometry helpers in ``utils.geometry.output_sites``.

``compute_output_sites``/``write_output_sites`` read from the case config and
disk (topobathy raster, mesh file, AoI shapefile) and are exercised end-to-end
by the notebooks instead; only the standalone numeric helpers are unit-tested
here.
"""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr
from shapely.geometry import Polygon, box

from utils.geometry.output_sites import (
    _resample_polyline,
    get_isobath_points,
    get_mesh_nodes_in_polygon,
)


class TestResamplePolyline(unittest.TestCase):
    def test_zero_length_polyline_returns_empty(self) -> None:
        pts = np.array([[0.0, 0.0], [0.0, 0.0]])
        out = _resample_polyline(pts, spacing=1.0)
        self.assertEqual(out.shape, (0, 2))

    def test_resamples_straight_line_at_spacing(self) -> None:
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        out = _resample_polyline(pts, spacing=2.0)
        self.assertGreaterEqual(len(out), 5)
        # First point matches the start; last point matches the end.
        np.testing.assert_allclose(out[0], [0.0, 0.0])
        np.testing.assert_allclose(out[-1], [10.0, 0.0], atol=1e-9)

    def test_finer_spacing_yields_more_points(self) -> None:
        pts = np.array([[0.0, 0.0], [10.0, 0.0]])
        coarse = _resample_polyline(pts, spacing=5.0)
        fine = _resample_polyline(pts, spacing=1.0)
        self.assertLess(len(coarse), len(fine))


class TestGetIsobathPoints(unittest.TestCase):
    def _bowl_topobathy(self) -> xr.DataArray:
        # A radially symmetric "bowl": depth = distance from centre - 5,
        # so the -0 isobath is a circle of radius 5 around the centre.
        x = np.linspace(-10, 10, 41)
        y = np.linspace(-10, 10, 41)
        xx, yy = np.meshgrid(x, y)
        z = np.hypot(xx, yy) - 5.0
        return xr.DataArray(z, coords={"y": y, "x": x}, dims=["y", "x"])

    def test_extracts_isobath_ring(self) -> None:
        da = self._bowl_topobathy()
        pts = get_isobath_points(da, level=0.0)
        self.assertGreater(len(pts), 0)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        np.testing.assert_allclose(radii, 5.0, atol=0.5)

    def test_no_contour_at_out_of_range_level_is_empty(self) -> None:
        da = self._bowl_topobathy()
        pts = get_isobath_points(da, level=1000.0)
        self.assertEqual(pts.shape, (0, 2))

    def test_aoi_polygon_drops_segments_entirely_outside_it(self) -> None:
        # Two disjoint bowls (two separate rings at level 0): one centred at
        # (0, 0), one far away at (100, 100). A segment is kept only if *any*
        # of its points fall inside the AoI (whole-segment filtering, not
        # per-point clipping) -- so an AoI around the origin bowl should drop
        # the far-away ring's segment entirely.
        x = np.linspace(-10, 110, 121)
        y = np.linspace(-10, 110, 121)
        xx, yy = np.meshgrid(x, y)
        z = np.minimum(np.hypot(xx, yy) - 5.0, np.hypot(xx - 100, yy - 100) - 5.0)
        da = xr.DataArray(z, coords={"y": y, "x": x}, dims=["y", "x"])

        aoi = box(-10.0, -10.0, 10.0, 10.0)
        pts = get_isobath_points(da, level=0.0, aoi_polygon=aoi)
        self.assertGreater(len(pts), 0)
        self.assertTrue((pts[:, 0] < 20.0).all())
        self.assertTrue((pts[:, 1] < 20.0).all())

    def test_spacing_resamples_output(self) -> None:
        da = self._bowl_topobathy()
        dense = get_isobath_points(da, level=0.0)
        sparse = get_isobath_points(da, level=0.0, spacing=2.0)
        self.assertLess(len(sparse), len(dense))

    def test_mismatched_dim_order_raises(self) -> None:
        # dims declared as (x, y) instead of the expected (y, x): shape then
        # disagrees with (y.size, x.size) inside the function.
        da = xr.DataArray(
            np.zeros((4, 3)), coords={"x": [0, 1, 2, 3], "y": [0, 1, 2]}, dims=["x", "y"]
        )
        with self.assertRaises(ValueError):
            get_isobath_points(da, level=0.0)

    def test_lon_lat_coordinate_names_supported(self) -> None:
        x = np.linspace(-10, 10, 21)
        y = np.linspace(-10, 10, 21)
        xx, yy = np.meshgrid(x, y)
        z = np.hypot(xx, yy) - 5.0
        da = xr.DataArray(z, coords={"lat": y, "lon": x}, dims=["lat", "lon"])
        pts = get_isobath_points(da, level=0.0)
        self.assertGreater(len(pts), 0)


class TestGetMeshNodesInPolygon(unittest.TestCase):
    def _mesh(self) -> xr.Dataset:
        # node 0, 1: wet, inside the (0,0)-(2,2) box below.
        # node 2: dry (z > 0), inside the box.
        # node 3: wet, outside the box.
        lon = np.array([0.5, 1.5, 1.0, 5.0])
        lat = np.array([0.5, 1.5, 1.0, 5.0])
        z = np.array([-10.0, -5.0, 3.0, -1.0])
        return xr.Dataset(
            {
                "mesh2d_node_x": ("node", lon),
                "mesh2d_node_y": ("node", lat),
                "mesh2d_node_z": ("node", z),
            }
        )

    def test_finds_nodes_inside_polygon(self) -> None:
        mesh = self._mesh()
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        nodes = get_mesh_nodes_in_polygon(poly, mesh)
        self.assertEqual({n["node_index"] for n in nodes}, {0, 1, 2})

    def test_wet_only_filters_positive_elevation(self) -> None:
        mesh = self._mesh()
        poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        nodes = get_mesh_nodes_in_polygon(poly, mesh, wet_only=True)
        self.assertEqual({n["node_index"] for n in nodes}, {0, 1})

    def test_no_nodes_inside_returns_empty(self) -> None:
        mesh = self._mesh()
        poly = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
        self.assertEqual(get_mesh_nodes_in_polygon(poly, mesh), [])


if __name__ == "__main__":
    unittest.main()
