"""Unit tests for site metadata, grouping and selection in ``utils.geometry``."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.topo_bathy.mesh import Mesh

from utils.geometry import (
    SITE_META,
    attach_site_meta,
    buoy_depth_check,
    check_model_sites,
    compute_output_sites,
    sites_summary,
    find_group,
    metric_groups,
    per_site_coord,
    resolve_sites,
    site_meta,
)


def _ds(sites: list[str], source: list[str], label: list[str]) -> xr.Dataset:
    return xr.Dataset(
        {"hs": (("sites",), np.zeros(len(sites)))},
        coords={
            "sites": sites,
            "site_source": ("sites", source),
            "site_label": ("sites", label),
        },
    )


class TestPerSiteCoord(unittest.TestCase):
    def test_reads_existing_coord(self) -> None:
        ds = _ds(["a", "b"], ["isobath", "buoy"], ["10m", "buoy_a"])
        self.assertEqual(per_site_coord(ds, "site_source"), {"a": "isobath", "b": "buoy"})

    def test_missing_coord_uses_default(self) -> None:
        ds = _ds(["a", "b"], ["isobath", "buoy"], ["10m", "buoy_a"])
        self.assertEqual(
            per_site_coord(ds, "no_such_coord", default="?"), {"a": "?", "b": "?"}
        )

    def test_scalar_coord_broadcasts(self) -> None:
        ds = _ds(["a", "b"], ["isobath", "buoy"], ["10m", "buoy_a"])
        ds = ds.assign_coords(constant="same-for-all")
        self.assertEqual(per_site_coord(ds, "constant"), {"a": "same-for-all", "b": "same-for-all"})


class TestMetricGroups(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = _ds(
            sites=["iso_10_0", "iso_10_1", "iso_20_0", "mesh_apia_0", "buoy_Example"],
            source=["isobath", "isobath", "isobath", "mesh", "buoy"],
            label=["10m", "10m", "20m", "Apia", "buoy_Example"],
        )

    def test_isobath_groups_sorted_by_depth(self) -> None:
        groups = metric_groups(self.ds)
        isobath_groups = [g for g in groups if g[1] == "isobath"]
        self.assertEqual([g[0] for g in isobath_groups], ["10m", "20m"])
        self.assertEqual(isobath_groups[0][2], ["iso_10_0", "iso_10_1"])

    def test_mesh_group_present(self) -> None:
        groups = metric_groups(self.ds)
        mesh_groups = [g for g in groups if g[1] == "mesh"]
        self.assertEqual(mesh_groups, [("Apia", "mesh", ["mesh_apia_0"])])

    def test_each_buoy_is_its_own_group(self) -> None:
        groups = metric_groups(self.ds)
        buoy_groups = [g for g in groups if g[1] == "buoy"]
        self.assertEqual(buoy_groups, [("buoy_Example", "buoy", ["buoy_Example"])])

    def test_buoy_detected_by_id_prefix_even_without_source_coord(self) -> None:
        ds = xr.Dataset(coords={"sites": ["buoy_X", "other"]})
        groups = metric_groups(ds)
        self.assertEqual(groups, [("buoy_X", "buoy", ["buoy_X"])])


class TestFindGroup(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = _ds(
            sites=["iso_10_0", "buoy_Example"],
            source=["isobath", "buoy"],
            label=["10m", "buoy_Example"],
        )

    def test_finds_known_group(self) -> None:
        name, source, ids = find_group(self.ds, "10m")
        self.assertEqual((name, source), ("10m", "isobath"))
        self.assertEqual(ids, ["iso_10_0"])

    def test_unknown_group_raises_with_available_list(self) -> None:
        with self.assertRaises(KeyError):
            find_group(self.ds, "nope")


class TestResolveSites(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = _ds(
            sites=["iso_10_0", "iso_10_1", "mesh_apia_0", "buoy_Example"],
            source=["isobath", "isobath", "mesh", "buoy"],
            label=["10m", "10m", "Apia", "buoy_Example"],
        )

    def test_none_returns_all_sites(self) -> None:
        self.assertEqual(
            resolve_sites(self.ds, None),
            ["iso_10_0", "iso_10_1", "mesh_apia_0", "buoy_Example"],
        )

    def test_group_name_selector(self) -> None:
        self.assertEqual(resolve_sites(self.ds, "10m"), ["iso_10_0", "iso_10_1"])

    def test_umbrella_selector(self) -> None:
        self.assertEqual(resolve_sites(self.ds, "buoy"), ["buoy_Example"])

    def test_direct_site_id_selector(self) -> None:
        self.assertEqual(resolve_sites(self.ds, "mesh_apia_0"), ["mesh_apia_0"])

    def test_list_selector_deduplicates_preserving_order(self) -> None:
        out = resolve_sites(self.ds, ["10m", "buoy", "iso_10_0"])
        self.assertEqual(out, ["iso_10_0", "iso_10_1", "buoy_Example"])

    def test_unknown_selector_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_sites(self.ds, "not-a-group")


class TestSiteMeta(unittest.TestCase):
    """Site metadata comes from the ok rows of the output-sites table."""

    TABLE = pd.DataFrame(
        {
            "lon": [1.0, 2.0],
            "lat": [3.0, 4.0],
            "source": ["buoy", "isobath"],
            "source_id": ["buoy_a", "10m_0"],
            "source_label": ["waves", "10m"],
            "nominal_depth": [5.0, 10.0],
            "name": ["a", None],
            "mesh_depth": [5.2, 10.1],
            "status": ["ok", "ok"],
        }
    )

    def test_attach_site_meta(self) -> None:
        ds = xr.Dataset({"hs": (("sites",), [9.0])}, coords={"sites": ["buoy_a"]})
        with mock.patch("utils.geometry.output_sites", return_value=self.TABLE):
            out = attach_site_meta(ds)
        self.assertEqual(str(out["site_source"].values[0]), "buoy")
        self.assertEqual(float(out["site_longitude"].values[0]), 1.0)
        self.assertEqual(float(out["site_mesh_depth"].values[0]), 5.2)
        self.assertEqual(set(out.coords) - {"sites"}, set(SITE_META))

    def test_site_meta_order_and_unknown(self) -> None:
        with mock.patch("utils.geometry.output_sites", return_value=self.TABLE):
            self.assertEqual(list(site_meta().sites.values), ["buoy_a", "10m_0"])
            self.assertEqual(list(site_meta(["10m_0"]).site_label.values), ["10m"])
            with self.assertRaises(KeyError):
                site_meta(["nope"])


class TestModelSites(unittest.TestCase):
    """Sites get the mesh depth and a status; summary and buoy check."""

    def setUp(self) -> None:
        mesh = Mesh(
            x=np.array([0.0, 1.0, 0.0, 1.0]),
            y=np.array([0.0, 0.0, 1.0, 1.0]),
            faces=np.array([[0, 1, 3, 2]]),
            z=np.array([0.0, -20.0, 0.0, -20.0]),  # 20 m deeper eastward
        )
        sites = pd.DataFrame(
            {
                "lon": [0.9, 0.1, 2.0, 0.5, 0.75],
                "lat": [0.5, 0.5, 0.5, 0.5, 0.5],
                "source": ["isobath", "isobath", "buoy", "buoy", "buoy"],
                "source_id": ["10m_0", "5m_0", "buoy_out", "buoy_ok", "buoy_far"],
                "source_label": ["10m", "5m", "x", "x", "x"],
                "nominal_depth": [10.0, 5.0, 8.0, 9.0, 4.0],
                "name": [None, None, "Out", "Ok", "Far"],
            }
        )
        self.mesh = mesh
        sim = {"bounds": {"wl": [-3.5, 3.5]}, "min_depth": 0.1}
        with mock.patch.dict("utils.geometry.SIMULATIONS", sim, clear=True):
            self.sites = check_model_sites(sites, mesh)

    def test_status(self) -> None:
        self.assertEqual(
            self.sites["status"].tolist(), ["ok", "can_dry", "outside_mesh", "ok", "ok"]
        )
        self.assertAlmostEqual(float(self.sites["mesh_depth"].iloc[0]), 18.0)

    def test_summary(self) -> None:
        summary = sites_summary(self.sites)
        self.assertEqual(summary.index.tolist(), ["10m", "5m", "buoy", "total"])
        self.assertEqual(summary.loc["buoy", ["sites", "ok", "outside_mesh"]].tolist(), [3, 2, 1])
        self.assertEqual(summary.loc["total", "can_dry"], 1)
        self.assertEqual(summary.loc["buoy", "median"], 12.5)

    def test_polygon_nodes(self) -> None:
        """Every model-mesh node inside a polygon is a site; dry nodes are flagged."""
        import geopandas as gpd
        from shapely.geometry import box

        poly = gpd.GeoDataFrame(geometry=[box(-0.1, -0.1, 1.1, 0.5)], crs="EPSG:4326")
        cfg = {"isobaths": [], "polygons": [{"name": "Port", "file": "port.shp"}],
               "include_buoys": False, "clip_to_aoi": False}
        sim = {"bounds": {"wl": [-3.5, 3.5]}, "min_depth": 0.1}
        with mock.patch.dict("utils.geometry.SIMULATIONS", sim, clear=True), \
                mock.patch("utils.geometry.model_grid", return_value=self.mesh), \
                mock.patch("utils.geometry.gpd.read_file", return_value=poly):
            sites = compute_output_sites(config=cfg)
        self.assertEqual(sites["source_id"].tolist(), ["Port_0", "Port_1"])  # southern nodes
        self.assertTrue((sites["source"] == "mesh").all())
        self.assertEqual(sites["nominal_depth"].tolist(), [0.0, 20.0])
        self.assertEqual(sites["status"].tolist(), ["can_dry", "ok"])

        cfg["polygons"][0].update(fraction=0.5)
        with mock.patch.dict("utils.geometry.SIMULATIONS", sim, clear=True), \
                mock.patch("utils.geometry.model_grid", return_value=self.mesh), \
                mock.patch("utils.geometry.gpd.read_file", return_value=poly):
            sites = compute_output_sites(config=cfg)
        self.assertEqual(sites["source_id"].tolist(), ["Port_1"])  # only wet nodes sampled

    def test_buoy_depth_check(self) -> None:
        check = buoy_depth_check(self.sites).set_index("source_id")
        self.assertFalse(check.loc["buoy_ok", "flag"])  # 10 m vs 9 m
        self.assertTrue(check.loc["buoy_far", "flag"])  # 15 m vs 4 m
        self.assertEqual(check.loc["buoy_far", "difference"], 11.0)


if __name__ == "__main__":
    unittest.main()
