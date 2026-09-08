"""Unit tests for SnapWave output-site metadata/grouping in ``utils.geometry.sites``."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from utils.geometry.sites import (
    attach_site_meta,
    find_group,
    metric_groups,
    per_site_coord,
    resolve_sites,
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


class TestAttachSiteMeta(unittest.TestCase):
    def test_copies_shared_metadata(self) -> None:
        template = xr.Dataset(
            {"hs": (("sites",), [1.0, 2.0])},
            coords={
                "sites": ["a", "b"],
                "site_source": ("sites", ["buoy", "isobath"]),
                "site_label": ("sites", ["buoy_a", "10m"]),
                "site_depth": ("sites", [-5.0, -10.0]),
                "site_longitude": ("sites", [1.0, 2.0]),
                "site_latitude": ("sites", [3.0, 4.0]),
            },
        )
        ds = xr.Dataset({"hs": (("sites",), [9.0])}, coords={"sites": ["a"]})
        out = attach_site_meta(ds, template)
        self.assertEqual(str(out["site_source"].values[0]), "buoy")
        self.assertEqual(float(out["site_longitude"].values[0]), 1.0)

    def test_no_overlap_returns_unchanged(self) -> None:
        template = xr.Dataset(
            {"hs": (("sites",), [1.0])},
            coords={"sites": ["a"], "site_source": ("sites", ["buoy"])},
        )
        ds = xr.Dataset({"hs": (("sites",), [9.0])}, coords={"sites": ["z"]})
        out = attach_site_meta(ds, template)
        self.assertNotIn("site_source", out.coords)


if __name__ == "__main__":
    unittest.main()
