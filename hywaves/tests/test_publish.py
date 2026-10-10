"""Unit tests for ``hywaves.publish`` (the ``publish`` command) and the
``portal.*`` staleness checks it enables in ``validate``."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from hywaves.cli import main
from hywaves.contract.validate import WARN, validate_case
from hywaves.publish import PublishError, publish_case


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


METRICS_HEADER = "goal,metamodel,mda_candidate,group,variable,n,bias,rmse,r2"


def _quiet(_msg: str) -> None:
    pass


class _CaseFixture(unittest.TestCase):
    """A minimal portal-ready case: config, goals, observations, one metrics file."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / "PubCase"
        self.case_dir.mkdir()
        self.config = {
            "name": "PubCase",
            "description": "A minimal publishable case.",
            "paths": {},
            "onshore_waves": "inputs/Observations/obs.json",
            "offshore_waves": [],
            "simulations": {"goals_file": "inputs/Goals/Goals.json"},
        }
        self._write_json(
            "inputs/Goals/Goals.json",
            {"goals": {"1": {"target": {"longitude": 1.0, "latitude": 2.0}, "angles": [10, 20]}}},
        )
        self._write_json(
            "inputs/Observations/obs.json",
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [1.1, 2.1]},
                        "properties": {"instrument_id": "B1", "name": "Buoy 1"},
                    }
                ],
            },
        )
        self._write_metrics("goal_01", ["1,isobaths,50,10m,hs,100,0.01,0.1,0.9"])
        self._write_config()

    def _write_json(self, rel: str, data) -> Path:
        path = self.case_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def _write_config(self) -> None:
        self._write_json("PubCase.json", self.config)

    def _write_metrics(self, goal_dir: str, rows: list[str], header: str = METRICS_HEADER) -> Path:
        path = self.case_dir / "outputs" / goal_dir / "isobaths" / "validation_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([header, *rows]) + "\n")
        return path

    def _publish(self, **kwargs):
        return publish_case(self.case_dir, log=_quiet, **kwargs)

    @property
    def portal(self) -> Path:
        return self.case_dir / "portal"

    def _manifest(self) -> dict:
        return json.loads((self.portal / "manifest.json").read_text())


class TestPublishBasics(_CaseFixture):
    def test_unpublishable_case_raises_and_writes_nothing(self) -> None:
        self.config["description"] = ""
        self._write_config()
        with self.assertRaises(PublishError) as ctx:
            self._publish()
        self.assertIn("config.description", str(ctx.exception))
        self.assertFalse(self.portal.exists())
        self.assertFalse((self.case_dir / "portal.tmp").exists())

    def test_minimal_case_bundle_layout(self) -> None:
        result = self._publish()
        for name in ("case.json", "goals.json", "observations.json", "validation_metrics.csv", "manifest.json"):
            self.assertTrue((self.portal / name).is_file(), name)
        self.assertEqual(json.loads((self.portal / "case.json").read_text())["name"], "PubCase")
        # Optional layers with nothing to build are skipped with a reason, not fatal.
        for layer in ("aoi", "isobaths", "output_sites", "gui_figures", "book"):
            self.assertIn(layer, result.skipped)
        self.assertFalse((self.case_dir / "portal.tmp").exists())
        self.assertFalse((self.case_dir / "portal.old").exists())

    def test_manifest_records_files_sources_and_skips(self) -> None:
        result = self._publish()
        manifest = self._manifest()
        self.assertEqual(manifest["case"], "PubCase")
        self.assertEqual(set(manifest["files"]), set(result.files))
        self.assertNotIn("manifest.json", manifest["files"])
        self.assertIn("PubCase.json", manifest["sources"])
        self.assertIn("outputs/goal_01/isobaths/validation_metrics.csv", manifest["sources"])
        self.assertIn("outputs/goal_*/*/validation_metrics.csv", manifest["source_patterns"])
        self.assertEqual(manifest["skipped"], result.skipped)

    def test_metrics_are_merged_with_union_of_columns(self) -> None:
        self._write_metrics(
            "goal_02",
            ["2,isobaths,50,10m,hs,100,0.02,0.2,0.8,isobath"],
            header=METRICS_HEADER + ",site_source",
        )
        self._publish()
        lines = (self.portal / "validation_metrics.csv").read_text().splitlines()
        self.assertEqual(lines[0], METRICS_HEADER + ",site_source")
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].startswith("1,") and lines[1].endswith(","))
        self.assertTrue(lines[2].endswith(",isobath"))

    def test_republish_replaces_the_whole_bundle(self) -> None:
        self._publish()
        (self.portal / "leftover.txt").write_text("stale")
        self._publish()
        self.assertFalse((self.portal / "leftover.txt").exists())

    def test_optional_csv_outputs_are_copied(self) -> None:
        mm = self.case_dir / "outputs" / "metamodel_validation" / "period_2020.csv"
        mm.parent.mkdir(parents=True)
        mm.write_text("a\n1\n")
        buoy = self.case_dir / "outputs" / "buoy_validation" / "offshore_summary_2020.csv"
        buoy.parent.mkdir(parents=True)
        buoy.write_text("a\n1\n")
        self._publish()
        self.assertTrue((self.portal / "metamodel_validation" / "period_2020.csv").is_file())
        self.assertTrue((self.portal / "buoy_validation" / "offshore_summary_2020.csv").is_file())

    def test_book_is_copied_but_not_checksummed_and_can_be_skipped(self) -> None:
        html = self.case_dir / "book" / "_build" / "html"
        html.mkdir(parents=True)
        (html / "index.html").write_text("<html></html>")

        result = self._publish()
        self.assertTrue(result.book)
        self.assertTrue((self.portal / "book" / "index.html").is_file())
        self.assertFalse(any(k.startswith("book/") for k in self._manifest()["files"]))

        result = self._publish(book=False)
        self.assertFalse(result.book)
        self.assertFalse((self.portal / "book").exists())


@unittest.skipUnless(_has("PIL"), "Pillow not installed")
class TestPublishImages(_CaseFixture):
    def _png(self, rel: str, size: tuple[int, int]) -> None:
        from PIL import Image

        path = self.case_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, "white").save(path)

    def test_large_figures_are_downscaled_small_ones_copied(self) -> None:
        from PIL import Image

        self._png("outputs/gui_figures/fig01_domain.png", (3000, 1500))
        self._png("outputs/buoy_validation/figures/buoy_B1.png", (800, 400))
        self._publish(max_image_px=1200)
        with Image.open(self.portal / "gui_figures" / "fig01_domain.png") as img:
            self.assertEqual(img.size, (1200, 600))
        small = self.portal / "buoy_validation" / "figures" / "buoy_B1.png"
        src = self.case_dir / "outputs" / "buoy_validation" / "figures" / "buoy_B1.png"
        self.assertEqual(small.read_bytes(), src.read_bytes())


@unittest.skipUnless(_has("shapefile"), "pyshp not installed")
class TestPublishAoi(_CaseFixture):
    def test_shapefile_becomes_geojson(self) -> None:
        import shapefile

        shp = self.case_dir / "inputs" / "AoI_Polygons" / "PubCase_AoI.shp"
        shp.parent.mkdir(parents=True)
        with shapefile.Writer(str(shp), shapeType=shapefile.POLYGON) as w:
            w.field("name", "C")
            w.poly([[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]])
            w.record("aoi")
        self.config["paths"] = {"aoi_file": "inputs/AoI_Polygons/PubCase_AoI.shp"}
        self._write_config()

        result = self._publish()
        self.assertNotIn("aoi", result.skipped)
        data = json.loads((self.portal / "aoi.geojson").read_text())
        self.assertEqual(data["type"], "FeatureCollection")
        geometry = data["features"][0]["geometry"]
        self.assertEqual(geometry["type"], "Polygon")
        self.assertEqual(len(geometry["coordinates"][0]), 5)
        self.assertIn("inputs/AoI_Polygons/PubCase_AoI.dbf", self._manifest()["sources"])


@unittest.skipUnless(
    all(_has(m) for m in ("rioxarray", "contourpy", "xarray", "numpy")),
    "raster/NetCDF stack not installed",
)
class TestPublishMapLayers(_CaseFixture):
    def test_isobaths_are_contoured_from_the_raster(self) -> None:
        import numpy as np
        import rioxarray  # noqa: F401 — registers the .rio accessor
        import xarray as xr

        # Depth deepens linearly eastward: -0 m at x=0 → -40 m at x=1.
        x = np.linspace(0, 1, 41)
        y = np.linspace(0, 1, 11)
        z = np.tile(-40 * x, (y.size, 1))
        da = xr.DataArray(z[None], dims=("band", "y", "x"), coords={"band": [1], "y": y, "x": x})
        tif = self.case_dir / "inputs" / "GEBCO" / "topo.tiff"
        tif.parent.mkdir(parents=True)
        da.rio.write_crs("EPSG:4326").rio.to_raster(tif)

        self.config["paths"] = {"topobathy_file": "inputs/GEBCO/topo.tiff"}
        self.config["simulations"]["output_sites_config"] = {
            "isobaths": [{"name": "10m", "level": -10}, {"name": "20m", "level": -20}]
        }
        self._write_config()

        result = self._publish()
        self.assertNotIn("isobaths", result.skipped)
        isobaths = json.loads((self.portal / "isobaths.json").read_text())
        self.assertEqual([i["name"] for i in isobaths], ["10m", "20m"])
        # The -10 m contour sits at x=0.25 along the whole domain.
        xs = [pt[0] for seg in isobaths[0]["segments"] for pt in seg]
        self.assertTrue(xs and all(abs(v - 0.25) < 1e-6 for v in xs))

    def test_output_sites_exclude_buoys(self) -> None:
        import numpy as np
        import xarray as xr

        ds = xr.Dataset(
            {
                "site_longitude": ("site", np.array([1.0, 2.0, 3.0])),
                "site_latitude": ("site", np.array([4.0, 5.0, 6.0])),
                "site_source": ("site", np.array(["isobath", "mesh", "buoy"])),
                "site_label": ("site", np.array(["10m", "", "B1"])),
            }
        )
        cases = self.case_dir / "SnapWave_CASES_Metamodel"
        cases.mkdir()
        ds.to_netcdf(cases / "output_sites_1.nc")
        self.config["simulations"].update(
            model="snapwave", snapwave={"metamodel_cases": "SnapWave_CASES_Metamodel"}
        )
        self._write_config()

        self._publish()
        sites = json.loads((self.portal / "output_sites.json").read_text())
        self.assertEqual(sites["source"], ["isobath", "mesh"])
        self.assertEqual(sites["lon"], [1.0, 2.0])

    def test_missing_declared_source_is_skipped_not_fatal(self) -> None:
        self.config["simulations"].update(
            model="snapwave", snapwave={"metamodel_cases": "/definitely/not/here"}
        )
        self._write_config()
        result = self._publish()
        self.assertIn("no output_sites_*.nc", result.skipped["output_sites"])

    def test_output_sites_fall_back_to_same_named_folder_in_case(self) -> None:
        import numpy as np
        import xarray as xr

        local = self.case_dir / "SnapWave_CASES_Metamodel"
        local.mkdir()
        xr.Dataset(
            {
                "site_longitude": ("site", np.array([7.0])),
                "site_latitude": ("site", np.array([8.0])),
                "site_source": ("site", np.array(["isobath"])),
            }
        ).to_netcdf(local / "output_sites_1.nc")
        # Declared path lives on another machine; only the .nc were copied here.
        self.config["simulations"].update(
            model="snapwave", snapwave={"metamodel_cases": "/elsewhere/PubCase/SnapWave_CASES_Metamodel"}
        )
        self._write_config()

        result = self._publish()
        self.assertNotIn("output_sites", result.skipped)
        sites = json.loads((self.portal / "output_sites.json").read_text())
        self.assertEqual(sites["lon"], [7.0])
        self.assertIn("SnapWave_CASES_Metamodel/output_sites_1.nc", self._manifest()["sources"])


class TestPortalStaleness(_CaseFixture):
    def _portal_findings(self):
        return [f for f in validate_case(self.case_dir) if f.code.startswith("portal.")]

    def test_unpublished_case_is_info_only(self) -> None:
        findings = self._portal_findings()
        self.assertEqual([f.code for f in findings], ["portal.missing"])

    def test_fresh_bundle_is_up_to_date(self) -> None:
        self._publish()
        self.assertEqual([f.code for f in self._portal_findings()], ["portal.ok"])

    def test_touched_but_unchanged_source_is_not_stale(self) -> None:
        self._publish()
        goals = self.case_dir / "inputs" / "Goals" / "Goals.json"
        stat = goals.stat()
        os.utime(goals, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        self.assertEqual([f.code for f in self._portal_findings()], ["portal.ok"])

    def test_edited_source_makes_bundle_stale(self) -> None:
        self._publish()
        self._write_json(
            "inputs/Goals/Goals.json",
            {"goals": {"1": {"target": {"longitude": 9.0, "latitude": 2.0}, "angles": [10, 20]}}},
        )
        findings = self._portal_findings()
        self.assertEqual([f.code for f in findings], ["portal.stale"])
        self.assertEqual(findings[0].severity, WARN)
        self.assertIn("Goals.json", findings[0].message)

    def test_new_output_file_makes_bundle_stale(self) -> None:
        self._publish()
        self._write_metrics("goal_02", ["2,isobaths,50,10m,hs,100,0.02,0.2,0.8"])
        findings = self._portal_findings()
        self.assertEqual([f.code for f in findings], ["portal.stale"])
        self.assertIn("goal_02", findings[0].message)

    def test_removed_source_makes_bundle_stale(self) -> None:
        self._publish()
        (self.case_dir / "outputs" / "goal_01" / "isobaths" / "validation_metrics.csv").unlink()
        codes = [f.code for f in self._portal_findings()]
        self.assertIn("portal.stale", codes)


class TestCliPublish(_CaseFixture):
    def _run(self, argv: list[str]) -> tuple[int, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue() + err.getvalue()

    def test_publish_exits_zero_and_reports_bundle(self) -> None:
        code, out = self._run(["publish", str(self.case_dir), "--no-book"])
        self.assertEqual(code, 0)
        self.assertIn("Portal bundle written", out)
        self.assertTrue((self.portal / "manifest.json").is_file())

    def test_publish_of_unready_case_exits_nonzero(self) -> None:
        self.config["description"] = ""
        self._write_config()
        code, out = self._run(["publish", str(self.case_dir)])
        self.assertNotEqual(code, 0)
        self.assertIn("not portal-ready", out)


if __name__ == "__main__":
    unittest.main()
