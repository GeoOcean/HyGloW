"""Unit tests for ``utils.simulation`` with bluemath's SnapWave wrappers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from bluemath_tk.wrappers.snapwave import (
    SnapWaveMetaModelWrapper,
    build_enclosure_polygon,
    read_forcing_table,
)
from shapely.geometry import box

from utils import simulation as sws

CASE_DIR = Path(__file__).resolve().parents[1]


def _sites() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_id": ["iso_5m_0", "buoy_A"],
            "source": ["isobath", "buoy"],
            "source_label": ["5m", "A"],
            "depth": [5.0, 12.0],
            "lon": [0.2, 0.6],
            "lat": [0.3, 0.4],
        }
    )


def _setup() -> sws.SnapWaveCaseSetup:
    aoi = box(0, 0, 1, 1)
    return sws.SnapWaveCaseSetup(
        aoi_gdf=gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326"),
        enclosure_polygon=build_enclosure_polygon(aoi),
        enclosure_buffer_m=0.0,
        goals_config={
            "1": {"lon": 0.0, "lat": 1.0, "angles": [0, 90]},
            "2": {"lon": 1.0, "lat": 1.0, "angles": [0, 90]},
        },
        output_sites_df=_sites(),
    )


class TestTemplatesDir(unittest.TestCase):
    def test_from_case_json(self) -> None:
        with mock.patch.dict(sws.SIMULATIONS, {"metamodel_templates": "my_tpl"}):
            self.assertTrue(sws.templates_dir("metamodel").endswith("my_tpl"))

    def test_bad_kind(self) -> None:
        with self.assertRaises(ValueError):
            sws.templates_dir("other")


class TestMetamodelCases(unittest.TestCase):
    def test_cases_end_to_end(self) -> None:
        """Goal 2 gets unit Hs at node 2; output is split per goal with site coords."""
        setup = _setup()
        kwargs = setup.metamodel_kwargs()
        self.assertEqual(kwargs["boundary_nodes"], [(0.0, 1.0), (1.0, 1.0)])

        with tempfile.TemporaryDirectory() as tmp:
            wrapper = SnapWaveMetaModelWrapper(
                templates_dir=str(CASE_DIR / "templates-metamodel"),
                metamodel_parameters={
                    "goal": [1, 2],
                    "active_node": [1, 2],
                    "tp": [8.0, 12.0],
                    "dir": [300.0, 310.0],
                    "spr": [20.0, 25.0],
                    "wl": [0.0, 0.5],
                },
                fixed_parameters={"gridfile": "mesh.nc"},
                output_dir=tmp,
                debug=False,
                **kwargs,
            )
            wrapper.build_cases()
            case_dir = Path(wrapper.cases_dirs[1])
            hs = read_forcing_table(case_dir / "hs.txt").iloc[0].to_numpy()
            np.testing.assert_array_equal(hs, [0.0, 1.0])

            for d in wrapper.cases_dirs:
                xr.Dataset(
                    {
                        "point_hm0": (("time", "stations"), [[0.4, 0.8]]),
                        "point_tp": (("time", "stations"), [[9.0, 9.5]]),
                        "point_wavdir": (("time", "stations"), [[300.0, 305.0]]),
                        "point_dirspr": (("time", "stations"), [[20.0, 22.0]]),
                        "station_x": (("stations",), [0.2, 0.6]),
                    },
                    coords={"time": [0.0]},
                ).to_netcdf(Path(d) / "output_sites.nc")
            written = wrapper.postprocess_cases()

            self.assertEqual(Path(written[2]).name, "output_sites_2.nc")
            with xr.open_dataset(written[2]) as ds:
                self.assertEqual(list(ds.sites.values), ["iso_5m_0", "buoy_A"])
                self.assertEqual(set(sws.SITE_COORDS) - set(ds.coords), set())
                self.assertEqual(float(ds.site_depth.sel(sites="buoy_A")), 12.0)
                self.assertEqual(float(ds.tp_forcing.squeeze()), 12.0)
                self.assertEqual(ds.case_num.item(), 1)
                self.assertEqual(
                    sorted(ds.data_vars),
                    ["dir", "dir_forcing", "hs", "spr", "spr_forcing", "tp",
                     "tp_forcing", "wl_forcing"],
                )


if __name__ == "__main__":
    unittest.main()
