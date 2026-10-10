"""Unit tests for ``utils.simulation`` with bluemath's SnapWave and SWAN wrappers."""

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
from bluemath_tk.wrappers.swan.swan_wrapper import SwanMetaModelWrapper
from scipy.spatial import Delaunay
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


def _setup(model: str = "snapwave") -> sws.HyWavesWrapperSetup:
    aoi = box(0, 0, 1, 1)
    return sws.HyWavesWrapperSetup(
        aoi_gdf=gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326"),
        enclosure_polygon=build_enclosure_polygon(aoi),
        enclosure_buffer_m=0.0,
        goals_config={
            "1": {"lon": 0.0, "lat": 1.0, "angles": [0, 90]},
            "2": {"lon": 1.0, "lat": 1.0, "angles": [0, 90]},
        },
        output_sites_df=_sites(),
        model=model,
    )


def _write_square_grid(path: Path, n: int = 6) -> None:
    """ADCIRC grid on [0, 1]^2 with the open boundary along the north edge (W->E)."""
    x, y = (v.ravel() for v in np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n)))
    tri = Delaunay(np.c_[x, y]).simplices
    north = [i for i in np.argsort(x) if np.isclose(y[i], 1)]
    rest = [i for i in range(len(x)) if np.isclose(x[i], 0) or np.isclose(x[i], 1) or np.isclose(y[i], 0)]
    land = [i for i in dict.fromkeys(rest) if i not in north[1:-1]]
    with path.open("w") as f:
        f.write(f"square\n{len(tri)} {len(x)}\n")
        for i in range(len(x)):
            f.write(f"{i + 1} {x[i]:.6f} {y[i]:.6f} 10.0\n")
        for k, t in enumerate(tri):
            f.write(f"{k + 1} 3 {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        f.write(f"1 ! open\n{len(north)}\n{len(north)}\n")
        f.write("".join(f"{i + 1}\n" for i in north))
        f.write(f"1 ! land\n{len(land)}\n{len(land)} 0\n")
        f.write("".join(f"{i + 1}\n" for i in land))


def _models_config(tmp: str) -> dict:
    """``simulations`` entries for both models, rooted at *tmp*."""
    block = lambda model, grid: {  # noqa: E731
        "grid_file": str(Path(tmp) / grid),
        "metamodel_templates": str(CASE_DIR / f"templates-{model}-metamodel"),
        "dynamic_templates": str(CASE_DIR / f"templates-{model}-dynamic"),
        "metamodel_cases": str(Path(tmp) / f"{model}_mm"),
        "dynamic_cases": str(Path(tmp) / f"{model}_dyn"),
    }
    return {
        "model": "snapwave",
        "snapwave": {**block("snapwave", "mesh.nc"), "slurm": {"cpus_per_task": 4}},
        "swan": block("swan", "mesh.grd"),
    }


_CASE = {
    "goal": [1, 2],
    "active_node": [1, 2],
    "hs": [1.0, 2.5],
    "tp": [8.0, 12.0],
    "dir": [300.0, 310.0],
    "spr": [20.0, 25.0],
    "wl": [0.0, 0.5],
}


class TestModelConfig(unittest.TestCase):
    def test_model_block_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sws.SIMULATIONS, _models_config(tmp)):
                self.assertTrue(sws.templates_dir("metamodel").endswith("templates-snapwave-metamodel"))
                self.assertTrue(sws.templates_dir("dynamic", "swan").endswith("templates-swan-dynamic"))
                self.assertEqual(sws.cases_dir("metamodel", "swan").name, "swan_mm")
                self.assertEqual(sws.grid_file("swan").name, "mesh.grd")
                self.assertEqual(sws.slurm_kwargs(max_parallel=5), {"cpus_per_task": 4, "max_parallel": 5})
                self.assertEqual(sws.slurm_kwargs("swan"), {})

    def test_bad_kind_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(sws.SIMULATIONS, _models_config(tmp)):
                with self.assertRaises(ValueError):
                    sws.templates_dir("other")
                with self.assertRaises(ValueError):
                    sws.grid_file("xbeach")


class TestGenerateCases(unittest.TestCase):
    def test_lhs_keeps_plausible_sea_states(self) -> None:
        """LHS per goal within its sector; steep samples are dropped."""
        sim = {
            "num_samples_per_goal": 200,
            "max_steepness": 0.05,
            "bounds": {"hs": [0.05, 8.0], "tp": [2, 15], "spr": [5, 40], "wl": [-1, 1]},
        }
        goals = {"1": {"angles": [-30, 30]}, "2": {"angles": [180, 270]}}
        with mock.patch.dict(sws.SIMULATIONS, sim):
            cases = sws.generate_cases(goals)
        self.assertEqual(
            list(cases.columns),
            ["goal", "active_node", "hs", "tp", "dir", "spr", "wl"],
        )
        steepness = cases["hs"] / (9.81 * cases["tp"] ** 2 / (2 * np.pi))
        self.assertLessEqual(float(steepness.max()), 0.05)
        per_goal = cases.groupby("goal").size()
        self.assertTrue(((per_goal > 50) & (per_goal < 200)).all())
        g2 = cases[cases["goal"] == "2"]
        self.assertTrue(g2["dir"].between(180, 270).all())


class TestMetamodelCases(unittest.TestCase):
    def test_cases_end_to_end(self) -> None:
        """Goal 2 gets its Hs at node 2; output is split per goal with site coords."""
        setup = _setup()
        kwargs = setup.metamodel_kwargs()
        self.assertEqual(kwargs["boundary_nodes"], [(0.0, 1.0), (1.0, 1.0)])

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sws.SIMULATIONS, _models_config(tmp)
        ):
            wrapper = setup.metamodel_wrapper(pd.DataFrame(_CASE), debug=False)
            self.assertIsInstance(wrapper, SnapWaveMetaModelWrapper)
            self.assertTrue(wrapper.fixed_parameters["gridfile"].endswith("mesh.nc"))
            wrapper.build_cases()
            case_dir = Path(wrapper.cases_dirs[1])
            hs = read_forcing_table(case_dir / "hs.txt").iloc[0].to_numpy()
            np.testing.assert_array_equal(hs, [0.0, 2.5])

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
            self.assertEqual(Path(written[2]).parent.name, "snapwave_mm")
            with xr.open_dataset(written[2]) as ds:
                self.assertEqual(list(ds.sites.values), ["iso_5m_0", "buoy_A"])
                self.assertEqual(set(sws.SITE_COORDS) - set(ds.coords), set())
                self.assertEqual(float(ds.site_depth.sel(sites="buoy_A")), 12.0)
                self.assertEqual(float(ds.tp_forcing.squeeze()), 12.0)
                self.assertEqual(float(ds.hs_forcing.squeeze()), 2.5)
                self.assertEqual(ds.case_num.item(), 1)
                self.assertEqual(
                    sorted(ds.data_vars),
                    ["dir", "dir_forcing", "hs", "hs_forcing", "spr", "spr_forcing",
                     "tp", "tp_forcing", "wl_forcing"],
                )


    def test_swan_cases_end_to_end(self) -> None:
        """SWAN: Hs at the active goal only; same per-goal output as SnapWave."""
        setup = _setup("swan")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            sws.SIMULATIONS, {**_models_config(tmp), "model": "swan"}
        ):
            _write_square_grid(Path(tmp) / "mesh.grd")
            cases = pd.DataFrame({**_CASE, "save_map": [True, False]})
            wrapper = setup.metamodel_wrapper(cases, debug=False)
            self.assertIsInstance(wrapper, SwanMetaModelWrapper)
            self.assertNotIn("save_map", wrapper.metamodel_parameters)
            wrapper.build_cases()
            text = (Path(wrapper.cases_dirs[1]) / "INPUT").read_text()
            self.assertIn("SET LEVEL=0.5", text)
            self.assertNotIn("SPECOUT 'sites'", text.replace("$ SPECOUT", ""))
            rows = [r for r in text.splitlines() if r.strip() and r.strip()[0].isdigit()]
            hs = [float(r.split()[1]) for r in rows]
            self.assertEqual(max(hs), 2.5)
            self.assertEqual(hs[0], 0.0)

            for d in wrapper.cases_dirs:
                (Path(d) / "output_sites.tab").write_text(
                    "0.4 9.0 300.0 20.0\n0.8 9.5 305.0 22.0\n"
                )
                (Path(d) / "norm_end").touch()
            written = wrapper.postprocess_cases()
            with xr.open_dataset(written[2]) as ds:
                self.assertEqual(list(ds.sites.values), ["iso_5m_0", "buoy_A"])
                self.assertEqual(float(ds.hs.sel(sites="buoy_A").squeeze()), 0.8)
                self.assertEqual(float(ds.hs_forcing.squeeze()), 2.5)
                self.assertEqual(set(sws.SITE_COORDS) - set(ds.coords), set())


if __name__ == "__main__":
    unittest.main()
