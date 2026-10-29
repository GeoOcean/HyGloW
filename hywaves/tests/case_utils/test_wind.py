"""Unit tests for ``utils.wind`` (training design, pairs, K fit) without model runs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from utils import wind

GOALS = {"1": {"angles": [250, 340]}, "2": {"angles": [270, 360]}}
BOUNDS = {"hs": [0.05, 8.0], "tp": [2, 15], "spr": [5, 45], "wl": [-1, 1]}
SITES = ["iso_10m_0", "iso_20m_0", "buoy_A"]


def _patches(tmp: str):
    sim = {
        "bounds": BOUNDS,
        "max_steepness": 0.06,
        "model": "snapwave",
        "snapwave": {"grid_file": "mesh.nc", "wind_cases": str(Path(tmp) / "cases")},
    }
    return (
        mock.patch.dict(wind.SIMULATIONS, sim),
        mock.patch.dict(wind.METAMODEL, {"wind_correction": {"num_cases": 12}}),
        mock.patch.object(wind, "output_dir", lambda: Path(tmp)),
    )


class TestWindCases(unittest.TestCase):
    def test_design_is_wind_sea_with_wind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2, p3 = _patches(tmp)
            with p1, p2, p3:
                cases = wind.generate_wind_cases(GOALS)
        self.assertEqual(len(cases), 12)
        self.assertTrue((cases["ws_weight"] >= wind.MIN_TRAINING_WEIGHT).all())
        steep = cases["hs"] / (9.81 * cases["tp"] ** 2 / (2 * np.pi))
        self.assertLessEqual(float(steep.max()), 0.06)
        self.assertTrue(cases["u10"].between(3.0, 35.0).all())
        self.assertTrue(cases["dir"].between(250, 360).all())
        angle = np.mod(cases["u10dir"] - cases["dir"] + 180, 360) - 180
        np.testing.assert_allclose(angle, cases["wind_wave_angle"], atol=1e-6)

    def test_case_parameters(self) -> None:
        cases = pd.DataFrame(
            {"hs": [2.0, 3.0], "tp": [6.0, 7.0], "dir": [300.0, 310.0], "spr": [30.0, 30.0],
             "wl": [0.1, 0.2], "u10": [14.0, 18.0], "u10dir": [290.0, 320.0]}
        )
        with_wind = wind.case_parameters(cases, 3, with_wind=True)
        without = wind.case_parameters(cases, 3, with_wind=False)
        self.assertEqual(with_wind["tref"], ["20000101 000000", "20000101 010000"])
        self.assertEqual(with_wind["hs_nodes"][1], [3.0, 3.0, 3.0])
        self.assertEqual(with_wind["u10"], [14.0, 18.0])
        self.assertEqual(without["u10"], [0.0, 0.0])
        self.assertEqual(without["hs_nodes"], with_wind["hs_nodes"])


class TestFitWindCorrection(unittest.TestCase):
    def test_pairs_fit_and_scores(self) -> None:
        rng = np.random.default_rng(1)
        n_cases = 40
        cases = pd.DataFrame(
            {"hs": rng.uniform(1, 5, n_cases), "tp": 7.0, "dir": 300.0, "spr": 30.0, "wl": 0.0,
             "u10": rng.uniform(8, 25, n_cases), "u10dir": rng.uniform(250, 350, n_cases),
             "wind_wave_angle": rng.uniform(-60, 60, n_cases), "ws_weight": rng.uniform(0.5, 1, n_cases)}
        )
        hs0 = cases["hs"].to_numpy()[:, None] * np.array([[0.6, 0.8, 0.9]])
        k = 1 + 0.4 * np.exp(-9.81 * hs0 / cases["u10"].to_numpy()[:, None] ** 2 * 5)
        with tempfile.TemporaryDirectory() as tmp:
            p1, p2, p3 = _patches(tmp)
            with p1, p2, p3:
                root = wind.wind_cases_dir(True).parent
                root.mkdir(parents=True)
                cases.to_csv(root / "wind_cases.csv", index=False)
                time = pd.date_range("2000-01-01", periods=n_cases, freq="h")
                for with_wind, values in ((False, hs0), (True, hs0 * k)):
                    xr.Dataset(
                        {"hs": (("time", "sites"), values)},
                        coords={"time": time, "sites": SITES, "site_label": ("sites", ["10m", "20m", "A"])},
                    ).to_netcdf(wind.postprocessed_path(with_wind))
                dirs = np.arange(0.0, 360.0, 5.0)
                xr.Dataset(
                    {
                        "fetch": (("sites", "dir"), np.full((3, 72), 4e4)),
                        "fetch_eff": (("sites", "dir"), np.full((3, 72), 4e4)),
                        "mean_depth": (("sites", "dir"), np.full((3, 72), 25.0)),
                        "site_depth": ("sites", [10.0, 20.0, 15.0]),
                        "point_x": ("sites", [4.0, 3.9, 4.1]),
                        "point_y": ("sites", [52.0, 52.1, 52.2]),
                    },
                    coords={"sites": SITES, "dir": dirs},
                ).to_netcdf(wind.fetch_table_path())

                pairs = wind.training_pairs()
                self.assertEqual(len(pairs), n_cases * 3)
                self.assertEqual(float(pairs["fetch"].iloc[0]), 4e4)
                correction, scores = wind.fit_wind_correction(test_fraction=0.25)
                k_df = wind.k_comparison(test_fraction=0.25)
                self.assertEqual(k_df["case"].nunique(), 10)
                for col in ("k_runs", "k_fitted", "k_formula_deep", "k_formula_depth"):
                    self.assertTrue(k_df[col].between(0.5, 4.0).all(), col)
                self.assertTrue(wind.correction_path().is_file())
                loaded = wind.load_wind_correction()
                self.assertEqual(loaded.n_samples, correction.n_samples)

                held = wind.holdout_cases(test_fraction=0.25)
                self.assertEqual(set(held.index), set(k_df["case"]))
                ids = list(held.index[:2])
                maps = wind.case_comparison(ids, test_fraction=0.25)
                self.assertEqual(dict(maps.sizes), {"case": 2, "sites": 3})
                np.testing.assert_allclose(maps["hs0"].sel(case=ids[0]), hs0[ids[0]], rtol=1e-6)
                np.testing.assert_allclose(maps["hs1"].sel(case=ids[0]), (hs0 * k)[ids[0]], rtol=1e-6)
                np.testing.assert_allclose(maps["latitude"], [52.0, 52.1, 52.2])
                self.assertIn("hs_ws_u2", maps.coords)
                ranges = wind.training_range()
                self.assertEqual(set(ranges), {"u10", "angle", "hs_ws_u2"})
                self.assertLess(ranges["u10"][0], ranges["u10"][1])
                curves = wind.k_response(SITES[:2], 300.0, n=7)
                self.assertEqual(curves["k_u10"].shape, (2, 7))
                self.assertEqual(curves["k_hs_ws_u2"].shape, (2, 7))
                for name in ("k_u10", "k_angle", "k_hs_ws_u2"):
                    self.assertTrue(bool(((curves[name] >= 0.5) & (curves[name] <= 4.0)).all()), name)
                np.testing.assert_allclose(curves["fetch"], 4e4)
                self.assertEqual(curves.attrs["wind_dir"], 300.0)
                with self.assertRaises(ValueError):
                    wind.case_comparison([int(next(i for i in range(n_cases) if i not in held.index))],
                                         test_fraction=0.25)

        rmse = scores.pivot_table(index="group", columns="estimate", values="rmse")
        self.assertTrue((rmse["wind correction"] < rmse["no correction"]).all())


class TestFormulaK(unittest.TestCase):
    def test_growth_curves(self) -> None:
        """Young seas grow more than developed ones; never below 1; depth limits growth."""
        u = np.array([15.0, 15.0, 15.0])
        hs = np.array([0.5, 2.0, 0.24 * 15.0**2 / 9.81])  # the last one fully developed
        k = wind.formula_k(hs, u, 30e3)
        self.assertGreater(k[0], k[1])
        self.assertGreater(k[1], 1.0)
        self.assertAlmostEqual(float(k[2]), 1.0)
        shallow = wind.formula_k(np.array([2.0]), np.array([15.0]), 30e3, depth=np.array([5.0]))
        self.assertAlmostEqual(float(shallow[0]), 1.0)  # already above the depth limit
        self.assertTrue(np.isnan(wind.formula_k(np.array([0.0]), np.array([10.0]), 1e4))[0])

if __name__ == "__main__":
    unittest.main()
