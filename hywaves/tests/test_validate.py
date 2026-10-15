"""Unit tests for ``hywaves.validate`` (the ``validate`` command)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hywaves.common import find_case_config
from hywaves.start import start_case
from hywaves.validate import ERROR, WARN, validate_case


def _errors(findings):
    return [f for f in findings if f.severity == ERROR]


def _warnings(findings):
    return [f for f in findings if f.severity == WARN]


def _minimal_config(**overrides) -> dict:
    config = {
        "name": "TestCase",
        "description": "A minimal valid case.",
        "paths": {},
        "onshore_waves": None,
        "offshore_waves": [],
        "simulations": {},
    }
    config.update(overrides)
    return config


class TestValidateCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / "TestCase"
        self.case_dir.mkdir()

    def _write_config(self, config: dict) -> None:
        (self.case_dir / "TestCase.json").write_text(json.dumps(config))

    def test_missing_directory_is_an_error(self) -> None:
        findings = validate_case(self.case_dir / "nope")
        self.assertEqual(len(_errors(findings)), 1)
        self.assertEqual(findings[0].code, "case.dir")

    def test_missing_config_is_an_error(self) -> None:
        findings = validate_case(self.case_dir)
        codes = [f.code for f in _errors(findings)]
        self.assertIn("config.file", codes)

    def test_malformed_json_is_an_error(self) -> None:
        (self.case_dir / "TestCase.json").write_text("{not json")
        findings = validate_case(self.case_dir)
        codes = [f.code for f in _errors(findings)]
        self.assertIn("config.parse", codes)

    def test_minimal_valid_config_has_no_errors_in_dev_mode(self) -> None:
        self._write_config(_minimal_config())
        findings = validate_case(self.case_dir, publish=False)
        self.assertEqual(_errors(findings), [])
        # Missing description/goals/observations are expected gaps -> warnings.
        self.assertGreater(len(_warnings(findings)), 0)

    def test_same_gaps_become_errors_in_publish_mode(self) -> None:
        config = _minimal_config()
        config["description"] = ""  # a "gap": missing description
        self._write_config(config)

        dev_findings = validate_case(self.case_dir, publish=False)
        publish_findings = validate_case(self.case_dir, publish=True)

        self.assertIn("config.description", [f.code for f in _warnings(dev_findings)])
        self.assertIn("config.description", [f.code for f in _errors(publish_findings)])

    def test_missing_name_is_always_an_error(self) -> None:
        config = _minimal_config()
        config["name"] = ""
        self._write_config(config)
        findings = validate_case(self.case_dir, publish=False)
        self.assertIn("config.name", [f.code for f in _errors(findings)])

    def test_todo_placeholder_is_a_gap(self) -> None:
        config = _minimal_config(description="TODO: fill this in")
        self._write_config(config)
        findings = validate_case(self.case_dir, publish=False)
        self.assertIn("config.todos", [f.code for f in _warnings(findings)])

    def test_goals_file_missing_is_a_gap_not_an_error(self) -> None:
        config = _minimal_config(simulations={"goals_file": "inputs/Goals/nope.json"})
        self._write_config(config)
        findings = validate_case(self.case_dir, publish=False)
        self.assertIn("goals.file", [f.code for f in _warnings(findings)])
        self.assertEqual(_errors(findings), [])

    def test_malformed_goals_structure_is_an_error(self) -> None:
        goals_path = self.case_dir / "inputs" / "Goals" / "Goals.json"
        goals_path.parent.mkdir(parents=True)
        goals_path.write_text(json.dumps({"goals": {"1": {"target": {}, "angles": [0]}}}))

        config = _minimal_config(simulations={"goals_file": "inputs/Goals/Goals.json"})
        self._write_config(config)

        findings = validate_case(self.case_dir)
        codes = [f.code for f in _errors(findings)]
        self.assertIn("goals.target", codes)
        self.assertIn("goals.angles", codes)

    def test_valid_goals_produce_info_not_errors(self) -> None:
        goals_path = self.case_dir / "inputs" / "Goals" / "Goals.json"
        goals_path.parent.mkdir(parents=True)
        goals_path.write_text(
            json.dumps(
                {
                    "goals": {
                        "1": {
                            "target": {"longitude": 1.0, "latitude": 2.0},
                            "angles": [10, 20],
                        }
                    }
                }
            )
        )
        config = _minimal_config(simulations={"goals_file": "inputs/Goals/Goals.json"})
        self._write_config(config)

        findings = validate_case(self.case_dir)
        self.assertEqual(_errors(findings), [])
        self.assertIn("goals.ok", [f.code for f in findings])

    def test_observations_must_be_a_feature_collection(self) -> None:
        obs_path = self.case_dir / "inputs" / "Observations" / "obs.json"
        obs_path.parent.mkdir(parents=True)
        obs_path.write_text(json.dumps({"type": "NotAFeatureCollection"}))

        config = _minimal_config(onshore_waves="inputs/Observations/obs.json")
        self._write_config(config)

        findings = validate_case(self.case_dir)
        self.assertIn("observations.type", [f.code for f in _errors(findings)])

    def test_duplicate_instrument_id_is_an_error(self) -> None:
        obs_path = self.case_dir / "inputs" / "Observations" / "obs.json"
        obs_path.parent.mkdir(parents=True)
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [1.0, 2.0]},
            "properties": {"instrument_id": "Dup"},
        }
        obs_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": [feature, feature]})
        )
        config = _minimal_config(onshore_waves="inputs/Observations/obs.json")
        self._write_config(config)

        findings = validate_case(self.case_dir)
        self.assertIn("observations.instrument", [f.code for f in _errors(findings)])

    def test_offshore_entry_without_name_is_an_error(self) -> None:
        config = _minimal_config(offshore_waves=[{"data_path": "x.zarr"}])
        self._write_config(config)
        findings = validate_case(self.case_dir)
        self.assertIn("offshore.name", [f.code for f in _errors(findings)])

    def test_no_metrics_files_is_a_gap(self) -> None:
        self._write_config(_minimal_config())
        findings = validate_case(self.case_dir, publish=False)
        self.assertIn("outputs.metrics", [f.code for f in _warnings(findings)])

    def test_metrics_csv_missing_required_columns_is_an_error(self) -> None:
        metrics_dir = self.case_dir / "outputs" / "goal_01" / "isobaths"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "validation_metrics.csv").write_text("goal,rmse\n1,0.1\n")
        self._write_config(_minimal_config())
        findings = validate_case(self.case_dir)
        self.assertIn("outputs.metrics", [f.code for f in _errors(findings)])

    def test_metamodel_absent_skips_engine_checks(self) -> None:
        self._write_config(_minimal_config())
        findings = validate_case(self.case_dir)
        self.assertIn("engine.skip", [f.code for f in findings])

    def test_engine_checks_flag_incomplete_metamodel_section(self) -> None:
        config = _minimal_config(metamodel={"mda_candidates": [0, -1]})
        self._write_config(config)
        findings = validate_case(self.case_dir)
        codes = [f.code for f in _errors(findings)]
        self.assertIn("engine.mda_candidates", codes)
        self.assertIn("engine.mda_inputs", codes)
        self.assertIn("engine.vars_to_predict", codes)
        self.assertIn("engine.sets", codes)
        self.assertIn("engine.max_steepness", codes)

    def test_engine_checks_sets_paths_and_bounds(self) -> None:
        metamodel = {
            "mda_candidates": [50],
            "mda_inputs": ["hs", "tp", "dir"],
            "mda_directional_variables": ["dir"],
            "sets": {"buoys": {"sites": "buoy"}, "bad": {}},
            "paths": {"pca_dir": "outputs/{goal}/pca", "gp_dir": "outputs/{goal}/{metamodel}/gp"},
        }
        simulations = {
            "goals_file": "Goals.json",
            "bounds": {"tp": [1, 15]},
            "max_steepness": 0.06,
        }
        config = _minimal_config(metamodel=metamodel)
        config["simulations"].update(simulations)
        self._write_config(config)
        messages = [f.message for f in _errors(validate_case(self.case_dir))]
        self.assertTrue(any("sets['bad']" in m for m in messages))
        self.assertTrue(any("paths.pca_dir" in m for m in messages))
        self.assertFalse(any("paths.gp_dir" in m for m in messages))
        self.assertTrue(any("bounds.hs" in m for m in messages))
        self.assertFalse(any("bounds.dir" in m or "bounds.tp" in m for m in messages))


    def test_engine_checks_satellite(self) -> None:
        config = _minimal_config(metamodel={"mda_candidates": [50]})
        config["satellite"] = {
            "data_path": "https://example.org/altimeter.zarr",
            "radius_km": -5,
            "train_fraction": 1.5,
            "flags": [1, "2"],
        }
        self._write_config(config)
        messages = [
            f.message for f in _errors(validate_case(self.case_dir))
            if f.code == "engine.satellite"
        ]
        self.assertTrue(any("radius_km" in m for m in messages))
        self.assertTrue(any("train_fraction" in m for m in messages))
        self.assertTrue(any("flags" in m for m in messages))
        self.assertFalse(any("data_path" in m for m in messages))  # URLs are not checked

    def test_offshore_hindcast_key(self) -> None:
        config = _minimal_config()
        config["offshore_waves"] = [
            {"name": "WHACS", "data_path": "w.zarr"},
            {"name": "WHACS_cal", "data_path": "c.zarr", "hindcast": "WHACS"},
            {"name": "X", "data_path": "x.zarr", "hindcast": "ERA5"},
        ]
        self._write_config(config)
        findings = validate_case(self.case_dir)
        warnings = [f.message for f in _warnings(findings) if f.code == "offshore.hindcast"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("'ERA5'", warnings[0])

    def test_engine_checks_site_filter(self) -> None:
        metamodel = {
            "site_filter": {"hs_ratio_max": 3.0, "hs_nan_max": 1.5, "tp_below": 1},
        }
        self._write_config(_minimal_config(metamodel=metamodel))
        messages = [
            f.message for f in _errors(validate_case(self.case_dir))
            if f.code == "engine.site_filter"
        ]
        self.assertTrue(any("'hs_nan_max' is a fraction" in m for m in messages))
        self.assertTrue(any("'tp_below' must end with" in m for m in messages))
        self.assertFalse(any("hs_ratio_max" in m for m in messages))

    def test_engine_checks_model_block(self) -> None:
        metamodel = {
            "mda_candidates": [50],
            "mda_inputs": ["hs", "tp", "dir"],
            "mda_directional_variables": ["dir"],
            "vars_to_predict": {"hs": {"vars_to_stack": ["hs"], "pca_variance": 0.99}, "tp": "raw"},
        }
        config = _minimal_config(metamodel=metamodel)
        config["simulations"].update({"model": "xbeach"})
        self._write_config(config)
        messages = [f.message for f in _errors(validate_case(self.case_dir))]
        self.assertTrue(any("simulations.model must be one of" in m for m in messages))

        config["simulations"].update(
            {"model": "swan", "swan": {"grid_file": "mesh.grd", "metamodel_templates": "t"}}
        )
        self._write_config(config)
        findings = validate_case(self.case_dir)
        messages = [f.message for f in _errors(findings)]
        self.assertTrue(any("simulations.swan.dynamic_cases" in m for m in messages))
        self.assertFalse(any("simulations.swan.grid_file must" in m for m in messages))
        self.assertIn("engine.tp_raw", {f.code for f in findings})

SITES_HEADER = "lon,lat,source,source_id,source_label,nominal_depth,name,mesh_depth,status"


class TestOutputSites(unittest.TestCase):
    """``simulations.output_sites_config`` and ``outputs/output_sites_{model}.csv``."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / "TestCase"
        (self.case_dir / "outputs").mkdir(parents=True)
        self.config = _minimal_config(
            simulations={
                "model": "snapwave",
                "snapwave": {"grid_file": "inputs/GRID/mesh.nc"},
                "output_sites_config": {"isobaths": [{"name": "10m", "level": -10}]},
            },
            metamodel={"sets": {"buoys": {"sites": "buoy"}, "iso": {"sites": "10m"}}},
        )

    def _validate(self, rows: list[str] | None = None, grid: str = "inputs/GRID/mesh.nc"):
        if rows is not None:
            text = "\n".join([f"# grid: {grid}", SITES_HEADER, *rows]) + "\n"
            (self.case_dir / "outputs" / "output_sites_snapwave.csv").write_text(text)
        (self.case_dir / "TestCase.json").write_text(json.dumps(self.config))
        return validate_case(self.case_dir)

    def test_missing_table_is_a_gap(self) -> None:
        findings = self._validate()
        self.assertIn("sites.table", [f.code for f in _warnings(findings)])

    def test_table_counts_ok_sites_and_resolves_sets(self) -> None:
        findings = self._validate(
            ["1,2,isobath,10m_0,10m,10,,9.9,ok", "1,2,buoy,buoy_B,waves,8,B,,outside_mesh"]
        )
        self.assertIn("1 of 2 output sites ok", " ".join(f.message for f in findings))
        self.assertNotIn("sites.sets", [f.code for f in _errors(findings)])

    def test_table_on_another_grid_is_an_error(self) -> None:
        findings = self._validate(["1,2,isobath,10m_0,10m,10,,9.9,ok"], grid="old_mesh.nc")
        self.assertIn("sites.table", [f.code for f in _errors(findings)])

    def test_set_selecting_no_ok_site_is_an_error(self) -> None:
        self.config["metamodel"]["sets"]["bay"] = {"sites": ["Bay", "10m_0"]}
        findings = self._validate(["1,2,isobath,10m_0,10m,10,,9.9,ok"])
        errors = [f.message for f in _errors(findings) if f.code == "sites.sets"]
        self.assertEqual(len(errors), 1)
        self.assertIn("'Bay'", errors[0])

    def test_bad_site_config_values_are_errors(self) -> None:
        self.config["simulations"]["min_depth"] = -1
        self.config["simulations"]["output_sites_config"] = {
            "isobaths": [{"name": "5m"}, {"level": -10, "smooth_m": 0}],
            "polygons": [{"name": "Bay", "file": "Bay.shp", "fraction": 1.5}, {"file": "x.shp"}],
            "clip_to_aoi": "yes",
        }
        codes = [f.code for f in _errors(self._validate())]
        self.assertEqual(codes.count("sites.isobaths"), 2)
        self.assertEqual(codes.count("sites.polygons"), 2)
        self.assertIn("sites.config", codes)
        self.assertIn("sites.min_depth", codes)


class TestFindCaseConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / "TestCase"
        self.case_dir.mkdir()

    def test_finds_exact_name_match(self) -> None:
        (self.case_dir / "TestCase.json").write_text("{}")
        self.assertEqual(find_case_config(self.case_dir), self.case_dir / "TestCase.json")

    def test_other_names_are_not_the_config(self) -> None:
        (self.case_dir / "testcase.json").write_text("{}")
        (self.case_dir / "config.json").write_text("{}")
        self.assertIsNone(find_case_config(self.case_dir))


class TestValidateAgainstRealScaffold(unittest.TestCase):
    """A freshly scaffolded case should pass in dev mode (warnings only)."""

    def test_fresh_case_has_no_errors_in_dev_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = start_case("FreshCase", tmp)
            findings = validate_case(case_dir, publish=False)
            self.assertEqual(_errors(findings), [])

    def test_fresh_case_has_errors_in_publish_mode(self) -> None:
        # A brand-new case is nowhere near portal-ready: gaps escalate.
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = start_case("FreshCase2", tmp)
            findings = validate_case(case_dir, publish=True)
            self.assertGreater(len(_errors(findings)), 0)


if __name__ == "__main__":
    unittest.main()
