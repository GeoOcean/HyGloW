"""Unit tests for ``hyglow.contract.validate`` (the ``validate`` command)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hyglow.contract.validate import ERROR, WARN, find_case_config, validate_case
from hyglow.scaffold import start_case


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
        metrics_dir = self.case_dir / "outputs" / "goal_01"
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


class TestFindCaseConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / "TestCase"
        self.case_dir.mkdir()

    def test_finds_exact_name_match(self) -> None:
        (self.case_dir / "TestCase.json").write_text("{}")
        self.assertEqual(find_case_config(self.case_dir), self.case_dir / "TestCase.json")

    def test_finds_lowercase_fallback(self) -> None:
        (self.case_dir / "testcase.json").write_text("{}")
        self.assertEqual(find_case_config(self.case_dir), self.case_dir / "testcase.json")

    def test_ignores_goals_and_observations_json(self) -> None:
        (self.case_dir / "Goals.json").write_text("{}")
        (self.case_dir / "observations.json").write_text("{}")
        self.assertIsNone(find_case_config(self.case_dir))

    def test_returns_none_when_ambiguous(self) -> None:
        (self.case_dir / "one.json").write_text("{}")
        (self.case_dir / "two.json").write_text("{}")
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
