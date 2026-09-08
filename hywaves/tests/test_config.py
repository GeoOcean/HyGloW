"""Unit tests for ``hywaves.config.CaseConfig``."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hywaves.config import CaseConfig, CaseConfigError, find_case_config


def _write_config(case_dir: Path, **overrides) -> None:
    config = {
        "name": "TestCase",
        "description": "desc",
        "paths": {"aoi_file": "inputs/AoI_Polygons/TestCase_AoI.shp"},
        "onshore_waves": "inputs/Observations/TestCase_observations.json",
        "offshore_waves": [{"name": "WHACS", "data_path": "x.zarr"}],
        "simulations": {"goals_file": "inputs/Goals/TestCase_Goals.json"},
        "metamodel": {"mda_candidates": [50, 100]},
    }
    config.update(overrides)
    (case_dir / "TestCase.json").write_text(json.dumps(config))


class TestCaseConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / "TestCase"
        self.case_dir.mkdir()
        _write_config(self.case_dir)

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(CaseConfigError):
            CaseConfig(self.case_dir / "nope")

    def test_missing_config_raises(self) -> None:
        empty_dir = self.case_dir.parent / "Empty"
        empty_dir.mkdir()
        with self.assertRaises(CaseConfigError):
            CaseConfig(empty_dir)

    def test_identity_properties(self) -> None:
        case = CaseConfig(self.case_dir)
        self.assertEqual(case.name, "TestCase")
        self.assertEqual(case.description, "desc")

    def test_name_falls_back_to_dir_name_when_unset(self) -> None:
        _write_config(self.case_dir, name="")
        case = CaseConfig(self.case_dir)
        self.assertEqual(case.name, "TestCase")  # the folder name

    def test_resolve_relative_and_absolute_paths(self) -> None:
        case = CaseConfig(self.case_dir)
        self.assertEqual(case.resolve("inputs/x.txt"), self.case_dir / "inputs" / "x.txt")
        absolute = Path("/tmp/somewhere/x.txt")
        self.assertEqual(case.resolve(absolute), absolute)

    def test_offshore_waves_normalizes_single_dict_to_list(self) -> None:
        _write_config(self.case_dir, offshore_waves={"name": "WHACS", "data_path": "x.zarr"})
        case = CaseConfig(self.case_dir)
        self.assertEqual(case.offshore_waves, [{"name": "WHACS", "data_path": "x.zarr"}])

    def test_get_offshore_wave_by_name_case_insensitive(self) -> None:
        case = CaseConfig(self.case_dir)
        entry = case.get_offshore_wave("whacs")
        self.assertEqual(entry["name"], "WHACS")

    def test_get_offshore_wave_unknown_name_raises(self) -> None:
        case = CaseConfig(self.case_dir)
        with self.assertRaises(KeyError):
            case.get_offshore_wave("nope")

    def test_get_offshore_wave_empty_raises(self) -> None:
        _write_config(self.case_dir, offshore_waves=[])
        case = CaseConfig(self.case_dir)
        with self.assertRaises(CaseConfigError):
            case.get_offshore_wave()

    def test_load_goals(self) -> None:
        goals_path = self.case_dir / "inputs" / "Goals" / "TestCase_Goals.json"
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
        case = CaseConfig(self.case_dir)
        goals = case.load_goals()
        self.assertEqual(set(goals), {"1"})

        slim = case.goals_for_snapwave(goals)
        self.assertEqual(slim["1"], {"lon": 1.0, "lat": 2.0, "angles": [10, 20]})

    def test_load_goals_without_goals_file_configured_raises(self) -> None:
        _write_config(self.case_dir, simulations={})
        case = CaseConfig(self.case_dir)
        with self.assertRaises(CaseConfigError):
            case.load_goals()

    def test_load_observations(self) -> None:
        obs_path = self.case_dir / "inputs" / "Observations" / "TestCase_observations.json"
        obs_path.parent.mkdir(parents=True)
        obs_path.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
        case = CaseConfig(self.case_dir)
        self.assertEqual(case.load_observations()["type"], "FeatureCollection")

    def test_load_observations_without_onshore_waves_raises(self) -> None:
        _write_config(self.case_dir, onshore_waves=None)
        case = CaseConfig(self.case_dir)
        with self.assertRaises(CaseConfigError):
            case.load_observations()

    def test_goal_metrics_files_and_figure_files(self) -> None:
        case = CaseConfig(self.case_dir)
        self.assertEqual(case.goal_metrics_files(), [])
        self.assertEqual(case.figure_files(), [])

        metrics_dir = self.case_dir / "outputs" / "goal_01"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "validation_metrics.csv").write_text("goal\n1\n")
        fig_dir = self.case_dir / "outputs" / "gui_figures"
        fig_dir.mkdir(parents=True)
        (fig_dir / "fig01_domain.png").touch()

        case.reload()
        self.assertEqual(len(case.goal_metrics_files()), 1)
        self.assertEqual(len(case.figure_files()), 1)

    def test_reload_picks_up_disk_changes(self) -> None:
        case = CaseConfig(self.case_dir)
        self.assertEqual(case.description, "desc")
        _write_config(self.case_dir, description="updated")
        case.reload()
        self.assertEqual(case.description, "updated")


class TestFindCaseConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / "TestCase"
        self.case_dir.mkdir()

    def test_no_candidates_raises(self) -> None:
        with self.assertRaises(CaseConfigError):
            find_case_config(self.case_dir)

    def test_ambiguous_candidates_raises(self) -> None:
        (self.case_dir / "one.json").write_text("{}")
        (self.case_dir / "two.json").write_text("{}")
        with self.assertRaises(CaseConfigError):
            find_case_config(self.case_dir)


if __name__ == "__main__":
    unittest.main()
