"""Unit tests for ``hywaves.scaffold`` (the ``start`` command)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from hywaves.scaffold import (
    ENGINE_DIRS,
    SUPPORT_DIRS,
    SYNCED_DIRS,
    TEMPLATE_DIR,
    ScaffoldError,
    _file_hashes,
    start_case,
)


class TestStartCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = Path(self._tmp.name)

    def test_creates_expected_structure(self) -> None:
        target = start_case("MyCase", self.dest)

        self.assertEqual(target, self.dest / "MyCase")
        self.assertTrue((target / "utils").is_dir())
        self.assertTrue((target / "tests").is_dir())
        self.assertTrue((target / "templates-metamodel").is_dir())
        self.assertTrue((target / "templates-dynamic").is_dir())
        self.assertTrue((target / "MyCase.json").is_file())
        self.assertTrue((target / "README.md").is_file())
        self.assertTrue((target / ".hywaves" / "manifest.json").is_file())
        # bluemesh2d is a regular environment dependency now, never vendored
        # from the template; no .gitignore/requirements.txt are generated
        # either — see the README's own Dependencies section instead.
        self.assertFalse((target / "bluemesh2d").exists())
        self.assertFalse((target / ".gitignore").exists())
        self.assertFalse((target / "requirements.txt").exists())

        for sub in ("AoI_Polygons", "GEBCO", "GRID", "Goals", "Observations", "WHACS"):
            self.assertTrue((target / "inputs" / sub / ".gitkeep").is_file())
        for sub in ("outputs", "logs"):
            self.assertTrue((target / sub / ".gitkeep").is_file())

    def test_case_json_and_readme_have_name_substituted(self) -> None:
        target = start_case("Wharf", self.dest)

        config = json.loads((target / "Wharf.json").read_text())
        self.assertEqual(config["name"], "Wharf")
        self.assertNotIn("__CASE_NAME__", json.dumps(config))

        readme = (target / "README.md").read_text()
        self.assertNotIn("__CASE_NAME__", readme)
        self.assertIn("Wharf", readme)

    def test_notebooks_copied_and_stripped_of_outputs(self) -> None:
        target = start_case("Atoll", self.dest)

        template_notebooks = sorted((TEMPLATE_DIR / "notebooks").glob("*.ipynb"))
        self.assertTrue(template_notebooks, "template ships no notebooks to copy")
        for nb_path in template_notebooks:
            copied = target / nb_path.name
            self.assertTrue(copied.is_file())
            nb = json.loads(copied.read_text())
            for cell in nb["cells"]:
                if cell.get("cell_type") == "code":
                    self.assertEqual(cell.get("outputs"), [])
                    self.assertIsNone(cell.get("execution_count"))

    def test_no_notebooks_flag_skips_notebooks(self) -> None:
        target = start_case("Barrier", self.dest, notebooks=False)
        self.assertEqual(list(target.glob("*.ipynb")), [])

        manifest = json.loads((target / ".hywaves" / "manifest.json").read_text())
        self.assertEqual(manifest["notebooks"], [])

    def test_manifest_engine_hashes_match_template(self) -> None:
        target = start_case("Lagoon", self.dest)

        manifest = json.loads((target / ".hywaves" / "manifest.json").read_text())
        expected = _file_hashes(TEMPLATE_DIR, SYNCED_DIRS)
        self.assertEqual(manifest["engine_files"], expected)
        self.assertTrue(expected, "template has no engine files to hash")

    def test_invalid_name_rejected(self) -> None:
        for bad in ("", "1Case", "My Case", "../escape", "Case/Sub"):
            with self.assertRaises(ScaffoldError):
                start_case(bad, self.dest)

    def test_existing_target_rejected(self) -> None:
        start_case("Duplicate", self.dest)
        with self.assertRaises(ScaffoldError):
            start_case("Duplicate", self.dest)

    def test_engine_and_support_dirs_are_disjoint_and_cover_synced_dirs(self) -> None:
        self.assertEqual(SYNCED_DIRS, ENGINE_DIRS + ("tests",))
        self.assertIn("tests", SUPPORT_DIRS)


class TestStartCaseFromExistingCase(unittest.TestCase):
    """``--from-case`` clones another case's engine + notebooks (outputs stripped)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self._orig_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._orig_cwd)

        self.source = self.root / "SourceCase"
        (self.source / "utils").mkdir(parents=True)
        (self.source / "utils" / "helper.py").write_text("VALUE = 1\n")

        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('hi')"],
                    "outputs": [{"output_type": "stream", "text": "hi\n"}],
                    "execution_count": 3,
                    "metadata": {},
                }
            ],
            "metadata": {"widgets": {"state": {}}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        (self.source / "00_Source_Notebook.ipynb").write_text(json.dumps(notebook))

    def test_clones_engine_and_strips_notebook_outputs(self) -> None:
        target = start_case("Clone", self.root / "out", from_case="SourceCase")

        self.assertEqual(
            (target / "utils" / "helper.py").read_text(),
            (self.source / "utils" / "helper.py").read_text(),
        )

        nb = json.loads((target / "00_Source_Notebook.ipynb").read_text())
        cell = nb["cells"][0]
        self.assertEqual(cell["outputs"], [])
        self.assertIsNone(cell["execution_count"])
        self.assertNotIn("widgets", nb.get("metadata", {}))

        manifest = json.loads((target / ".hywaves" / "manifest.json").read_text())
        self.assertEqual(manifest["source"], "SourceCase")

    def test_unknown_source_case_raises(self) -> None:
        with self.assertRaises(ScaffoldError):
            start_case("Clone2", self.root / "out", from_case="NoSuchCase")

    def test_source_equals_target_rejected(self) -> None:
        with self.assertRaises(ScaffoldError):
            start_case("SourceCase", self.root, from_case="SourceCase")


if __name__ == "__main__":
    unittest.main()
