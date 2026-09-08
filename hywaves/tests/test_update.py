"""Unit tests for ``hywaves.update`` (the ``update`` command).

``plan_update``/``update_case`` are exercised against a synthetic template +
case pair (via monkeypatched ``TEMPLATE_DIR``) so the classification logic
(up_to_date/update/add/modified/orphaned) can be tested precisely without
touching the real ``hywaves/template/``. A separate integration test at the
bottom scaffolds from the *real* template to guard against actual breakage.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hywaves import update as update_mod
from hywaves.scaffold import start_case
from hywaves.update import UpdateError, plan_update, update_case


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _quiet_update_case(*args, **kwargs):
    """``update_case`` narrates its plan to stdout; keep test output clean."""
    with contextlib.redirect_stdout(io.StringIO()):
        return update_case(*args, **kwargs)


class TestPlanUpdate(unittest.TestCase):
    """Classification logic, against a small synthetic template/case pair."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self.template_dir = self.root / "template"
        _write(self.template_dir / "utils" / "a.py", "A = 1\n")
        _write(self.template_dir / "utils" / "b.py", "B = 1\n")
        _write(self.template_dir / "tests" / "test_a.py", "# test a\n")

        patcher = mock.patch.object(update_mod, "TEMPLATE_DIR", self.template_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.case_dir = self.root / "MyCase"
        _write(self.case_dir / "utils" / "a.py", "A = 1\n")
        _write(self.case_dir / "utils" / "b.py", "B = 1\n")
        _write(self.case_dir / "tests" / "test_a.py", "# test a\n")

    def _write_manifest(self, engine_files: dict[str, str]) -> None:
        manifest = {"engine_files": engine_files}
        _write(
            self.case_dir / ".hywaves" / "manifest.json",
            json.dumps(manifest),
        )

    def test_all_up_to_date_when_identical(self) -> None:
        plan = plan_update(self.case_dir)
        self.assertEqual(sorted(plan["up_to_date"]), ["tests/test_a.py", "utils/a.py", "utils/b.py"])
        self.assertEqual(plan["update"], [])
        self.assertEqual(plan["add"], [])
        self.assertEqual(plan["modified"], [])
        self.assertEqual(plan["orphaned"], [])

    def test_new_template_file_is_add(self) -> None:
        _write(self.template_dir / "utils" / "c.py", "C = 1\n")
        plan = plan_update(self.case_dir)
        self.assertEqual(plan["add"], ["utils/c.py"])

    def test_orphaned_file_not_in_template(self) -> None:
        _write(self.case_dir / "utils" / "legacy.py", "LEGACY = 1\n")
        plan = plan_update(self.case_dir)
        self.assertEqual(plan["orphaned"], ["utils/legacy.py"])

    def test_untouched_file_that_diverged_from_template_is_update(self) -> None:
        # No manifest: without shipped-state provenance every diverging file
        # is treated conservatively as locally modified, never auto-synced.
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")
        plan = plan_update(self.case_dir)
        self.assertEqual(plan["modified"], ["utils/a.py"])

        # With a manifest recording the shipped hash for the *old* content,
        # and the case file still matching that shipped hash (untouched by
        # the user), the file is safe to auto-sync.
        import hashlib

        old_hash = hashlib.sha256(b"A = 1\n").hexdigest()
        self._write_manifest({"utils/a.py": old_hash, "utils/b.py": hashlib.sha256(b"B = 1\n").hexdigest()})
        plan = plan_update(self.case_dir)
        self.assertEqual(plan["update"], ["utils/a.py"])

    def test_locally_modified_file_is_modified_even_with_manifest(self) -> None:
        import hashlib

        shipped_hash = hashlib.sha256(b"A = 1\n").hexdigest()
        self._write_manifest({"utils/a.py": shipped_hash, "utils/b.py": hashlib.sha256(b"B = 1\n").hexdigest()})

        # Template moved on AND the user edited their local copy too.
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")
        _write(self.case_dir / "utils" / "a.py", "A = 99  # user tweak\n")

        plan = plan_update(self.case_dir)
        self.assertEqual(plan["modified"], ["utils/a.py"])

    def test_empty_template_raises(self) -> None:
        empty_template = self.root / "empty_template"
        empty_template.mkdir()
        with mock.patch.object(update_mod, "TEMPLATE_DIR", empty_template):
            with self.assertRaises(UpdateError):
                plan_update(self.case_dir)

    def test_not_a_directory_raises(self) -> None:
        with self.assertRaises(UpdateError):
            update_case(self.root / "does-not-exist")

    def test_refuses_to_update_template_from_itself(self) -> None:
        with self.assertRaises(UpdateError):
            update_case(self.template_dir)


class TestUpdateCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

        self.template_dir = self.root / "template"
        _write(self.template_dir / "utils" / "a.py", "A = 1\n")

        patcher = mock.patch.object(update_mod, "TEMPLATE_DIR", self.template_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.case_dir = self.root / "MyCase"
        _write(self.case_dir / "utils" / "a.py", "A = 1\n")

    def test_dry_run_writes_nothing(self) -> None:
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")
        _quiet_update_case(self.case_dir, dry_run=True)
        self.assertEqual((self.case_dir / "utils" / "a.py").read_text(), "A = 1\n")
        self.assertFalse((self.case_dir / ".hywaves" / "manifest.json").exists())

    def test_syncs_untouched_file_and_writes_manifest(self) -> None:
        import hashlib

        shipped_hash = hashlib.sha256(b"A = 1\n").hexdigest()
        _write(
            self.case_dir / ".hywaves" / "manifest.json",
            json.dumps({"engine_files": {"utils/a.py": shipped_hash}}),
        )
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")

        _quiet_update_case(self.case_dir)

        self.assertEqual((self.case_dir / "utils" / "a.py").read_text(), "A = 2\n")
        manifest = json.loads((self.case_dir / ".hywaves" / "manifest.json").read_text())
        new_hash = hashlib.sha256(b"A = 2\n").hexdigest()
        self.assertEqual(manifest["engine_files"]["utils/a.py"], new_hash)

    def test_keeps_locally_modified_file_without_force(self) -> None:
        _write(self.case_dir / "utils" / "a.py", "A = 99  # my tweak\n")
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")

        _quiet_update_case(self.case_dir)
        self.assertEqual((self.case_dir / "utils" / "a.py").read_text(), "A = 99  # my tweak\n")

    def test_force_overwrites_locally_modified_file(self) -> None:
        _write(self.case_dir / "utils" / "a.py", "A = 99  # my tweak\n")
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")

        _quiet_update_case(self.case_dir, force=True)
        self.assertEqual((self.case_dir / "utils" / "a.py").read_text(), "A = 2\n")


class TestUpdateAgainstRealTemplate(unittest.TestCase):
    """Regression check: a case just scaffolded from the real template is
    immediately fully in sync — nothing to add, update, or flag as modified.
    """

    def test_freshly_scaffolded_case_is_fully_up_to_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = start_case("FreshCase", tmp)
            plan = plan_update(case_dir)
            self.assertEqual(plan["update"], [])
            self.assertEqual(plan["add"], [])
            self.assertEqual(plan["modified"], [])
            self.assertGreater(len(plan["up_to_date"]), 0)


if __name__ == "__main__":
    unittest.main()
