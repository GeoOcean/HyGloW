"""Unit tests for ``hywaves.update`` (the ``update`` command).

``plan_update``/``update_case`` are exercised against a synthetic template +
case pair (via monkeypatched ``TEMPLATE_DIR``) so the classification logic
(up_to_date/update/add/modified/orphaned) can be tested precisely without
touching the real ``hywaves/template/``. A separate integration test at the
bottom scaffolds from the *real* template to guard against actual breakage.
"""

from __future__ import annotations

import contextlib
import hashlib
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_manifest(case_dir: Path, engine_files: dict[str, str]) -> None:
    _write(case_dir / ".hywaves" / "manifest.json", json.dumps({"engine_files": engine_files}))


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

        patcher = mock.patch.object(update_mod, "TEMPLATE_DIR", self.template_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.case_dir = self.root / "MyCase"
        _write(self.case_dir / "utils" / "a.py", "A = 1\n")
        _write(self.case_dir / "utils" / "b.py", "B = 1\n")
        _write_manifest(
            self.case_dir,
            {"utils/a.py": _sha("A = 1\n"), "utils/b.py": _sha("B = 1\n")},
        )

    def test_case_without_manifest_raises(self) -> None:
        (self.case_dir / ".hywaves" / "manifest.json").unlink()
        with self.assertRaises(UpdateError):
            plan_update(self.case_dir)

    def test_all_up_to_date_when_identical(self) -> None:
        plan = plan_update(self.case_dir)
        self.assertEqual(sorted(plan["up_to_date"]), ["utils/a.py", "utils/b.py"])
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
        # The case file still matches its shipped hash: safe to auto-sync.
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")
        plan = plan_update(self.case_dir)
        self.assertEqual(plan["update"], ["utils/a.py"])

    def test_locally_modified_file_is_modified(self) -> None:
        # Template moved on AND the user edited their local copy too.
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")
        _write(self.case_dir / "utils" / "a.py", "A = 99  # user tweak\n")

        plan = plan_update(self.case_dir)
        self.assertEqual(plan["modified"], ["utils/a.py"])

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
        _write_manifest(self.case_dir, {"utils/a.py": _sha("A = 1\n")})
        self.manifest_path = self.case_dir / ".hywaves" / "manifest.json"

    def test_dry_run_writes_nothing(self) -> None:
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")
        before = self.manifest_path.read_text()
        _quiet_update_case(self.case_dir, dry_run=True)
        self.assertEqual((self.case_dir / "utils" / "a.py").read_text(), "A = 1\n")
        self.assertEqual(self.manifest_path.read_text(), before)

    def test_syncs_untouched_file_and_records_toolkit(self) -> None:
        _write(self.template_dir / "utils" / "a.py", "A = 2\n")

        _quiet_update_case(self.case_dir)

        self.assertEqual((self.case_dir / "utils" / "a.py").read_text(), "A = 2\n")
        manifest = json.loads(self.manifest_path.read_text())
        self.assertEqual(manifest["engine_files"]["utils/a.py"], _sha("A = 2\n"))
        self.assertIn("version", manifest["bluemath_tk"])
        self.assertIn("commit", manifest["bluemath_tk"])

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
