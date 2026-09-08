"""Unit tests for ``hywaves.cli`` (``python -m hywaves ...``)."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from hywaves import __version__
from hywaves.cli import main


def _run(argv: list[str]) -> tuple[int, str]:
    """Run the CLI, returning (exit_code, combined stdout+stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue() + err.getvalue()


class TestCliStart(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = Path(self._tmp.name)

    def test_start_creates_case_and_exits_zero(self) -> None:
        code, out = _run(["start", "--name", "CliCase", "--dest", str(self.dest)])
        self.assertEqual(code, 0)
        self.assertTrue((self.dest / "CliCase" / "CliCase.json").is_file())
        self.assertIn("Created case", out)

    def test_start_no_notebooks_flag(self) -> None:
        code, _ = _run(
            ["start", "--name", "CliCase2", "--dest", str(self.dest), "--no-notebooks"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(list((self.dest / "CliCase2").glob("*.ipynb")), [])

    def test_start_invalid_name_exits_nonzero(self) -> None:
        code, _ = _run(["start", "--name", "1Bad", "--dest", str(self.dest)])
        self.assertNotEqual(code, 0)

    def test_start_existing_target_exits_nonzero(self) -> None:
        _run(["start", "--name", "CliCase3", "--dest", str(self.dest)])
        code, _ = _run(["start", "--name", "CliCase3", "--dest", str(self.dest)])
        self.assertNotEqual(code, 0)


class TestCliValidate(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = Path(self._tmp.name)
        _run(["start", "--name", "ValCase", "--dest", str(self.dest)])
        self.case_dir = self.dest / "ValCase"

    def test_dev_mode_passes(self) -> None:
        code, out = _run(["validate", str(self.case_dir)])
        self.assertEqual(code, 0)
        self.assertIn("PASSED", out)

    def test_publish_mode_fails_on_fresh_case(self) -> None:
        code, out = _run(["validate", str(self.case_dir), "--publish"])
        self.assertNotEqual(code, 0)
        self.assertIn("error(s)", out)


class TestCliUpdate(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dest = Path(self._tmp.name)
        _run(["start", "--name", "UpdCase", "--dest", str(self.dest)])
        self.case_dir = self.dest / "UpdCase"

    def test_dry_run_on_fresh_case_exits_zero(self) -> None:
        code, out = _run(["update", str(self.case_dir), "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("Dry run", out)

    def test_update_on_nonexistent_case_exits_nonzero(self) -> None:
        code, out = _run(["update", str(self.dest / "NoSuchCase")])
        self.assertNotEqual(code, 0)
        self.assertIn("error", out.lower())


class TestCliMisc(unittest.TestCase):
    def test_version_flag(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--version"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn(__version__, buf.getvalue())

    def test_no_command_exits_nonzero(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main([])
        self.assertNotEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
