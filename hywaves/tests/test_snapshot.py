"""Unit tests for ``hywaves.snapshot`` (``python -m hywaves snapshot``)."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from hywaves.cli import main
from hywaves.snapshot import SnapshotError, list_snapshots, snapshot_case


class TestSnapshot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.case_dir = Path(self._tmp.name) / "SnapCase"
        portal = self.case_dir / "portal"
        (portal / "book").mkdir(parents=True)
        (portal / "book" / "index.html").write_text("<html></html>")
        (portal / "validation_metrics.csv").write_text("goal\n1\n")
        (portal / "manifest.json").write_text(
            json.dumps({"schema": 1, "case": "SnapCase", "files": {}})
        )

    def test_copies_portal_and_book_with_tagged_manifest(self) -> None:
        target = snapshot_case(self.case_dir, "hs1", note="unit Hs")
        self.assertEqual(target, self.case_dir / "portal_hs1")
        self.assertTrue((target / "validation_metrics.csv").is_file())
        self.assertTrue((target / "book" / "index.html").is_file())
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(manifest["snapshot"]["tag"], "hs1")
        self.assertEqual(manifest["snapshot"]["note"], "unit Hs")
        self.assertEqual(manifest["case"], "SnapCase")
        self.assertNotIn("snapshot", json.loads((self.case_dir / "portal" / "manifest.json").read_text()))
        self.assertEqual(list_snapshots(self.case_dir), [target])

    def test_refuses_bad_or_taken_tags_and_missing_portal(self) -> None:
        for bad in ("HS1", "hs 1", "_x", ""):
            with self.assertRaises(SnapshotError):
                snapshot_case(self.case_dir, bad)
        snapshot_case(self.case_dir, "hs1")
        with self.assertRaises(SnapshotError):
            snapshot_case(self.case_dir, "hs1")
        with self.assertRaises(SnapshotError):
            snapshot_case(Path(self._tmp.name) / "NoPortal", "v1")

    def test_cli(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = main(["snapshot", str(self.case_dir), "v2", "--note", "test"])
        self.assertEqual(code, 0)
        self.assertIn("portal_v2", out.getvalue())


if __name__ == "__main__":
    unittest.main()
