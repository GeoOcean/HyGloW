"""The tests of the template's ``utils/`` (``case_utils/``), run on a freshly started case.

They import ``utils`` from a case, so they run inside one. On an existing case::

    cd MyCase && PYTHONPATH=. python -m unittest discover -s ../hywaves/tests/case_utils -t ../hywaves/tests/case_utils
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hywaves.start import start_case

CASE_UTILS_TESTS = Path(__file__).resolve().parent / "case_utils"


class TestTemplateUtils(unittest.TestCase):
    def test_template_tests_pass_on_a_new_case(self) -> None:
        tests = str(CASE_UTILS_TESTS)
        with tempfile.TemporaryDirectory() as tmp:
            case = start_case("TemplateCheck", tmp, notebooks=False)
            run = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", tests, "-t", tests],
                cwd=case,
                env={**os.environ, "PYTHONPATH": str(case)},
                capture_output=True,
                text=True,
                timeout=1200,
            )
        self.assertEqual(run.returncode, 0, run.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
