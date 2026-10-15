"""The template's own tests (``hywaves/template/tests``), run against a freshly scaffolded case.

Cases do not carry tests: the template's ``utils/`` is tested here, through a
case made with ``start``. To run them against an existing case::

    cd MyCase && PYTHONPATH=. python -m unittest discover -s ../hywaves/template/tests -t ../hywaves/template/tests
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

from hywaves.scaffold import TEMPLATE_DIR, start_case


class TestTemplateUtils(unittest.TestCase):
    def test_template_tests_pass_on_a_new_case(self) -> None:
        tests = str(TEMPLATE_DIR / "tests")
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
