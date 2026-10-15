"""Notebook logging configuration."""

from __future__ import annotations

import unittest

from bluemath_tk.core import models

from utils.config import configure_notebook_logging


class TestLoggingConfig(unittest.TestCase):
    def test_models_console_level(self) -> None:
        self.addCleanup(models.set_models_console_level, models.MODELS_CONSOLE_LEVEL)
        configure_notebook_logging("CRITICAL")
        self.assertEqual(models.MODELS_CONSOLE_LEVEL, "CRITICAL")


if __name__ == "__main__":
    unittest.main()
