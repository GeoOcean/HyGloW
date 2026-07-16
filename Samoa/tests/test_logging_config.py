"""Notebook logging configuration."""

from __future__ import annotations

import logging
import unittest

from utils.config import configure_notebook_logging


class TestLoggingConfig(unittest.TestCase):
    def test_configure_notebook_logging_sets_console_only(self) -> None:
        logger = logging.getLogger("hyglow.test_logging")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        logger.propagate = False
        file_handler = logging.FileHandler("/dev/null")
        console_handler = logging.StreamHandler()
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        configure_notebook_logging(console_level="WARNING")

        self.assertEqual(console_handler.level, logging.WARNING)
        self.assertEqual(file_handler.level, logging.NOTSET)
        self.assertEqual(logger.level, logging.INFO)


if __name__ == "__main__":
    unittest.main()
