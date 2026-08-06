"""Notebook logging configuration."""

from __future__ import annotations

import contextlib
import io
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


class TestBluemathModelConsoleQuieting(unittest.TestCase):
    """bluemath_tk model loggers (PCA, MDA, RBF, GP, ...) shouldn't flood the
    console -- see ``_quiet_bluemath_model_console`` in utils.config.
    """

    def setUp(self) -> None:
        from bluemath_tk.core.models import BlueMathModel

        self.BlueMathModel = BlueMathModel
        configure_notebook_logging()

    def test_fresh_model_instantiation_is_quiet_on_console(self) -> None:
        # BlueMathModel.__init__ re-creates the console handler on every
        # instantiation (via set_logger_name) at bluemath_tk's own default
        # level (WARNING) unless our patch intercepts it.
        class _DummyModel(self.BlueMathModel):
            def __init__(self) -> None:
                super().__init__()

        with contextlib.redirect_stderr(io.StringIO()) as buf:
            model = _DummyModel()
            model.logger.warning("routine banner, e.g. 'Using PCA'")

        self.assertEqual(buf.getvalue(), "")

    def test_logger_restored_by_name_reference_is_repaired(self) -> None:
        # logging.Logger pickles by *name reference* (Logger.__reduce__ ->
        # getLogger(name)), not by value -- so a model loaded from disk
        # (load_model/load_models) gets a `_logger` that's just whatever the
        # process's logger for that name currently is. If nothing has
        # configured that name yet this kernel, it has no handlers and
        # propagate=True, so every record falls through to the root logger.
        name = "_HyglowTestUnpickledModel"
        bare_logger = logging.getLogger(name)
        bare_logger.handlers = []
        bare_logger.propagate = True

        class _DummyModel(self.BlueMathModel):
            def __init__(self, logger: logging.Logger) -> None:
                self._logger = logger
                self._exclude_attributes = []
                self.num_workers = 1

        model = _DummyModel(bare_logger)

        with contextlib.redirect_stderr(io.StringIO()) as buf:
            model.logger.warning("should not leak to the console via root")

        self.assertEqual(buf.getvalue(), "")
        self.assertFalse(model.logger.propagate)
        self.assertTrue(model.logger.handlers)


if __name__ == "__main__":
    unittest.main()
