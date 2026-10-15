"""HyGloW — hybrid global wave downscaling toolkit.

Source of truth for the case checks (``contract/validate.py``), the case template
(``template/``) and the CLI (``python -m hywaves start/update/validate/
publish/snapshot``). Cases created with ``start`` are self-contained folders:
the case code (``utils/``) is vendored into each case, and the case manifest
records the hywaves version and bluemath_tk revision it works with (see
:mod:`hywaves.manifest`).
"""

__version__ = "0.15.0"
