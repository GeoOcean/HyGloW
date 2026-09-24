"""HyGloW — hybrid global wave downscaling toolkit.

Source of truth for the case contract, the ``CaseConfig`` loader, and the
scaffolding CLI. Cases created with ``python -m hywaves start`` are fully
self-contained folders: the engine code (``utils/``) is vendored into each
case so it can run on any machine (laptop or HPC) without installing this
package — ``bluemesh2d`` and the rest of the scientific stack are regular
environment dependencies, see a case's own README.
"""

__version__ = "0.5.0"

# Format version of the internal .hywaves/manifest.json bookkeeping file
# (written by `start`/`update`, never hand-edited). Case config JSON has no
# schema_version of its own — see the Changelog in contract/CONTRACT.md.
SCHEMA_VERSION = 1
