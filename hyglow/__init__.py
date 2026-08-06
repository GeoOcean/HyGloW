"""HyGloW — hybrid global wave downscaling toolkit.

Source of truth for the case contract, the ``CaseConfig`` loader, and the
scaffolding CLI. Cases created with ``python -m hyglow start`` are fully
self-contained folders: the engine code (``utils/``) is vendored into each
case so it can run on any machine (laptop or HPC) without installing this
package — ``bluemesh2d`` is the one exception, a regular pip dependency.
"""

__version__ = "0.3.0"

# Format version of the internal .hyglow/manifest.json bookkeeping file
# (written by `start`/`update`, never hand-edited). Case config JSON has no
# schema_version of its own — see the Changelog in contract/CONTRACT.md.
SCHEMA_VERSION = 1
