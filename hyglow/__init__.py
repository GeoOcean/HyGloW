"""HyGloW — hybrid global wave downscaling toolkit.

Source of truth for the case contract, the ``CaseConfig`` loader, and the
scaffolding CLI. Cases created with ``python -m hyglow start`` are fully
self-contained folders: the engine code is vendored into each case so it can
run on any machine (laptop or HPC) without installing this package.
"""

__version__ = "0.3.0"

SCHEMA_VERSION = 1
