"""HyGloW — hybrid global wave downscaling toolkit.

One module per command (``python -m hywaves <command>``): :mod:`.start`,
:mod:`.validate` (the case rules), :mod:`.publish`, :mod:`.snapshot`,
:mod:`.update`; :mod:`.common` holds what they share (template layout, case
config, case manifest with the bluemath_tk revision). ``template/`` is the
case every ``start`` copies.
"""

__version__ = "0.15.0"
