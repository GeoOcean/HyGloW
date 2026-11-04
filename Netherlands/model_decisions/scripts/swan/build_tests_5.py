"""Komen with both forcings and friction 0.019, scored at the buoys.

Run from Netherlands/ with GDAL_DATA and PROJ_DATA set.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "SWAN_test")
from build_tests_2 import CASES, ROOT, copy_case  # noqa: E402

FRICTION = ("FRICTION JONSWAP CONSTANT 0.038", "FRICTION JONSWAP CONSTANT 0.019")

folders = [
    copy_case(source, CASES / f"{variant}_fric" / source.name, [FRICTION])
    for variant in ("whacs_map", "era5cal")
    for source in sorted(p for p in (CASES / variant).iterdir() if p.is_dir())
]
(ROOT / "case_dirs_5.txt").write_text("\n".join(str(f) for f in folders) + "\n")
script = (ROOT / "swan_array_2.sh").read_text()
for old, new in (
    ("swan_test2", "swan_test5"),
    ("slurm_logs_2", "slurm_logs_5"),
    ("case_dirs_2", "case_dirs_5"),
    ("1-205%96", f"1-{len(folders)}%48"),
):
    script = script.replace(old, new)
(ROOT / "swan_array_5.sh").write_text(script)
(ROOT / "slurm_logs_5").mkdir()
print(len(folders), "cases")
