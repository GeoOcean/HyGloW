"""Second SnapWave round on the 39 hours, on top of enc_upw: friction and no wind.
Run from SWAN_test/snapwave/: python build_2.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(".").resolve()
SOURCE = ROOT / "dyn" / "enc_upw"
FRICTION = {"enc_upw_fw02": 0.02, "enc_upw_fw04": 0.04}


def set_key(text: str, key: str, value: object) -> str:
    line = f"{key:<15}= {value}"
    text, n = re.subn(rf"^{key}\s*=.*$", line, text, flags=re.M)
    return text if n else f"{line}\n{text}"


def copy_hour(hour: Path, variant: str) -> Path:
    folder = ROOT / "dyn" / variant / hour.name
    shutil.copytree(hour, folder, ignore=shutil.ignore_patterns("*.nc", "*.log", "*.upw"))
    return folder


def main() -> None:
    cases = []
    for hour in sorted(p for p in SOURCE.iterdir() if p.is_dir()):
        for variant, fw in FRICTION.items():
            folder = copy_hour(hour, variant)
            inp = folder / "snapwave.inp"
            inp.write_text(set_key(inp.read_text(), "fw", fw))
            cases.append(folder)
        folder = copy_hour(hour, "enc_upw_nowind")
        inp = folder / "snapwave.inp"
        text = re.sub(r"^(u10|u10dir)\s*=.*\n", "", inp.read_text(), flags=re.M)
        inp.write_text(set_key(set_key(text, "sector", 180), "niter", 200))
        (folder / "u10.txt").unlink()
        (folder / "u10dir.txt").unlink()
        cases.append(folder)
    (ROOT / "case_dirs_2.txt").write_text("\n".join(str(f) for f in cases) + "\n")
    print(len(cases))


if __name__ == "__main__":
    main()
