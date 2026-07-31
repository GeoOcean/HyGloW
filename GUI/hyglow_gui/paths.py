from __future__ import annotations

from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GUI_DIR.parent
CATALOG_PATH = GUI_DIR / "catalog" / "cases.yaml"


def case_root(case: dict) -> Path:
    return REPO_ROOT / case["case_dir"]


def resolve_case(case: dict, rel: str | Path) -> Path:
    return case_root(case) / rel
