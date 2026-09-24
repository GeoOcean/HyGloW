from __future__ import annotations

from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GUI_DIR.parent
CATALOG_PATH = GUI_DIR / "catalog" / "cases.yaml"


def case_root(case: dict) -> Path:
    return REPO_ROOT / case["case_dir"]


def portal_root(case: dict) -> Path:
    """The case's published bundle (``python -m hywaves publish``) — the only thing the GUI reads."""
    return case_root(case) / "portal"
