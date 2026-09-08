"""Case configuration loader — one instance per case folder.

Unlike the vendored ``utils.config`` module inside each case (which binds to
its own folder at import time for notebook convenience), :class:`CaseConfig`
is instantiable for *any* case directory. It is what the GUI, the validator,
and multi-case tooling use::

    from hywaves.config import CaseConfig

    case = CaseConfig("Samoa")
    goals = case.load_goals()
    zarr = case.open_offshore_waves("WHACS")   # lazy xarray import
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CaseConfigError(RuntimeError):
    """Raised when a case folder or its config JSON is unusable."""


def find_case_config(case_dir: Path) -> Path:
    """Locate the case config JSON inside *case_dir*.

    Convention: ``{Case}/{Case}.json`` (case-insensitive fallback). If exactly
    one other top-level JSON exists that is not a goals/observations file, it
    is accepted as a fallback.
    """
    name = case_dir.name
    for candidate in (case_dir / f"{name}.json", case_dir / f"{name.lower()}.json"):
        if candidate.is_file():
            return candidate
    fallbacks = sorted(
        p
        for p in case_dir.glob("*.json")
        if "goals" not in p.name.lower() and "observations" not in p.name.lower()
    )
    if len(fallbacks) == 1:
        return fallbacks[0]
    raise CaseConfigError(
        f"No case config JSON found under {case_dir} "
        f"(expected {name}.json; found {len(fallbacks)} candidates)"
    )


class CaseConfig:
    """Configuration for one HyGloW case folder."""

    def __init__(self, case_dir: str | Path):
        """Load the config JSON for *case_dir* immediately (see :meth:`reload`).

        Raises
        ------
        CaseConfigError
            If *case_dir* does not exist or no config JSON can be found.
        """
        self.case_dir = Path(case_dir).resolve()
        if not self.case_dir.is_dir():
            raise CaseConfigError(f"Case directory does not exist: {self.case_dir}")
        self.config_path = find_case_config(self.case_dir)
        self.data: dict[str, Any] = {}
        self.reload()

    def reload(self) -> dict[str, Any]:
        """Re-read the config JSON from disk."""
        with self.config_path.open(encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Display name from the config JSON, falling back to the folder name."""
        return str(self.data.get("name") or self.case_dir.name)

    @property
    def description(self) -> str:
        """Long description from the config JSON (``""`` if unset)."""
        return str(self.data.get("description") or "")

    # ------------------------------------------------------------------
    # Sections
    # ------------------------------------------------------------------

    @property
    def paths(self) -> dict[str, Any]:
        """The config JSON's ``paths`` section (``{}`` if unset)."""
        return self.data.get("paths", {})

    @property
    def onshore_waves(self) -> str | None:
        """Relative path to the observations GeoJSON, or ``None`` if unset."""
        return self.data.get("onshore_waves")

    @property
    def offshore_waves(self) -> list[dict[str, Any]]:
        """The config JSON's ``offshore_waves`` catalog, always as a list."""
        block = self.data.get("offshore_waves", [])
        return [block] if isinstance(block, dict) else list(block)

    @property
    def simulations(self) -> dict[str, Any]:
        """The config JSON's ``simulations`` section (``{}`` if unset)."""
        return self.data.get("simulations", {})

    @property
    def output_sites_config(self) -> dict[str, Any]:
        """The config JSON's ``simulations.output_sites_config`` section (``{}`` if unset)."""
        return self.simulations.get("output_sites_config", {})

    @property
    def metamodel(self) -> dict[str, Any]:
        """The config JSON's ``metamodel`` section (``{}`` if unset)."""
        return self.data.get("metamodel", {})

    # ------------------------------------------------------------------
    # Paths and files
    # ------------------------------------------------------------------

    def resolve(self, path: str | Path) -> Path:
        """Absolute path under the case directory (absolute paths pass through)."""
        p = Path(path)
        return p if p.is_absolute() else self.case_dir / p

    @property
    def outputs_dir(self) -> Path:
        """The case's ``outputs/`` directory."""
        return self.case_dir / "outputs"

    def goal_metrics_files(self) -> list[Path]:
        """All ``outputs/goal_*/validation_metrics.csv`` files."""
        return sorted(self.outputs_dir.glob("goal_*/validation_metrics.csv"))

    def figure_files(self) -> list[Path]:
        """Showcase figures published to the GUI portal."""
        return sorted((self.outputs_dir / "gui_figures").glob("fig*.png"))

    # ------------------------------------------------------------------
    # Goals and observations
    # ------------------------------------------------------------------

    def load_goals(self, path: str | Path | None = None) -> dict[str, dict[str, Any]]:
        """Goals mapping from ``simulations.goals_file`` (or *path* override)."""
        goals_file = path or self.simulations.get("goals_file")
        if not goals_file:
            raise CaseConfigError(f"{self.name}: simulations.goals_file is not set")
        with self.resolve(goals_file).open(encoding="utf-8") as f:
            return json.load(f)["goals"]

    def goals_for_snapwave(
        self, goals: dict[str, dict[str, Any]] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Slim ``{id: {lon, lat, angles}}`` mapping for SnapWave wrappers."""
        goals = goals or self.load_goals()
        return {
            key: {
                "lon": goal["target"]["longitude"],
                "lat": goal["target"]["latitude"],
                "angles": goal["angles"],
            }
            for key, goal in goals.items()
        }

    def load_observations(self) -> dict[str, Any]:
        """Raw observations GeoJSON FeatureCollection."""
        if not self.onshore_waves:
            raise CaseConfigError(f"{self.name}: onshore_waves is not set")
        with self.resolve(self.onshore_waves).open(encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Offshore hindcasts
    # ------------------------------------------------------------------

    def get_offshore_wave(self, name: str | None = None) -> dict[str, Any]:
        """One ``offshore_waves`` entry by (case-insensitive) name."""
        entries = self.offshore_waves
        if not entries:
            raise CaseConfigError(f"{self.name}: offshore_waves is empty")
        if name is None:
            return entries[0]
        key = name.casefold()
        for entry in entries:
            if str(entry.get("name", "")).casefold() == key:
                return entry
        names = [e.get("name") for e in entries]
        raise KeyError(f"Unknown offshore wave {name!r}; available: {names}")

    def open_offshore_waves(self, name: str | None = None):
        """Open the processed offshore zarr as a lazy xarray Dataset."""
        import xarray as xr  # deferred: keep CaseConfig importable without xarray

        entry = self.get_offshore_wave(name)
        path = entry.get("data_path")
        if not path:
            raise CaseConfigError(
                f"offshore wave {entry.get('name')!r} has no data_path"
            )
        return xr.open_zarr(self.resolve(path))

    def __repr__(self) -> str:  # pragma: no cover
        return f"CaseConfig({self.name!r}, {self.case_dir})"
