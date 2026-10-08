"""``hywaves publish``: export a case's portal bundle to ``{Case}/portal/``.

The bundle is the only thing the portal (GUI) reads — see CONTRACT.md §6.
It holds small, self-describing files: the contract files copied under fixed
names, the heavy map layers (depth contours from the topobathy raster,
output sites from the metamodel NetCDF, the AoI shapefile) pre-computed into
light JSON, and web-sized figures. A deployment then needs the bundle alone,
never the raw inputs or ``outputs/``.

Workflow::

    python -m hywaves validate MyCase/ --publish   # optional: publish runs it too
    python -m hywaves publish MyCase/              # → MyCase/portal/
    git add MyCase/portal                          # book/ is git-ignored

Only the stdlib is needed to build a bare bundle. The pre-computed layers
use the case environment's scientific stack (``pyshp`` for the AoI,
``rioxarray`` + ``contourpy`` for depth contours, ``xarray`` for output
sites, ``Pillow`` for downscaling figures), each imported lazily — a missing
package or source file skips just that layer, with the reason recorded in
the manifest, rather than failing the whole publish.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import __version__
from .contract.validate import ERROR, find_case_config, validate_case

PORTAL_DIRNAME = "portal"
MANIFEST_NAME = "manifest.json"
# Manifest format version — bump when its keys change shape.
MANIFEST_SCHEMA = 1

# Longest side, in pixels, of any published PNG. The GUI shows figures at
# most ~1140 CSS px wide; 2400 keeps them sharp on 2x displays while
# cutting 3000-3600 px notebook exports to roughly half their weight.
MAX_IMAGE_PX = 2400

# Display-only resampling for isobath lines: independent of the case config's
# per-isobath `spacing_m` (that one spaces discrete SnapWave output sites along
# the contour; this one just keeps the drawn line light).
ISOBATH_DISPLAY_SPACING_DEG = 0.02

# ~1 m at the equator — far below anything visible on the map.
COORD_DECIMALS = 5

_RASTER_X_NAMES = ("x", "lon", "longitude")
_RASTER_Y_NAMES = ("y", "lat", "latitude")


class PublishError(RuntimeError):
    """Raised when a case cannot be published (e.g. it fails ``validate --publish``)."""


class LayerSkipped(Exception):
    """A single optional layer can't be built — recorded in the manifest, not fatal."""


@dataclass
class PublishResult:
    """What :func:`publish_case` produced."""

    portal: Path
    files: dict[str, dict] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    book: bool = False

    @property
    def size_bytes(self) -> int:
        return sum(f["size"] for f in self.files.values())


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of *path*, streamed (inputs can be hundreds of MB)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round(value: float) -> float:
    return round(float(value), COORD_DECIMALS)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


class _Bundle:
    """Staging-dir writer that records every source file the bundle depends on."""

    def __init__(self, case_dir: Path, staging: Path):
        self.case_dir = case_dir
        self.staging = staging
        self.sources: dict[str, dict] = {}
        self.source_patterns: list[str] = []

    def source(self, path: Path) -> Path:
        """Record *path* as a bundle input (size, mtime, sha256) and return it."""
        stat = path.stat()
        try:
            key = path.resolve().relative_to(self.case_dir).as_posix()
        except ValueError:
            key = str(path.resolve())  # declared outside the case folder
        self.sources[key] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }
        return path

    def glob(self, pattern: str) -> list[Path]:
        """Case-relative glob, remembered so ``validate`` can spot new/removed files."""
        self.source_patterns.append(pattern)
        return sorted(self.case_dir.glob(pattern))

    def copy(self, src: Path, rel: str) -> None:
        dest = self.staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source(src), dest)

    def case_path(self, rel: str) -> Path:
        """Resolve a config-declared path (relative to the case, or absolute)."""
        return self.case_dir / rel


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------


def _publish_aoi(bundle: _Bundle, config: dict) -> None:
    """AoI shapefile → ``aoi.geojson`` (a FeatureCollection, one feature per shape)."""
    rel = config.get("paths", {}).get("aoi_file")
    if not isinstance(rel, str) or not rel:
        raise LayerSkipped("paths.aoi_file is not declared")
    path = bundle.case_path(rel)
    if not path.is_file():
        raise LayerSkipped(f"AoI file not found: {rel}")
    try:
        import shapefile
    except ImportError:
        raise LayerSkipped("pyshp is not installed") from None

    bundle.source(path)
    for sidecar in (".shx", ".dbf", ".prj"):
        if (side := path.with_suffix(sidecar)).is_file():
            bundle.source(side)

    def _round_coords(coords):
        if isinstance(coords[0], (int, float)):
            return [_round(c) for c in coords[:2]]
        return [_round_coords(c) for c in coords]

    features = []
    for shape in shapefile.Reader(str(path)).shapes():
        geometry = dict(shape.__geo_interface__)
        if geometry.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        geometry["coordinates"] = _round_coords(geometry["coordinates"])
        features.append({"type": "Feature", "geometry": geometry, "properties": {}})
    if not features:
        raise LayerSkipped(f"no polygon shapes in {rel}")
    _write_json(
        bundle.staging / "aoi.geojson", {"type": "FeatureCollection", "features": features}
    )


def _resample_polyline(points, spacing: float):
    """Resample an (N, 2) polyline at approximately uniform arc length."""
    import numpy as np

    pts = np.asarray(points, dtype=float)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total <= 0.0:
        return pts
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    ds = np.arange(0.0, total, float(spacing))
    if ds.size == 0 or (total - ds[-1]) > 0.25 * spacing:
        ds = np.append(ds, total)
    out = np.empty((ds.size, 2), dtype=float)
    for k, d in enumerate(ds):
        i = int(np.searchsorted(cum, d, side="right") - 1)
        i = max(0, min(i, len(seg_len) - 1))
        t = 0.0 if seg_len[i] == 0 else (d - cum[i]) / seg_len[i]
        out[k] = pts[i] * (1.0 - t) + pts[i + 1] * t
    return out


def _publish_isobaths(bundle: _Bundle, config: dict) -> None:
    """Depth contours from the topobathy raster → ``isobaths.json``.

    One entry per ``simulations.output_sites_config.isobaths`` spec:
    ``{"name", "level", "segments"}`` where ``segments`` is a list of
    ``[[lon, lat], ...]`` lines (a level can have several disjoint contours,
    e.g. one per island).

    A spec's ``smooth_m`` / ``min_length_m`` are applied as for the output
    sites (via ``bluemath_tk.waves.hywaves.output_points``), so the drawn lines
    are the contours the sites sit on.
    """
    rel = config.get("paths", {}).get("topobathy_file")
    specs = config.get("simulations", {}).get("output_sites_config", {}).get("isobaths") or []
    if not specs:
        raise LayerSkipped("no simulations.output_sites_config.isobaths declared")
    if not isinstance(rel, str) or not rel:
        raise LayerSkipped("paths.topobathy_file is not declared")
    path = bundle.case_path(rel)
    if not path.is_file():
        raise LayerSkipped(f"topobathy raster not found: {rel}")
    import contourpy
    import numpy as np
    import rioxarray
    from bluemath_tk.waves.hywaves.output_points import polyline_length_m, smooth_raster

    bundle.source(path)
    da = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    xname = next((n for n in _RASTER_X_NAMES if n in da.coords), None)
    yname = next((n for n in _RASTER_Y_NAMES if n in da.coords), None)
    if xname is None or yname is None:
        raise LayerSkipped(f"no x/y (or lon/lat) coordinates in {rel}")

    generators: dict = {}

    def _contours(smooth_m):
        """Contour generator of the raster, smoothed by *smooth_m* (cached)."""
        if smooth_m not in generators:
            raster = smooth_raster(da, float(smooth_m)) if smooth_m else da
            generators[smooth_m] = contourpy.contour_generator(
                x=raster[xname].values,
                y=raster[yname].values,
                z=np.asarray(raster.values, dtype=float),
                line_type=contourpy.LineType.Separate,
            )
        return generators[smooth_m]

    isobaths = []
    for spec in specs:
        level = spec.get("level")
        if level is None:
            continue
        segments = [s for s in _contours(spec.get("smooth_m")).lines(float(level)) if len(s) >= 2]
        min_length = spec.get("min_length_m")
        if min_length:
            segments = [s for s in segments if polyline_length_m(s) >= float(min_length)]
        segments = [_resample_polyline(s, ISOBATH_DISPLAY_SPACING_DEG) for s in segments]
        isobaths.append(
            {
                "name": str(spec.get("name") or f"{abs(level):g}m"),
                "level": float(level),
                "segments": [[[_round(x), _round(y)] for x, y in seg] for seg in segments],
            }
        )
    _write_json(bundle.staging / "isobaths.json", isobaths)


def _publish_output_sites(bundle: _Bundle, config: dict) -> None:
    """Isobath/mesh output-site coordinates → ``output_sites.json``.

    Site layout is shared across every goal, so one ``output_sites_*.nc``
    gives the case-wide table. Buoy sites are excluded — the observations
    GeoJSON already carries them. Written column-wise
    (``{"lon": [...], "lat": [...], "source": [...], "label": [...]}``) —
    thousands of sites, so no per-row key repetition.
    """
    rel = config.get("simulations", {}).get("metamodel_cases")
    if not isinstance(rel, str) or not rel:
        raise LayerSkipped("simulations.metamodel_cases is not declared")
    # The declared folder first; failing that, a same-named folder inside the
    # case — for when the SnapWave runs live elsewhere (e.g. an HPC scratch
    # path in the config) and only the output_sites_*.nc were copied locally.
    candidates = [bundle.case_path(rel), bundle.case_dir / Path(rel).name]
    nc_files = next(
        (found for d in candidates if d.is_dir() and (found := sorted(d.glob("output_sites_*.nc")))),
        None,
    )
    if not nc_files:
        raise LayerSkipped(
            f"no output_sites_*.nc under {rel} or {Path(rel).name}/ in the case folder"
        )
    try:
        import xarray as xr
    except ImportError:
        raise LayerSkipped("xarray is not installed") from None

    bundle.source(nc_files[0])
    with xr.open_dataset(nc_files[0]) as ds:
        required = {"site_longitude", "site_latitude", "site_source"}
        if not required.issubset(ds.coords.keys() | ds.data_vars.keys()):
            raise LayerSkipped(f"{nc_files[0].name} lacks {sorted(required)}")
        lons = ds["site_longitude"].values
        lats = ds["site_latitude"].values
        sources = ds["site_source"].values.astype(str)
        labels = (
            ds["site_label"].values.astype(str)
            if "site_label" in ds.variables
            else [""] * len(lons)
        )
    keep = [i for i, s in enumerate(sources) if s != "buoy"]
    _write_json(
        bundle.staging / "output_sites.json",
        {
            "lon": [_round(lons[i]) for i in keep],
            "lat": [_round(lats[i]) for i in keep],
            "source": [str(sources[i]) for i in keep],
            "label": [str(labels[i]) for i in keep],
        },
    )


def _publish_metrics(bundle: _Bundle) -> None:
    """Merge every ``outputs/goal_*/*/validation_metrics.csv`` into one ``validation_metrics.csv``.

    Each row already carries its ``goal`` and ``metamodel`` columns (contract
    §4), so nothing is lost. Columns are the union across files, in
    first-seen order.
    """
    files = bundle.glob("outputs/goal_*/*/validation_metrics.csv")
    if not files:
        raise LayerSkipped("no outputs/goal_*/*/validation_metrics.csv")
    fieldnames: list[str] = []
    rows: list[dict] = []
    for path in files:
        with bundle.source(path).open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames += [c for c in reader.fieldnames or [] if c not in fieldnames]
            rows.extend(reader)
    with (bundle.staging / "validation_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _publish_csvs(bundle: _Bundle, pattern: str, dest_dir: str) -> None:
    files = bundle.glob(pattern)
    if not files:
        raise LayerSkipped(f"no {pattern}")
    for path in files:
        bundle.copy(path, f"{dest_dir}/{path.name}")


def _publish_images(
    bundle: _Bundle, pattern: str, dest_dir: str, max_px: int, log: Callable[[str], None]
) -> None:
    """Copy PNGs, downscaling any whose longest side exceeds *max_px*."""
    files = bundle.glob(pattern)
    if not files:
        raise LayerSkipped(f"no {pattern}")
    try:
        from PIL import Image
    except ImportError:
        Image = None
        log("  ! Pillow is not installed — figures copied at full size")

    for path in files:
        rel = f"{dest_dir}/{path.name}"
        if Image is None:
            bundle.copy(path, rel)
            continue
        bundle.source(path)
        dest = bundle.staging / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            if max(img.size) <= max_px:
                shutil.copyfile(path, dest)
                continue
            scale = max_px / max(img.size)
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            img.resize(size, Image.LANCZOS).save(dest, optimize=True)


def _publish_book(bundle: _Bundle) -> None:
    """Built Jupyter Book (``book/_build/html``) → ``book/`` — git-ignored, deployed by hand."""
    html = bundle.case_dir / "book" / "_build" / "html"
    if not (html / "index.html").is_file():
        raise LayerSkipped("no built book (book/_build/html/index.html)")
    shutil.copytree(html, bundle.staging / "book")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def publish_case(
    case_dir: str | Path,
    *,
    book: bool = True,
    max_image_px: int = MAX_IMAGE_PX,
    log: Callable[[str], None] = print,
) -> PublishResult:
    """Validate *case_dir* in publish mode, then (re)build ``{case_dir}/portal/``.

    The bundle is built in a sibling ``portal.tmp/`` and swapped in only
    once complete, so a failed publish never leaves a half-written or stale
    ``portal/`` behind.

    Parameters
    ----------
    case_dir
        Path to the case folder.
    book
        Copy a built Jupyter Book into ``portal/book/`` (if one exists).
    max_image_px
        Longest side, in pixels, of any published PNG.
    log
        Progress sink (``print`` by default; tests pass a no-op).

    Raises
    ------
    PublishError
        If the case fails ``validate --publish``.
    """
    case_dir = Path(case_dir).resolve()
    findings = validate_case(case_dir, publish=True)
    errors = [f for f in findings if f.severity == ERROR]
    if errors:
        details = "\n".join(f"  {f}" for f in errors)
        raise PublishError(
            f"{case_dir.name} is not portal-ready ({len(errors)} error(s)):\n{details}\n"
            f"Fix these (see `python -m hywaves validate {case_dir.name} --publish`) and retry."
        )

    config_path = find_case_config(case_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    portal = case_dir / PORTAL_DIRNAME
    staging = case_dir / f"{PORTAL_DIRNAME}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    bundle = _Bundle(case_dir, staging)
    result = PublishResult(portal=portal)
    try:
        # Contract files — validated above, so these exist at publish.
        bundle.copy(config_path, "case.json")
        bundle.copy(bundle.case_path(config["simulations"]["goals_file"]), "goals.json")
        bundle.copy(bundle.case_path(config["onshore_waves"]), "observations.json")

        layers: list[tuple[str, Callable[[], None]]] = [
            ("aoi", lambda: _publish_aoi(bundle, config)),
            ("isobaths", lambda: _publish_isobaths(bundle, config)),
            ("output_sites", lambda: _publish_output_sites(bundle, config)),
            ("validation_metrics", lambda: _publish_metrics(bundle)),
            (
                "metamodel_validation",
                lambda: _publish_csvs(
                    bundle, "outputs/metamodel_validation/*.csv", "metamodel_validation"
                ),
            ),
            (
                "buoy_validation",
                lambda: _publish_csvs(
                    bundle, "outputs/buoy_validation/*_summary_*.csv", "buoy_validation"
                ),
            ),
            (
                "buoy_figures",
                lambda: _publish_images(
                    bundle, "outputs/buoy_validation/figures/*.png", "buoy_validation/figures", max_image_px, log
                ),
            ),
            (
                "gui_figures",
                lambda: _publish_images(
                    bundle, "outputs/gui_figures/*.png", "gui_figures", max_image_px, log
                ),
            ),
        ]
        if book:
            layers.append(("book", lambda: _publish_book(bundle)))

        for name, build in layers:
            try:
                build()
                log(f"  ✓ {name}")
            except LayerSkipped as exc:
                result.skipped[name] = str(exc)
                log(f"  - {name}: skipped ({exc})")
        result.book = (staging / "book").is_dir()

        # Checksums for the committed part only — the book is a deploy-time extra.
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            rel = path.relative_to(staging).as_posix()
            if rel.startswith("book/"):
                continue
            result.files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}

        manifest = {
            "schema": MANIFEST_SCHEMA,
            "hywaves_version": __version__,
            "case": config.get("name", case_dir.name),
            "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "max_image_px": max_image_px,
            "book": result.book,
            "skipped": result.skipped,
            "files": result.files,
            "sources": bundle.sources,
            "source_patterns": bundle.source_patterns,
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Swap in: move the old bundle aside first so the rename never collides.
    retired = case_dir / f"{PORTAL_DIRNAME}.old"
    if retired.exists():
        shutil.rmtree(retired)
    if portal.exists():
        os.replace(portal, retired)
    os.replace(staging, portal)
    shutil.rmtree(retired, ignore_errors=True)
    return result
