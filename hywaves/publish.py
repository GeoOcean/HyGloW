"""``hywaves publish``: export a case's portal bundle to ``{Case}/portal/``.

The bundle is the only thing the portal (GUI) reads.
It holds small, self-describing files: the contract files copied under fixed
names, the heavy map layers (depth contours from the topobathy raster, the
output sites of ``outputs/output_sites_{model}.csv``, the AoI shapefile)
pre-computed into light JSON, and web-sized figures. A deployment then needs
the bundle alone, never the raw inputs or ``outputs/``.

Workflow::

    python -m hywaves validate MyCase/ --publish   # optional: publish runs it too
    python -m hywaves publish MyCase/              # → MyCase/portal/
    git add MyCase/portal                          # book/ is git-ignored

The report (``portal/book/``) is a Jupyter Book of the case README and its
numbered notebooks, built by ``publish`` itself (saved outputs, nothing is
re-executed) in the case's git-ignored ``tmp/book/``: no ``book/`` folder in
the case. Its links are absolute, for the route the GUI serves it under
(``{HYGLOW_URL_BASE}/case-book/{case id}``).

Publishing runs in the case environment (``pyshp`` for the AoI,
``rioxarray`` + ``contourpy`` + bluemath_tk for depth contours, ``Pillow``
for figures). A missing *source file* skips just that layer, with the reason
recorded in the manifest; the portal shows it as "not published yet". The
manifest, and the case manifest, record the bluemath_tk revision the
published results were made with.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import __version__
from .contract.validate import ERROR, find_case_config, model_block, validate_case
from .manifest import bluemath_tk_revision, write_manifest

PORTAL_DIRNAME = "portal"
MANIFEST_NAME = "manifest.json"

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
    book_url: str | None = None

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
    import shapefile

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

    As for the output sites (``utils.geometry.compute_output_sites``), the
    raster is blanked outside the model mesh and each spec's ``smooth_m`` /
    ``min_length_m`` are applied, so the drawn lines are the contours the
    sites sit on.
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
    grid_rel = model_block(config).get("grid_file", "")
    grid = bundle.case_path(grid_rel)
    if not grid.is_file():
        raise LayerSkipped(f"model grid not found: {grid_rel}")
    import contourpy
    import numpy as np
    import rioxarray
    from bluemath_tk.topo_bathy.mesh import mask_raster_to_mesh, read_mesh
    from bluemath_tk.waves.hywaves.output_points import polyline_length_m, smooth_raster

    bundle.source(path)
    bundle.source(grid)
    da = rioxarray.open_rasterio(path).squeeze("band", drop=True)
    da = mask_raster_to_mesh(da, read_mesh(grid))

    generators: dict = {}

    def _contours(smooth_m):
        """Contour generator of the raster, smoothed by *smooth_m* (cached)."""
        if smooth_m not in generators:
            raster = smooth_raster(da, float(smooth_m)) if smooth_m else da
            generators[smooth_m] = contourpy.contour_generator(
                x=raster["x"].values,
                y=raster["y"].values,
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
    """The ``ok`` isobath/mesh sites of ``outputs/output_sites_{model}.csv`` → ``output_sites.json``.

    Buoys are left out (the observations GeoJSON already carries them).
    Written column-wise (``{"lon": [...], "lat": [...], "source": [...],
    "label": [...]}``) — thousands of sites, so no per-row key repetition.
    """
    rel = f"outputs/output_sites_{config.get('simulations', {}).get('model')}.csv"
    path = bundle.case_path(rel)
    if not path.is_file():
        raise LayerSkipped(f"{rel} not found (notebook 03)")
    with bundle.source(path).open(encoding="utf-8", newline="") as f:
        rows = [
            row
            for row in csv.DictReader(line for line in f if not line.startswith("#"))
            if row["status"] == "ok" and row["source"] != "buoy"
        ]
    _write_json(
        bundle.staging / "output_sites.json",
        {
            "lon": [_round(row["lon"]) for row in rows],
            "lat": [_round(row["lat"]) for row in rows],
            "source": [row["source"] for row in rows],
            "label": [row["source_label"] for row in rows],
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
    bundle: _Bundle, pattern: str, dest_dir: str, max_px: int
) -> None:
    """Copy PNGs, downscaling any whose longest side exceeds *max_px*."""
    from PIL import Image

    files = bundle.glob(pattern)
    if not files:
        raise LayerSkipped(f"no {pattern}")
    for path in files:
        rel = f"{dest_dir}/{path.name}"
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


#: Where ``publish`` builds the report, inside the case (git-ignored).
BOOK_BUILD_DIR = Path("tmp") / "book"


def book_url(case_dir: Path, url_base: str | None = None, case_id: str | None = None) -> str:
    """Route the GUI serves the case's report under: ``[/<url_base>]/case-book/<case id>``.

    Parameters
    ----------
    case_dir
        Case folder.
    url_base
        The GUI's URL prefix; default the ``HYGLOW_URL_BASE`` environment
        variable (unset = served at the root).
    case_id
        The case id in the GUI catalog; default the folder name in lowercase.
    """
    base = (url_base if url_base is not None else os.environ.get("HYGLOW_URL_BASE", "")).strip("/")
    cid = case_id or case_dir.name.lower()
    return f"/{base}/case-book/{cid}" if base else f"/case-book/{cid}"


def book_notebooks(case_dir: Path) -> list[Path]:
    """The case's numbered notebooks (``01_*.ipynb``, ``05b_*.ipynb``…), in order."""
    return sorted(case_dir.glob("[0-9][0-9]*_*.ipynb"))


def write_book_config(case_dir: Path, build_dir: Path, config: dict) -> Path:
    """Write the book's ``myst.yml``: the case README, then every numbered notebook."""
    files = [case_dir / "README.md"] if (case_dir / "README.md").is_file() else []
    files += book_notebooks(case_dir)
    name = config.get("name", case_dir.name)
    description = " ".join(str(config.get("description") or "").split()).replace('"', "'")
    toc = "\n".join(f"    - file: {os.path.relpath(f, build_dir)}" for f in files)
    text = (
        "# Written by `python -m hywaves publish` (see mystmd.org/guide/frontmatter)\n"
        "version: 1\n"
        "project:\n"
        f"  id: {uuid.uuid5(uuid.NAMESPACE_URL, 'hyglow/' + case_dir.name)}\n"
        f'  title: "HyGloW — {name}"\n'
        f'  description: "{description}"\n'
        "  toc:\n"
        f"{toc}\n"
        "site:\n"
        "  template: book-theme\n"
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    path = build_dir / "myst.yml"
    path.write_text(text, encoding="utf-8")
    return path


def build_book(case_dir: Path, config: dict, url: str, timeout: int = 3600) -> Path:
    """Build the case's report with ``jupyter-book`` in ``{case}/tmp/book/``; return its HTML folder.

    The notebooks' saved outputs are rendered as they are (nothing is
    re-executed). This jupyter-book/mystmd version fetches the theme's
    runtime JS/CSS into ``_build/templates/.../public/build/`` but never
    copies it into the static export, so it is copied to
    ``_build/html/myst_assets_folder/`` here.

    Raises
    ------
    LayerSkipped
        If ``jupyter-book`` is not installed or the build fails.
    """
    exe = shutil.which("jupyter-book")
    if exe is None:
        raise LayerSkipped("jupyter-book is not installed (pip install jupyter-book)")
    build_dir = case_dir / BOOK_BUILD_DIR
    write_book_config(case_dir, build_dir, config)
    html = build_dir / "_build" / "html"
    if html.exists():
        shutil.rmtree(html)
    try:
        run = subprocess.run(
            [exe, "build", "--html"], cwd=build_dir, env={**os.environ, "BASE_URL": url},
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise LayerSkipped(f"jupyter-book build took longer than {timeout} s") from None
    if run.returncode != 0 or not (html / "index.html").is_file():
        tail = (run.stderr or run.stdout).strip().splitlines()[-1:] or ["no output"]
        raise LayerSkipped(f"jupyter-book build failed: {tail[0]}")
    theme = next((build_dir / "_build" / "templates").glob("site/**/public/build"), None)
    if theme is not None:
        shutil.copytree(theme, html / "myst_assets_folder", dirs_exist_ok=True)
    return html


def _publish_book(bundle: _Bundle, config: dict, url: str) -> None:
    """The case's report (:func:`build_book`) → ``book/`` — git-ignored, deployed by hand."""
    html = build_book(bundle.case_dir, config, url)
    shutil.copytree(html, bundle.staging / "book")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def publish_case(
    case_dir: str | Path,
    *,
    book: bool = True,
    url_base: str | None = None,
    case_id: str | None = None,
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
        Build the case's report (README + numbered notebooks, saved outputs)
        into ``portal/book/`` (needs ``jupyter-book``; slow on big notebooks).
    url_base, case_id
        Route of the report in the GUI (see :func:`book_url`): the GUI's
        ``HYGLOW_URL_BASE`` (default that environment variable) and the case
        id of the GUI catalog (default the folder name in lowercase).
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
                    bundle, "outputs/buoy_validation/figures/*.png", "buoy_validation/figures", max_image_px
                ),
            ),
            (
                "gui_figures",
                lambda: _publish_images(
                    bundle, "outputs/gui_figures/*.png", "gui_figures", max_image_px
                ),
            ),
        ]
        url = book_url(case_dir, url_base, case_id)
        if book:
            layers.append(("book", lambda: _publish_book(bundle, config, url)))

        for name, build in layers:
            try:
                build()
                log(f"  ✓ {name}")
            except LayerSkipped as exc:
                result.skipped[name] = str(exc)
                log(f"  - {name}: skipped ({exc})")
        result.book = (staging / "book").is_dir()
        result.book_url = url if result.book else None

        # Checksums for the committed part only — the book is a deploy-time extra.
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            rel = path.relative_to(staging).as_posix()
            if rel.startswith("book/"):
                continue
            result.files[rel] = {"size": path.stat().st_size, "sha256": sha256_file(path)}

        revision = bluemath_tk_revision()
        manifest = {
            "hywaves_version": __version__,
            "bluemath_tk": revision,
            "case": config.get("name", case_dir.name),
            "model": config.get("simulations", {}).get("model"),
            "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "max_image_px": max_image_px,
            "book": result.book,
            "book_url": result.book_url,
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
    write_manifest(case_dir, bluemath_tk=revision)
    return result
