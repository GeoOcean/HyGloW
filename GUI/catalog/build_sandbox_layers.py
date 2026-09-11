"""One-off: derive the small(ish), static reference-layer files used by the
GUI's sandbox-mode world view from the full QGIS/HyWorld source data, which
is far too heavy to ship into a browser map as-is:

- world_coastline.json     <- QGIS/HyWorld/land/i/GSHHS_i_L1.shp   (~7.3MB)
- depth200_world.json      <- QGIS/HyWorld/miscellaneous/200mDepthAll.gpkg (66MB)
- coastal_buffer_fill.json <- QGIS/HyWorld/miscellaneous/50kmdistancepolygon.gpkg (3.6MB)
- coastal_buffer_line.json <- QGIS/HyWorld/miscellaneous/50kmOr200mLine.gpkg (28MB)

The depth200/coastal_buffer_* outputs keep every feature (no
area/length-based dropping) at a light simplify tolerance — as close to the
source resolution as is still reasonable to ship as one static file, since
they're opt-in toggle layers rather than always-on.

WHACS hindcast coverage (QGIS/HyWorld/hindcasts/whacs_coords.gpkg, 2.5M
points) isn't built here: rendering it as real points is too heavy, and any
polygon/decimation stand-in misrepresents what's actually a point dataset —
so it's left out entirely for now rather than shown as something it isn't.

Not part of the app's runtime — run manually and commit the outputs. Needs
shapely + geopandas (not GUI runtime dependencies, so install them
separately to re-run).
"""

import json
from pathlib import Path

import geopandas as gpd
import shapefile
import shapely.geometry as sg

REPO_ROOT = Path(__file__).resolve().parents[2]
QGIS_DIR = REPO_ROOT / "QGIS" / "HyWorld"
OUT_DIR = Path(__file__).resolve().parent


def _write(name: str, data) -> None:
    path = OUT_DIR / name
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"{name}: {path.stat().st_size / 1024:.1f} KB")


def build_coastline() -> None:
    """Intermediate-res official GSHHS land polygons -> ring list.

    "i" (intermediate) rather than "l" (low) or "f" (full) — a visibly
    crisper coastline than low-res while still staying under ~1MB simplified.
    """
    r = shapefile.Reader(str(QGIS_DIR / "land" / "i" / "GSHHS_i_L1.shp"))
    rings = []
    for shape, rec in zip(r.iterShapes(), r.iterRecords(), strict=True):
        if rec["area"] <= 500:  # drops tiny islets, keeps ~99.3% of land area
            continue
        poly = sg.shape(shape.__geo_interface__)
        polys = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
        for p in polys:
            simplified = p.simplify(0.02, preserve_topology=True)
            if simplified.is_empty:
                continue
            ext = list(simplified.exterior.coords)
            if len(ext) >= 4:
                rings.append([[round(x, 3), round(y, 3)] for x, y in ext])
    _write("world_coastline.json", rings)


def build_depth200_world() -> None:
    """Global 200m isobath (per-case isobaths already do the same, just for one case).

    Every one of the 32k source features is kept — this is an opt-in toggle
    layer, not always-on, so it can afford to be the highest-resolution one.
    """
    gdf = gpd.read_file(QGIS_DIR / "miscellaneous" / "200mDepthAll.gpkg")
    lines = []
    for geom in gdf.geometry:
        simplified = geom.simplify(0.01, preserve_topology=True)
        coords = [[round(x, 3), round(y, 3)] for x, y in simplified.coords]
        if len(coords) >= 2:
            lines.append(coords)
    _write("depth200_world.json", lines)


def build_coastal_buffer_fill() -> None:
    """50km-from-coast buffer polygon (single global multipolygon) -> ring list.

    Every one of the 324 source parts is kept, see build_depth200_world.
    """
    gdf = gpd.read_file(QGIS_DIR / "miscellaneous" / "50kmdistancepolygon.gpkg")
    geom = gdf.geometry.iloc[0]
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    rings = []
    for p in polys:
        simplified = p.simplify(0.01, preserve_topology=True)
        if simplified.is_empty:
            continue
        ext = list(simplified.exterior.coords)
        if len(ext) >= 4:
            rings.append([[round(x, 3), round(y, 3)] for x, y in ext])
    _write("coastal_buffer_fill.json", rings)


def build_coastal_buffer_line() -> None:
    """50km-or-200m boundary (single global multiline) -> line list.

    Every one of the 10689 source parts is kept, see build_depth200_world.
    """
    gdf = gpd.read_file(QGIS_DIR / "miscellaneous" / "50kmOr200mLine.gpkg")
    geom = gdf.geometry.iloc[0]
    lines_in = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
    lines = []
    for line in lines_in:
        simplified = line.simplify(0.01, preserve_topology=True)
        coords = [[round(x, 3), round(y, 3)] for x, y in simplified.coords]
        if len(coords) >= 2:
            lines.append(coords)
    _write("coastal_buffer_line.json", lines)


if __name__ == "__main__":
    build_coastline()
    build_depth200_world()
    build_coastal_buffer_fill()
    build_coastal_buffer_line()
