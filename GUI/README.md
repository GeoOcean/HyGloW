# HyGloW Results Explorer

Lightweight Dash portal for HyGloW use cases. Reads each case's standard files:

- `{Case}/{Case}.json` — config, offshore forcing, metamodel settings
- `{Case}/inputs/Goals/{Case}_Goals.json` — goal locations (via `simulations.goals_file`)
- `{Case}/inputs/Observations/{Case}_observations.json` — buoy GeoJSON (via `onshore_waves`)
- `{Case}/outputs/goal_*/validation_metrics.csv` — **Goals Validation** tab (written by `val.sweep()`, notebook 06)
- `{Case}/outputs/metamodel_validation/*.csv` — **Metamodel Validation** tab (notebook 09's save-metrics cell)
- `{Case}/outputs/buoy_validation/*_summary_*.csv` + `.../figures/<site>.png` — **Buoy Validation** tab (notebooks 10a/10b's save cells)
- `{Case}/outputs/gui_figures/*.png` — **Final Plots** tab (a case's own `12_Final_Plots.ipynb`)
- `{Case}/book/_build/html/` — rendered Jupyter Book report (optional; see below)

Portal metadata (authors, institution, pipeline status, **continent** for the
"Browse cases" picker below) lives in `catalog/cases.yaml`. A case can also pin
`metamodel.mda_candidate` there (an int) — the fixed MDA train size the three
validation tabs read; without it, the GUI falls back to the largest MDA
candidate the case has actually fitted.

## Run locally

```bash
cd GUI
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050

The app is **map-first**: a full-screen basemap with floating panels (case
metadata, use-case picker, legend, validation drawer) inspired by
competition-style registries such as
[ShoreShop 3](https://shoreshop3.netlify.app/data). A Map/Satellite toggle
(top-center of the map) switches the basemap between OpenStreetMap and Esri
World Imagery — the same tokenless raster source the case-plot notebooks use.
Buoys render as star markers; click one to see its validation summary PNG
(density scatter + time series), if the case has generated one.

## Landing: a world map, not a case

The app opens on a one-time **welcome modal** — a short intro plus a note
that HyGloW's metamodel is built on
[BlueMath](https://github.com/GeoOcean/BlueMath_tk), the GeoOcean group's
core toolkit. Closing it (either "🧭 Start discovering" or the X) reveals
the actual landing page underneath: a world map, centered and zoomed to
show it whole, with every catalog case's **real AoI** drawn and labeled —
this is the default state, not something you opt into. From there:

- **Click a case's area on the map** to open it directly — the click target
  is a labeled marker at that AoI's centroid, styled the same
  sky-blue as the AoI shown once a case is loaded, so "this is an AoI" reads
  consistently everywhere in the app.
- **🗺️ Browse cases** (header button) opens a separate modal to pick by
  continent (from each case's `continent` field in `catalog/cases.yaml`)
  then by name — useful once the catalog outgrows "just click the map".
- **‹ Back to explore** (top of a loaded case's sidebar) returns to the
  landing view.

Case-specific panels (sidebar, dropdown, legend, results drawer) stay hidden
until a case is actually picked, whichever way you picked it.

## The landing map is also the sandbox

Alongside the case AoIs, the same world view is where you sketch a new area
of interest before it's a real case:

- **Draw** — the map's toolbar (top right) adds a rectangle and a freeform-
  path tool; pan is the default drag mode so panning around doesn't
  accidentally start a shape. Drawing a new shape always replaces whatever
  was drawn before (rect and freeform can otherwise pile up with no way to
  tell which one export would use), so there's only ever one AoI on the map.
- **Export** — name it and click **Export as shapefile** to download a
  `<name>_AoI.zip` containing `.shp/.shx/.dbf/.prj` in WGS84 — the same
  format and CRS every case's own `aoi_file` already uses. Unzip it into a
  new case's `inputs/AoI_Polygons/` to bootstrap that case's config.
- **Reference layers** — a checklist toggles use-case AoIs, continent
  blocks, coastline, the global 200m isobath, and the 50km coastal buffer
  (fill and line) on and off without disturbing a shape already drawn. The
  continent/coastline/isobath/buffer layers are pre-simplified static files
  — see "Sandbox reference-layer data" below for where they come from and
  how to regenerate them; the case AoIs are read live from each case's own
  `aoi_file` (`loaders.all_case_aois`).

The drawn shape lives in *paper* (screen-fraction) coordinates from Plotly's
draw tools, not lon/lat, since a screen rectangle isn't a rectangle in a map
projection — `assets/sandbox.js` converts it using the underlying MapLibre
instance's own `unproject()`, which already accounts for the current pan/zoom.

## Results tabs

Four tabs in the bottom drawer, each reading a different contract file (see
above) and independent of goal/MDA selection in the UI — there's no goal
picker; the tabs already aggregate across every goal or drill into one buoy:

- **Goals Validation** — boxplot of holdout RMSE across every goal, faceted
  by variable, one box per isobath contour band.
- **Metamodel Validation** — bar chart of the same shape, but for the
  domain-wide linear-summation-vs-dynamic-SnapWave comparison (one
  reconstructed period, not per-goal).
- **Buoy Validation** — the same boxplot style again, one panel each for
  offshore and onshore buoys, boxes colored by model (bulk / spectral
  metamodel / closest-goal offshore hindcast).
- **Final Plots** — clickable thumbnails of the case's own
  `12_Final_Plots.ipynb` figures (domain map with the mesh, a
  dynamic-vs-metamodel reconstruction comparison, LHS training cases, and
  the reconstruction spectra network); clicking one opens it full-size in a
  modal (mirrors the buoy-marker-click convention). Degrades per-figure
  (`loaders.available_gui_figures`) for a case that hasn't run that
  notebook, or ran an older version with fewer figures.

Chart and map colors (contour bands, buoy/goal markers, per-model colors,
the dark Plotly theme) all come from one file, `hywaves_gui/colors.py` — edit
there to restyle, rather than hunting through `map_figure.py`/`callbacks.py`.

## Deploy

```bash
cd GUI
gunicorn app:server -b 0.0.0.0:8050
```

Deploy the full HyGloW repo so case paths resolve, or copy lightweight exports later.

## Add a case

1. Add an entry to `catalog/cases.yaml` with `case_dir`, `continent` (groups
   it in the "Browse cases" picker — reuses an existing continent's group if
   one matches, otherwise starts a new one) and provenance fields, plus
   `metamodel.mda_candidate` once you've settled on a production MDA size.
2. Ensure the case follows the standard JSON / goals / observations layout
   (including `paths.aoi_file` — that's what draws its AoI on the landing
   map).
3. Run notebook 06 (`val.sweep()`) for every goal, then 09 and 10a/10b with
   their save cells, so the three tabs above have data.
4. Restart the app.

A sandbox-drawn AoI shapefile (see above) is a reasonable starting point for
a brand-new case's `inputs/AoI_Polygons/` before working through the rest of
the pipeline notebooks.

## Sandbox reference-layer data

The sandbox's coastline/isobath/buffer layers are static files under
`catalog/` (`world_coastline.json`, `depth200_world.json`,
`coastal_buffer_fill.json`, `coastal_buffer_line.json`), pre-simplified
offline by `catalog/build_sandbox_layers.py` from the full-resolution source
data in `QGIS/HyWorld/` — which is 10s-100s of MB and far too heavy to load
into a browser map directly. The continent blocks aren't derived from data
at all; they're hand-drawn boxes in `hywaves_gui/world_regions.py`, deliberately
rough rather than tracing real coastlines.

That script needs `shapely` and `geopandas`, which aren't GUI runtime
dependencies — install them separately, then:

```bash
cd GUI
python catalog/build_sandbox_layers.py
```

Rerun it if you want different simplification tolerances, or after the
source data in `QGIS/HyWorld/` changes. The outputs are small enough
(a few hundred KB to ~4MB) to commit directly.

WHACS hindcast coverage isn't offered as a sandbox layer — a script comment
in `build_sandbox_layers.py` explains why (rendering the real 2.5M points is
too heavy, and any decimated/polygon stand-in misrepresents a point dataset
as something it isn't).

## Report (Jupyter Book)

The header's **📖 Report** link opens a rendered [Jupyter Book](https://jupyterbook.org)
built from a case's own numbered notebooks (see `{Case}/book/README.md`, e.g.
`NewSouthWales/book/`) — a narrated, sequential write-up alongside the tabs
above. It's a static site the GUI just serves; the link is disabled
automatically for any case without a `book/_build/html/index.html`. It does
**not** rebuild on its own — rerun the case's `book` build after notebooks
change (see that case's `book/README.md`).
