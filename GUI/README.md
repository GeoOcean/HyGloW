# HyGloW Results Explorer

Lightweight Dash portal for HyGloW use cases. It reads **only** each case's
published bundle, `{Case}/portal/`, written by

```bash
python -m hywaves publish {Case}/
```

(see `hywaves/contract/CONTRACT.md` §6). Never the raw inputs or `outputs/`,
so a deployment needs the bundles alone — a few MB per case:

| Bundle file (`{Case}/portal/…`)                   | Used for                                                    | Built from                                                     |
| ------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------- |
| `case.json`                                       | name, description, offshore forcing, legend isobaths        | `{Case}/{Case}.json`                                           |
| `goals.json`, `observations.json`                 | goal and buoy markers                                       | `simulations.goals_file`, `onshore_waves`                      |
| `aoi.geojson`                                     | AoI outline (case map + landing map)                        | `paths.aoi_file` shapefile                                     |
| `isobaths.json`                                   | depth-contour lines                                         | `paths.topobathy_file` raster, contoured at publish            |
| `output_sites.json`                               | isobath/mesh output-site dots                               | first `output_sites_*.nc` under `simulations.metamodel_cases` (or a same-named folder in the case) |
| `validation_metrics.csv`                          | **Goals Validation** tab                                    | every `outputs/goal_*/validation_metrics.csv` (notebook 06)    |
| `metamodel_validation/*.csv`                      | **Metamodel Validation** tab                                | notebook 09's save-metrics cell                                |
| `buoy_validation/*_summary_*.csv`, `…/figures/*.png` | **Buoy Validation** tab + buoy-click popups              | notebooks 10a/10b's save cells                                 |
| `gui_figures/*.png`                               | **Final Plots** tab                                         | the case's own `12_Final_Plots.ipynb`                          |
| `book/`                                           | **📖 Report** link (optional; git-ignored, see below)       | `book/_build/html/`                                            |
| `manifest.json`                                   | "Published" date in the case panel                          | written by `publish`                                           |

Any missing file just empties that layer/tab (a "not published yet"
message), and a case with no `portal/` at all still appears in the catalog.

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
  how to regenerate them; the case AoIs come from each case's published
  `portal/aoi.geojson` (`loaders.all_case_aois`).

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

The repo checkout is enough — `GUI/` plus each case's committed `portal/`
(no `inputs/`, `outputs/`, or model runs needed; they're git-ignored anyway).
Install `requirements.txt` (no raster/NetCDF stack) and run:

```bash
cd GUI
gunicorn app:server -b 127.0.0.1:8050
```

**Behind a reverse proxy under a sub-path** (e.g. `https://host/hyglow/`),
set `HYGLOW_URL_BASE` — every page, asset, figure and report route then
lives under it:

```bash
HYGLOW_URL_BASE=/hyglow/ gunicorn app:server -b 127.0.0.1:8050
```

with the proxy forwarding the prefix unchanged, e.g. nginx:

```nginx
location /hyglow {
    proxy_pass http://localhost:8050/hyglow;
}
```

(In a systemd unit: `Environment=HYGLOW_URL_BASE=/hyglow/`.)

**Reports** (`portal/book/`) are not in git — copy them to the server by
hand after publishing, e.g.
`rsync -a NewSouthWales/portal/book/ server:/deploys/HyGloW/NewSouthWales/portal/book/`.
A book's links are baked in at build time, so build it for the same prefix
first: `HYGLOW_URL_BASE=/hyglow/ NewSouthWales/book/build.sh`, then re-run
`python -m hywaves publish NewSouthWales`.

**Updating a case** on the server = re-run `publish` locally, commit
`{Case}/portal/`, pull on the server, restart the service.

## Add a case

1. Add an entry to `catalog/cases.yaml` with `case_dir`, `continent` (groups
   it in the "Browse cases" picker — reuses an existing continent's group if
   one matches, otherwise starts a new one) and provenance fields, plus
   `metamodel.mda_candidate` once you've settled on a production MDA size.
2. Ensure the case follows the standard JSON / goals / observations layout
   (including `paths.aoi_file` — that's what draws its AoI on the landing
   map).
3. Run notebook 06 (`val.sweep()`) for every goal, then 09, 10a/10b and 12
   with their save cells, so the tabs above have data.
4. `python -m hywaves publish {Case}/` — validates the case, then writes
   `{Case}/portal/`. Commit that folder (its `book/` is git-ignored).
   `python -m hywaves validate {Case}/` warns whenever the bundle has fallen
   behind its sources.
5. Restart the app.

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
above. It's a static site the GUI just serves, from the bundle's
`portal/book/` (copied there by `publish`); the link is disabled
automatically for any case without a `portal/book/index.html`. It does
**not** rebuild on its own — rerun the case's `book` build after notebooks
change (see that case's `book/README.md`), then `publish` again. The book is
git-ignored (tens of MB of static site): copy it to a deployment by hand, as
described under **Deploy**.
