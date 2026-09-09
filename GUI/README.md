# HyGloW Results Explorer

Lightweight Dash portal for HyGloW use cases. Reads each case's standard files:

- `{Case}/{Case}.json` — config, offshore forcing, metamodel settings
- `{Case}/inputs/Goals/{Case}_Goals.json` — goal locations (via `simulations.goals_file`)
- `{Case}/inputs/Observations/{Case}_observations.json` — buoy GeoJSON (via `onshore_waves`)
- `{Case}/outputs/goal_*/validation_metrics.csv` — **Goals Validation** tab (written by `val.sweep()`, notebook 06)
- `{Case}/outputs/metamodel_validation/*.csv` — **Metamodel Validation** tab (notebook 09's save-metrics cell)
- `{Case}/outputs/buoy_validation/*_summary_*.csv` + `.../figures/<site>.png` — **Buoy Validation** tab (notebooks 10a/10b's save cells)
- `{Case}/book/_build/html/` — rendered Jupyter Book report (optional; see below)

Portal metadata (authors, institution, pipeline status) lives in `catalog/cases.yaml`.
A case can also pin `metamodel.mda_candidate` there (an int) — the fixed MDA
train size all three validation tabs read; without it, the GUI falls back to
the largest MDA candidate the case has actually fitted.

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

## Results tabs

Three tabs in the bottom drawer, each reading a different contract file (see
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

1. Add an entry to `catalog/cases.yaml` with `case_dir` and provenance fields,
   plus `metamodel.mda_candidate` once you've settled on a production MDA size.
2. Ensure the case follows the standard JSON / goals / observations layout.
3. Run notebook 06 (`val.sweep()`) for every goal, then 09 and 10a/10b with
   their save cells, so the three tabs above have data.
4. Restart the app.

## Report (Jupyter Book)

The header's **📖 Report** link opens a rendered [Jupyter Book](https://jupyterbook.org)
built from a case's own numbered notebooks (see `{Case}/book/README.md`, e.g.
`NewSouthWales/book/`) — a narrated, sequential write-up alongside the tabs
above. It's a static site the GUI just serves; the link is disabled
automatically for any case without a `book/_build/html/index.html`. It does
**not** rebuild on its own — rerun the case's `book` build after notebooks
change (see that case's `book/README.md`).
