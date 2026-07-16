# HyGloW Results Explorer

Lightweight Dash portal for HyGloW use cases. Reads each case's standard files:

- `{Case}/{Case}.json` — config, offshore forcing, metamodel settings
- `{Case}/inputs/Goals/{Case}_Goals.json` — goal locations (via `simulations.goals_file`)
- `{Case}/inputs/Observations/{Case}_observations.json` — buoy GeoJSON (via `onshore_waves`)
- `{Case}/outputs/goal_*/validation_metrics.csv` — holdout validation
- `{Case}/outputs/course_figures/fig*.png` — showcase figures

Portal metadata (authors, institution, pipeline status) lives in `catalog/cases.yaml`.

## Run locally

```bash
cd GUI
pip install -r requirements.txt
python app.py
```

Open http://localhost:8050

The app is **map-first**: a full-screen OpenStreetMap basemap with floating panels
(case metadata, filters, validation drawer) inspired by competition-style registries
such as [ShoreShop 3](https://shoreshop3.netlify.app/data). Click goals on the map
to inspect them.

## Deploy

```bash
cd GUI
gunicorn app:server -b 0.0.0.0:8050
```

Deploy the full HyGloW repo so case paths resolve, or copy lightweight exports later.

## Add a case

1. Add an entry to `catalog/cases.yaml` with `case_dir` and provenance fields.
2. Ensure the case follows the standard JSON / goals / observations layout.
3. Run validation and course-figure notebooks so outputs exist.
4. Restart the app.
