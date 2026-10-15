# HyGloW Results Explorer

A Dash portal for the HyGloW cases. It reads only each case's `portal/`
(written by `python -m hywaves publish {Case}/`), never inputs or `outputs/`,
so a deployment needs the repository and the bundles alone.

```bash
cd GUI
pip install -r requirements.txt
python app.py            # http://localhost:8050
```

The landing page is a world map with every case's area; click one to open
its maps and the validation tabs (holdout, linear summation, buoys, final
plots, report). The same map draws and exports a new area of interest as a
shapefile for `inputs/AoI_Polygons/`. Earlier versions of a case
(`portal_<tag>/`, from `python -m hywaves snapshot`) appear in the case
dropdown.

## Add a case

1. Run its notebooks 06, 09, 10 and 12 (their save cells), then
   `python -m hywaves publish {Case}/` and commit `{Case}/portal/`.
2. Add it to `catalog/cases.yaml` (`case_dir`, `continent`, authors;
   optionally `metamodel.mda_candidate`, the MDA size the tabs show, default
   the largest).
3. Restart the app.

## Deploy

```bash
cd GUI
HYGLOW_URL_BASE=/hyglow/ gunicorn app:server -b 127.0.0.1:8050
```

`HYGLOW_URL_BASE` only when served under a sub-path (the proxy forwards the
prefix unchanged, e.g. nginx `location /hyglow { proxy_pass
http://localhost:8050/hyglow; }`). Updating = publish locally, commit, pull
on the server, restart.

Reports (`portal/book/`, a Jupyter Book of the case README and notebooks
built by `publish`, needs `jupyter-book`) are git-ignored: copy them to the
server by hand, published with the same prefix (`publish --url-base
/hyglow/`).

The world reference layers in `catalog/` (coastline, 200 m isobath, coastal
buffer) are simplified from `QGIS/HyWorld/` by
`catalog/build_sandbox_layers.py` (needs shapely and geopandas).
