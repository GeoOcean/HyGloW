# NewSouthWales — Jupyter Book

Renders this case's own numbered notebooks (`../01_*.ipynb` … `../12_*.ipynb`)
into a single narrated static site — the report linked from the GUI's
**📖 Report** button.

It renders each notebook's **already-saved outputs as-is** — it does not
re-execute them (`--execute` is off by default), so it's fast and doesn't
need the SnapWave/simulation environment at build time. That also means the
book only reflects what was in each notebook the last time *you* ran and
saved it — rerun and save a notebook first if you want its latest outputs in
the report.

## Build

```bash
pip install jupyter-book   # one-time, only needed to build the report
cd NewSouthWales/book
./build.sh                               # local GUI (served at /)
HYGLOW_URL_BASE=/hyglow/ ./build.sh      # deployed GUI served under /hyglow/
```

Don't run `jupyter-book build --html` directly — use `build.sh`. It wraps
two required things `jupyter-book` doesn't do on its own:

1. **`BASE_URL`**, matching the GUI's serving route (`/case-book/<case_id>`,
   under the GUI's `HYGLOW_URL_BASE` prefix if it has one — see
   `GUI/app.py`'s `serve_book`) — baked into every internal link and
   asset path at build time. Wrong or missing, and the report loads with
   broken styling and dead nav links (every URL resolves against the site
   root instead of the actual mount path).
2. A post-build copy of the site theme's runtime JS/CSS bundle into
   `_build/html/myst_assets_folder/`. This version of jupyter-book/mystmd
   fetches that bundle into `_build/templates/.../public/build/` but never
   copies it into the exported `_build/html/` — every page references
   `/myst_assets_folder/...` paths that otherwise just don't exist in the
   static export, so the report loads completely unstyled. See the comment
   at the top of `build.sh` — if a future jupyter-book/mystmd release fixes
   this upstream, that copy step becomes a no-op you can safely delete.

Output goes to `_build/html/` (gitignored — a local/build-machine artifact,
not committed). The GUI serves the report from the case's portal bundle, so
after rebuilding run `python -m hywaves publish NewSouthWales` (it copies
`_build/html/` into `portal/book/`, also git-ignored) and refresh the GUI.
For a deployment, copy `portal/book/` to the server by hand — see
`GUI/README.md` → Deploy.

## Editing the table of contents

Notebooks and their order are listed under `project.toc` in `myst.yml`. Add,
remove, or reorder entries there (paths are relative to this `book/`
directory, e.g. `../13_New_Notebook.ipynb`).
