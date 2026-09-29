# HyGloW presentation

A self-contained HTML deck introducing HyGloW (the open project and portal) and
HyWaves (the hybrid downscaling method) for a general audience. It uses
the same structure and look as the DEME workshop deck. No build step and no
server: open `index.html` in a browser.

## Presenting

- Arrow keys / space / on-screen buttons to navigate, `f` fullscreen, click a
  figure to zoom, top-right button toggles light/dark.
- **`s` (or the "Full/Short deck" button) switches between the full deck
  (21 slides) and a short one (17)**. The short deck drops the slides marked
  `data-depth="deep"` (goals & sampling, spectral partitions, holdout
  convergence, cost). Use short for non-technical or time-boxed audiences.
- Per-audience text (title, subtitle, presenter, venue, default mode) lives in
  `data/config.js`. Edit it by hand.
- **`mainCase`** in `config.js` picks the case the deck walks through (goals,
  spectra, map comparison, headline buoy numbers, GUI screenshots). It is listed first
  in every chart and card. **`featuredBuoy`** picks the "one site, up close" buoy.
  Switching the main case needs its GUI screenshots in `assets/gui/<Case>/`
  (currently Netherlands and NewSouthWales).

## Keeping it in sync

Numbers and charts come from `data/data.js`, generated from each case's
published `portal/` bundle (the same files the GUI reads):

```bash
python3 presentation/data/build_data.py
```

Rerun it after `python -m hywaves publish <Case>/`. Case figures are referenced
directly from `../<Case>/portal/gui_figures/` and
`../NewSouthWales/portal/buoy_validation/figures/`, so re-publishing updates them
with no extra step. The deck therefore has to stay inside the repo checkout
(or be copied together with the `portal/` folders).

**Hand-maintained snapshots** (not regenerated):

| What | Where it comes from |
| --- | --- |
| Cost slide (40 ms / 104 ms / ~11 h) | `NewSouthWales/README.md` timing table (`09_extra_Reconstruction_Timing.ipynb`) |
| Per-site spread slide (`data/persite/<Case>_linear_summation.csv`) | `data/persite/compute_persite.py`, run inside the case (needs its models + dynamic SnapWave runs); `build_data.py` summarises it |
| `assets/gui/**/*.png` | Screenshots of the Results Explorer, per case, 2026-09-28 |
| GeoOcean figures (141+ pubs, partners), `bluemath_diagram.png`, `hybrid-diagram.png`, `partner_logos.png` | DEME workshop deck assets |

## Files

```
presentation/
├── index.html        all slide content
├── style.css         theme (DEME deck base + HyGloW additions)
├── deck.js           navigation, short/full mode, lightbox, data-driven slides & SVG charts
├── assets/           logos, hybrid diagram, GUI screenshots
└── data/
    ├── build_data.py regenerates data.js from the portal bundles (stdlib only)
    ├── data.js       generated, do not hand-edit
    └── config.js     hand-edited per-audience metadata
```
