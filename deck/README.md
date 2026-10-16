# HyGloW deck

An HTML slide deck introducing HyGloW (the open project and portal) and HyWaves
(the hybrid downscaling method) for a general audience. Published on GitHub
Pages by `.github/workflows/deck.yml` on every push to `main` that touches
`deck/`, a case `portal/` or `GUI/catalog/cases.yaml`.

## Build and preview

```bash
python deck/build.py            # needs Pillow; writes deck/_site/ (git-ignored)
```

Open `deck/_site/index.html` in a browser. The build reads only the cases'
published `portal/` bundles (the files the GUI reads) and
`GUI/catalog/cases.yaml`. It copies just the images the deck shows, as WebP no
wider than 2000 px, with lazy loading. After `python -m hywaves publish <Case>/`
and committing `portal/`, the online deck updates itself.

## Presenting

- Arrow keys / space / on-screen buttons to navigate, `f` fullscreen, click a
  figure to zoom, top-right button toggles light/dark.
- `s` (or the "Full/Short deck" button) hides the slides marked
  `data-depth="deep"` (goals sketch, spectra, MDA convergence).
- `config.json` holds per-audience text (title, subtitle, presenter, venue,
  default mode), `mainCase` (the case the deck walks through: figures, headline
  numbers, GUI screenshots) and `featuredBuoy` (the "one site, up close" buoy,
  a buoy name from the main case's buoy summary).

## Hand-maintained

| What | Where |
| --- | --- |
| `assets/gui/<Case>/case_{metamodel,buoys,goals,final}.png` | Results Explorer screenshots of the main case, 2026-09-28 |
| GeoOcean facts (141+ pubs, partners), `bluemath_diagram.png`, `partner_logos.png` | GeoOcean material |
| Toolkit / wrappers slides | bluemath-tk snippets and the BinWaves Paradip example |

Everything else on the HyWaves slides (runs per goal, sites, buoy errors,
convergence, reconstruction time) comes from the portals.

## Files

```
deck/
├── index.html    slide content
├── style.css     theme
├── deck.js       navigation, short/full mode, lightbox, data-driven slides & SVG charts
├── config.json   hand-edited per-audience metadata
├── build.py      -> _site/ (index.html, data.js, WebP images)
└── assets/       logos, diagrams, GUI screenshots
```
