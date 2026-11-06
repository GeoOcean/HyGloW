# Netherlands dashboard

Every result of the Netherlands case for 2013 in one web app: buoys, offshore hindcasts
(WHACS, WHACS_cal), the HyWaves linear sum (MDA 600, CODEC water level; bulk and spectral),
the hourly SnapWave runs with and without ERA5 wind, the metamodel training cases, the
satellite calibration and the numerical tests. Case-only; the public results explorer is
`GUI/` (it reads only `portal/`).

```bash
cd Netherlands/dashboard
python build_store.py              # ~6 min, writes store/ (4.2 GB, git-ignored)
python app.py --port 8050          # forward the port, open http://localhost:8050
```

`build_store.py --only cube conditions` rebuilds parts (`sites`, `offshore`, `cube`,
`conditions`, `wind_grid`, `metamodel`, `calibration`, `numerical_tests`); rebuild after re-running
notebooks 06-08 or the 2013 reconstruction (`tmp/reconstruct_2013.py`). The app reads only
`store/` (and the findings text from this README and `hywaves/WIND.md`). The first
view of each estimate loads it (~1 s); the app keeps it in memory (up to ~8 GB).

| Page | What |
| --- | --- |
| Overview | The case on a map, headline numbers, 2013 timeline (click to open an hour) |
| Explorer | Two estimates at any hour, their difference, wind and goal arrows, play; click a site for its series and skill. The URL keeps the view |
| Compare | Any estimate against any reference over any hours (filters, presets, box selection): metric map, density + QQ, coast × wind direction, per group |
| Error budget | Spectral metamodel − buoy split into partitions, method, missing wind, model + forcing, per buoy and condition |
| Wind | What SnapWave's wind adds, by coast and direction, against the buoys; analog hours; non-converged hours |
| Storms | Peaks of every estimate at every buoy per event, series |
| Metamodel | Training cases and their fields, holdout skill vs MDA size, linear-sum metrics |
| Calibration | The two steps (directional fit per goal, one tail map for all goals): passes raw / directional / mapped, sector factors, the record's quantiles, the deep-buoy check (1979-2023), the change from the goals to the coast |
| Numerical tests | Decisions, buoy scores and goal-case ratios of `numerical_tests/` |
| Findings | Conclusions so far and a guided tour into the pages |

Code: `build_store.py` (store) from `sources.py` (the case results), `data.py` (store access, metrics: reference first),
`figures.py`, `components.py`, `pages/` (one file per page).

## What it shows (first iteration, SnapWave, 2013)

Conclusions only; the numbers are on the pages named (the Findings page links into each).

- **The linear sum works.** The bulk metamodel matches the dynamic run without wind (like
  for like) at every isobath and at the buoys; the method error is a negligible part of
  the error at the buoys (Compare, Error budget).
- **But it is high in the biggest seas near the coast**, most at 5 m and more with
  WHACS_cal (Compare: p99.9 bias, 5 m). Likely depth-limited breaking, which does not add
  linearly: each goal alone breaks less than all together. To check.
- **The missing wind is the largest error** at the buoys, and it grows with the sea; with
  wind the dynamic run is the best estimate (Error budget, Wind; numbers and options in
  `hywaves/WIND.md`).
- **What SnapWave's wind adds depends on the sea already there** and on the fetch inside
  the domain: little where WHACS already brings a grown sea from offshore (NW), most with
  wind from land or along the coast (E-S); SW wind adds most in Holland and the Wadden, E
  wind at the outer Belgium-Zeeland sites. At the buoys the run with wind is too high with
  S-SW wind and about right with NW wind: SnapWave grows too much sea along the coast (Wind).
- **Spectral Tp, direction and spreading are poor**, while spectral Hs is fine (Compare,
  Explorer). To check how the summed spectrum gives Tp, Dir, Spr. Spreading is too narrow
  without wind, in the metamodel and the dynamic run.
- **The satellite calibration helps the storm tail** at the buoys and in Xaver; the
  offshore change shrinks towards the coast (Calibration, Storms).
