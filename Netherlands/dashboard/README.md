# Netherlands dashboard

Every result of the Netherlands case for 2013 in one web app: buoys, offshore hindcasts
(WHACS, WHACS_cal), the HyWaves linear sum (MDA 600, CODEC water level; bulk and spectral),
the hourly SnapWave runs with and without ERA5 wind, the metamodel training cases, the
satellite calibration and the numerical tests. Case-only, like `analysis/`; the public
results explorer is `GUI/` (it reads only `portal/`).

```bash
cd Netherlands/dashboard
python build_store.py              # ~6 min, writes store/ (4.2 GB, git-ignored)
python app.py --port 8050          # forward the port, open http://localhost:8050
```

`build_store.py --only cube conditions` rebuilds parts (`sites`, `offshore`, `cube`,
`conditions`, `wind_grid`, `metamodel`, `calibration`, `numerical_tests`); rebuild after re-running
notebooks 06-08 or the 2013 reconstruction (`tmp/reconstruct_2013.py`). The app reads only
`store/` (and the findings text from `analysis/README.md` and `hywaves/WIND.md`). The first
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

Code: `build_store.py` (store), `data.py` (store access, metrics: reference first),
`figures.py`, `components.py`, `pages/` (one file per page).
