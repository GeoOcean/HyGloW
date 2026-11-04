"""Write and execute the two decision notebooks. Run from ``Netherlands/model_decisions/``:

    python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
KERNEL = {"display_name": "bluemath-312", "language": "python", "name": "python3"}

SETUP = """import pandas as pd
from IPython.display import Image, display

import decisions as d

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", 250)"""

SWAN = [
    ("md", """# SWAN: how the template configuration was chosen

Netherlands, 2013. Every SWAN option in `hywaves/template/templates-swan-*/INPUT` was
tested against **27 buoys** (Hs; bias = model − buoy, RMSE in m) on the same
**39 test hours** (8 calm, 20 storm, 11 Xaver), then the open points on **120 other
hours**. Goal cases (no wind) were compared with SnapWave on **40 metamodel cases**
spread over steepness. The numbers come from `results/*.csv` (see the README); the
runs themselves live in `runs/` (not tracked).

Groups: `calm`, `storm` (January–February storms), `xaver` (December 2013); `obs a-b m`
bins by the observed Hs (bins by observation regress to the mean: the top bins look
worse than the model is, so compare variants within a bin, not bins between them);
`d` is the mesh depth at the buoy."""),
    ("code", SETUP),
    ("md", """## 1. Source terms (dynamic runs with wind)

Westhuysen (SWAN's `GEN3` default), Komen and ST6 (`GEN3 ST6 5.7E-7 8.0E-6 4.0 4.0 UP
HWANG VECTAU TRUE10` + `SSWELL ZIEGER`), all with the old WHACS_cal forcing. SnapWave's
production run is shown for reference."""),
    ("code", 'd.buoy_table("swan_source_terms")'),
    ("code", """d.plot_by_class(
    "swan_source_terms",
    ["SWAN Westhuysen (old WHACS_cal)", "SWAN Komen (old WHACS_cal)", "SWAN ST6 (old WHACS_cal)"],
    "Source terms, 39 hours",
);"""),
    ("md", """**Decision: `GEN3 KOMEN`.** Best from 2 m up (storm RMSE 0.35 vs 0.41 Westhuysen and
ST6). ST6 is worse everywhere above 3 m, needs 16 GB per task and runs twice as long
(median 37 vs 20 min). Komen hits the 50-iteration limit in 6 of 39 hours (accuracy
≥ 83 %)."""),
    ("md", """## 2. Whitecapping in the no-wind goal cases

The goal cases have no wind. SWAN still applies whitecapping, which removes the energy
of steep seas. Hs ratio SWAN / SnapWave (energy weighted, sites with ≥ 25 % of the
offshore Hs), by case steepness and depth."""),
    ("code", 'd.ratio_table("swan_whitecapping")'),
    ("md", """**Decision: `OFF WCAP` in the goal cases** (and in the no-wind runs that train K, so
both sides of K share the physics). Steep cases go from 0.75 to 0.95 of SnapWave; gentle
ones do not change. With wind, Komen switches whitecapping back on."""),
    ("md", """## 3. Offshore forcing

Same physics (Komen), three forcings: the old WHACS_cal (directional energy fit
only), the new WHACS_cal (the same fit plus a satellite-only quantile mapping of the
storm tail, notebook 07b) and ERA5_cal."""),
    ("code", 'd.buoy_table("swan_forcing_numerics", ["SWAN Komen (old WHACS_cal)", "SWAN Komen (WHACS_cal)", "SWAN Komen (ERA5_cal)"])'),
    ("code", """d.plot_by_class(
    "swan_forcing_numerics",
    ["SWAN Komen (old WHACS_cal)", "SWAN Komen (WHACS_cal)", "SWAN Komen (ERA5_cal)"],
    "Forcing, 39 hours",
);"""),
    ("md", """**Decision: the quantile mapping stays in 07b** (Xaver RMSE 0.72 → 0.62; against 9
deep buoys 1979–2023: RMSE 0.27 → 0.25, p99.99 −1.06 → −0.14 m). ERA5_cal is better
still in Xaver (0.47). The forcing for the dynamic runs is **open**: below 3 m the
two tie, and the storm advantage of ERA5_cal rests mostly on Xaver (section 5)."""),
    ("md", """## 4. Numerics with no effect

A water-level field instead of one level per case, `BREAKING BKD`, 100 instead of 50
iterations, no triads, and 5° instead of 10° directions."""),
    ("code", 'd.buoy_table("swan_forcing_numerics", ["SWAN Komen (old WHACS_cal)", "SWAN Komen (old WHACS_cal), water-level field", "SWAN Komen (old WHACS_cal), BREAKING BKD"])'),
    ("code", 'd.ratio_table("swan_friction_triads")[["SWAN OFF WCAP", "SWAN OFF WCAP, no triads"]]'),
    ("code", 'd.load("swan_directions").describe().round(3)'),
    ("md", """**Decision: keep the simple settings.** The water-level field changes the scores by
≤ 0.02 m; `BKD` changes nothing below 5 m and is worse at 5–6 m (RMSE 0.74 → 0.96,
7 pairs); triads do not change Hs; 100 iterations
change Hs by +0.003 m on average (p99 0.056 m) at up to twice the cost; 5° directions
leave 37 of 40 cases within 1 % (two narrow-swell cases move, and only 0.6–1.6 % of
the partition energy has a spreading under 10°) at twice the cost. `NUMERIC STOPC`
stays at the manual defaults."""),
    ("md", """## 5. Bottom friction

JONSWAP 0.038 (the manual's value for sandy bottoms) against 0.019 (its value for
smoother seafloors). First the 39 hours, then 120 other hours of 2013 that the
choice was not tuned on."""),
    ("code", 'd.buoy_table("swan_friction", ["SWAN Komen (WHACS_cal)", "SWAN Komen (WHACS_cal), friction 0.019", "SWAN Komen (ERA5_cal)", "SWAN Komen (ERA5_cal), friction 0.019"])'),
    ("code", 'd.buoy_table("swan_120h")'),
    ("code", """d.plot_by_class(
    "swan_120h",
    ["SWAN Komen (WHACS_cal), friction 0.038", "SWAN Komen (WHACS_cal), friction 0.019", "SWAN Komen (ERA5_cal), friction 0.019"],
    "Friction and forcing, 120 other hours",
);"""),
    ("md", """**Decision: friction 0.019** in both templates. On the 120 hours RMSE 0.31 → 0.30,
better above 3 m (3–4 m 0.39 → 0.36, 6–12 m 1.39 → 1.27), slightly worse below 1 m
(bias +0.06 → +0.09). On these hours ERA5_cal again wins above 4 m, but 8 of them
fall on the Xaver days and 3 on St Jude."""),
    ("md", """## 6. Why the dynamic runs have wind

Same hours with and without wind (goal physics), then 500 hours of 2013 without wind."""),
    ("code", 'd.buoy_table("swan_wind")'),
    ("code", 'd.buoy_table("swan_no_wind_500h")'),
    ("md", """**Decision: dynamic runs keep wind, and the linear sum needs the wind correction K.**
Without wind Hs is low everywhere (500 hours: bias −0.30, RMSE 0.50 m); wind adds 22 %
in calm hours and 4–6 % in storms (median Hs ratio). Stationary runs stay: a
non-stationary Xaver run gained no Hs, and the lag study found none."""),
    ("md", """## 7. Wind input

`AGROW` (linear growth, off unless requested) and ERA5 wind × 1.1 (ERA5 storm winds
are often low)."""),
    ("code", 'd.buoy_table("swan_wind_input")'),
    ("md", """**Decision: neither.** `AGROW` changes nothing. Wind × 1.1 helps Xaver (RMSE 0.56 →
0.50) but spoils everything else (calm 0.17 → 0.25, storm 0.33 → 0.39): ERA5 wind is
right on average."""),
    ("md", """## Template physics (result)

Goal cases (`templates-swan-metamodel/INPUT`):

```
OFF QUAD
OFF WCAP
BREAKING CONSTANT 1.0 0.73
FRICTION JONSWAP CONSTANT 0.019
TRIAD
NUMERIC STOPC 0.005 0.01 0.005 99.5 STATIONARY 50
```

Dynamic runs (`templates-swan-dynamic/INPUT`): `GEN3 KOMEN` with wind, else
`OFF QUAD` + `OFF WCAP`; the rest as above; `TABLE` with `HSIGN TPS TMM10 DIR DSPR` and
2D spectra at the buoys (`SPECOUT`). Grid: `CIRCLE 36 0.03 1.0 36`."""),
]

SNAPWAVE = [
    ("md", """# SnapWave: how the template configuration was chosen

Netherlands, 2013, build `snapwave/b663c81-wind1` (GeoOcean wind fixes on upstream
`b663c81`). The same 27 buoys and 39 test hours as the SWAN notebook, the new WHACS_cal
forcing; 40 metamodel cases against SWAN (`OFF WCAP`, friction 0.038); 24 of the
hours where wind mode did not converge; and the convergence of all 8760 production
hours of 2013. Numbers from `results/*.csv`."""),
    ("code", SETUP),
    ("md", """## 1. Boundary enclosure

SnapWave forces every mesh edge node inside `encfile` (the manual: the enclosure must
"limit the boundary points to those on the outer edge of the mesh that are at sea").
HyWaves used the minimum rectangle around the AoI, which held the whole mesh, so the
coast was forced with the goal spectra and radiated energy into the domain. Now
`bluemath_tk.topo_bathy.mesh.open_boundary_nodes` takes the outline through the goals,
out to where the lateral boundaries meet the coast (two corners), and
`open_boundary_enclosure` puts a thin polygon around it: 768 nodes, the 767 of SWAN's
`.grd` open boundary plus one corner node."""),
    ("code", 'display(Image(filename="results/figures/snapwave_enclosure.png", width=650))'),
    ("md", """## 2. Dynamic runs at the buoys

AoI rectangle → open-boundary enclosure → plus upwind refraction (`upwindref = 1`),
then bottom friction `fw` and no wind on top. SWAN (Komen, WHACS_cal, friction 0.038)
for reference."""),
    ("code", 'd.buoy_table("snapwave_setup")'),
    ("code", """d.plot_by_class(
    "snapwave_setup",
    ["SnapWave (WHACS_cal), AoI rectangle", "SnapWave (WHACS_cal), open-boundary enclosure",
     "SnapWave (WHACS_cal), enclosure + upwindref 1", "SWAN Komen (WHACS_cal)"],
    "SnapWave setup, 39 hours",
);"""),
    ("code", """d.plot_by_class(
    "snapwave_setup",
    ["SnapWave (WHACS_cal), enclosure + upwindref 1", "SnapWave (WHACS_cal), enclosure + upwindref 1, fw 0.02",
     "SnapWave (WHACS_cal), enclosure + upwindref 1, fw 0.04", "SnapWave no wind (WHACS_cal), enclosure + upwindref 1"],
    "SnapWave friction and wind, 39 hours",
);"""),
    ("md", """**Decisions.**

- **Enclosure**: correct by construction; at the buoys calm RMSE 0.35 → 0.32, depth
  < 15 m bias +0.11 → +0.06.
- **`upwindref = 1`** (monotone refraction): storm bias +0.20 → +0.15; Xaver slightly
  worse (−0.29 → −0.32).
- **`fw` stays at the default 0.01**: 0.02 fixes moderate seas but worsens storms
  (Xaver RMSE 0.66 → 0.79); 0.04 helps calm hours more and ruins storms (Xaver 1.06).
- **The positive bias is SnapWave's wind input**: without wind SnapWave matches SWAN
  (calm bias −0.18 vs −0.20 for SWAN); with wind it adds ~0.4 m in calm hours where
  SWAN adds ~0.2. That lives in the source's growth curves, not in a setting; the
  SnapWave K is trained on SnapWave with/without wind, so it inherits it."""),
    ("md", """## 3. Goal cases: refraction, iterations, sector

Hs ratio SnapWave / SWAN (`OFF WCAP`) on the 40 metamodel cases."""),
    ("code", 'd.ratio_table("snapwave_setup")'),
    ("md", """**Decisions.** The enclosure lowers Hs near the coast (mean −0.012 m, p1 −0.30 m over
all sites). Upwind refraction brings gentle seas from 1.09–1.17 to 1.03 of SWAN and
makes **every** goal case converge (largest final error 1e-5; with the central scheme
7 of 40 ended above 0.01, one at 0.999). `niter` counts sweeps (4 per iteration):
200 instead of 40 changes Hs by ≤ 0.005 m (p99) but lets upwind cases finish (median
13 iterations = 52 sweeps); SnapWave stops when converged, so it costs nothing.
`sector = 360` gains nothing at twice the time."""),
    ("md", """## 4. Wind mode that does not converge

In the 2013 production run, wind mode did not converge in 241 of 8760 hours (final
error > 0.01): mostly swell (mean Tp > 10 s) under light wind (~3 m/s), with Hs up to
~10 m under 0.5 m offshore. 24 of
them under each variant (`dt`: pseudo time step; `iter`: niter 800; `nodcg`:
`jadcgdx = 0`):"""),
    ("code", 'd.load("snapwave_unstable_hours").round(3)'),
    ("md", """No setting fixes it (the best, `dt = 3600`, still leaves 14 of 24 unstable, up to
11.6 m). But the final error separates the hours cleanly; all 8760 production hours
by final error:"""),
    ("code", 'd.load("snapwave_convergence_2013").round(3)'),
    ("md", """**Decision: postprocessing drops runs whose last error is above 0.01**
(`bluemath_tk.wrappers.snapwave.MAX_ERROR`, read from `wrapper_out.log`): metamodel cases
are left out of the training set, dynamic and wind-correction hours become NaN.
Below 0.003 runs are sound (median Hs max / offshore 1.0–1.2); above 0.01 the
median is 3–10 ×. With the new settings converged runs end at ~1e-5 and the failing
hours at 0.6–2.4."""),
    ("md", """## Template settings (result)

`templates-snapwave-*/snapwave.inp`: `upwindref = 1`, `niter = 200`, `sector = 180`
(360 with wind, forced by the build), `gamma = 0.7`, `fw` default 0.01; the enclosure
comes from `open_boundary_nodes` / `open_boundary_enclosure`
(`utils.simulation.HyWavesWrapperSetup`)."""),
]


def build(name: str, cells: list[tuple[str, str]]) -> None:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = KERNEL
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / name)
    print(name, len(nb.cells), "cells")


if __name__ == "__main__":
    build("01_SWAN_Decisions.ipynb", SWAN)
    build("02_SnapWave_Decisions.ipynb", SNAPWAVE)
