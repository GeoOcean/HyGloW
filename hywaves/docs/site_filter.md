# Output sites in training: site filter and dry points

Since hywaves 0.12.0. Configured in `metamodel.site_filter`; the code is in
bluemath_tk `waves/hywaves/metamodel.py` (`mask_dry_sites`,
`detect_removed_sites`, `drop_sites`, `fill_dry_sites`) and the case layer
`utils/metamodel.py` (`load_cases_goal`).

## Summary

The metamodel cases give every output site one value per case (`hs`, `tp`,
`dir`, `spr`). Some sites have no valid value in some cases: they are outside
the model grid, or they fall dry at low water levels. Before the MDA and the
PCA, each goal's cases go through four steps:

1. **Mask dry points.** Where `hs` is NaN or 0, every field becomes NaN, so
   SnapWave and SWAN report a dry point the same way.
2. **Detect bad sites.** A site is removed when it breaks any
   `site_filter` rule in its cases, including `hs_nan_max`: dry (or
   outside the grid) in more than that fraction of cases.
3. **Drop them.** They are listed in `outputs/goal_XX/split.json`
   (`removed_sites`) and are never trained, predicted, reconstructed or
   published.
4. **Fill the few dry cases left** (training only): `hs = 0`, and `tp`,
   `dir`, `spr` take that case's forcing values.

After step 4 the training fields have no NaN, so the PCA's
`pca_preprocessing.nan_threshold_to_drop` no longer drops anything; it is
only a safety net.

## The rules

`metamodel.site_filter` maps `{var}_{rule}` to a threshold. A site is removed
if it breaks any rule in any case of the goal:

| Rule | Removes a site when | Example | Catches |
| --- | --- | --- | --- |
| `{var}_max` | `var > threshold` | `"hs_max": 20` | blow-ups |
| `{var}_min` | `var < threshold` | `"tp_min": 0.5` | collapsed periods |
| `{var}_ratio_max` | `var / {var}_forcing > threshold` | `"hs_ratio_max": 3` | numerical anomalies (nearshore Hs far above the offshore Hs) |
| `{var}_nan_max` | the fraction of cases with `var` NaN `> threshold` | `"hs_nan_max": 0.01` | sites outside the grid (always NaN) and sites dry in more than 1 % of cases |

The bound rules skip NaN, so a dry case never trips them; only `nan_max`
counts dry cases. The template config uses
`{"hs_ratio_max": 3, "tp_min": 0.5, "hs_nan_max": 0.01}`.

## Why a site can be dry or missing

- **Outside the grid.** Isobath sites are traced on the topobathy raster
  (`paths.topobathy_file`), which may cover estuaries, harbours and tidal
  basins that the computational mesh closes off. Both models return no value
  there in every case.
- **Dry at low water level.** The mesh bathymetry is coarser and smoother than
  the raster, so a "5 m" isobath site can be 1–2 m deep in the model. The
  metamodel cases sample the water level (`simulations.bounds.wl`), and at the
  low end such a site falls dry.

How each model writes a dry or outside point:

| | outside the grid | dry |
| --- | --- | --- |
| SnapWave | NaN | `hs = 0`, `tp` = forcing, `dir = 270`, `spr` NaN |
| SWAN | exception value → NaN in every field | exception value → NaN in every field |

`mask_dry_sites` turns both into "every field NaN". `hs = 0` exactly is taken
as dry: a wet point keeps some energy, however small (in the Netherlands
SnapWave cases the smallest non-zero `hs` is 0.01 mm, and exact zeros only
appear at the shallow sites that fall dry).

## Why fill the remaining dry cases with the forcing

A site dry in at most `hs_nan_max` of the cases is kept, and those few cases
still need values, because the PCA and the GP see every training case.

- `hs = 0` is the physics: no water, no waves.
- `tp`, `dir`, `spr` have no physical value at a dry point. The case forcing
  is a smooth function of the GP inputs, so it adds no jump for the GP to
  learn; a constant (the old `dir = 270` placeholder, or the PCA's fill of 0)
  does.
- In the reconstruction they do not matter: a contribution with `hs ≈ 0` adds
  no energy to the spectral sum, so its `tp`, `dir` and `spr` drop out.

The holdout validation (`06`) loads its reference with `fill_dry=False`: dry
reference cases stay NaN and the metrics skip them, so the filled values
never count as truth.

## What the PCA did before (and why it was not enough)

`PCA` keeps a site column only if it is valid in more than
`nan_threshold_to_drop` (0.99) of the cases, and fills the rest with
`value_to_replace_nans`. A dropped column is not reconstructed: the PCA
inverse returns NaN there, for every prediction. That worked like a silent
site filter, but per variable. In the Netherlands SnapWave fit (hywaves
0.11.0) the 39 outside-grid sites were dropped from every PCA, while 9 dry
sites kept `hs` and `dir` (with `dir = 270` as data) and lost `spr`. The site
filter now makes one decision per site, recorded in `split.json`.

## Choosing `hs_nan_max`

- `0.0` removes every site that is ever dry in the cases. The case water level
  (`simulations.bounds.wl`) is often wider than the real one, so this can
  remove sites that are only dry at water levels that never happen.
- `0.01` (default) matches the PCA threshold: sites dry in more than 1 % of
  the cases go, and the PCA never has to drop a column.
- Larger values keep more sites, but more of their training cases are filled
  values rather than model output.

Compare against the real water levels at the goals (`forcing.water_level`)
before raising it.

## Inspecting

```python
from utils import metamodel as mm

mm.report_training_data(goal)             # removed sites, dry fractions after the filter
mm.removed_sites(goal)                    # from split.json
mm.load_cases_goal(goal, apply_site_filter=False)  # masked cases, before the filter
```

The reconstruction uses the sites kept in **every** goal
(`common_model_sites`), so a site removed for one goal is removed everywhere.

## Netherlands (2026-10-12)

With `hs_nan_max = 0.01` on the SnapWave cases (629 per goal, wl in
[−3.5, 3.5] m), every goal removes the same **48 of 407 sites**, all isobath
sites (no buoys):

- 39 outside the mesh (18 on 5 m, 17 on 10 m, 4 on 20 m): Eastern and Western
  Scheldt mouths, Rotterdam harbour, Marsdiep and behind Vlieland and
  Terschelling, the Ems mouth past the mesh's eastern end;
- 9 dry in 4.9–35 % of the cases (mesh depth 0.4–3.1 m), e.g. `5m_22`, dry
  from wl ≈ −1 m.

`5m_68`, dry only at wl ≤ −3.46 m (0.6 % of the cases), is kept with 4 filled
cases. CODEC's lowest water level at the goals is −3.1 m (goal 1) and −1.5 to
−2.3 m elsewhere.

The lasting fix is to trace the isobath sites on the mesh's own bathymetry,
inside the mesh; that changes the output sites, so it waits for the next full
re-run of the cases.
