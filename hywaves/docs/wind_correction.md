# The wind correction

How HyGloW adds the locally generated wind sea to a HyWaves reconstruction,
why it is built this way, how to use it, and what it has shown so far.
Available since hywaves 0.11.0; optional (`metamodel.wind_correction`).

## In one paragraph

For every hour and every output site: each goal's (or goal × partition's)
offshore sea state says how much of it is wind sea (`w`, 0–1, from its
steepness and spreading); the wind-sea energy reaching the site is the sum of
`w · hs²` over the goals' predicted contributions; with the wind at the site,
the fetch and the depth along that wind, a model trained on SnapWave (or SWAN)
runs with and without wind gives the growth factor `K` of that wind sea; one
extra contribution with the missing energy `(K² − 1) · E_ws` is added, from
the wind direction, and summed with the goals as a spectrum. The swell is
never scaled.

## Why it is needed

The metamodel cases are stationary runs **without wind**: each one puts waves
at a single goal and lets the model propagate them to the coast. Over the tens
of kilometres between the goals and the coast, a real storm keeps feeding
energy into the sea, while the model, without wind, only loses it to breaking
and friction. So the reconstruction misses part of the nearshore Hs, most in
storms.

The Xaver pilot (December 2013, 48 h, 7 buoys, `Netherlands/SWAN_test/`) put a
number on it. Without wind, SnapWave was 1.42 m too low against the buoys and
SWAN 1.88 m. With a uniform 20 m/s wind those biases became −0.53 m and
−0.21 m.

## Why not simply run the metamodel cases with wind

Each metamodel case is one goal's contribution, and the reconstruction **sums**
the contributions. Wind blows over the whole domain, so every case with wind
would grow its own copy of the local wind sea, and the sum would count it once
per goal. Wind growth is also not additive: it slows down as the sea gets
closer to fully developed, which depends on the **total** energy, not on one
goal's share. So the cases stay without wind, and the wind is added once,
after the goals are predicted and before they are summed.

## How it works

For each time step and output site, between the goals' predictions and the
spectral sum:

```
offshore forcing of every goal x partition  ──►  metamodel  ──►  contributions at the site
          │                                                         │
          └─ wind-sea weight w (steepness, spreading)               │
                                                                    ▼
             E_ws = Σ w · hs²     (wind-sea energy at the site, summed once)
             wind U, θ            (ERA5 at the site, or inferred from the sea state)
             K = f(g·Hs_ws/U², g·F(θ)/U², g·d/U², g·d̄(θ)/U², cos(wind − wave))
                                                                    │
             one more contribution: Hs = √((K² − 1) · E_ws), from θ, wind-sea Tp
                                                                    ▼
             JONSWAP × Cartwright per contribution → sum → spectrum → bulk stats
```

1. **Which part is wind sea.** Each contribution's *offshore* forcing gets a
   weight `w` from 0 (swell) to 1 (young wind sea): a linear ramp in
   deep-water steepness `Hs / L0` from 0.02 to 0.03, times a ramp in
   directional spreading from 20° to 25°. The ramps keep everything continuous
   in the sea-state parameters. In spectral mode the weight is per partition,
   so swell partitions stay untouched.
2. **Wind-sea energy at the site:** `E_ws = Σ w · hs²` over goals and
   partitions, summed once.
3. **The wind.** `"forcing"`: ERA5 interpolated to the site. `"inferred"`:
   each wind-sea contribution implies a wind from its own direction at the
   fully developed speed (`max(√(g·Hs/0.24), g·Tp/7.69)`, a lower bound),
   weighted by its energy at the site.
4. **The growth factor `K`** = Hs(with wind) / Hs(without wind) of the wind
   sea. It's predicted from five dimensionless numbers: how developed the
   wind sea already is (`g·Hs_ws/U²`), the fetch upwind of the site
   (`g·F/U²`), the depth at the site and along the fetch (`g·d/U²`,
   `g·d̄/U²`), and the cosine of the wind–wave angle.
5. **One extra contribution** carries the missing energy `(K² − 1) · E_ws`,
   from the wind direction, with the wind-sea Tp (energy-weighted, or from the
   growth curves with `added_tp: "growth"`) and 30° spreading. It joins the
   sum like any goal. The bulk Hs therefore grows by exactly K on its wind-sea
   part, and Tp and direction shift as they should.

### The fetch

`bluemath_tk.waves.fetch.fetch_table` walks a ray upwind from every output
site, every 5°, through the triangles of the model grid, until it leaves the
domain (250 m steps). It keeps the distance, the mean depth along the ray, and
whether the ray left through the open boundary or hit land. Saville's
effective fetch (±45°, cos²-weighted) smooths it around islands and inlets.
It is computed once per case (`outputs/wind/fetch_table_{model}.nc`).

### Where K comes from

A pure growth-curve formula (Kahma & Calkoen with the Breugem & Holthuijsen
depth limit, the curves SnapWave's own wind source uses) got moderate seas
right in the pilot but gave K = 1 above Hs ≈ 4 m, where the models still
added 15–35 %. So K is **learned from the model itself**, using those same
dimensionless numbers:

- **Training cases** (notebook `05b`): about 100 full-domain stationary cases,
  every goal forced with the same sea state, each run **twice**: without wind
  (the metamodel's own settings) and with a uniform wind. The design is an LHS
  of Hs, Tp, direction, spreading, water level, wind speed as 0.8–2× the fully
  developed speed, and wind–wave angle within ±60°. It keeps plausible sea
  states with `w ≥ 0.5`, then MDA picks the cases.
- **Target:** every case × site pair gives the wind-sea K,
  `K² = 1 + (Hs₁² − Hs₀²) / (w · Hs₀²)`, consistent with how it is applied.
- **Model:** a gradient-boosting regression of `log K` (scikit-learn),
  `WindCorrection` in `bluemath_tk.waves.hywaves.wind_correction`. It
  returns K = 1 without wind (< 1 m/s), without wind sea (< 5 cm), or outside
  the grid. K is learned per numerical model: SnapWave and SWAN respond
  differently to wind.

### What the five inputs of K mean

They are made dimensionless with the wind speed, using the length `U²/g` and
the time `U/g` that wind-wave growth scales with: the same dimensionless
situation gives the same *relative* growth at any wind speed or site, which is
why ~100 training cases can cover the whole coast and every wind.

| Input | Meaning | Effect on K |
| --- | --- | --- |
| `g·Hs_ws/U²` | How developed the wind sea already is (0.24 = fully developed: the wind cannot raise it further) | The strongest: young seas under strong wind grow a lot, developed ones little |
| `g·F/U²` | Fetch left: distance the wind acts upwind of the site inside the domain | More fetch, more growth, flattening out |
| `g·d/U²` | Depth at the site: breaking and friction limit growth in shallow water | Shallow sites get K ≈ 1 |
| `g·d̄/U²` | Mean depth along the fetch: shallow banks upwind limit growth on the way | As above, along the path |
| `cos(angle)` | Wind relative to the wind-sea direction | Following wind grows most |

`K` is not a formula of these: it is what the model runs showed for similar
situations, learned by a gradient-boosting regression on all the training
pairs.

### Growth formula vs fitted model

Notebook `05b` (section 5, `wind.k_comparison()` and
`viz.plot_wind_k_comparison()`) puts three estimates side by side on training
cases held out from the fit: the `K` the runs show (the reference), the fitted
model, and the growth formula (`wind.formula_k`: the wind sea treated as grown
over its equivalent fetch, grown over the fetch left, in deep water or capped
by the depth limit at the site). Netherlands, SnapWave, 20 held-out cases,
7,265 site-cases (all three bounded to 0.5–4):

| Estimate of K | RMSE | bias | median |
| --- | --- | --- | --- |
| runs (reference) | — | — | 1.25 |
| fitted model | 0.38 | −0.06 | 1.23 |
| formula, depth-limited | 0.51 | −0.00 | 1.23 |
| formula, deep water | 0.84 | +0.30 | 1.54 |

The depth-limited formula gets the typical K right, but scatters more and
saturates at the cap (K = 4) for very young seas (`g·Hs_ws/U²` < 0.02), where
the runs give 1.5–2.5; in storms it tends to stay below the runs, because over
shallow seas the wind also keeps the existing sea from decaying, which growth
curves do not describe. The fitted model follows the runs in both regimes.

## A worked example

Xaver's peak, **5 December 2013 19:00**, buoy **IJmuidenMunitiestort_1**
(Netherlands), bulk mode, `buoys` set, MDA 550, ERA5 wind.

**1 · Wind-sea weight per goal** (from the goal's *offshore* forcing) and what
reaches the buoy:

| Goal | offshore Hs / Tp / spreading | steepness | w | nearshore Hs | w · hs² |
| --- | --- | --- | --- | --- | --- |
| 3 | 3.95 m / 8.5 s / 27.0° | 0.035 | 1.00 | 2.94 m | 8.63 |
| 4 | 4.52 m / 9.2 s / 24.7° | 0.034 | 0.94 | 3.15 m | 9.32 |
| 5 | 5.28 m / 10.6 s / 23.4° | 0.030 | 0.67 | 0.84 m | 0.48 |
| 8 | 6.67 m / 12.0 s / 20.9° | 0.029 | 0.17 | 0.00 m | 0.00 |
| 1, 2, 6, 7, 9, 10 | … | … | … | ≤ 0.41 m | ≤ 0.17 |

(Goal 4: steepness 0.034 is above 0.03, so 1 on the first ramp; spreading
24.7° gives (24.7 − 20) / 5 = 0.94 on the second; w = 0.94.)

**2 · Wind-sea energy:** `E_ws = Σ w·hs² = 18.75 m²` → `Hs_ws = 4.33 m`,
direction 312° (energy-weighted). The total energy is 19.58 m², so 96 % of it
is wind sea here.

**3 · Wind at the buoy** (ERA5): 18.5 m/s from 310°; wind–wave angle −2°.
(The inferred wind would be ≈ 12.7 m/s, the fully developed lower bound.)

**4 · Fetch and depth along 310°:** effective fetch 29.6 km, depth 24.7 m at
the buoy, 26.1 m on average upwind.

**5 · K:** with `U²/g = 34.9 m`, the inputs are `g·Hs_ws/U² = 0.124` (half way
to fully developed), `g·F/U² = 846`, `g·d/U² = 0.705`, `g·d̄/U² = 0.746`,
`cos(angle) = 0.999` → **K = 1.156**. Changing one input at a time:

| Hs_ws (m) | 1 | 2 | 3 | **4.33** | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K | 2.67 | 1.55 | 1.31 | **1.156** | 1.12 | 1.10 | 1.08 |

| U (m/s) | 8 | 12 | 15 | **18.5** | 22 | 26 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K | 1.02 | 1.06 | 1.12 | **1.156** | 1.26 | 1.29 | 1.35 |

| fetch (km) | 2 | 10 | 20 | **29.6** | 50 | 80 | 120 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| K | 1.04 | 1.10 | 1.13 | **1.156** | 1.20 | 1.21 | 1.22 |

| site depth (m) | 5 | 10 | 15 | **24.7** | 35 | 50 |
| --- | --- | --- | --- | --- | --- | --- |
| K | 0.98 | 1.05 | 1.11 | **1.156** | 1.18 | 1.21 |

| wind–wave angle | **−2°** | 30° | 60° | 90° | 135° | 180° |
| --- | --- | --- | --- | --- | --- | --- |
| K | **1.156** | 1.12 | 1.11 | 1.11 | 1.11 | 1.11 |

The growth formula for the same hour: the 4.33 m sea is "already grown" over
≈ 150 km at 18.5 m/s; 29.6 km more gives 4.70 m, **K = 1.085**. Its
shallow-water cap (3.62 m at 24.7 m depth) would even allow no growth at all.

**6 · Extra contribution:** `Hs_add = √((1.156² − 1) · 18.75) = 2.51 m`, from
310°, Tp 8.86 s (energy-weighted Tp of the wind-sea contributions), spreading
30°.

**7 · Spectral sum** (10 goals + the wind sea):

| | Hs | Tp | dir |
| --- | --- | --- | --- |
| without correction | √19.58 = 4.43 m | 8.84 s | 315° |
| with correction | √(19.58 + 6.29) = 5.09 m | 8.85 s | 313° |

The wind-sea part went from 4.33 to 4.33 × 1.156 = 5.01 m; the rest is
unchanged. (The buoy has no record at 19:00; its nearest, at 17:00, read
5.15 m — an order of magnitude check, not a comparison.)

## Using it

1. In the case JSON, `metamodel.wind_correction` (template defaults):

   ```json
   "wind_correction": {
       "enabled": false,
       "wind": "forcing",
       "num_cases": 100,
       "speed_ratio": [0.8, 2.0],
       "wind_wave_angle": [-60.0, 60.0],
       "u10_range": [3.0, 35.0],
       "steepness_range": [0.02, 0.03],
       "spr_range": [20.0, 25.0],
       "speed_factor": 1.0,
       "added_spr": 30.0,
       "added_tp": "wind_sea"
   }
   ```

   `"wind": "forcing"` needs `forcing.wind` (e.g. ERA5); `"inferred"` needs no
   wind data.
2. **Notebook `05b_Wind_Correction`:** fetch table → training cases (built
   twice, run on the cluster with the wind-fixed SnapWave build
   `snapwave/b663c81-wind1`) → postprocess → `wind.fit_wind_correction()`. It
   saves `outputs/wind/wind_correction_{model}.pkl` and prints holdout scores.
3. **Reconstruct:** `mm.reconstruct(..., wind_correction=True)`, or set
   `enabled: true` to make it the default. The output gains `wind_k` (K, 1
   where nothing was added). The reconstruction cache keeps corrected and
   uncorrected results apart.
4. **Validate like with like** (notebook 09). The uncorrected reconstruction
   goes against dynamic runs **without** wind; the corrected one against
   dynamic runs **with** wind. Against the buoys (10) both are scored on their
   own.

## Results so far

All numbers are Hs, with the reference first (bias = estimate − reference).

| Test | Reference | Without correction | With correction |
| --- | --- | --- | --- |
| Pilot, Xaver, train day 1 → test day 2, 378 sites (uniform 20 m/s, wind from the waves) | SnapWave with wind | bias −0.51, RMSE 0.73 m | bias +0.01, RMSE 0.14 m |
| Netherlands `05b`, 20 held-out training cases, 7,360 site-cases | SnapWave with wind | bias −0.52, RMSE 0.85 m | bias −0.09, RMSE 0.48 m |
| Netherlands, 2012–2013, 27 buoys with data, MDA 550, `buoys` set, ERA5 wind (median over buoys) | buoys | bulk: bias −0.10, RMSE 0.31 m, R² 0.78 · spectral: bias −0.08, RMSE 0.30 m, R² 0.79 | bulk: bias +0.08, RMSE 0.29 m, R² 0.83 · spectral: bias +0.11, RMSE 0.28 m, R² 0.83 |

For comparison, over the same 2012–2013 buoys:
- the dynamic SnapWave runs with ERA5 wind: Hs bias +0.19 m, RMSE 0.42 m;
- WHACS at the closest goal: bias +0.19 m, RMSE 0.41 m.

## Limitations and open points

- **SnapWave has one period per node.** Wind energy added on top of swell takes
  the swell's period, so its runs, and the K learned from them, may
  overestimate shoaling and breaking in mixed seas. SWAN resolves frequency;
  compare once the SWAN pass is done.
- **Training winds are uniform.** Real wind fields vary over the domain; K
  sees only the wind at the site and the fetch along its direction.
- **Angles beyond ±60°** (opposing or cross winds) are outside training.
  Tree models do not extrapolate: K stays at its ±60° value (1.11 in the
  example, for 90°, 135° and 180° alike), which is likely too high for
  opposing winds. Notebook 08's forcing figure shows how often that happens
  (about 3 % of wind-sea hours in the Netherlands for 2012–2014); widening
  the training angles would fix it.
- **Refits differ slightly.** The regression holds out a random 10 % for early
  stopping when it has more than 10,000 samples, so refitting changes scores
  in the third decimal.
- **Very shallow sites** (≈ 5 m) are dominated by breaking; the models even give
  K slightly below 1 there.
- **The inferred wind is a lower bound** (the fully developed speed). With wind
  data, prefer `"wind": "forcing"`.
- **Tp is not improved.** At the Dutch buoys, Tp has negative median R² for
  every estimate (the metamodel passes the offshore Tp through), corrected or
  not.

## Code map

| Piece | Where |
| --- | --- |
| Wind-sea weight, inferred wind, growth curves | `bluemath_tk/waves/wind.py` |
| Fetch table, effective fetch | `bluemath_tk/waves/fetch.py` |
| `WindCorrection`, `wind_sea_contribution` | `bluemath_tk/waves/hywaves/wind_correction.py` |
| Hook in the spectral sum | `bluemath_tk/waves/hywaves/reconstruction.py` (`wind_correction=` of `sum_partition_spectra_batched`) |
| Case layer (config, fetch per case, training design, fit, formula vs fitted K, reconstruction inputs) | `hywaves/template/utils/wind.py` |
| K comparison figure | `hywaves/template/utils/viz.py` (`plot_wind_k_comparison`) |
| `reconstruct(wind_correction=...)` | `hywaves/template/utils/metamodel.py` |
| Notebook | `hywaves/template/notebooks/05b_Wind_Correction.ipynb` |
