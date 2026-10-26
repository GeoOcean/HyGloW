# HyWaves: the method

HyWaves turns an offshore wave hindcast into nearshore waves without running
a spectral wave model for every hour. The code is in
[bluemath_tk](https://github.com/GeoOcean/BlueMath_tk) (`waves/hywaves/`);
this folder holds the case template, the case checks and the command line.

## Physics in one page

**Goals.** The offshore sea enters the nearshore domain through a few points
on its open boundary, the *goals*, each with the sector of directions from
which waves can reach the coast. Each goal takes its waves from the nearest
node of an offshore hindcast (WHACS, ERA5, ...), as bulk sea state or as
spectral partitions (wind sea and swells).

**Training cases.** For every goal, a few hundred stationary runs of a
nearshore model (SnapWave or SWAN) force waves at that goal only, sampling its
sea states with a Latin hypercube: significant height Hs, peak period Tp,
direction, directional spreading and water level. Implausibly steep seas
(Hs / L0 above a limit) are dropped. Each run gives Hs, Tp, direction and
spreading at the output sites (isobaths, buoys, mesh areas), which are chosen
inside the model mesh and always wet.

**Metamodel.** Per goal, a maximum-dissimilarity selection (MDA) picks the
training cases; the fields at the sites are compressed with one PCA per
variable (direction as its two components), and a Gaussian process maps the
offshore sea state to the principal components. The rest of the cases check
it (holdout).

**Reconstruction by linear summation.** For each hour, every goal (and every
offshore partition) is predicted at the sites, turned into a directional
spectrum (JONSWAP in frequency, cos²ˢ in direction), and the spectra of all
goals are **added as energy**; the nearshore bulk parameters (or the full
spectrum) come from the sum. This works because wave energy propagates
linearly away from breaking and wind growth, and because both models spread
each goal's energy along the open boundary with weights that add up to one,
so the sum of single-goal runs equals a run with every goal forced.

**What the sum leaves out, and the options for it.**

- *Local wind.* The training runs have no wind (with wind, every goal's run
  would grow its own copy of the local sea). An optional **wind correction**
  adds the wind sea once, after the goals are predicted: its growth factor K
  is learned from full-domain runs with and without wind, as a function of
  how developed the sea is, the fetch and the depth (all scaled with the wind
  speed).
- *Water level.* A metamodel input: constant, or a tide and surge series per
  goal.
- *Offshore bias.* The hindcast can be checked and calibrated against
  satellite altimetry per goal and direction sector before reconstructing.
- *Non-linear processes* (strong breaking, wave-current interaction, time lag
  across a large domain) are what the **dynamic runs** check: the model run
  hour by hour with every goal, wind and water level, against which the
  reconstruction is validated, and buoys are the final check.

## Running it: resources and timings

Fitting and reconstructing are the heavy steps. Before each, one call looks
at the machine (cores, free memory, GPU) and **proposes** settings using at
most 75% of it; you read the proposal, change it if you want, and run with
it. Nothing is guessed silently, and no queue system is needed (SLURM is only
used for the SnapWave / SWAN runs).

```python
res = mm.fit_resources()                   # prints the machine, the fits to do, the proposal
mm.fit_all(resources=res)

res = mm.reconstruction_resources("2013", 550, metamodel="isobaths", spectral=True)
res.workers = 20                           # optional: leave cores to others
ds = mm.reconstruct("2013", 550, metamodel="isobaths", spectral=True, resources=res)
```

How the work is split:

- **Fit**: every (goal, metamodel set, MDA size) is an independent small
  fit, run as its own single-thread process, many at a time.
- **Reconstruction**: per call (up to 12 months x all sites), three steps:
  1. the GP of every goal and offshore partition, each in its own process
     (it only depends on the offshore sea states, not on the sites);
  2. the PCA inverse to the sites, threaded;
  3. the spectral sum in worker processes. Every site and hour is
     independent, so the sum is cut in small site x hour blocks (0.5-2 GiB
     per worker, whatever the period or number of sites); a small sum, such
     as one site, runs in the main process.
- **Cache**: results are stored per site and calendar month
  (`tmp/reconstruct/`, one store per setup); a request reads what is stored
  and computes only what is missing. Hours with no energy from any goal are
  NaN. The reconstruction proposal also prints the disk the spectra would
  take, so you can decide on `save_spectra` (~6 MB per site and month).

Timings on one 48-core node (Xeon Gold 5418Y, 36 cores used, shared with
other jobs), Netherlands case (10 goals), WHACS, **MDA 550, with water level
and wind correction**, nothing cached, October 2026:

| Task | Time | Peak memory |
| --- | --- | --- |
| 38 buoys, one year, bulk | 44 s | 53 GiB |
| 38 buoys, one year, spectral | 84 s | 76 GiB |
| 38 buoys, one year, spectral, spectra saved | 97 s | 77 GiB |
| One site, one year, bulk / spectral | 22 s / 41 s | 11 / 26 GiB |
| One site, 1979-2023 (45 years), bulk / spectral | ~17 min / ~31 min (extrapolated) | |
| 4,971 isobath sites, one month, bulk | 149 s | 52 GiB |
| One site and year read back from the cache | 1 s | 1 GiB |

Before these changes the same requests took 105 s (38 buoys, bulk), 173 s
(spectral) and 60 / 97 s (one site, bulk / spectral, per year); a bulk month
of 1,875 sites took ~11 min in one process. For the GP fits: 30 fits (MDA
50, 30 epochs) and their MDA took 16 s in 30 processes; one fit uses many
threads poorly (38 buoys, MDA 550, 100 epochs: 108 s on 1 thread, 58 s on
36). The proposal prints its own time estimate for each request (an order
of magnitude on other machines).

## Commands

```bash
python -m hywaves start --name MyCase   # start a new case
python -m hywaves validate MyCase/      # check the structure and data of the case
python -m hywaves publish MyCase/       # build its portal (MyCase/portal/, read by the GUI)
python -m hywaves snapshot MyCase/ v1   # keep the current portal under a tag
python -m hywaves update MyCase/        # bring the templates code (utils/) to the case
```

One file per command in this folder (`start.py`, `validate.py`,
`publish.py`, `snapshot.py`, `update.py`), plus `common.py` for what they
share; `template/` is what `start` copies. `--help` on any command explains
its options.

Tests (all in `tests/`; `tests/case_utils/` checks the template's `utils/`
inside a freshly started case):

```bash
python -m unittest discover -s hywaves/tests -t .
```
