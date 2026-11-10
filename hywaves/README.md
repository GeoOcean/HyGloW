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
direction, directional spreading and water level. Hs and Tp are sampled
independently, so the design also holds seas that do not occur (a 9 m sea at
4 s): samples steeper than `simulations.max_steepness` (Hs / L0, with the
deep-water wavelength L0 = g Tp² / 2π; 0.06, beyond which waves break by
whitecapping) are dropped, not resampled, so a goal keeps fewer cases than
sampled. The limit is a minimum period per height (at 0.06: 3.3 s for 1 m,
7.3 s for 5 m, 10.3 s for 10 m). Each run gives Hs, Tp, direction and
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
  would grow its own copy of the local sea), so the sum leaves out the sea the
  local wind grows over the domain. The dynamic runs can be forced with wind
  (ERA5) to show how much that is, or with no boundary waves to give the sea the
  wind alone grows. Not in the reconstruction yet; the way found (a wind-only
  term damped by the share of the sea already there), what was tried and the
  numbers are in [`WIND.md`](WIND.md).
- *Water level.* A metamodel input: constant, or a tide and surge series per
  goal.
- *Offshore bias.* The hindcast can be calibrated against satellite
  altimetry before reconstructing (notebook 07b), in two steps. First, per
  goal, the level: each partition Hs is multiplied by a factor for its family
  (wind sea or swell) and the 22.5° sector of its own direction, and the bulk
  Hs follows the change in partition energy. Then, over the passes of all
  goals, the storm tail: one quantile map of the bulk Hs (a constant offset
  beyond p99.9), whose ratio map(Hs)/Hs multiplies the bulk and every
  partition, so the sea/swell shares of the first step are kept. The
  altimeter only sees bulk Hs: how the tail correction is shared among the
  partitions is a choice, not data. In the Netherlands WHACS record
  (1979-2023) it hardly matters: swell holds at most 0.6 % of the energy in
  hours above 4 m, and putting the extra energy in the wind sea alone would
  change swell Hs by less than 3 cm (p90). Check it again where storms can
  be swell. Deep buoys near the goals, never used in the fit, are the
  independent check.
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
- **Reconstruction**: in calls of up to 12 months x all sites; the GP of
  every goal runs in its own process, then the spectra of all goals are
  summed per site and hour in worker processes. Results are cached per site
  and month (`tmp/reconstruct/`), so a request only computes what is missing.
  Details and open points: [`RECONSTRUCTION.md`](RECONSTRUCTION.md).

Timings, Netherlands (10 goals, 38 buoys), WHACS, MDA 550, with water level,
on one 48-core node (36 cores used, shared), nothing cached, October 2026:

| Task | Time | Peak memory |
| --- | --- | --- |
| 38 buoys, one year, bulk / spectral | 44 s / 84 s | 53 / 76 GiB |
| 38 buoys, 1979-2023, spectral, spectra saved | ~73 min, ~90 GB of spectra | 77 GiB |
| One buoy, 1980-2023, bulk | 14 min | 11 GiB |
| One buoy and year read back from the cache | 1 s | 1 GiB |

Fits: 30 fits (MDA 50, 30 epochs) and their MDA take 16 s in 30 processes.
The proposal prints its own time estimate for each request (an order of
magnitude on other machines).

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
