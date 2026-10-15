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

## Commands

```bash
python -m hywaves start --name MyCase        # a new case from template/
python -m hywaves validate MyCase/ [--publish]
python -m hywaves publish MyCase/            # MyCase/portal/, read by the GUI
python -m hywaves snapshot MyCase/ <tag>     # keep this portal as a version
python -m hywaves update MyCase/             # bring the case's utils/ to the template
```

`python -m hywaves <command> --help` explains each one.
