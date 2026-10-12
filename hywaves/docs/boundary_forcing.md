# Boundary forcing: how the goals are spread along the open boundary

The goals are points on the open boundary, but the models force a whole
boundary. This page describes how each model fills the stretch between
goals, and why both are consistent with the linear summation of the
reconstruction.

## The rule that matters

The reconstruction adds the goal contributions **as energy** (spectra). It
matches the dynamic runs (every goal forced at once) only if, at every
boundary point,

1. each metamodel case imposes a **fraction w(goal, point) of its goal's
   energy**, and
2. the fractions of all goals add up to 1.

A dynamic run then imposes `Σ w·E_goal` at that point, which is exactly the
sum of the metamodel contributions. Both models follow this rule; they use
different weights `w`.

## SnapWave

`snapwave_boundaries.f90` (`find_boundary_indices`, `update_boundaries`):
every open-boundary mesh node takes the **two nearest goals** (straight-line
distance, `boundary.txt` points) and mixes their directional **spectra** with
inverse-distance weights `w1 = d2 / (d1 + d2)`, `w2 = 1 − w1`.

- Between two goals the weight falls almost linearly along the boundary.
- Beyond the first and last goal, the two nearest goals are still the last
  two, so the energy decays towards the boundary end (a corner far from goal
  1 still gets part of goal 2's energy).
- The spectra are mixed as they are, so two goals with different directions
  give a bimodal spectrum in between.

## SWAN

`BOUNDSPEC SIDE 1 CW|CCW VARIABLE PAR` with one row per goal
(`swan_utils.boundary_par_table`). SWAN builds a JONSWAP spectrum per row
and, for each boundary vertex between two rows, **interpolates the spectra**
linearly in the distance along the boundary (`SINTRP` in `swanser.f`):

- in a metamodel case the neighbouring rows have Hs 0, and SWAN then takes
  `w · spectrum` exactly, so the **energy is linear** and Hs decays as
  `√w`: halfway between the goal and its neighbour Hs is `H / √2`, not `H/2`;
- in a dynamic run both rows have energy: SWAN keeps the energy linear
  (`w1·E1 + w2·E2`) but shifts both spectra to the interpolated mean
  direction and frequency ("peakedness maintained"), so the result is one
  peak, not SnapWave's two;
- the boundary ends: `boundary_par_table` repeats the first goal at the start
  of the boundary and the last goal at its end, so the stretch between a
  boundary end and its nearest goal takes **that goal's forcing in full**
  (SWAN would leave it unforced otherwise).

`swan_utils.boundary_hs_profile(open_xy, rows)` returns the Hs SWAN imposes
at each boundary vertex, and `swan_plots.plot_case` colours the open boundary
with it.

## Comparing the two

| | SnapWave | SWAN |
| --- | --- | --- |
| Weights | two nearest goals, inverse distance | neighbouring rows, linear in distance along the boundary |
| Interpolated quantity | spectra (energy) | spectra (energy) |
| Two goals with different directions (dynamic) | bimodal mix | one peak at the mean direction |
| Boundary ends | decay, mixed with the second goal | the nearest goal, in full |
| Consistent with the linear summation | yes (energy) | yes (energy) |

The direction difference only matters in the dynamic reference (09), and
only between goals with different sea states. The boundary-end difference
changes what each goal's cases see near the corners of the domain, so the
same goal gives slightly different nearshore fields in the two models there.

## Checked on Netherlands case 01/0000 (2026-10-12)

Goal 1, Hs 7.92 m, Tp 13.9 s, dir 262°, spr 21°, wl −2.30 m, SWAN 41.51,
field output on the grid:

- from the start of the boundary to goal 1 (the southern edge): Hs 7.92–7.94 m
  at the boundary vertices, the full goal forcing;
- from goal 1 to goal 2: SWAN's boundary Hs is 1.02 × the energy-linear
  profile (median; 10th–90th percentile 1.016–1.044), against 1.37 × a
  Hs-linear one;
- beyond goal 2 the imposed energy is 0, but the boundary vertices still
  show up to 1 m near goal 2, falling to 0 by goal 10: waves from goal 1
  travelling along the boundary and leaving through it (SWAN only imposes
  the directions that point into the domain).
