# TODO

Open work across HyGloW (state on 2026-11-16; details in `CLAUDE.md`, "Where we are").

## Method

- [ ] Breaking near the coast: try different ceilings for the depth cap (per site depth band,
      per goal, from the dynamic runs instead of the training cases); it barely acts in Samoa
      and the linear sum is still high at 5 m (some 10 m).
- [ ] Spectral mode: Tp, Dir and Spr are poor against the dynamic runs and the buoys (bulk
      mode tracks direction better); find where the spectral sum loses them.
- [ ] Storms under the buoys at their peak (Cyclone Gita: every estimate well under the buoy,
      including SWAN): the forcing, no local wind. Revisit with the damped wind-only term
      (`hywaves/WIND.md`) if it is built.
- [ ] SnapWave on a closed ring forces the whole spectrum on every boundary node, also where
      the waves leave the domain (SWAN only imposes the incoming directions): lee-side
      boundaries keep the offshore Hs (`Samoa/numerical_tests/README.md`). Decide whether it
      matters and add it to the upstream issue.
- [ ] Satellite calibration step 1 overshoots moderate seas (open since 07b).

## Cases

- [ ] **Samoa**: check 08b-12, publish `portal/` (snapshot the old one first), commit it.
- [ ] Samoa: score King Poloa (2024 on, after WHACS and CODEC) with ERA5 and `wl=0`.
- [ ] **Netherlands**: re-run with the Samoa setup (05-06: sector 360 and the spread bounds;
      08 with wind2); `hywaves update` + the `tm01` config keys + the template 08b; a mean
      period for `tm01` in `utils/buoy_archive.py` (check what Waterinfo/CMEMS give); rebuild
      the dashboard store.
- [ ] Netherlands: SWAN pass (05-12 with `model: swan`, `tp` as PCA target, snapshot `swan`;
      `Netherlands.json` is on `model: swan` now).
- [ ] Netherlands: buoy spectra for notebook 11.
- [ ] Netherlands: decide from the dashboard what moves into 09-12.
- [ ] **NorthCarolina**: check the first run (chain launched 2026-11-16); copy the template
      08b with the next `update` (the old 08b calls removed functions); compare with the
      delivered daily SnapWave run; CDIP spectra for its CDIP buoys; ERA5 needs `EDH_API_KEY`.
- [ ] **NewSouthWales**: check its first 0.17 run (queued after North Carolina), copy the
      template 08b with the next `update`, publish.

## Code and repos

- [ ] Send the SnapWave upstream issue
      (`/lustre/geocean/WORK/users/javiert/software/snapwave/upstream_issue.md`).
- [ ] bluemath_tk: commit `waves/cdip.py` (`tm01`); merge `feature/hywaves-depth-cap` into
      `develop`.
- [ ] HyGloW: PR to `main` with the 0.17 work, the case migrations and 08b.

## Paper

- [ ] Sketch for the paper: method (MDA -> PCA -> GP per goal, linear sum of goal spectra),
      cases (Netherlands, Samoa, North Carolina, New South Wales), validation against dynamic
      runs and buoys, the boundary-forcing lesson (one direction vs per point, checked with
      SWAN), timings, limits (breaking, wind, spectral shape).
