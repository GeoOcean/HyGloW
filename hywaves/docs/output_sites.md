# Output sites: on the mesh, always wet, then the site filter

Code: bluemath_tk `topo_bathy/mesh.py` (`read_mesh`, `depth_at_points`,
`mesh_to_raster`, `mask_raster_to_mesh`), `waves/hywaves/output_points.py`
(`site_status`), `waves/hywaves/metamodel.py` (`mask_zero_energy`,
`detect_removed_sites`, `drop_sites`, `fill_zero_energy`); case layer
`utils/geometry.py` (`compute_output_sites`, `output_sites`,
`topobathy_from_mesh`) and `utils/metamodel.py` (`load_cases_goal`).

## One rule: a site is inside the mesh and never dry

Sites are decided **once, before any run**, on the mesh the model runs on:

1. **Isobaths** are traced on `paths.topobathy_file` after blanking it outside
   the model mesh (`mask_raster_to_mesh`). Smoothing respects the blanked
   cells, so contours stop at the mesh edge. A case without a tiff builds one
   from its grid: `utils.geometry.topobathy_from_mesh()`.
2. **Every site** (isobath, mesh polygon node, buoy) gets its `mesh_depth`,
   linear inside the triangles (`depth_at_points`; quads are split): the depth
   the models see and the one used from here on. The contour level, or the
   buoy `measured_depth` of the observations GeoJSON, stays as `nominal_depth`.
3. A site is `ok` only if `mesh_depth + wl_min >= min_depth`, with `wl_min` the
   lowest metamodel water level (`simulations.bounds.wl[0]`) and `min_depth`
   the shallowest depth the models compute (`simulations.min_depth`, default
   0.1, rendered into SnapWave `hmin` and SWAN `DEPMIN`). Outside the mesh the
   depth is NaN, so the same test drops it (`status`: `ok`, `outside_mesh`,
   `can_dry`).
4. Notebook 03 builds the table after the observations, shows
   `sites_summary` (per group: kept, dropped, mesh-depth spread),
   `buoy_depth_check` (measured against mesh depth, flags large differences)
   and the maps `plot_output_sites` / `plot_site_depths`, and saves it, dropped
   sites included, as `outputs/output_sites_{model}.csv` (first line
   `# grid: <grid_file>`). Every later step reads its `ok` sites
   (`output_sites()`); a table built on another grid is refused. Rebuild with `output_sites(recompute=True)`,
   which invalidates runs made with the old sites.

Smoothing (`output_sites_config.isobaths[].smooth_m`) trades homogeneous depths
for smooth lines. Netherlands (5 km spacing): `smooth_m` 1000 → 500 → 250 m
gives a median |mesh − nominal| on the 20 m isobath of 0.48 → 0.24 → 0.08 m
and 352 → 506 → 649 sites (longer, wigglier contours).

Polygons (`output_sites_config.polygons[]`: `name`, `file`) take every
model-mesh node inside them as a site (`{name}_{node}`). Optional `fraction`
(and `seed`) keeps about that share of the always-wet nodes, one random node
per equal-area cell (`spatial_subsample`), so the subset covers the whole
polygon even where the mesh is much finer. Netherlands Den Haag: 1414 wet
nodes → 283 sites at `fraction` 0.2 (nearest neighbour 0.57 km median).

Mesh formats (`read_mesh`): UGRID `.nc` with triangles, quads or a mix (fill
values for missing nodes, `start_index` honoured), and ADCIRC `fort.14` /
`.grd`. Elevation is positive up (as GEBCO), site `depth` positive down.

## What `hs = 0` means

Because every site is wet in every case, `hs = 0` is a result: this goal sends
no energy to that site in that case (e.g. waves from the land side of the goal
sector). SWAN writes exact zeros there, SnapWave tiny values. The other fields
are meaningless (SWAN NaN, SnapWave placeholders), so:

- `mask_zero_energy`: where `hs = 0`, `tp`/`dir`/`spr` become NaN, `hs` stays 0;
- `fill_zero_energy` (training only): those fields take the case forcing, a
  smooth function of the GP inputs; it does not matter in the reconstruction,
  where `hs = 0` adds no energy;
- the holdout reference (`fill_zero=False`) compares `hs = 0` and skips the
  other fields there.

## The site filter

`metamodel.site_filter` rules, checked per goal on the masked cases
(`detect_removed_sites`); the removed sites go to `split.json`
`removed_sites` and the reconstruction keeps the sites kept in **every** goal.

| Rule | Drops a site when | Example | Catches |
| --- | --- | --- | --- |
| `{var}_max` / `{var}_min` | any case is above / below | `"tp_min": 0.5` | placeholder periods |
| `{var}_ratio_max` | `var / {var}_forcing` exceeds it in any case | `"hs_ratio_max": 3` | numerical blow-ups |
| `{var}_nan_max` | the fraction of cases with `var` NaN exceeds it | `"hs_nan_max": 0.01` | model failures |

The bound rules skip NaN. With sites on the mesh and always wet, `hs_nan_max`
should remove almost nothing; if it does, look at those sites (a node of the
model can still be shallower than the interpolated depth at the site).

## History

hywaves 0.12.0 treated `hs = 0` as dry and counted it in `hs_nan_max`. With
SWAN, whose shadow zeros are exact, each goal removed 82–297 of the 407
Netherlands sites and only 14 were left in every goal. 0.14.0 moved the dry
decision to the geometry, before the runs.

## Inspecting

```python
from utils.geometry import compute_output_sites, output_sites
from utils import metamodel as mm

output_sites(all_sites=True)              # saved table, dropped sites included
compute_output_sites()                    # rebuild in memory (try other smooth_m)
mm.removed_sites(goal)                    # from split.json
mm.load_cases_goal(goal, apply_site_filter=False)  # masked cases, before the filter
```
