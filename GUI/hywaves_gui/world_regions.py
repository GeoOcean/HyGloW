"""Hand-drawn, deliberately blocky continent regions for the sandbox-mode
world view — simple rectangles covering land AND water, not real coastlines
(see map_figure.build_sandbox_map for the actual coastline reference layer).
Purely a "which broad region am I looking at" backdrop, so rough edges are
expected and fine — but the rings below are seamed against each other at
shared lon/lat edges (9, 34, -37, 66.5, -93, -50, ...) so the whole globe is
covered with no gaps; where two neighbors still overlap slightly (e.g. Asia
into the Indian Ocean south of it, or Oceania stretching into the empty
Southeast Pacific) that's a deliberate over-cover, never an empty patch.
"""

from __future__ import annotations

WORLD_REGIONS: list[dict] = [
    {
        "name": "North Pole",
        "color": "#7dd3fc",
        "rings": [[[-180, 66.5], [180, 66.5], [180, 90], [-180, 90]]],
    },
    {
        "name": "North America",
        "color": "#fb7185",
        "rings": [[[-180, 9], [-50, 9], [-50, 66.5], [-180, 66.5]]],
    },
    {
        "name": "South America",
        "color": "#fbbf24",
        "rings": [[[-93, -58], [-32, -58], [-32, 9], [-93, 9]]],
    },
    {
        "name": "Europe",
        "color": "#34d399",
        "rings": [[[-50, 34], [40, 34], [40, 66.5], [-50, 66.5]]],
    },
    {
        "name": "Africa",
        "color": "#60a5fa",
        "rings": [[[-50, -37], [52, -37], [52, 34], [-50, 34]]],
    },
    {
        "name": "Asia",
        "color": "#a78bfa",
        "rings": [[[40, -37], [180, -37], [180, 66.5], [40, 66.5]]],
    },
    {
        "name": "Oceania",
        "color": "#f472b6",
        "rings": [
            [[90, -50], [180, -50], [180, -11], [90, -11]],
            [[-180, -37], [-93, -37], [-93, 9], [-180, 9]],
        ],
    },
    {
        "name": "South Pole",
        "color": "#94a3b8",
        "rings": [[[-180, -90], [180, -90], [180, -37], [-180, -37]]],
    },
]
