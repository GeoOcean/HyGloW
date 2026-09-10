"""Hand-drawn, deliberately blocky continent regions for the sandbox-mode
world view — simple rectangles covering land AND water, not real coastlines
(see map_figure.build_sandbox_map for the actual coastline reference layer).
Purely a "which broad region am I looking at" backdrop, so rough edges and
minor overlaps between neighbors are expected and fine.
"""

from __future__ import annotations

WORLD_REGIONS: list[dict] = [
    {
        "name": "North America",
        "color": "#fb7185",
        "rings": [[[-180, 5], [-50, 5], [-50, 90], [-180, 90]]],
    },
    {
        "name": "South America",
        "color": "#fbbf24",
        "rings": [[[-93, -58], [-32, -58], [-32, 13], [-93, 13]]],
    },
    {
        "name": "Europe",
        "color": "#34d399",
        "rings": [[[-25, 34], [40, 34], [40, 90], [-25, 90]]],
    },
    {
        "name": "Africa",
        "color": "#60a5fa",
        "rings": [[[-20, -37], [52, -37], [52, 34], [-20, 34]]],
    },
    {
        "name": "Asia",
        "color": "#a78bfa",
        "rings": [[[40, -11], [180, -11], [180, 90], [40, 90]]],
    },
    {
        "name": "Oceania",
        "color": "#f472b6",
        "rings": [
            [[90, -50], [180, -50], [180, -11], [90, -11]],
            [[-180, -30], [-130, -30], [-130, 5], [-180, 5]],
        ],
    },
    {
        "name": "Antarctica",
        "color": "#94a3b8",
        "rings": [[[-180, -90], [180, -90], [180, -40], [-180, -40]]],
    },
]
