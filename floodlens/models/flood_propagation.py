"""
Flood propagation — turns a predicted river stage (meters) into a spatial
flood extent grid.

Method: hydrologically-connected flood fill (BFS from the river channel),
NOT a naive "everything below elevation X is flooded" threshold. The
naive approach wrongly floods isolated low-elevation depressions that
have no water path from the river. Flood-filling only cells reachable
from the river channel through non-rising terrain is a standard
simplification used in first-pass flood-extent modeling and is honest
about being a simplification (no infiltration, drainage, or levee-breach
dynamics) — those are called out explicitly in the README as roadmap
items for a real satellite/hydrological-model version.
"""

from __future__ import annotations

from collections import deque
import numpy as np


def flood_extent_for_level(
    dem: np.ndarray,
    river_mask: np.ndarray,
    river_level_m: float,
    river_baseline_m: float,
) -> np.ndarray:
    """
    Returns a boolean grid: True where the cell is flooded when the river
    reaches river_level_m.

    The river channel itself sits at dem == -0.5 (below plain level). A
    cell floods if its elevation is at or below the *rise above baseline*
    the river has achieved, AND it is reachable from the river without
    crossing a higher cell (hydrological connectivity).
    """
    rows, cols = dem.shape
    rise_above_baseline = max(river_level_m - river_baseline_m, 0.0)

    # Effective flood surface elevation across the plain.
    flood_surface = rise_above_baseline

    flooded = np.zeros((rows, cols), dtype=bool)
    visited = np.zeros((rows, cols), dtype=bool)
    q = deque()

    river_cells = np.argwhere(river_mask)
    for r, c in river_cells:
        visited[r, c] = True
        flooded[r, c] = True
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                if dem[nr, nc] <= flood_surface:
                    flooded[nr, nc] = True
                    q.append((nr, nc))

    return flooded


def flood_timeline(
    dem: np.ndarray,
    river_mask: np.ndarray,
    forecast_df,
    river_baseline_m: float,
    sample_every_n_hours: int = 3,
) -> dict[int, np.ndarray]:
    """Compute flood extent grids at sampled hours across the forecast
    horizon, so downstream code can determine *when* each cell/asset
    first floods without recomputing per-hour for every asset."""
    timeline = {}
    for i in range(0, len(forecast_df), sample_every_n_hours):
        hour = i + 1
        level = float(forecast_df["predicted_level_m"].iloc[i])
        timeline[hour] = flood_extent_for_level(dem, river_mask, level, river_baseline_m)
    # Always include the final hour too.
    last_hour = len(forecast_df)
    if last_hour not in timeline:
        level = float(forecast_df["predicted_level_m"].iloc[-1])
        timeline[last_hour] = flood_extent_for_level(dem, river_mask, level, river_baseline_m)
    return timeline
