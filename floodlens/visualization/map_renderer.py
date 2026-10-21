"""
Renders the flood extent + assets as a PNG heatmap — the visual centerpiece
for a demo. No basemap/tile dependency (keeps this offline-runnable);
just DEM shading + flood overlay + building/road/shelter markers.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def render_flood_map(
    dem: np.ndarray,
    flooded_grid: np.ndarray,
    buildings,
    roads: list,
    shelters,
    hour_label: str,
    out_path: str,
):
    fig, ax = plt.subplots(figsize=(8, 8))

    terrain_cmap = LinearSegmentedColormap.from_list(
        "terrain_dry", ["#dff0d8", "#a8d08d", "#6fa85a"]
    )
    ax.imshow(dem, cmap=terrain_cmap, origin="upper")

    flood_overlay = np.ma.masked_where(~flooded_grid, np.ones_like(dem))
    ax.imshow(flood_overlay, cmap=LinearSegmentedColormap.from_list("flood", ["#3fa9f5", "#0b3d91"]),
              origin="upper", alpha=0.65)

    if buildings is not None and len(buildings):
        risk_colors = {"CRITICAL": "#d90429", "HIGH": "#f77f00", "WATCH": "#fcbf49", "SAFE": "#606c38"}
        for _, b in buildings.iterrows():
            color = risk_colors.get(getattr(b, "risk_level", "SAFE"), "#606c38") \
                if hasattr(b, "risk_level") else "#606c38"
            ax.plot(b["col"], b["row"], marker="s", markersize=2, color=color)

    for road in roads:
        pts = road["points"]
        rows_ = [p[0] for p in pts]
        cols_ = [p[1] for p in pts]
        ax.plot(cols_, rows_, color="#4a4e69", linewidth=1.5, linestyle="--", alpha=0.8)

    if shelters is not None and len(shelters):
        for _, s in shelters.iterrows():
            ax.plot(s["col"], s["row"], marker="*", markersize=16, color="#ffd60a",
                     markeredgecolor="black", markeredgewidth=0.5)

    ax.set_title(f"FloodLens — Predicted Flood Extent ({hour_label})", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_impact_table_png(building_impacts, out_path: str, top_n: int = 15):
    """A simple table image of the most urgent buildings — useful as a
    second demo artifact alongside the map."""
    critical = sorted(
        [b for b in building_impacts if b.hours_to_flood is not None],
        key=lambda b: b.hours_to_flood,
    )[:top_n]

    fig, ax = plt.subplots(figsize=(11, 0.35 * len(critical) + 0.5))
    ax.axis("off")
    # Shorten shelter names for the table (full name is in the JSON export).
    def _short_shelter(name):
        if not name:
            return "-"
        return name.replace("Char Bandar Primary School Shelter", "Primary School Shelter") \
                    .replace("Union Cyclone Shelter #12", "Cyclone Shelter #12")

    rows = [[b.id, b.type, f"{b.hours_to_flood}h", b.risk_level,
             _short_shelter(b.nearest_shelter_name), f"{b.distance_to_shelter_km} km"]
            for b in critical]
    table = ax.table(
        cellText=rows,
        colLabels=["ID", "Type", "Time to Flood", "Risk", "Nearest Shelter", "Distance"],
        loc="upper left", cellLoc="left",
        colWidths=[0.10, 0.13, 0.15, 0.13, 0.32, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path
