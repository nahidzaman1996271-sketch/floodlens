import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
fig.patch.set_facecolor('#0B1D2A')
ax.set_facecolor('#0B1D2A')

ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)

# Title
ax.text(7, 6.4, 'FloodLens — End-to-End Prediction & Response Pipeline', 
        ha='center', va='center', color='#FFFFFF', fontsize=18, fontweight='bold', fontfamily='sans-serif')
ax.text(7, 6.0, 'Transforming raw hydrological sensor data into street-level survival decisions', 
        ha='center', va='center', color='#7C93A3', fontsize=11, fontfamily='sans-serif')

def draw_card(ax, x, y, w, h, title, items, color, border_color):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.1,rounding_size=0.15',
                                 facecolor=color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.35, title, ha='center', va='center', color='#FFFFFF', 
            fontsize=11.5, fontweight='bold')
    for i, item in enumerate(items):
        ax.text(x + 0.25, y + h - 0.85 - i*0.42, f'- {item}', ha='left', va='center', 
                color='#D3E0EA', fontsize=10)

# Stage 1: Data Ingestion
draw_card(ax, 0.5, 1.6, 3.6, 4.0, '1. Multi-Modal Ingestion', [
    'River Gauge History (BWDB)',
    'Hourly Monsoon Water Levels',
    '72h Rainfall Forecast Grid',
    'Topographic DEM (Elevation)',
    'Spatial Infrastructure (OSM)',
    'Buildings, Roads, Shelters'
], '#0F2436', '#2E7DA6')

# Stage 2: Hydrological & Geospatial ML
draw_card(ax, 4.8, 1.6, 4.4, 4.0, '2. AI & Hydrological Physics Engine', [
    "Holt Damped-Trend Forecast",
    "Dynamic Rainfall-Coupling Term",
    "Time-to-Danger & Severe Horizon",
    "Hydrologic Flood-Fill (BFS)",
    "Terrain-Constrained Propagation",
    "Building Submersion Countdown",
    "Road Passability Risk Index"
], '#0F2436', '#4FA9DB')

# Stage 3: Decision & Action
draw_card(ax, 9.9, 1.6, 3.6, 4.0, '3. Decision Layer & Action', [
    'Household Risk Stratification',
    'Nearest Safe Shelter Routing',
    'Claude LLM Plain Alerts',
    'Grounded Prompt (No Hallucinations)',
    'Bangla & English Localization',
    'FastAPI Endpoints (/predict)',
    'Live Monitor & Offline Fallback'
], '#0F2436', '#4FA381')

# Arrows
def draw_arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='#4FA9DB', edgecolor='#4FA9DB', width=2.5, headwidth=8, shrink=0.05))
    if label:
        ax.text((x1+x2)/2, y1 + 0.22, label, ha='center', va='center', color='#7C93A3', fontsize=9)

draw_arrow(ax, 4.1, 3.6, 4.8, 3.6, 'Raw Signals')
draw_arrow(ax, 9.2, 3.6, 9.9, 3.6, 'Impact Grid')

# Footer highlights
banner = patches.FancyBboxPatch((0.5, 0.4), 13.0, 0.85, boxstyle='round,pad=0.1,rounding_size=0.1',
                                facecolor='#12283A', edgecolor='#1F3B50', linewidth=1)
ax.add_patch(banner)
ax.text(0.9, 0.82, 'Key Differentiator:', color='#D9B34F', fontsize=11, fontweight='bold')
ax.text(2.8, 0.82, 'From abstract gauge readings ("water stage 6.5m") to hyper-local survival instructions ("house floods in 7h, move to shelter 2.4km now").', 
        color='#EAF0F2', fontsize=9.8)

plt.tight_layout()
plt.savefig('D:/cloudcamp hackathon/algorithm/floodlens_project/assets/system_architecture.png', 
            dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
print('Rendered system_architecture.png successfully')
