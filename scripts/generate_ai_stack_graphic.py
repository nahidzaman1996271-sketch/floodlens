"""
Renders assets/ai_stack_architecture.png: the FloodLens AI stack after the
DuckDB, local-LLM (Ollama), RAG / Graph RAG, n8n and CI additions.
Same visual style as generate_architecture_graphic.py.

    python scripts/generate_ai_stack_graphic.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "ai_stack_architecture.png")

BG = "#0B1D2A"
CARD = "#0F2436"
TEXT = "#D3E0EA"
MUTED = "#7C93A3"
BLUE, TEAL, AMBER, PURPLE, GREEN = "#2E7DA6", "#1FA39A", "#E0A030", "#8B6CD9", "#4CAF6A"

fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.axis("off")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)

ax.text(8, 8.55, "FloodLens — AI Stack: Forecast → DuckDB → RAG / Graph RAG → Local-first Alerts",
        ha="center", va="center", color="#FFFFFF", fontsize=17, fontweight="bold")
ax.text(8, 8.15, "Runs fully offline on a laptop (Ollama on a 4 GB GTX 1650); uses Claude when an API key is available",
        ha="center", va="center", color=MUTED, fontsize=11)


def card(x, y, w, h, title, items, border, fs=9.6, step=0.37):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.14",
                                        facecolor=CARD, edgecolor=border, linewidth=1.8))
    ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="center", color="#FFFFFF",
            fontsize=11, fontweight="bold")
    for i, item in enumerate(items):
        ax.text(x + 0.17, y + h - 0.75 - i * step, "• " + item, ha="left", va="center",
                color=TEXT, fontsize=fs)


def arrow(x1, y1, x2, y2, color=MUTED, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.17, text, ha="center", va="center",
                color=color, fontsize=8.3, style="italic")


# ---- top row: the four stages ------------------------------------------------
TOP_Y, TOP_H = 4.2, 3.6
card(0.3, TOP_Y, 3.55, TOP_H, "1. Forecast pipeline", [
    "River gauge history (BWDB-style)",
    "72 h rainfall forecast",
    "Holt damped-trend level forecast",
    "DEM flood-fill, hour by hour",
    "Building & road impact (hours)",
    "Nearest-shelter routing",
    "Deterministic, auditable numbers",
], BLUE)

card(4.2, TOP_Y, 3.55, TOP_H, "2. DuckDB analytical store", [
    "results/floodlens.duckdb",
    "runs, gauge_history",
    "rainfall / level forecasts",
    "building_impacts, road_impacts",
    "rag_chunks: text + context",
    "embedding FLOAT[768] array",
    "array_cosine_similarity search",
], TEAL)

card(8.1, TOP_Y, 3.8, TOP_H, "3. RAG + Graph RAG", [
    "6 flood-guidance docs → 30 chunks",
    "Variable / semantic chunking",
    "Contextual retrieval (LLM context)",
    "nomic-embed-text embeddings",
    "Hybrid: vector + BM25 (RRF)",
    "HyDE · LLM reranker · CRAG",
    "Knowledge graph 121 nodes/419 edges",
], PURPLE, fs=9.3)

card(12.25, TOP_Y, 3.45, TOP_H, "4. Bilingual alerts (EN+BN)", [
    "① Claude API (if key set)",
    "② Ollama local: qwen2.5:3b",
    "    (llama3.2 · gemma2 · phi3.5)",
    "③ llama.cpp / LM Studio",
    "    (OpenAI-compatible)",
    "④ Deterministic template",
    "Grounded by graph facts + RAG",
], AMBER, fs=9.3)

mid = TOP_Y + TOP_H / 2
arrow(3.9, mid, 4.15, mid, BLUE)
arrow(7.8, mid, 8.05, mid, TEAL)
arrow(11.95, mid, 12.2, mid, PURPLE)

# ---- bottom row: delivery & automation ---------------------------------------
BOT_Y, BOT_H = 0.75, 2.2
card(0.3, BOT_Y, 3.0, BOT_H, "FastAPI on Render", [
    "/predict  /alerts",
    "/ask  (RAG answers)",
    "/buildings/{id}/context",
], GREEN, step=0.4)
card(3.55, BOT_Y, 3.0, BOT_H, "MCP server", [
    "7 read-only tools, stdio",
    "Claude Code / Desktop",
    "same cached pipeline",
], GREEN, step=0.4)
card(6.8, BOT_Y, 3.0, BOT_H, "n8n hourly workflow", [
    "GET /predict → danger ≤ 12 h?",
    "GET /alerts → EN+BN digest",
    "→ Slack / Discord / Telegram",
], GREEN, step=0.4)
card(10.05, BOT_Y, 2.8, BOT_H, "GitHub Actions", [
    "pytest on every push",
    "forecast + RAG eval",
    "every 6 h → artifacts",
], GREEN, step=0.4)
card(13.1, BOT_Y, 2.6, BOT_H, "Web dashboard", [
    "GitHub Pages",
    "72 h map slider",
    "Leaflet GIS view",
], GREEN, step=0.4)

ax.plot([0.5, 15.5], [3.22, 3.22], color="#1E3A4F", lw=1.2)
ax.text(8, 3.32, "Delivery & automation", ha="center", va="bottom", color=MUTED, fontsize=9.5)

ax.text(8, 0.32, "55 automated tests  ·  RAG eval: contextual hybrid hit@1 = 1.00, all configs hit@3 = 1.00  ·  "
        "3/3 alerts drafted by a local model  ·  all safety numbers computed, never generated",
        ha="center", va="center", color=MUTED, fontsize=9.5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.savefig(OUT, facecolor=BG, bbox_inches="tight", pad_inches=0.2)
print(f"Wrote {OUT}")
