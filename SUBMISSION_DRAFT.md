# 🌊 FloodLens — CloudCamp Bangladesh Submission Draft

> **Event**: CloudCamp Bangladesh — International AI Builders Congress  
> **Phase**: "Idea to Unicorn" — Preliminary Submission  
> **Team Name**: Candy Crush  
> **Team Members**: Nahid Ibn Zaman, Farhan Ishraq Ifti, Tahmid Rashid Pranjol, Mahmuda Khanum  

---

## 1. STORY (~250–300 words)
*(Copy-paste into CloudCamp portal "Story" section)*

**The Problem:**  
Bangladesh's flood early-warning systems tell communities an abstract river gauge number — "water level may rise" or "danger stage reached at 6.5 meters." That information is accurate for hydrologists, but completely useless for a family living in a rural haor or floodplain basin. At 2:00 AM, a household doesn't know whether they have 45 minutes or 12 hours before floodwaters enter their home, whether the village access road is already submerged, or where the nearest dry shelter actually is. The tragic result is warnings that are heard, but not acted upon until it is too late.

**Our Approach:**  
FloodLens transforms raw flood data into a street-level personal survival plan. Instead of "danger stage reached," each household or community health worker receives an actionable, plain-language instruction: *"Flooding predicted here in 6 hours. Village Access Road 1 is impassable. Nearest safe refuge: Char Bandar Primary School Shelter (2.07 km away via Upazila Connector Road). Move medicine, documents, and livestock now."*

We achieved this by uniting three distinct technologies:
1. **Time-series forecasting** of the river’s rate-of-rise using Holt’s damped-trend exponential smoothing coupled with rainfall forecasts;
2. **Hydrologically-connected 2D terrain propagation** using breadth-first search flood-fill over digital elevation models (DEM) to determine exact street-level inundation timing;
3. **A grounded Claude LLM alert layer** that converts calculated hazard metrics into plain-language bilingual instructions (Bangla & English) without hallucinating numbers.

**Status:**  
A fully functional, end-to-end prototype tested with 24 passing integration tests, complete with a REST API, real-time live monitoring dashboard, and clear integration paths for live BWDB gauge feeds and open SRTM terrain data.

---

## 2. PROOF & VERIFIABLE METRICS
*(Copy-paste into CloudCamp portal "Proof" section)*

* **Live Interactive Web App**: https://nahidzaman1996271-sketch.github.io/floodlens/ (Interactive 72h flood progression scrub slider, Leaflet GIS mapping, road access table, and bilingual alerts).
* **Live Cloud Backend API (Render)**: https://floodlens-api-vq31.onrender.com (FastAPI microservice with live endpoints `/predict`, `/alerts`, `/scenario/summary`).
* **Live Interactive Swagger API Docs**: https://floodlens-api-vq31.onrender.com/docs
* **GitHub Repository**: https://github.com/nahidzaman1996271-sketch/floodlens (Full source code, 24 unit/integration tests, assets, and live interactive monitor).
* **Commit Code / Submission SHA**: `f23be2c` (Full audit trail, chronological commit hashes, and timestamps documented in README).
* **Test Suite Verification**: 24/24 tests passing (`pytest tests/ -v`). Verifies forecast monotonicity, physical elevation sanity, shelter isolation on high ground, and API contracts.
* **Concrete Impact Figures (from `results/prediction_export.json`)**:
  * Locality: Char Bandar Union (synthetic haor-floodplain profile, 60×60 elevation grid)
  * Current Gauge Level: 5.08 m | Danger Level: 6.5 m | Severe Level: 8.0 m
  * **Hours to Danger Threshold**: 6.0 hours
  * **Hours to Severe Threshold**: 13.0 hours
  * **Total Buildings Monitored**: 102
  * **Buildings at Risk**: 91 (89.2%)
  * **Critical Category (< 10h submersion)**: 89 buildings
  * **Road Infrastructure Impact**: 2 of 3 roads impassable within 1 hour; 1 main connector road remains passable for 13 hours.
* **Visual Artifacts Attached**:
  * `assets/hero_mockup.png` — Mobile survival plan UI showing shelter routing and prioritized evacuation.
  * `assets/flood_map_hour_1.png`, `assets/flood_map_hour_25.png`, `assets/flood_map_hour_72.png` — Visual wavefront progression from baseline to severe inundation.
  * `assets/impact_table_top15.png` — Household-by-household risk ranking and shelter triage.
  * `assets/system_architecture.png` — Complete end-to-end architecture pipeline diagram.

---

## 3. AI-NATIVE BUILD
*(Copy-paste into CloudCamp portal "AI-Native Build" section)*

* **Constrained LLM Agent (`floodlens/alerts/claude_alert.py`)**:
  * We engineered a strictly bounded system prompt with Anthropic's Claude 3.5 Sonnet.
  * The prompt enforces zero hallucination: the model is prohibited from inventing distances, timings, or coordinates. It strictly phrases pre-computed numerical outputs into urgent, empathetic, human instructions.
* **Fail-Safe Reliability Design**:
  * In disaster response, an API outage must not silence warnings. We designed a deterministic local template fallback that triggers automatically if network connection or API limits drop.
* **AI-Assisted Architecture**:
  * The water-level forecasting module adapts a time-to-failure modeling pattern originally proven in enterprise GPU cluster telemetry, demonstrating cross-domain AI-native engineering.

---

## 4. DATA LIFECYCLE
*(Copy-paste into CloudCamp portal "Data Lifecycle" section)*

* **Current Prototype Phase**:
  * Evaluated on hydrologically realistic synthetic profiles (`floodlens/data/synthetic.py`) calibrated to real Bangladesh Water Development Board (BWDB) monsoon gauge patterns and haor topography.
* **Production Integration Roadmap**:
  1. **BWDB Real-Time Telemetry API**: Stream hourly sensor gauge readings across 100+ national river monitoring stations.
  2. **Copernicus & SRTM 30m Digital Elevation Models (DEM)**: Ingest satellite elevation rasters for Bangladesh's high-risk flood zones (Sylhet, Sunamganj, Kurigram, Sirajganj).
  3. **OpenStreetMap (OSM) Footprints**: Ingest building boundaries, school/cyclone shelter locations, and rural road classifications.
  4. **Bangladesh Meteorological Department (BMD) Forecasts**: Direct ingestion of numerical weather prediction (NWP) rainfall accumulation grids.

---

## 5. PROMPT LIBRARY
*(Copy-paste into CloudCamp portal "Prompt Library" section)*

### Primary Alert System Prompt (`ALERT_SYSTEM_PROMPT`)

```text
You are FloodLens Alert, an emergency notification assistant for Bangladesh flood response.
Your job is to take structured building-level flood predictions and generate ONE short, urgent,
plain-language alert (2-3 sentences max) for the resident or a community health worker.

Rules:
1. State clearly when flooding is expected (in hours).
2. State the risk level (CRITICAL, HIGH, WATCH).
3. Name the nearest safe shelter and distance in km.
4. Give 1-2 prioritized, realistic actions based on building type (e.g., move medicine first for clinics/elderly,
   unlash livestock, move dry food/documents to high ground).
5. Do NOT invent numbers, distances, or shelter names not present in the input.
6. Tone: calm, direct, actionable, urgent without being hysterical.
```

---

## 6. VIDEO PITCH SCRIPT & YOUTUBE NOTES
*(From `FloodLens_Final_Script.docx` & `youtube tricks.txt`)*

* **Length**: ~3:00 minutes
* **Speaker**: Nahid Ibn Zaman (with team Candy Crush: Farhan Ishraq Ifti, Tahmid Rashid Pranjol, Mahmuda Khanum)
* **Outline**:
  * `[0:00 - 0:08]` Introduction to team & CloudCamp Hackathon.
  * `[0:08 - 0:35]` The Problem: Bangladesh monsoon flood warnings lack street-level action.
  * `[0:35 - 1:05]` The Solution: Turning raw flood data into personal survival plans.
  * `[1:05 - 2:05]` Demo & Flow: Multi-model architecture, DEM flood propagation, Claude alert generation.
  * `[2:05 - 2:45]` Impact: Piloting with local disaster management offices.
  * `[2:45 - 3:00]` Sign-off: Link to GitHub repo, slide deck, and live monitor.
