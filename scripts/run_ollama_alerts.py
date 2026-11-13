"""
Generates FloodLens bilingual (EN + BN) alerts with a LOCAL open-weight model
served by Ollama -- no cloud API, no internet needed once the model is pulled.

Setup:
    ollama pull qwen2.5:3b          # or llama3.2:3b, gemma2:2b, ...
    ollama serve                    # if not already running

Run:
    python scripts/run_ollama_alerts.py                 # uses qwen2.5:3b
    OLLAMA_MODEL=llama3.2:3b python scripts/run_ollama_alerts.py

Writes results/ollama_alerts_sample.json with the model name, latency and the
generated alerts, as proof of the local-LLM path.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import urllib.request
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floodlens.pipeline import run_pipeline
from floodlens.alerts.claude_alert import (
    generate_alerts_for_critical, OLLAMA_DEFAULT_HOST, OLLAMA_DEFAULT_MODEL,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(limit: int = 3):
    host = os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST)
    if not host.startswith("http"):
        host = "http://" + host
    model = os.environ.get("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)

    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/version", timeout=5) as r:
            version = json.loads(r.read().decode()).get("version", "unknown")
    except Exception as exc:
        print(f"Ollama is not reachable at {host} ({exc}). Start it with `ollama serve`.")
        sys.exit(1)

    print(f"Ollama {version} at {host}, model {model}")
    building_impacts = run_pipeline()["building_impacts"]

    start = time.perf_counter()
    alerts = generate_alerts_for_critical(building_impacts, limit=limit, use_api=False, use_ollama=True)
    elapsed = time.perf_counter() - start

    for a in alerts:
        print(f"\n[{a.source}] {a.building_id} ({a.risk_level})")
        print(f"  EN: {a.message}")
        print(f"  BN: {a.message_bn}")

    local = sum(a.source == "ollama_local" for a in alerts)
    print(f"\n{local}/{len(alerts)} alerts generated locally in {elapsed:.1f}s")
    if local == 0:
        print("No alert came from Ollama -- check the model is pulled: ollama pull " + model)
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "ollama_alerts_sample.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "runtime": "ollama",
            "ollama_version": version,
            "model": model,
            "machine": f"{platform.system()} {platform.machine()}",
            "alerts_generated_locally": local,
            "total_seconds": round(elapsed, 2),
            "alerts": [asdict(a) for a in alerts],
        }, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
