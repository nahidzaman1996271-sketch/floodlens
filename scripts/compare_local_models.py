"""
Compares open-weight models running locally in Ollama on the FloodLens alert
task, so we can pick the best small model for offline union-level deployment.

    ollama pull qwen2.5:3b && ollama pull llama3.2:3b && ollama pull gemma2:2b && ollama pull phi3.5
    python scripts/compare_local_models.py
    python scripts/compare_local_models.py qwen2.5:3b llama3.2:3b   # custom list

For each model it drafts alerts for the 3 most urgent buildings (grounded
with Graph RAG facts + retrieved guidance) and measures:
    json_ok         reply parsed as {"en", "bn"}
    hours_faithful  the English text contains the exact hours-to-flood figure
    shelter_named   the English text names the assigned shelter
    has_bangla      the Bangla text actually contains Bengali script
    seconds         wall-clock latency per alert
Writes results/local_model_comparison.json.
"""

from __future__ import annotations

import json
import os
import platform
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floodlens.pipeline import run_pipeline
from floodlens.alerts.claude_alert import ALERT_SYSTEM_PROMPT, _payload, ollama_chat
from floodlens.rag.retriever import FloodRAG

DEFAULT_MODELS = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "phi3.5"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
BENGALI = re.compile(r"[ঀ-৿]")


def installed_models(host: str) -> set[str]:
    with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as r:
        return {m["name"] for m in json.loads(r.read().decode())["models"]}


def main(models: list[str]):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    try:
        have = installed_models(host)
    except Exception as exc:
        print(f"Ollama not reachable at {host}: {exc}")
        sys.exit(1)

    result = run_pipeline()
    rag = FloodRAG(use_llm=False).build()
    rag.attach_graph(result)
    urgent = sorted([b for b in result["building_impacts"] if b.hours_to_flood is not None],
                    key=lambda b: b.hours_to_flood)[:3]
    contexts = {b.id: rag.context_for_building(b) for b in urgent}

    report = {"machine": f"{platform.system()} {platform.machine()}", "ollama_host": host, "models": {}}
    for model in models:
        if model not in have and f"{model}:latest" not in have:
            print(f"skip {model}: not pulled (ollama pull {model})")
            report["models"][model] = {"skipped": "not pulled"}
            continue
        rows = []
        for b in urgent:
            t = time.perf_counter()
            try:
                raw = ollama_chat([
                    {"role": "system", "content": ALERT_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(_payload(b, contexts[b.id]), ensure_ascii=False)},
                ], model=model, host=host, timeout=300)
                err = None
            except Exception as exc:
                raw, err = "", str(exc)
            secs = time.perf_counter() - t
            try:
                data = json.loads(raw)
                en, bn = str(data.get("en", "")), str(data.get("bn", ""))
                ok = bool(en)
            except Exception:
                en, bn, ok = raw, "", False
            rows.append({
                "building_id": b.id, "seconds": round(secs, 1), "json_ok": ok,
                "hours_faithful": str(b.hours_to_flood) in en,
                "shelter_named": bool(b.nearest_shelter_name) and b.nearest_shelter_name.split()[0] in en,
                "has_bangla": bool(BENGALI.search(bn)), "en": en[:400], "bn": bn[:400], "error": err,
            })
        n = len(rows)
        summary = {k: round(sum(r[k] for r in rows) / n, 2)
                   for k in ("json_ok", "hours_faithful", "shelter_named", "has_bangla")}
        summary["avg_seconds"] = round(sum(r["seconds"] for r in rows) / n, 1)
        report["models"][model] = {"summary": summary, "samples": rows}
        print(f"{model:14s} {summary}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "local_model_comparison.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_MODELS)
