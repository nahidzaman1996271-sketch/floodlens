"""
Turns raw structured predictions into actionable, plain-language alerts.

This is the "from warning to action" layer of the product: instead of a
household or a health worker having to interpret a gauge chart, they get
a short instruction. We use the Claude API to generate this text when a
key is available. If it isn't, we try a local open-weight model served by
Ollama (OLLAMA_HOST, default http://localhost:11434; model OLLAMA_MODEL,
default qwen2.5:3b) so alerts can still be drafted fully offline on a
laptop at a union parishad office. If neither is reachable, we fall back to
a deterministic template so the demo never breaks. All paths produce the
same structured alert shape.
"""

from __future__ import annotations

import os
import json
import urllib.request
from dataclasses import dataclass, field

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5:3b"


ALERT_SYSTEM_PROMPT = """You are the alert-drafting layer of FloodLens, a flood early-warning \
system for a specific locality in Bangladesh. You are given structured, already-computed \
flood predictions (hours until flooding, risk level, nearest shelter, distance). Your ONLY \
job is to turn that structured data into a short, calm, actionable alert message for the \
affected household or facility in English and Bangla.

Rules:
- Do not invent numbers. Use only the hours/distances/names given to you.
- Keep English message under 60 words, followed by Bangla translation.
- Give concrete actions in priority order (move people first, then documents/medicine, then \
livestock/valuables).
- Name the specific shelter and distance given.
- No alarmist language beyond what the risk level warrants. CRITICAL = urgent and direct. \
WATCH = informative, no panic.
- If "facts" (from the FloodLens knowledge graph) or "guidance" (retrieved from the FloodLens \
knowledge base) are provided, use them to choose and order the actions, and mention a road \
closure or an alternative shelter if the facts say so. Never copy numbers that are not given.
- Output JSON format with two keys: "en" and "bn"."""


@dataclass
class Alert:
    building_id: str
    risk_level: str
    message: str
    source: str  # "claude_api", "ollama_local" or "template_fallback"
    message_bn: str = ""  # Bangla localized alert
    rag_sources: list = field(default_factory=list)  # knowledge-base chunk ids used, if any


def _template_fallback_bn(impact) -> str:
    """Deterministic Bangla translation matching the English fallback."""
    if impact.hours_to_flood is None:
        return (f"বর্তমান ৭২ ঘণ্টার পূর্বাভাসে এই এলাকার জন্য কোনো বন্যার ঝুঁকি নেই। "
                f"নিয়মিত সরকারি আবহাওয়া বুলেটিন ও নির্দেশিকা অনুসরণ করুন।")

    urgency = "এখনই নিরাপদ স্থানে যান" if impact.risk_level == "CRITICAL" else \
              "আশ্রয়ে যাওয়ার প্রস্তুতি নিন" if impact.risk_level == "HIGH" else "সতর্ক থাকুন"
    shelter_clause = (
        f" নিকটতম আশ্রয়কেন্দ্র: {impact.nearest_shelter_name} "
        f"({impact.distance_to_shelter_km} কিমি দূরে)।"
        if impact.nearest_shelter_name else ""
    )
    priority_clause = ""
    if impact.priority == "medical_priority":
        priority_clause = " রোগী এবং জরুরি ওষুধপত্র অগ্রাধিকার ভিত্তিতে সরিয়ে নিন।"
    elif impact.priority == "vulnerable_group_priority":
        priority_clause = " শিশু, প্রবীণ ও গর্ভবতী নারীদের আগে সরিয়ে নিন।"

    return (
        f"{urgency}: আনুমানিক {impact.hours_to_flood} ঘণ্টার মধ্যে এখানে পানি প্রবেশ করবে "
        f"({impact.risk_level} ঝুঁকি)।{shelter_clause}{priority_clause} "
        f"জরুরি কাগজপত্র, ওষুধ ও বিশুদ্ধ পানি সাথে নিন। সময় থাকলে গবাদিপশু উঁচু স্থানে সরিয়ে নিন।"
    )


def _template_fallback(impact) -> str:
    """Deterministic, judge-Q&A-proof fallback: no network dependency,
    same structured inputs, same style of output."""
    if impact.hours_to_flood is None:
        return (f"No flooding predicted in the current 72-hour forecast window for this "
                f"location. Continue to monitor updates.")

    urgency = "Move now" if impact.risk_level == "CRITICAL" else \
              "Prepare to move" if impact.risk_level == "HIGH" else "Stay alert"
    shelter_clause = (
        f" Nearest shelter: {impact.nearest_shelter_name} "
        f"({impact.distance_to_shelter_km} km away)."
        if impact.nearest_shelter_name else ""
    )
    priority_clause = ""
    if impact.priority == "medical_priority":
        priority_clause = " Prioritize patients and medical supplies first."
    elif impact.priority == "vulnerable_group_priority":
        priority_clause = " Prioritize children, elderly, and people with mobility needs first."

    return (
        f"{urgency}: flooding predicted here in approximately {impact.hours_to_flood} hours "
        f"({impact.risk_level} risk).{shelter_clause}{priority_clause} "
        f"Take essential documents, medicine, and water. Move livestock to higher ground if time allows."
    )


def _payload(impact, rag_context: dict | None = None) -> dict:
    payload = {
        "building_id": impact.id,
        "type": impact.type,
        "hours_to_flood": impact.hours_to_flood,
        "risk_level": impact.risk_level,
        "nearest_shelter_name": impact.nearest_shelter_name,
        "distance_to_shelter_km": impact.distance_to_shelter_km,
        "priority": impact.priority,
    }
    if rag_context:
        payload["facts"] = rag_context.get("facts", [])
        payload["guidance"] = rag_context.get("guidance", [])
    return payload


def _parse_llm_json(raw: str, impact) -> tuple[str, str]:
    """Parse the {"en", "bn"} JSON an LLM returns; tolerate plain text."""
    raw = (raw or "").strip()
    try:
        data = json.loads(raw)
        return data.get("en", raw), data.get("bn") or _template_fallback_bn(impact)
    except Exception:
        if raw.startswith("{"):
            # Truncated / malformed JSON (e.g. hit the token cap): don't show
            # raw JSON to residents -- an empty message makes the caller fall back.
            return "", _template_fallback_bn(impact)
        return raw, _template_fallback_bn(impact)


def _claude_alert(impact, rag_context: dict | None = None) -> Alert | None:
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=ALERT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(_payload(impact, rag_context), ensure_ascii=False)}],
    )
    raw = "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    en_msg, bn_msg = _parse_llm_json(raw, impact)
    if not en_msg:
        return None
    return Alert(building_id=impact.id, risk_level=impact.risk_level,
                 message=en_msg, source="claude_api", message_bn=bn_msg)


def ollama_chat(messages: list[dict], model: str | None = None, host: str | None = None,
                timeout: float | None = None) -> str:
    """Call a local Ollama server's /api/chat endpoint (JSON mode, no streaming)
    using only the standard library. Returns the assistant message text.
    timeout defaults to OLLAMA_TIMEOUT (seconds, default 120): CPU-only or
    low-RAM machines can need well over a minute for the first token."""
    if timeout is None:
        timeout = float(os.environ.get("OLLAMA_TIMEOUT", "120"))
    host = (host or os.environ.get("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST).rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    body = json.dumps({
        "model": model or os.environ.get("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL),
        "messages": messages,
        "format": "json",
        "stream": False,
        # Small models in JSON mode can loop on one phrase (seen with qwen2.5:3b
        # writing Bangla) and never close the object: penalise repeats and cap
        # the length so a runaway reply fails fast and falls back cleanly.
        "options": {"temperature": 0.2, "repeat_penalty": 1.15,
                    "num_predict": int(os.environ.get("OLLAMA_NUM_PREDICT", "1024"))},
    }).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("content", "")


def _ollama_alert(impact, rag_context: dict | None = None) -> Alert | None:
    raw = ollama_chat([
        {"role": "system", "content": ALERT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(_payload(impact, rag_context), ensure_ascii=False)},
    ])
    en_msg, bn_msg = _parse_llm_json(raw, impact)
    if not en_msg:
        return None
    return Alert(building_id=impact.id, risk_level=impact.risk_level,
                 message=en_msg, source="ollama_local", message_bn=bn_msg)


def _local_openai_alert(impact, rag_context: dict | None = None) -> Alert | None:
    """Any OpenAI-compatible local server: llama.cpp llama-server, LM Studio, vLLM."""
    from floodlens.rag.llm import openai_compatible_chat
    raw = openai_compatible_chat([
        {"role": "system", "content": ALERT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(_payload(impact, rag_context), ensure_ascii=False)},
    ], json_mode=True)
    en_msg, bn_msg = _parse_llm_json(raw, impact)
    if not en_msg:
        return None
    return Alert(building_id=impact.id, risk_level=impact.risk_level,
                 message=en_msg, source="openai_compatible_local", message_bn=bn_msg)


def generate_alert(impact, use_api: bool | None = None, use_ollama: bool | None = None,
                   rag_context: dict | None = None) -> Alert:
    """
    impact: a BuildingImpact (from impact_assessment.py).
    use_api: force True/False for the Claude API; if None, auto-detect based on
             ANTHROPIC_API_KEY presence.
    use_ollama: force True/False for the local Ollama model; if None, it is
             tried when OLLAMA_HOST or FLOODLENS_USE_OLLAMA=1 is set. Passing
             use_api=False with use_ollama left as None disables all LLM
             backends (template only), which keeps tests deterministic.
    Backends are tried in order Claude API -> Ollama -> template, falling back
    gracefully on any error.
    """
    should_try_api = use_api if use_api is not None else bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_ollama is None:
        should_try_ollama = use_api is not False and bool(
            os.environ.get("OLLAMA_HOST") or os.environ.get("FLOODLENS_USE_OLLAMA") == "1"
        )
    else:
        should_try_ollama = use_ollama

    backends = []
    if should_try_api:
        backends.append(_claude_alert)
    if should_try_ollama:
        backends.append(_ollama_alert)
    if use_api is not False and os.environ.get("LOCAL_LLM_BASE_URL"):
        backends.append(_local_openai_alert)

    sources = list(rag_context.get("sources", [])) if rag_context else []
    for backend in backends:
        try:
            alert = backend(impact, rag_context)
            if alert is not None:
                alert.rag_sources = sources
                return alert
        except Exception:
            continue  # fall through -- the demo must never crash on network/key/model issues

    return Alert(
        building_id=impact.id, risk_level=impact.risk_level,
        message=_template_fallback(impact), source="template_fallback",
        message_bn=_template_fallback_bn(impact),
    )


def generate_alerts_for_critical(building_impacts, limit: int = 10, use_api: bool | None = None,
                                 use_ollama: bool | None = None, rag=None) -> list[Alert]:
    """rag: an optional floodlens.rag.retriever.FloodRAG (with graph attached).
    When given, every alert is grounded with Graph RAG facts + retrieved guidance."""
    critical = [b for b in building_impacts if b.risk_level in ("CRITICAL", "HIGH")]
    critical.sort(key=lambda b: (b.hours_to_flood is None, b.hours_to_flood))
    out = []
    for b in critical[:limit]:
        ctx = None
        if rag is not None:
            try:
                ctx = rag.context_for_building(b)
            except Exception:
                ctx = None
        out.append(generate_alert(b, use_api=use_api, use_ollama=use_ollama, rag_context=ctx))
    return out
