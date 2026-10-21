"""
Turns raw structured predictions into actionable, plain-language alerts.

This is the "from warning to action" layer of the product: instead of a
household or a health worker having to interpret a gauge chart, they get
a short instruction. We use the Claude API to generate this text when a
key is available; if it isn't (offline judging environment, no network,
no key configured), we fall back to a deterministic template so the demo
never breaks. Both paths produce the same structured alert shape.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass


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
- Output JSON format with two keys: "en" and "bn"."""


@dataclass
class Alert:
    building_id: str
    risk_level: str
    message: str
    source: str  # "claude_api" or "template_fallback"
    message_bn: str = ""  # Bangla localized alert


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


def generate_alert(impact, use_api: bool | None = None) -> Alert:
    """
    impact: a BuildingImpact (from impact_assessment.py).
    use_api: force True/False; if None, auto-detect based on ANTHROPIC_API_KEY
             presence, falling back gracefully on any API error.
    """
    should_try_api = use_api if use_api is not None else bool(os.environ.get("ANTHROPIC_API_KEY"))

    if should_try_api:
        try:
            import anthropic
            client = anthropic.Anthropic()
            payload = {
                "building_id": impact.id,
                "type": impact.type,
                "hours_to_flood": impact.hours_to_flood,
                "risk_level": impact.risk_level,
                "nearest_shelter_name": impact.nearest_shelter_name,
                "distance_to_shelter_km": impact.distance_to_shelter_km,
                "priority": impact.priority,
            }
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                system=ALERT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": json.dumps(payload)}],
            )
            raw = "".join(
                block.text for block in response.content if getattr(block, "type", "") == "text"
            ).strip()
            # Try to parse json
            try:
                data = json.loads(raw)
                en_msg = data.get("en", raw)
                bn_msg = data.get("bn", _template_fallback_bn(impact))
            except Exception:
                en_msg = raw
                bn_msg = _template_fallback_bn(impact)

            if en_msg:
                return Alert(building_id=impact.id, risk_level=impact.risk_level,
                             message=en_msg, source="claude_api", message_bn=bn_msg)
        except Exception:
            pass  # fall through to template — the demo must never crash on network/key issues

    return Alert(
        building_id=impact.id, risk_level=impact.risk_level,
        message=_template_fallback(impact), source="template_fallback",
        message_bn=_template_fallback_bn(impact),
    )


def generate_alerts_for_critical(building_impacts, limit: int = 10, use_api: bool | None = None) -> list[Alert]:
    critical = [b for b in building_impacts if b.risk_level in ("CRITICAL", "HIGH")]
    critical.sort(key=lambda b: (b.hours_to_flood is None, b.hours_to_flood))
    return [generate_alert(b, use_api=use_api) for b in critical[:limit]]
