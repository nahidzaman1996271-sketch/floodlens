"""Tests for the local Ollama alert backend (Ollama itself is mocked)."""

import io
import json

import pytest

from floodlens.alerts import claude_alert
from floodlens.alerts.claude_alert import generate_alert
from floodlens.pipeline import run_pipeline


@pytest.fixture
def critical_impact():
    impacts = run_pipeline()["building_impacts"]
    return next(b for b in impacts if b.risk_level == "CRITICAL" and b.nearest_shelter_name)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ollama_backend_used_when_enabled(monkeypatch, critical_impact):
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        content = json.dumps({"en": "Move now to the shelter.", "bn": "এখনই আশ্রয়কেন্দ্রে যান।"})
        return _FakeResponse(json.dumps({"message": {"content": content}}).encode())

    monkeypatch.setattr(claude_alert.urllib.request, "urlopen", fake_urlopen)
    alert = generate_alert(critical_impact, use_api=False, use_ollama=True)

    assert alert.source == "ollama_local"
    assert alert.message == "Move now to the shelter."
    assert alert.message_bn == "এখনই আশ্রয়কেন্দ্রে যান।"
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["format"] == "json"
    assert captured["body"]["stream"] is False
    assert captured["body"]["messages"][0]["role"] == "system"


def test_ollama_failure_falls_back_to_template(monkeypatch, critical_impact):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(claude_alert.urllib.request, "urlopen", boom)
    alert = generate_alert(critical_impact, use_api=False, use_ollama=True)
    assert alert.source == "template_fallback"


def test_use_api_false_disables_ollama_by_default(monkeypatch, critical_impact):
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")

    def should_not_be_called(req, timeout=None):
        raise AssertionError("Ollama must not be called when use_api=False")

    monkeypatch.setattr(claude_alert.urllib.request, "urlopen", should_not_be_called)
    alert = generate_alert(critical_impact, use_api=False)
    assert alert.source == "template_fallback"


def test_openai_compatible_local_server_backend(monkeypatch, critical_impact):
    """llama.cpp llama-server / LM Studio speak the OpenAI chat API."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:8080/v1")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        content = json.dumps({"en": "Leave now.", "bn": "এখনই বের হন।"})
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

    from floodlens.rag import llm
    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    alert = generate_alert(critical_impact, use_ollama=False)
    assert alert.source == "openai_compatible_local"
    assert captured["url"] == "http://localhost:8080/v1/chat/completions"
