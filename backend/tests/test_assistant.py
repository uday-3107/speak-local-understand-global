"""Module 7 tests: prompt template, JSON parsing, Ollama client mock (no models)."""
import json

import pytest
import requests
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.core.config import settings
from backend.core.db import get_db
from backend.main import app
from scripts.ai_translation_assistant import (JsonAssistant, OllamaLlm,
                                              PromptTemplate)


class TestPromptTemplate:
    def test_build_contains_system_and_inputs(self):
        p = PromptTemplate().build("hello world", "नमस्ते दुनिया", "en", "hi", "Chemistry")
        assert "# System" in p and "# User" in p
        assert "Chemistry" in p
        assert "hello world" in p
        assert "नमस्ते दुनिया" in p

    def test_default_subject(self):
        p = PromptTemplate().build("x", "y", "en", "hi")
        assert "general" in p


class TestJsonAssistant:
    def test_parses_json_answer(self):
        raw = '{"translation": "नमस्ते", "explanation": "अभिवादन", "study_note": "याद रखें"}'
        parsed = JsonAssistant.parse(raw)
        assert parsed["translation"] == "नमस्ते"
        assert parsed["explanation"] == "अभिवादन"

    def test_falls_back_to_raw_text(self):
        parsed = JsonAssistant.parse("नमस्ते दुनिया")
        assert parsed["translation"] == "नमस्ते दुनिया"


class TestOllamaClient:
    def test_ping_returns_true_when_running(self, monkeypatch):
        class _Resp:
            status_code = 200
        monkeypatch.setattr("requests.get", lambda *a, **k: _Resp())
        assert OllamaLlm().ping() is True

    def test_ping_false_on_error(self, monkeypatch):
        def _boom(*a, **k):
            raise requests.RequestException("conn refused")

        monkeypatch.setattr("requests.get", _boom)
        assert OllamaLlm().ping() is False

    def test_generate_posts_and_parses(self, monkeypatch):
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"response": json.dumps({"translation": "X"})}

        def _post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs["json"]
            return _Resp()

        monkeypatch.setattr("requests.post", _post)
        text, latency = OllamaLlm(url="http://localhost:11434").generate("hi")
        assert captured["json"]["model"] == "mistral"
        assert captured["json"]["stream"] is False
        assert "X" in text
        assert latency >= 0


@pytest.fixture
async def client():
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()
    await engine.dispose()


class TestAssistantEndpoint:
    async def test_answers_via_ollama(self, client, monkeypatch):
        def _fake_generate(prompt):
            assert "### System" in prompt
            assert "Question: summarize" in prompt
            return "The lecture covered Newton's laws.", 1234

        monkeypatch.setattr("backend.api.assistant.llm.generate", _fake_generate)
        r = await client.post(
            "/api/v1/assistant",
            json={"question": "summarize", "role": "student"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "The lecture covered Newton's laws."
        assert body["model"] == "mistral"
        assert body["latency_ms"] == 1234

    async def test_context_is_sent_to_llm(self, client, monkeypatch):
        captured = {}

        def _fake_generate(prompt):
            captured["prompt"] = prompt
            return "ok", 5

        monkeypatch.setattr("backend.api.assistant.llm.generate", _fake_generate)
        r = await client.post(
            "/api/v1/assistant",
            json={
                "question": "explain",
                "role": "student",
                "context": [
                    {"source_text": "hello", "translated_text": "नमस्ते", "target_lang": "hi"},
                ],
            },
        )
        assert r.status_code == 200
        assert "hello" in captured["prompt"]
        assert "नमस्ते" in captured["prompt"]

    async def test_503_when_ollama_down(self, client, monkeypatch):
        def _boom(prompt):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr("backend.api.assistant.llm.generate", _boom)
        r = await client.post(
            "/api/v1/assistant", json={"question": "hi"}
        )
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "ollama_offline"