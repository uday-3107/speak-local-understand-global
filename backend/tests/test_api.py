import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.core.config import settings
from backend.core.db import get_db
from backend.main import app


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


async def test_health(client):
    r = await client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "up"}


async def test_session_crud(client):
    r = await client.post("/api/v1/sessions", json={"subject": "physics", "source_lang": "en"})
    assert r.status_code == 201
    sid = r.json()["id"]
    uuid.UUID(sid)

    r = await client.post(
        f"/api/v1/sessions/{sid}/segments",
        json={
            "source_text": "hello world",
            "source_lang": "en",
            "translated_text": "नमस्ते दुनिया",
            "target_lang": "hi",
            "model_used": "nllb",
            "latency_ms": 120,
        },
    )
    assert r.status_code == 201
    segment_id = r.json()["id"]

    r = await client.get(f"/api/v1/sessions/{sid}/segments")
    assert r.status_code == 200
    assert len(r.json()) == 1

    r = await client.post(
        "/api/v1/sessions/feedback",
        json={"segment_id": segment_id, "rating": True, "comment": "good"},
    )
    assert r.status_code == 201


async def test_join_with_code(client):
    r = await client.post("/api/v1/sessions", json={"subject": "maths", "source_lang": "en"})
    assert r.status_code == 201
    joined = r.json()
    code = joined["join_code"]
    assert code and len(code) == 6

    r = await client.post(
        "/api/v1/sessions/join",
        json={"code": code.lower(), "target_lang": "hi"},
    )
    assert r.status_code == 200
    assert r.json()["id"] == joined["id"]


async def test_join_rejects_bad_or_ended(client):
    r = await client.post(
        "/api/v1/sessions/join",
        json={"code": "ZZZZZZ"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "session_not_found"

    r = await client.post("/api/v1/sessions", json={"subject": "ended soon", "source_lang": "en"})
    sid, code = r.json()["id"], r.json()["join_code"]
    await client.post(f"/api/v1/sessions/{sid}/end")

    r = await client.post(
        "/api/v1/sessions/join",
        json={"code": code},
    )
    assert r.status_code == 404


async def test_transcript_download(client):
    r = await client.post("/api/v1/sessions", json={"subject": "physics", "source_lang": "en"})
    assert r.status_code == 201
    sid = r.json()["id"]
    r = await client.post(
        f"/api/v1/sessions/{sid}/segments",
        json={
            "source_text": "hello world",
            "source_lang": "en",
            "translated_text": "नमस्ते दुनिया",
            "target_lang": "hi",
            "model_used": "nllb",
            "latency_ms": 120,
        },
    )
    assert r.status_code == 201

    r = await client.get(f"/api/v1/sessions/{sid}/transcript")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "attachment" in r.headers["content-disposition"]
    body = r.text
    assert "physics" in body
    assert "hello world" in body
    assert "नमस्ते दुनिया" in body

    r = await client.get("/api/v1/sessions/00000000-0000-0000-0000-000000000000/transcript")
    assert r.status_code == 404


async def test_error_shape(client):
    r = await client.get("/api/v1/sessions/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    assert "error" in r.json()
    assert "code" in r.json()["error"]
    assert "message" in r.json()["error"]

    r = await client.post(
        "/api/v1/sessions",
        json={"source_lang": "en", "subject": 123},
    )
    assert r.status_code == 422
    assert "error" in r.json()


def test_websocket_transcribes_and_persists(client, monkeypatch):
    import numpy as np
    from starlette.testclient import TestClient

    from backend.api import websocket as ws_module
    from backend.core.db import SessionLocal
    from backend.core.db import get_db as original_get_db
    from backend.ml_models.whisper_service import SttResult

    def fake_translate(text: str, src: str, tgt: str) -> tuple[str, int, str]:
        assert text == "hello world"
        return "नमस्ते दुनिया", 50, "nllb-test"

    class FakeWhisper:
        def transcribe_np(self, audio, language=None, beam_size=5, word_timestamps=False) -> SttResult:
            assert len(audio) > 0
            return SttResult(
                text="hello world",
                language="en",
                duration_s=0.6,
                latency_ms=100,
                model="small",
                segments=[(0.0, 1.0, "hello world")],
                words=[(0.0, 0.6, "hello"), (0.6, 1.1, "world.")],
            )

    monkeypatch.setattr(ws_module, "translate", fake_translate)
    monkeypatch.setattr(ws_module, "whisper", FakeWhisper())
    monkeypatch.setattr(
        ws_module,
        "decode_wav_to_float",
        lambda raw: np.concatenate(
            [np.full(16000, 0.5, dtype=np.float32), np.zeros(16000, dtype=np.float32)]
        ),
    )

    async def override_get_db():
        async with SessionLocal() as session:
            yield session

    app.dependency_overrides[original_get_db] = override_get_db
    try:
        with TestClient(app) as tc:
            r = tc.post("/api/v1/sessions", json={"subject": "ws test", "source_lang": "en"})
            assert r.status_code == 201
            sid = r.json()["id"]

            with tc.websocket_connect(f"/ws/{sid}") as ws:
                ws.send_json({"type": "audio", "data": "AAAA" + "x" * 16, "target": "hi"})
                assert ws.receive_json()["type"] == "status"
                msg = ws.receive_json()
                assert msg["type"] == "segment"
                assert msg["payload"]["source_text"] == "hello world"
                assert msg["payload"]["translated_text"] == "नमस्ते दुनिया"
                assert msg["payload"]["model_used"] == "small->nllb-test"

            r = tc.get(f"/api/v1/sessions/{sid}/segments")
            assert r.status_code == 200
            segments = r.json()
            assert len(segments) == 1
            assert segments[0]["translated_text"] == "नमस्ते दुनिया"
    finally:
        app.dependency_overrides.pop(original_get_db, None)