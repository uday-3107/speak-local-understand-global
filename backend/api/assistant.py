"""Module 7 live wiring: POST /api/v1/assistant — real Ollama (Mistral) answers.

Frontend sends the question plus the recent captions it already has; the
backend optionally augments context from the DB (session_id) and streams the
question through the tested OllamaLlm client (scripts/ai_translation_assistant).
Answers are free-text English (chat-style), grounded in the provided lecture
context. Each exchange is appended to data/processed/assistant/live.jsonl.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.core.errors import AppError
from backend.schemas.common import (
    AssistantContextItem,
    AssistantRequest,
    AssistantResponse,
)
from backend.services import session_service
from scripts.ai_translation_assistant import OllamaLlm

router = APIRouter(prefix="/assistant", tags=["assistant"])

llm = OllamaLlm()
OUT_FILE = Path("data/processed/assistant/live.jsonl")

SYSTEM_PROMPT = (
    "You are an AI study assistant for a live multilingual classroom lecture. "
    "You receive recent captions: the original spoken sentence and its machine "
    "translation into the display language. Answer the question in English, in "
    "1-3 short sentences, using ONLY the provided context. If the context does "
    "not cover the question, say so instead of guessing. No markdown, no lists."
)


def build_prompt(question: str, context: list[dict], role: str) -> str:
    if context:
        lines = [
            f"{i}. [{c.get('source_lang', '?')}] {c['source_text']}"
            f" -> [{c.get('target_lang', '?')}] {c['translated_text']}"
            for i, c in enumerate(context, 1)
        ]
        ctx_block = "\n".join(lines)
    else:
        ctx_block = "(no captions yet — the lecture is just starting)"
    user = (
        f"Role: {role}\n"
        f"Recent lecture captions (oldest to newest):\n{ctx_block}\n\n"
        f"Question: {question}\nAnswer:"
    )
    return f"### System\n{SYSTEM_PROMPT}\n\n### User\n{user}\n\n### Assistant"


@router.post("", response_model=AssistantResponse)
async def ask_assistant(
    payload: AssistantRequest,
    db: AsyncSession = Depends(get_db),
) -> AssistantResponse:
    context = [c.model_dump() for c in payload.context or []]
    if not context and payload.session_id:
        segments = await session_service.list_segments(db, payload.session_id)
        context = [
            {
                "source_text": s.source_text,
                "source_lang": s.source_lang,
                "translated_text": s.translated_text,
                "target_lang": s.target_lang,
            }
            for s in segments[-6:]
        ]

    prompt = build_prompt(payload.question, context, payload.role)

    async def _generate() -> tuple[str, int]:
        return await asyncio.to_thread(llm.generate, prompt)

    try:
        answer, latency_ms = await _generate()
    except Exception as exc:
        raise AppError(
            "ollama_offline",
            "AI assistant is unavailable — Ollama is not running "
            "(start it with `ollama serve`).",
            status_code=503,
        ) from exc

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "ts": time.time(),
                    "role": payload.role,
                    "question": payload.question,
                    "answer": answer,
                    "model": llm.model,
                    "latency_ms": latency_ms,
                    "context": context,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    return AssistantResponse(
        answer=answer,
        model=llm.model,
        latency_ms=latency_ms,
        question=payload.question,
    )