import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.core.errors import NotFoundError
from backend.models import Recording, Session, TranscriptSegment
from backend.schemas.common import (
    FeedbackCreate,
    FeedbackRead,
    RecordingRead,
    SessionCreate,
    SessionJoin,
    SessionRead,
    TranscriptSegmentCreate,
    TranscriptSegmentRead,
)
from backend.services import feedback_service, session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("/join", response_model=SessionRead)
async def join_session(payload: SessionJoin, db: AsyncSession = Depends(get_db)) -> Session:
    session = await session_service.join_session(db, payload.code)
    if not session:
        raise NotFoundError("session_not_found", "no live lecture with that code")
    return session


@router.post("", response_model=SessionRead, status_code=201)
async def create_session(payload: SessionCreate, db: AsyncSession = Depends(get_db)) -> Session:
    return await session_service.create_session(db, payload)


@router.get("", response_model=list[SessionRead])
async def list_sessions(db: AsyncSession = Depends(get_db)) -> list[Session]:
    return await session_service.list_sessions(db)


@router.get("/{session_id}", response_model=SessionRead)
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Session:
    session = await session_service.get_session(db, session_id)
    if not session:
        raise NotFoundError("session_not_found", "session not found")
    return session


@router.get("/{session_id}/segments", response_model=list[TranscriptSegmentRead])
async def list_segments(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> list[TranscriptSegment]:
    return await session_service.list_segments(db, session_id)


@router.post("/{session_id}/segments", response_model=TranscriptSegmentRead, status_code=201)
async def create_segment(
    session_id: uuid.UUID,
    payload: TranscriptSegmentCreate,
    db: AsyncSession = Depends(get_db),
) -> TranscriptSegment:
    return await session_service.create_segment(db, session_id, payload)


@router.post("/{session_id}/end", response_model=SessionRead)
async def end_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Session:
    session = await session_service.end_session(db, session_id)
    if not session:
        raise NotFoundError("session_not_found", "session not found")
    return session


@router.post("/{session_id}/start", response_model=SessionRead)
async def start_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Session:
    session = await session_service.start_session(db, session_id)
    if not session:
        raise NotFoundError("session_not_found", "session not found or already ended")
    return session


@router.get("/{session_id}/recordings", response_model=list[RecordingRead])
async def list_recordings(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> list[Recording]:
    return await session_service.list_recordings(db, session_id)


@router.get("/{session_id}/transcript")
async def download_transcript(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> PlainTextResponse:
    session = await session_service.get_session(db, session_id)
    if not session:
        raise NotFoundError("session_not_found", "session not found")
    segments = await session_service.list_segments(db, session_id)

    lines = [
        f"Session transcript — {session.subject or 'Lecture'}",
        f"Session: {session.id}",
        f"Languages: {session.source_lang} -> {session.target_lang}",
        f"Started: {session.started_at.isoformat()}",
        f"Ended: {session.ended_at.isoformat() if session.ended_at else '—'}",
        f"Captions: {len(segments)}",
        "=" * 60,
    ]
    for s in segments:
        stamp = s.timestamp.strftime("%H:%M:%S")
        lines.append(f"[{stamp}] ({s.source_lang}) {s.source_text}")
        lines.append(f"[{stamp}] ({s.target_lang}) {s.translated_text}")
        lines.append("")

    return PlainTextResponse(
        "\n".join(lines).rstrip() + "\n",
        headers={
            "Content-Disposition": f'attachment; filename="transcript_{session_id}.txt"'
        },
    )


@router.get("/{session_id}/recording/download")
async def download_recording(
    session_id: uuid.UUID, db: AsyncSession = Depends(get_db)
) -> FileResponse:
    recording = await session_service.latest_recording(db, session_id)
    if not recording:
        raise NotFoundError("recording_not_found", "no recording for this session")
    return FileResponse(
        recording.file_path,
        media_type="audio/wav",
        filename=f"recording_{session_id}.wav",
    )


@router.post("/feedback", response_model=FeedbackRead, status_code=201)
async def create_feedback(payload: FeedbackCreate, db: AsyncSession = Depends(get_db)):
    return await feedback_service.create_feedback(db, payload)
