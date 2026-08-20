import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Recording, Session, TranscriptSegment
from backend.schemas.common import SessionCreate, TranscriptSegmentCreate

JOIN_CODE_ALPHABET = "023456789ABCDEFGHJKMNPQRSTUVWXYZ"


def make_join_code(session_id: uuid.UUID) -> str:
    """Deterministic 6-char code (no 1/I/L/O) derived from the session id."""
    value = session_id.int & ((1 << 30) - 1)
    code = ""
    for _ in range(6):
        code = JOIN_CODE_ALPHABET[value % 32] + code
        value //= 32
    return code


async def create_session(db: AsyncSession, payload: SessionCreate) -> Session:
    session = Session(
        subject=payload.subject,
        source_lang=payload.source_lang,
        target_lang=payload.target_lang,
        join_code=make_join_code(uuid.uuid4()),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def join_session(db: AsyncSession, code: str) -> Session | None:
    result = await db.execute(
        select(Session).where(
            Session.join_code == code.upper().strip(),
            Session.ended_at.is_(None),
        )
    )
    return result.scalar_one_or_none()


async def end_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    session = await db.get(Session, session_id)
    if not session:
        return None
    session.ended_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(session)
    return session


async def start_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    """Mark the session as officially started; resets started_at to now."""
    session = await db.get(Session, session_id)
    if not session or session.ended_at:
        return None
    session.started_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(session)
    return session


async def list_sessions(db: AsyncSession) -> list[Session]:
    result = await db.execute(select(Session).order_by(Session.started_at.desc()))
    return list(result.scalars().all())


async def get_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    return await db.get(Session, session_id)


async def list_segments(db: AsyncSession, session_id: uuid.UUID) -> list[TranscriptSegment]:
    result = await db.execute(
        select(TranscriptSegment)
        .where(TranscriptSegment.session_id == session_id)
        .order_by(TranscriptSegment.timestamp.asc())
    )
    return list(result.scalars().all())


async def create_segment(
    db: AsyncSession, session_id: uuid.UUID, payload: TranscriptSegmentCreate
) -> TranscriptSegment:
    segment = TranscriptSegment(session_id=session_id, **payload.model_dump())
    db.add(segment)
    await db.commit()
    await db.refresh(segment)
    return segment


async def list_recordings(db: AsyncSession, session_id: uuid.UUID) -> list[Recording]:
    result = await db.execute(
        select(Recording)
        .where(Recording.session_id == session_id)
        .order_by(Recording.created_at.asc())
    )
    return list(result.scalars().all())


async def latest_recording(db: AsyncSession, session_id: uuid.UUID) -> Recording | None:
    result = await db.execute(
        select(Recording)
        .where(Recording.session_id == session_id)
        .order_by(Recording.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()