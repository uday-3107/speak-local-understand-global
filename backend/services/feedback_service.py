from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Feedback
from backend.schemas.common import FeedbackCreate


async def create_feedback(db: AsyncSession, payload: FeedbackCreate) -> Feedback:
    feedback = Feedback(**payload.model_dump())
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    return feedback