import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import User
from backend.schemas.common import UserCreate, UserUpdate


async def create_user(db: AsyncSession, payload: UserCreate) -> User:
    user = User(
        name=payload.name,
        role=payload.role,
        preferred_language=payload.preferred_language,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def get_user(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    return await db.get(User, user_id)


async def update_user(db: AsyncSession, user_id: uuid.UUID, payload: UserUpdate) -> User | None:
    user = await db.get(User, user_id)
    if not user:
        return None
    if payload.name is not None:
        user.name = payload.name
    if payload.role is not None:
        user.role = payload.role
    if payload.preferred_language is not None:
        user.preferred_language = payload.preferred_language
    await db.commit()
    await db.refresh(user)
    return user
