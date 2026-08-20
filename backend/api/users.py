import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db
from backend.core.errors import NotFoundError
from backend.models import User
from backend.schemas.common import UserCreate, UserRead, UserUpdate
from backend.services import user_service

router = APIRouter(prefix="/users", tags=["users"])


@router.post("", response_model=UserRead, status_code=201)
async def create_user(payload: UserCreate, db: AsyncSession = Depends(get_db)) -> User:
    return await user_service.create_user(db, payload)


@router.get("/{user_id}", response_model=UserRead)
async def get_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> User:
    user = await user_service.get_user(db, user_id)
    if not user:
        raise NotFoundError("user_not_found", "user not found")
    return user


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: uuid.UUID, payload: UserUpdate, db: AsyncSession = Depends(get_db)
) -> User:
    user = await user_service.update_user(db, user_id, payload)
    if not user:
        raise NotFoundError("user_not_found", "user not found")
    return user
