import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ApiError(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ApiError


class HealthResponse(BaseModel):
    status: str
    database: str


class UserCreate(BaseModel):
    name: str
    role: str = "student"
    preferred_language: str = "en"


class UserUpdate(BaseModel):
    name: str | None = None
    role: str | None = None
    preferred_language: str | None = None


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    role: str
    preferred_language: str
    created_at: datetime


class SessionCreate(BaseModel):
    subject: str = ""
    source_lang: str
    target_lang: str = "hi"


class SessionJoin(BaseModel):
    code: str
    target_lang: str = "hi"


class SessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    subject: str
    join_code: str | None
    source_lang: str
    target_lang: str
    started_at: datetime
    ended_at: datetime | None


class RecordingRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    language: str
    duration_s: float
    created_at: datetime


class TranscriptSegmentCreate(BaseModel):
    source_text: str
    source_lang: str
    translated_text: str
    target_lang: str
    model_used: str = ""
    latency_ms: int = 0


class TranscriptSegmentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    session_id: uuid.UUID
    source_text: str
    source_lang: str
    translated_text: str
    target_lang: str
    model_used: str
    timestamp: datetime
    latency_ms: int


class FeedbackCreate(BaseModel):
    segment_id: uuid.UUID
    rating: bool
    comment: str = ""


class FeedbackRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    segment_id: uuid.UUID
    rating: bool
    comment: str


class AssistantContextItem(BaseModel):
    source_text: str
    source_lang: str = ""
    translated_text: str
    target_lang: str = ""


class AssistantRequest(BaseModel):
    question: str
    role: str = "student"
    session_id: uuid.UUID | None = None
    context: list[AssistantContextItem] | None = None


class AssistantResponse(BaseModel):
    answer: str
    model: str
    latency_ms: int
    question: str