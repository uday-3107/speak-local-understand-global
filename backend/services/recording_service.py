"""Persist lecture recordings to disk (Module 1: speech data collection).

The live WS endpoint appends every decoded PCM chunk here; when the session
ends the recording is flushed to a WAV file under data/recordings/ and a row
is written to the `recordings` table (session, language, duration).
"""
import asyncio
import os
import uuid
import wave

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import Recording

RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "recordings",
)


class RecordingWriter:
    """Collects float PCM chunks for one session and writes a 16-bit WAV at close."""

    def __init__(self, session_id: uuid.UUID, language: str = "en") -> None:
        self.session_id = session_id
        self.language = language
        self._chunks: list[np.ndarray] = []
        self._samples = 0

    def append(self, audio: np.ndarray) -> None:
        if audio is None or len(audio) == 0:
            return
        self._chunks.append(audio.astype(np.float32))
        self._samples += len(audio)

    @property
    def duration_s(self) -> float:
        return self._samples / 16000

    def write(self) -> Recording | None:
        """Flush to disk + return the Recording row. Returns None if empty."""
        if self._samples == 0:
            return None
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        file_path = os.path.join(RECORDINGS_DIR, f"{self.session_id}.wav")
        pcm = np.concatenate(self._chunks)
        pcm16 = np.clip(pcm, -1.0, 1.0)
        pcm16 = (pcm16 * 32767).astype(np.int16)
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm16.tobytes())
        return Recording(
            session_id=self.session_id,
            language=self.language,
            duration_s=round(self._samples / 16000, 2),
            file_path=file_path,
        )


async def commit_recording(recording: Recording) -> None:
    """Persist a flushed Recording row (call after write() in a session context)."""
    from backend.core.db import SessionLocal

    async with SessionLocal() as db:
        db.add(recording)
        await db.commit()
