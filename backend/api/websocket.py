import asyncio
import base64
import io
import re
import threading
import uuid
import wave

import av
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.db import SessionLocal
from backend.core.errors import AppError
from backend.ml_models import translate
from backend.ml_models.whisper_service import WhisperService, SttResult
from backend.schemas.common import TranscriptSegmentCreate
from backend.services import session_service
from backend.services.recording_service import RecordingWriter, commit_recording

router = APIRouter(tags=["websocket"])

whisper = WhisperService()

MAX_CHUNK_BYTES = 5 * 1024 * 1024


def decode_wav_to_float(raw: bytes) -> np.ndarray:
    """Parse a 16-bit PCM WAV (produced by the frontend WebAudio capture)."""
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav_file:
            if wav_file.getsampwidth() != 2:
                raise ValueError("expected 16-bit PCM")
            if wav_file.getnchannels() != 1:
                raise ValueError("expected mono audio")
            data = wav_file.readframes(wav_file.getnframes())
            rate = wav_file.getframerate()
    except (wave.Error, EOFError) as exc:
        raise AppError("invalid_audio", f"invalid wav audio: {exc}") from exc
    if not data:
        raise AppError("empty_audio", "empty audio chunk")
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if rate != 16000:
        ratio = rate / 16000.0
        x = np.linspace(0, len(samples) - 1, max(1, int(len(samples) / ratio)))
        samples = np.interp(x, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def decode_webm_to_wav(raw: bytes) -> bytes:
    """Decode an audio/webm blob (MediaRecorder) to PCM16 16kHz wav bytes.

    Uses PyAV (bundled with faster-whisper) so no external ffmpeg binary is needed.
    """
    container = av.open(io.BytesIO(raw))
    stream = container.streams.audio[0]
    frames: list[np.ndarray] = []
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            frames.append(resampled.to_ndarray())

    if not frames:
        raise ValueError("no decodable audio frames in chunk")

    pcm = np.concatenate([f.reshape(-1) for f in frames])
    int16 = np.clip(pcm, -32768, 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(int16.tobytes())
    return buffer.getvalue()


class LiveTranscriber:
    """Per-connection pause-anchored STT+MT.

    The frontend already flushes a chunk when it hears ~900ms of trailing
    silence, so each received chunk is (mostly) a complete utterance. The
    server simply accumulates audio until it also sees a real pause (or the
    buffer reaches FORCE_FLUSH_S during long unbroken speech), then transcribes
    the whole buffer and emits it as one segment, resetting the buffer. No
    overlapping windows, no text matching, no dropped or duplicated words.
    """

    MAX_WINDOW = 16000 * 12
    FORCE_FLUSH_S = 6.0
    MIN_SILENCE_S = 0.45
    MIN_SPEECH_S = 1.5
    SILENCE_RMS = 0.01
    # Interim captions: while the lecturer is still speaking (no pause yet),
    # periodically transcribe the accumulated buffer and emit an unpersisted
    # preview so viewers see text mid-speech. The next pause produces the
    # final segment carrying the same interim_id, which replaces the preview.
    INTERIM_AT_S = 3.0
    INTERIM_STEP_S = 1.5

    def __init__(self) -> None:
        self.language = "en"
        self.target = "hi"
        self.stt_model = "whisper"
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_transcribed_secs = 0.0
        self._interim_id: str | None = None
        # STT releases the GIL, so to_thread calls can overlap; serialize
        # buffer mutations + transcription per connection.
        self._lock = threading.Lock()

    def _trailing_silence(self) -> float:
        i = len(self._buffer)
        step = 16000 // 5
        while i > 0:
            w = self._buffer[max(0, i - step):i]
            if float(np.sqrt(np.mean(w * w))) >= self.SILENCE_RMS:
                break
            i = max(0, i - step)
        return (len(self._buffer) - i) / 16000

    def stage(self, audio: np.ndarray, language: str | None, target: str) -> list[dict]:
        """Buffer management + STT only (no MT). Returns staged phrases."""
        with self._lock:
            return self._stage_locked(audio, language, target)

    def _stage_locked(self, audio: np.ndarray, language: str | None, target: str) -> list[dict]:
        self.language = language or self.language
        self.target = target
        self._buffer = np.concatenate([self._buffer, audio])[-self.MAX_WINDOW:]
        secs = len(self._buffer) / 16000
        if secs < self.MIN_SPEECH_S:
            return []
        silence = self._trailing_silence()
        final = silence >= self.MIN_SILENCE_S or secs >= self.FORCE_FLUSH_S
        if not final:
            # Interim preview only: enough audio, and it has grown since the
            # last transcription so we are not re-transcribing every chunk.
            if secs < self.INTERIM_AT_S or secs - self._last_transcribed_secs < self.INTERIM_STEP_S:
                return []
        try:
            stt: SttResult = whisper.transcribe_np(
                self._buffer, language=self.language, beam_size=1
            )
        except AppError:
            if final:
                self._buffer = np.zeros(0, dtype=np.float32)
                self._last_transcribed_secs = 0.0
                self._interim_id = None
            else:
                # Keep buffering; skip retrying until more audio arrives.
                self._last_transcribed_secs = secs
            return []
        self.stt_model = stt.model or self.stt_model
        if not stt.text:
            if final:
                self._buffer = np.zeros(0, dtype=np.float32)
                self._last_transcribed_secs = 0.0
                self._interim_id = None
            else:
                self._last_transcribed_secs = secs
            return []
        phrase = stt.text.strip()
        if final:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._last_transcribed_secs = 0.0
            interim_id, interim = self._interim_id, False
            self._interim_id = None
        else:
            self._last_transcribed_secs = secs
            self._interim_id = str(uuid.uuid4())
            interim_id, interim = self._interim_id, True
        return [{"phrase": phrase, "interim": interim, "interim_id": interim_id}]


transcribers: dict[WebSocket, LiveTranscriber] = {}


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    transcriber = LiveTranscriber()
    transcribers[websocket] = transcriber
    recorder = RecordingWriter(uuid.UUID(session_id))
    pending: asyncio.Task | None = None
    in_flight: set[asyncio.Task] = set()

    def schedule_emit(item: dict) -> None:
        """MT -> persist -> send, chained after the previous emit so messages
        reach the client in order while translation overlaps the next STT."""
        nonlocal pending
        prev = pending  # captured now — reading `pending` inside _run would
        # make the first task await itself (it is assigned before it runs).

        async def _run() -> None:
            if prev is not None:
                try:
                    await prev
                except asyncio.CancelledError:
                    pass
            interim = item["interim"]
            try:
                translated, mt_latency_ms, mt_model = await asyncio.to_thread(
                    translate, item["phrase"], transcriber.language, transcriber.target
                )
                payload = TranscriptSegmentCreate(
                    source_text=item["phrase"],
                    source_lang=transcriber.language,
                    target_lang=transcriber.target,
                    translated_text=translated,
                    model_used=f"{transcriber.stt_model}->{mt_model}",
                    latency_ms=mt_latency_ms,
                ).model_dump()
                if not interim:
                    async with SessionLocal() as db:
                        segment = await session_service.create_segment(
                            db, uuid.UUID(session_id), TranscriptSegmentCreate(**payload)
                        )
                    payload["id"] = str(segment.id)
                await websocket.send_json(
                    {
                        "type": "segment",
                        "interim": interim,
                        "interim_id": item["interim_id"],
                        "payload": payload,
                    }
                )
            except AppError as exc:
                await websocket.send_json({"type": "error", "message": exc.message})
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": f"processing failed: {exc}"})

        task = asyncio.create_task(_run())
        in_flight.add(task)
        task.add_done_callback(in_flight.discard)
        pending = task

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if msg_type != "audio":
                await websocket.send_json({"type": "error", "message": f"unknown message type: {msg_type}"})
                continue

            await websocket.send_json({"type": "status", "state": "processing"})
            try:
                raw = base64.b64decode(message["data"])
                language = message.get("language")
                target = message.get("target", "hi")
                audio = decode_wav_to_float(raw)
                recorder.append(audio)
                staged = await asyncio.to_thread(
                    transcriber.stage, audio, language, target
                )
                for item in staged:
                    schedule_emit(item)
            except AppError as exc:
                await websocket.send_json({"type": "error", "message": exc.message})
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": f"processing failed: {exc}"})
    except WebSocketDisconnect:
        pass
    finally:
        transcribers.pop(websocket, None)
        if in_flight:
            await asyncio.gather(*in_flight, return_exceptions=True)
        recording = recorder.write()
        if recording is not None:
            await commit_recording(recording)