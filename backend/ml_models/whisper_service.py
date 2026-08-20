from dataclasses import dataclass
import io
from time import perf_counter

import numpy as np

from backend.core.config import settings
from backend.core.errors import AudioError, ModelError
from backend.ml_models.base import Service

_MODELS: dict[str, object] = {}


def _load(size: str):
    if size not in _MODELS:
        from faster_whisper import WhisperModel

        try:
            _MODELS[size] = WhisperModel(size, device="cpu", compute_type="int8")
        except Exception as exc:
            raise ModelError("stt_load_failed", f"failed to load whisper {size}: {exc}") from exc
    return _MODELS[size]


@dataclass
class SttResult:
    text: str
    language: str
    duration_s: float
    latency_ms: int
    model: str = ""
    segments: list = None
    words: list = None


class WhisperService(Service):
    name = "faster-whisper"

    def _pick_model(self, language: str | None) -> str:
        return settings.whisper_model_telugu if language == "te" else settings.whisper_model

    def transcribe(self, audio_path: str, language: str | None = None, beam_size: int = 5) -> SttResult:
        if not audio_path:
            raise AudioError("empty_audio", "no audio provided")
        return self._run(_load(self._pick_model(language)), audio_path, language, beam_size)

    def transcribe_bytes(self, audio_bytes: bytes, language: str | None = None, beam_size: int = 5) -> SttResult:
        if not audio_bytes:
            raise AudioError("empty_audio", "no audio provided")
        return self._run(_load(self._pick_model(language)), io.BytesIO(audio_bytes), language, beam_size)

    def transcribe_np(self, audio: np.ndarray, language: str | None = None, beam_size: int = 5, word_timestamps: bool = False) -> SttResult:
        if audio is None or len(audio) == 0:
            raise AudioError("empty_audio", "no audio provided")
        return self._run(_load(self._pick_model(language)), audio, language, beam_size, word_timestamps)

    def _run(self, model, audio, language: str | None, beam_size: int, word_timestamps: bool = False) -> SttResult:
        size = self._pick_model(language)
        try:
            start = perf_counter()
            seg_gen, info = model.transcribe(
                audio,
                beam_size=beam_size,
                language=language,
                vad_filter=True,
                word_timestamps=word_timestamps,
            )
            segments = list(seg_gen)
            seg_list = [(s.start, s.end, s.text.strip()) for s in segments]
            word_list = []
            if word_timestamps:
                for s in segments:
                    for w in s.words or []:
                        word_list.append((float(w.start), float(w.end), str(w.word)))
            text = " ".join(t for _, _, t in seg_list if t).strip()
            latency_ms = int((perf_counter() - start) * 1000)
            if not text:
                raise AudioError("silence", "no speech detected in audio")
            return SttResult(
                text=text,
                language=getattr(info, "language", "") or (language or ""),
                duration_s=getattr(info, "duration", 0.0),
                latency_ms=latency_ms,
                model=size,
                segments=seg_list,
                words=word_list,
            )
        except AudioError:
            raise
        except Exception as exc:
            raise ModelError("stt_failed", f"whisper transcription failed: {exc}") from exc