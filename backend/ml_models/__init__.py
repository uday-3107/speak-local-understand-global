from backend.core.config import settings
from backend.ml_models.indic2_service import Indic2Service
from backend.ml_models.marian_service import MarianService
from backend.ml_models.nllb_service import NllbService
from backend.ml_models.whisper_service import WhisperService

whisper = WhisperService()
nllb = NllbService()
marian = MarianService()
indic2 = Indic2Service()


def translate(text: str, src: str, tgt: str) -> tuple[str, int, str]:
    """Route translation by configured backend and language pair.

    Returns (translated_text, latency_ms, model_used).
    """
    if settings.translation_backend == "marian" and marian.supports(src, tgt):
        result, latency = marian.translate(text, src, tgt)
        return result, latency, marian.name
    if settings.translation_backend == "indic2":
        if indic2.supports(src, tgt):
            result, latency = indic2.translate(text, src, tgt)
            return result, latency, indic2.name
        # indic-indic / en-indic checkpoints don't cover X->English; NLLB does.
        result, latency = nllb.translate(text, src, tgt)
        return result, latency, f"{nllb.name}(fallback)"
    result, latency = nllb.translate(text, src, tgt)
    return result, latency, nllb.name