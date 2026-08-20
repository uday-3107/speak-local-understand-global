from backend.core.errors import ModelError, NotFoundError
from backend.ml_models.base import Service, resolve_device

_LOADED: dict = {}

_MARIAN_PAIRS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
}


def _load(pair_key: tuple[str, str]):
    if pair_key not in _LOADED:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            device = resolve_device()
            model_id = _MARIAN_PAIRS[pair_key]
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
            _LOADED[pair_key] = (model, tokenizer, device)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError("marian_load_failed", f"failed to load Marian model: {exc}") from exc
    return _LOADED[pair_key]


class MarianService(Service):
    name = "marianmt"

    def supports(self, src: str, tgt: str) -> bool:
        return (src, tgt) in _MARIAN_PAIRS

    def translate(self, text: str, src: str, tgt: str) -> tuple[str, int]:
        pair = (src, tgt)
        if pair not in _MARIAN_PAIRS:
            raise NotFoundError("unsupported_pair", f"no Marian model for {src}->{tgt}")
        import torch

        model, tokenizer, device = _load(pair)
        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            result, latency_ms = self._timed(
                model.generate, **inputs, max_new_tokens=200
            )
            with torch.no_grad():
                decoded = tokenizer.batch_decode(result, skip_special_tokens=True)[0]
            return decoded, int(latency_ms)
        except Exception as exc:
            raise ModelError("marian_failed", f"MarianMT translation failed: {exc}") from exc