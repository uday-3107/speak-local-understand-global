import torch

from backend.core.config import settings
from backend.core.errors import ModelError
from backend.ml_models.base import Service, resolve_device

LANG_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu"}

_LOADED: dict = {}


def _get():
    if not _LOADED:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            device = resolve_device()
            tokenizer = AutoTokenizer.from_pretrained(settings.nllb_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(settings.nllb_model).to(device)
            _LOADED["tokenizer"] = tokenizer
            _LOADED["model"] = model
            _LOADED["device"] = device
        except Exception as exc:
            raise ModelError("nllb_load_failed", f"failed to load {settings.nllb_model}: {exc}") from exc
    return _LOADED


class NllbService(Service):
    name = "nllb"

    def translate(self, text: str, src: str, tgt: str) -> tuple[str, int]:
        loaded = _get()
        tokenizer, model, device = loaded["tokenizer"], loaded["model"], loaded["device"]
        src = LANG_CODES.get(src, src)
        tgt = LANG_CODES.get(tgt, tgt)
        try:
            tokenizer.src_lang = src
            inputs = tokenizer(text, return_tensors="pt").to(device)
            result, latency_ms = self._timed(
                model.generate,
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt),
                max_new_tokens=256,
            )
            with torch.no_grad():
                decoded = tokenizer.batch_decode(result, skip_special_tokens=True)[0]
            return decoded, int(latency_ms)
        except Exception as exc:
            raise ModelError("nllb_failed", f"NLLB translation failed: {exc}") from exc