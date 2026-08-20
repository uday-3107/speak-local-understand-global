from backend.core.errors import ModelError
from backend.ml_models.base import Service, resolve_device

# IndicTrans2 ships three direction-specific checkpoints. en-indic-1B is
# trained for En->Indic only (any non-English source produces garbage),
# indic-indic-1B covers Indic<->Indic, and indic-en-1B (not downloaded)
# would cover Indic->En. X->English is routed to NLLB by the caller.
_MODELS = {
    "en": "ai4bharat/indictrans2-en-indic-1B",
    "indic": "ai4bharat/indictrans2-indic-indic-1B",
}

_LANG_TAGS = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu"}
_ISO = {"en": "en", "hi": "hi", "te": "te"}
_SCRIPT = {"en": "Latn", "hi": "Deva", "te": "Telu"}

# Scripts the official pipeline does NOT transliterate into Devanagari.
_NO_TRANSLIT_SCRIPTS = {"Arab", "Aran", "Olck", "Mtei", "Latn"}

_LOADED: dict = {}


def _patch_tokenizer_class(model_id: str):
    """IndicTrans2's remote tokenizer never calls ``super().__init__()``, so the
    `_special_tokens_map` attribute the modern transformers `__setattr__` expects
    is never created (transformers >= ~4.49). Initialize it explicitly on the
    class as a compatibility shim (no cache files touched)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "tokenization_indictrans.IndicTransTokenizer", model_id
    )
    if not hasattr(cls, "_special_tokens_map"):
        cls._special_tokens_map = {}


def _patch_onnx_stub():
    """transformers 5.x removed the `transformers.onnx` module, which the remote
    `configuration_indictrans.py` still imports at module scope (ONNX-export
    classes only — never used at inference). Provide harmless stubs."""
    import sys
    import types

    if "transformers.onnx" in sys.modules:
        return
    onnx_mod = types.ModuleType("transformers.onnx")

    class OnnxConfig:
        default_fixed_batch = 2
        default_fixed_sequence = 8

    class OnnxSeq2SeqConfigWithPast(OnnxConfig):
        pass

    utils_mod = types.ModuleType("transformers.onnx.utils")

    def compute_effective_axis_dimension(*_a, **_k):
        return 0

    onnx_mod.OnnxConfig = OnnxConfig
    onnx_mod.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
    utils_mod.compute_effective_axis_dimension = compute_effective_axis_dimension
    sys.modules["transformers.onnx"] = onnx_mod
    sys.modules["transformers.onnx.utils"] = utils_mod


def _patch_model_class(model_id: str):
    """transformers 5.x calls ``init_weights`` -> ``tie_weights(recompute_mapping=...)``,
    but the remote `IndicTransForConditionalGeneration.tie_weights` has the old
    4.x signature. Accept (and ignore) the extra kwarg."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module(
        "modeling_indictrans.IndicTransForConditionalGeneration", model_id
    )
    original = cls.tie_weights

    try:
        if getattr(cls.tie_weights, "_orig_tie_weights", None) is None:
            def _patched(*args, **kwargs):
                return original(*args)

            _patched._orig_tie_weights = original
            cls.tie_weights = _patched
    except Exception:
        pass


def _fix_positional_buffers(model):
    """transformers 5.x constructs the model on meta device; the remote model's
    sinusoidal positional buffers are `persistent=False` and never materialized,
    so they come back NaN. Regenerate them from the model's own size values."""
    for mod in (
        model.model.encoder.embed_positions,
        model.model.decoder.embed_positions,
    ):
        mod.make_weights(mod.weights.shape[0], mod.embedding_dim, mod.padding_idx)


def _load(model_id: str) -> dict:
    """Load (and patch) one IndicTrans2 checkpoint. Returns
    {"tokenizer", "model", "device"}."""
    if model_id not in _LOADED:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        try:
            _patch_onnx_stub()
            _patch_tokenizer_class(model_id)
            _patch_model_class(model_id)
            device = resolve_device()
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                src_lang="eng_Latn",
                tgt_lang="hin_Deva",
                trust_remote_code=True,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, trust_remote_code=True
            ).to(device)
            _fix_positional_buffers(model)
            model.eval()
            _LOADED[model_id] = {
                "tokenizer": tokenizer,
                "model": model,
                "device": device,
            }
        except Exception as exc:
            raise ModelError(
                "indic2_load_failed", f"failed to load {model_id}: {exc}"
            ) from exc
    return _LOADED[model_id]


def _preprocess(text: str, src: str) -> str:
    """Official IndicTrans2 preprocess: normalize + tokenize + transliterate
    non-Devanagari/non-Latin sources into Devanagari (the model's internal
    script-unified space)."""
    from indicnlp.tokenize import indic_tokenize

    text = text.strip()
    if src == "en":
        return text
    if _SCRIPT[src] in _NO_TRANSLIT_SCRIPTS:
        return " ".join(indic_tokenize.trivial_tokenize(text, _ISO[src]))
    try:
        from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

        normalizer = IndicNormalizerFactory().get_normalizer(_ISO[src])
        text = normalizer.normalize(text)
    except Exception:
        pass
    tokens = indic_tokenize.trivial_tokenize(text, _ISO[src])
    from indicnlp.transliterate.unicode_transliterate import (
        UnicodeIndicTransliterator,
    )

    dev = UnicodeIndicTransliterator.transliterate(
        " ".join(tokens), _ISO[src], "hi"
    )
    return dev.replace(" ् ", "्")


def _postprocess(text: str, tgt: str) -> str:
    """Official IndicTrans2 postprocess: transliterate Devanagari output back
    into the target script (identity for Devanagari/Latin targets)."""
    if tgt == "en" or _SCRIPT[tgt] in _NO_TRANSLIT_SCRIPTS:
        return text.strip()
    from indicnlp.transliterate.unicode_transliterate import (
        UnicodeIndicTransliterator,
    )

    xlated = UnicodeIndicTransliterator.transliterate(text, "hi", _ISO[tgt])
    from indicnlp.tokenize import indic_detokenize

    return indic_detokenize.trivial_detokenize(xlated, _ISO[tgt]).strip()


def _decode(model, tokenizer, inp, mask, num_beams: int, max_new_tokens: int) -> str:
    """Manual decode loop. transformers 5.x `generate()` passes
    `EncoderDecoderCache` objects the remote 4.32-era model code can't handle,
    so we run the encoder + decoder steps ourselves with the legacy tuple KV
    cache (shape (bsz, seq, heads, head_dim) per layer)."""
    import torch

    with torch.no_grad():
        enc = model.model.encoder(input_ids=inp, attention_mask=mask)
        ehs, emask = enc.last_hidden_state, mask

        if num_beams <= 1:
            past, ids = None, [model.config.decoder_start_token_id]
            for _ in range(max_new_tokens):
                dec_ids = torch.tensor([[ids[-1]]], device=inp.device)
                dec_mask = torch.ones(1, len(ids), device=inp.device)
                dec = model.model.decoder(
                    input_ids=dec_ids,
                    attention_mask=dec_mask,
                    encoder_hidden_states=ehs,
                    encoder_attention_mask=emask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = dec.past_key_values
                tok = model.lm_head(dec.last_hidden_state[:, -1]).argmax(-1).item()
                if tok == model.config.eos_token_id:
                    break
                ids.append(tok)
            return tokenizer.decode(ids[1:], skip_special_tokens=True)

        # Sequential beam search (one forward pass per hypothesis per step).
        hyps = [
            {
                "ids": [model.config.decoder_start_token_id],
                "past": None,
                "score": 0.0,
            }
        ]
        completed = []
        for _ in range(max_new_tokens):
            if not hyps:
                break
            new = []
            for h in hyps:
                dec_ids = torch.tensor([[h["ids"][-1]]], device=inp.device)
                dec_mask = torch.ones(1, len(h["ids"]), device=inp.device)
                dec = model.model.decoder(
                    input_ids=dec_ids,
                    attention_mask=dec_mask,
                    encoder_hidden_states=ehs,
                    encoder_attention_mask=emask,
                    past_key_values=h["past"],
                    use_cache=True,
                )
                logprobs = torch.log_softmax(
                    model.lm_head(dec.last_hidden_state[:, -1])[0], dim=-1
                )
                top = torch.topk(logprobs, num_beams)
                for i in range(num_beams):
                    tok = top.indices[i].item()
                    ns = h["score"] + top.values[i].item()
                    if tok == model.config.eos_token_id:
                        completed.append(
                            (
                                ns / (len(h["ids"]) ** 0.6),
                                h["ids"][1:] + [tok],
                            )
                        )
                    else:
                        new.append(
                            {
                                "ids": h["ids"] + [tok],
                                "score": ns,
                                "past": dec.past_key_values,
                            }
                        )
            new.sort(key=lambda x: -x["score"])
            hyps = new[:num_beams]
            if completed and len(completed) >= num_beams:
                break
        if completed:
            completed.sort(key=lambda x: -x[0])
            return tokenizer.decode(completed[0][1][:-1], skip_special_tokens=True)
        if hyps:
            return tokenizer.decode(hyps[0]["ids"][1:], skip_special_tokens=True)
        return ""


class Indic2Service(Service):
    name = "indictrans2"

    def supports(self, src: str, tgt: str) -> bool:
        """X->English is not supported by the two cached checkpoints."""
        return tgt != "en" and src in _LANG_TAGS and tgt in _LANG_TAGS

    def translate(
        self,
        text: str,
        src: str,
        tgt: str,
        num_beams: int = 1,
        max_new_tokens: int = 256,
    ) -> tuple[str, int]:
        if not self.supports(src, tgt):
            raise ModelError(
                "unsupported_pair",
                f"IndicTrans2 has no checkpoint for {src}->{tgt}",
            )
        import torch

        model_id = _MODELS["en"] if src == "en" else _MODELS["indic"]
        loaded = _load(model_id)
        tokenizer, model, device = (
            loaded["tokenizer"],
            loaded["model"],
            loaded["device"],
        )
        src_lang = _LANG_TAGS[src]
        tgt_lang = _LANG_TAGS[tgt]
        try:
            processed = _preprocess(text, src)
            tagged = f"{src_lang} {tgt_lang} {processed}"
            inputs = tokenizer(tagged, return_tensors="pt").to(device)
            result, latency_ms = self._timed(
                _decode,
                model,
                tokenizer,
                inputs["input_ids"],
                inputs["attention_mask"],
                num_beams,
                max_new_tokens,
            )
            with torch.no_grad():
                return _postprocess(result, tgt), int(latency_ms)
        except Exception as exc:
            raise ModelError(
                "indic2_failed", f"IndicTrans2 translation failed: {exc}"
            ) from exc
