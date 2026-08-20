"""indic2_nan_check.py — read-only: isolate the NaN + cache errors.

No downloads, no cache writes, no other models touched. Loads ONLY the cached
1B model (fp32 then fp16), runs a forward pass on CPU / MPS and one greedy
generate with the 5.x legacy-cache flag, prints finite/stats per config.

Run from repo root:
    /opt/anaconda3/bin/python -m scripts.indic2_nan_check
"""
from __future__ import annotations

import gc
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch


def log(msg: str) -> None:
    print(f"[nan] {msg}")


def forward_check(tag: str, model, tokenizer, ids, mask, decoder_start: int = 2):
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask,
                    decoder_input_ids=torch.tensor([[decoder_start]]).to(ids.device))
    l = out.logits
    nan = (~torch.isfinite(l)).sum().item()
    log(f"{tag}: logits={tuple(l.shape)} nan_count={nan} "
        f"min={l.min().item() if nan == 0 else 'nan'} "
        f"max={l.max().item() if nan == 0 else 'nan'}")
    return nan == 0


def gen_check(tag: str, model, tokenizer, ids, mask):
    model.config.use_cache = False
    model.generation_config.use_cache = False
    model._supports_cache_class = False
    t0 = time.perf_counter()
    try:
        gen = model.generate(input_ids=ids, attention_mask=mask,
                             max_new_tokens=8, use_cache=False)
        dt = time.perf_counter() - t0
        log(f"{tag}: {dt:.1f}s out={tuple(gen.shape)} "
            f"raw='{tokenizer.batch_decode(gen, skip_special_tokens=False)[0][:80]}' "
            f"plain='{tokenizer.batch_decode(gen, skip_special_tokens=True)[0][:80]}'")
    except Exception as exc:
        log(f"{tag}: EXC {type(exc).__name__}: {exc}")


def main() -> None:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from backend.ml_models.indic2_service import _MODEL_ID, _patch_model_class, _patch_onnx_stub, _patch_tokenizer_class

    hi = pd.read_parquet("data/raw/in22_gen/pairs/eng-hin.parquet")["target"].iloc[0]
    tagged = f"hin_Deva tel_Telu {hi}"

    _patch_tokenizer_class()
    _patch_onnx_stub()
    _patch_model_class()
    tokenizer = AutoTokenizer.from_pretrained(
        _MODEL_ID, src_lang="eng_Latn", tgt_lang="hin_Deva", trust_remote_code=True)
    inputs = tokenizer(tagged, return_tensors="pt")
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    # --- MPS fp32, eval mode ---
    m = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID, trust_remote_code=True).to("mps").eval()
    ok = forward_check("mps fp32 eval", m, tokenizer, ids.to("mps"), mask.to("mps"))
    if ok:
        gen_check("mps fp32 eval greedy", m, tokenizer, ids.to("mps"), mask.to("mps"))
    del m; gc.collect(); torch.mps.empty_cache()

    # --- CPU fp32 ---
    m = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_ID, trust_remote_code=True).eval()
    ok = forward_check("cpu fp32 eval", m, tokenizer, ids, mask)
    if ok:
        gen_check("cpu fp32 eval greedy", m, tokenizer, ids, mask)
    del m; gc.collect()

    # --- MPS fp16 ---
    m = AutoModelForSeq2SeqLM.from_pretrained(
        _MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16).to("mps").eval()
    ok = forward_check("mps fp16 eval", m, tokenizer, ids.to("mps"), mask.to("mps"))
    if ok:
        gen_check("mps fp16 eval greedy", m, tokenizer, ids.to("mps"), mask.to("mps"))
    del m; gc.collect(); torch.mps.empty_cache()
    log("done")


if __name__ == "__main__":
    main()