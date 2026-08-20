"""indic2_layer_nan.py — read-only: layer-by-layer NaN hunt inside the encoder.

No downloads, no cache writes, no other models touched. Prints which layer /
stage of the cached 1B encoder first produces non-finite values, plus the
runtime-resolved attention implementation.

Run from repo root:
    /opt/anaconda3/bin/python -m scripts.indic2_layer_nan
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch


def log(msg: str) -> None:
    print(f"[l] {msg}")


def chk(t: torch.Tensor, tag: str) -> bool:
    ok = bool(torch.isfinite(t).all())
    if not ok:
        log(f"NON-FINITE at {tag}: {tuple(t.shape)}")
    return ok


def main() -> None:
    from backend.ml_models.indic2_service import _load

    loaded = _load()
    tokenizer, model, _ = loaded["tokenizer"], loaded["model"], loaded["device"]
    model = model.to("cpu").eval()
    log(f"attn_impl (runtime) = {getattr(model.config, '_attn_implementation', None)}")

    hi = pd.read_parquet("data/raw/in22_gen/pairs/eng-hin.parquet")["target"].iloc[0]
    ids = tokenizer(f"hin_Deva tel_Telu {hi}", return_tensors="pt")["input_ids"]
    mask = torch.ones_like(ids)

    enc = model.model.encoder
    with torch.no_grad():
        emb = enc.embed_tokens(ids) * enc.embed_scale
        chk(emb, "embed_tokens")
        pos = enc.embed_positions(ids, emb)
        hs = emb + pos
        chk(hs, "embed+pos")
        am = _mask = torch.ones(1, 1, ids.shape[1], ids.shape[1])
        log(f"4d mask: {tuple(am.shape)} all_zero={bool((am == 0).all())}")

        for i, layer in enumerate(enc.layers):
            o = layer(hs, attention_mask=am)
            hs = o[0] if isinstance(o, tuple) else o
            if not chk(hs, f"encoder layer {i}"):
                l = layer
                for j, sub in enumerate(["self_attn"]):
                    pass
                break

        if torch.isfinite(hs).all():
            log("all 18 encoder layers finite — NaN must be downstream")
        dec_emb = model.model.decoder.embed_tokens(torch.tensor([[2]]))
        chk(dec_emb, "decoder embed")
    log("done")


if __name__ == "__main__":
    main()