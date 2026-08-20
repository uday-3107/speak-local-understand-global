"""indic2_where_nan.py — read-only: locate where NaN appears in the forward.

No downloads, no cache writes, no other models touched. Loads ONLY the cached
1B model on CPU and runs: (1) param NaN scan, (2) encoder forward alone,
(3) decoder forward alone, (4) decoder layer-by-layer — printing finiteness
at each step to pinpoint the exact failing op.

Run from repo root:
    /opt/anaconda3/bin/python -m scripts.indic2_where_nan
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch


def log(msg: str) -> None:
    print(f"[nan] {msg}")


def finite_stats(t: torch.Tensor, tag: str) -> None:
    t = t.detach()
    nan = (~torch.isfinite(t)).sum().item()
    log(f"{tag}: {tuple(t.shape)} nan_count={nan} "
        f"min={t.min().item() if nan == 0 else 'nan'} "
        f"max={t.max().item() if nan == 0 else 'nan'}")


def main() -> None:
    from backend.ml_models.indic2_service import _load

    loaded = _load()
    tokenizer, model, _ = loaded["tokenizer"], loaded["model"], loaded["device"]
    model = model.to("cpu").eval()
    log(f"loaded on cpu")

    bad = [(n, p.numel()) for n, p in model.named_parameters()
           if not torch.isfinite(p).all()]
    log(f"params with NaN/Inf: {bad if bad else 'none'}")

    hi = pd.read_parquet("data/raw/in22_gen/pairs/eng-hin.parquet")["target"].iloc[0]
    tagged = f"hin_Deva tel_Telu {hi}"
    ids = tokenizer(tagged, return_tensors="pt")["input_ids"]
    mask = torch.ones_like(ids)

    with torch.no_grad():
        enc = model.model.encoder(input_ids=ids, attention_mask=mask)
        finite_stats(enc.last_hidden_state, "encoder out")

        dec_in = torch.tensor([[2]])
        dec = model.model.decoder(
            input_ids=dec_in, attention_mask=torch.ones_like(dec_in),
            encoder_hidden_states=enc.last_hidden_state,
            encoder_attention_mask=mask, use_cache=False)
        finite_stats(dec.last_hidden_state, "decoder out (1 token)")

        layer = model.model.decoder.layers[0]
        hs = enc.last_hidden_state
        for i, (name, p) in enumerate(layer.named_parameters()):
            if not torch.isfinite(p).all():
                log(f"layer0 param non-finite: {name}")

        x = model.model.decoder.embed_tokens(dec_in) * model.model.decoder.scale_embedding \
            if hasattr(model.model.decoder, "scale_embedding") else model.model.decoder.embed_tokens(dec_in)
        finite_stats(x, "decoder embed token")
        out0, _, _ = layer(
            hidden_states=x,
            attention_mask=torch.ones_like(dec_in),
            encoder_hidden_states=enc.last_hidden_state,
            encoder_attention_mask=mask,
            use_cache=False,
        )
        finite_stats(out0[0] if isinstance(out0, tuple) else out0, "decoder layer0 out")
    log("done")


if __name__ == "__main__":
    main()