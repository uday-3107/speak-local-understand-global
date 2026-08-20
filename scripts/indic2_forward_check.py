"""indic2_forward_check.py — read-only forward/generate experiment.

No downloads, no cache writes, no other models touched. Loads ONLY the cached
1B model, runs one forward pass and three generate variants (greedy / cached /
beam / cpu) and prints logits stats + raw outputs — to locate why generation
emits </s> immediately.

Run from repo root:
    /opt/anaconda3/bin/python -m scripts.indic2_forward_check
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch


def log(msg: str) -> None:
    print(f"[fwd] {msg}")


def main() -> None:
    from transformers import __version__ as tv

    log(f"transformers={tv} torch={torch.__version__} "
        f"mps_available={torch.backends.mps.is_available()}")

    from backend.ml_models.indic2_service import _load

    loaded = _load()
    tokenizer, model, device = loaded["tokenizer"], loaded["model"], loaded["device"]
    log(f"device={device} dtype={next(model.parameters()).dtype}")

    hi = pd.read_parquet("data/raw/in22_gen/pairs/eng-hin.parquet")["target"].iloc[0]
    tagged = f"hin_Deva tel_Telu {hi}"
    inputs = tokenizer(tagged, return_tensors="pt")
    ids, mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
    log(f"input ids={ids.shape} max_id={ids.max().item()} src_vocab={tokenizer.src_vocab_size} "
        f"tgt_vocab={tokenizer.tgt_vocab_size}")

    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask,
                    decoder_input_ids=torch.tensor([[2]]).to(device))
    logits = out.logits
    n_nan = (~torch.isfinite(logits)).sum().item()
    top = torch.topk(logits[0, -1], 5)
    top_tokens = [(tokenizer.tgt_decoder.get(i.item(), f"id{i}"), round(v.item(), 3))
                  for i, v in zip(top.indices, top.values)]
    log(f"logits={tuple(logits.shape)} finite={n_nan == 0} nan_count={n_nan} "
        f"min={logits.min().item():.3f} max={logits.max().item():.3f} "
        f"mean={logits.mean().item():.3f} std={logits.std().item():.3f}")
    log(f"top-5 next-token: {top_tokens}")

    for name, kw in [("greedy", {}), ("greedy+cache", {"use_cache": True}),
                     ("beam5", {"num_beams": 5})]:
        t0 = time.perf_counter()
        try:
            gen = model.generate(input_ids=ids, attention_mask=mask,
                                 max_new_tokens=16, **kw)
            dt = time.perf_counter() - t0
            raw = tokenizer.batch_decode(gen, skip_special_tokens=False)[0][:80]
            plain = tokenizer.batch_decode(gen, skip_special_tokens=True)[0][:80]
            log(f"{name}: {dt:.1f}s out={tuple(gen.shape)} "
                f"raw='{raw}' plain='{plain}'")
        except Exception as exc:
            log(f"{name}: EXC {type(exc).__name__}: {exc}")

    cpu_model = model.to("cpu")
    t0 = time.perf_counter()
    try:
        gen = cpu_model.generate(input_ids=ids.cpu(), attention_mask=mask.cpu(),
                                 max_new_tokens=8)
        dt = time.perf_counter() - t0
        log(f"cpu greedy: {dt:.1f}s out={tuple(gen.shape)} "
            f"plain='{tokenizer.batch_decode(gen, skip_special_tokens=True)[0][:80]}'")
    except Exception as exc:
        log(f"cpu greedy: EXC {type(exc).__name__}: {exc}")
    log("done")


if __name__ == "__main__":
    main()