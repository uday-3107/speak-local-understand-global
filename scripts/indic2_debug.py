"""indic2_debug.py — print-only diagnostic for the cached IndicTrans2 en-indic-1B.

Question: why did the probe return BLEU 0.0 / script 0.0 / ~200ms per sentence
(no real generation) with the wrapper in backend/ml_models/indic2_service.py?

Tries two tokenizer input styles and prints the raw tokenizer/model internals:
  A) wrapper style:  f"{src_lang} {tgt_lang} {text}"  (what the probe used)
  B) official style: tokenizer(text) with src_lang/tgt_lang as construction
     kwargs (the API documented by ai4bharat)

Strictly read-only: offline-enforced (HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE),
loads ONLY the cached 1B model, prints to stdout, writes NO files, prints
HF cache size before/after so nothing can be hidden.

Run from repo root:
    /opt/anaconda3/bin/python -m scripts.indic2_debug
"""
from __future__ import annotations

import os
import subprocess
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import torch

PAIRS = "data/raw/in22_gen/pairs"


def log(msg: str) -> None:
    print(f"[debug] {msg}")


def cache_size() -> str:
    r = subprocess.run(["du", "-sh", os.path.expanduser("~/.cache/huggingface")],
                       capture_output=True, text=True)
    return r.stdout.split()[0] if r.returncode == 0 else "n/a"


def main() -> None:
    before = cache_size()
    log(f"HF cache BEFORE: {before}")

    from transformers import AutoTokenizer

    from backend.ml_models.indic2_service import _MODEL_ID, _load

    loaded = _load()
    tokenizer, model, device = loaded["tokenizer"], loaded["model"], loaded["device"]
    log(f"model on {device}, dtype={next(model.parameters()).dtype}")

    added = getattr(tokenizer, "added_tokens_decoder", {})
    specials = [str(t) for t in added.values()
                if t and ("_Latn" in str(t) or "_Deva" in str(t) or "_Telu" in str(t) or str(t)[:2] == "<2")]
    log(f"added special tokens: {len(added)} total; lang/tag tokens sample: {specials[:16]}")

    en_hi = pd.read_parquet(f"{PAIRS}/eng-hin.parquet")
    en_te = pd.read_parquet(f"{PAIRS}/eng-tel.parquet")
    en_s, hi_s, te_s = en_hi["source"].iloc[0], en_hi["target"].iloc[0], en_te["target"].iloc[0]

    with torch.no_grad():
        # A) wrapper style (what the probe used): inline "src_lang tgt_lang text"
        tagged = f"hin_Deva tel_Telu {hi_s}"
        ids = tokenizer(tagged, return_tensors="pt").input_ids.to(device)
        log(f"[A wrap hi->te] tagged='{tagged[:80]}'")
        log(f"[A wrap hi->te] in_ids={ids.shape[1]} decode='{tokenizer.decode(ids[0])[:120]}'")
        t0 = time.perf_counter()
        out = model.generate(input_ids=ids, num_beams=5, max_new_tokens=32, use_cache=False)
        dt = time.perf_counter() - t0
        log(f"[A wrap hi->te] generate {dt:.1f}s out_ids={out.shape[1]} "
            f"raw='{tokenizer.batch_decode(out, skip_special_tokens=False)[0][:160]}' "
            f"plain='{tokenizer.batch_decode(out, skip_special_tokens=True)[0][:160]}'")

        # B) official style: src/tgt langs as tokenizer kwargs at construction
        try:
            tok2 = AutoTokenizer.from_pretrained(
                _MODEL_ID, src_lang="hin_Deva", tgt_lang="tel_Telu", trust_remote_code=True
            )
            ids2 = tok2(hi_s, return_tensors="pt").input_ids.to(device)
            log(f"[B offc hi->te] in_ids={ids2.shape[1]} decode='{tok2.decode(ids2[0])[:120]}'")
            t0 = time.perf_counter()
            out2 = model.generate(input_ids=ids2, num_beams=5, max_new_tokens=32, use_cache=False)
            dt = time.perf_counter() - t0
            log(f"[B offc hi->te] generate {dt:.1f}s out_ids={out2.shape[1]} "
                f"raw='{tok2.batch_decode(out2, skip_special_tokens=False)[0][:160]}' "
                f"plain='{tok2.batch_decode(out2, skip_special_tokens=True)[0][:160]}'")
        except Exception as exc:
            log(f"[B offc hi->te] FAILED: {exc!r}")

        # C) quick input sanity for the other two source languages, wrapper style
        for label, tagged in [("en->hi", f"eng_Latn hin_Deva {en_s}"),
                              ("te->en wrap", f"tel_Telu eng_Latn {te_s}")]:
            ids = tokenizer(tagged, return_tensors="pt").input_ids.to(device)
            log(f"[C {label}] in_ids={ids.shape[1]} decode='{tokenizer.decode(ids[0])[:120]}'")

    after = cache_size()
    log(f"HF cache AFTER:  {after}  ({'UNCHANGED' if before == after else 'CHANGED?!'})")
    log("done — nothing was written")


if __name__ == "__main__":
    main()