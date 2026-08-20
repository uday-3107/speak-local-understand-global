"""indic2_inspect.py — READ-ONLY diagnosis of the cached IndicTrans2 vocab files.

No downloads. No cache writes. No model loading (no torch model in RAM).
Only reads these cached files:
  * config.json               — expected vocab sizes
  * dict.SRC.json / dict.TGT  — token -> id maps
  * model.SRC / model.TGT     — SentencePiece models
  * model.safetensors header  — embedding/lm_head shapes (read-only)

Goal: find which file is corrupt/mismatched (why the tokenizer emits garbage
tokens and the model immediately emits </s>).

Run from repo root:
    /opt/anaconda3/bin/python -m scripts.indic2_inspect
"""
from __future__ import annotations

import json

import sentencepiece as spm
from safetensors import safe_open

SNAP = ("/Users/gunjiudaynarayana/.cache/huggingface/hub/"
        "models--ai4bharat--indictrans2-en-indic-1B/"
        "snapshots/10e65a9951a1e922cd109a95e8aba9357b62144b")

TAGS = ["hin_Deva", "tel_Telu", "eng_Latn"]
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]


def log(msg: str) -> None:
    print(f"[inspect] {msg}")


def main() -> None:
    cfg = json.load(open(f"{SNAP}/config.json"))
    log(f"config: vocab_size={cfg.get('vocab_size')} "
        f"encoder_vocab={cfg.get('encoder_vocab_size')} "
        f"decoder_vocab={cfg.get('decoder_vocab_size')} "
        f"d_model={cfg.get('d_model')} model_type={cfg.get('model_type')}")

    dicts = {}
    for name in ["dict.SRC.json", "dict.TGT.json"]:
        d = json.load(open(f"{SNAP}/{name}"))
        dicts[name] = d
        sample = list(d.items())[:6]
        log(f"{name}: entries={len(d)} first={sample}")
        log(f"  lang tags present: { {t: d.get(t) for t in TAGS} }")
        log(f"  specials present:  { {t: d.get(t) for t in SPECIALS} }")
        log(f"  max id={max(d.values())} min id={min(d.values())} "
            f"id range ok={len(d) > max(d.values())}")

    for side, spm_fp, dict_fp in [
        ("SRC", "model.SRC", "dict.SRC.json"),
        ("TGT", "model.TGT", "dict.TGT.json"),
    ]:
        sp = spm.SentencePieceProcessor(model_file=f"{SNAP}/{spm_fp}")
        d = dicts[dict_fp]
        pieces = sp.EncodeAsPieces("सेवा संबंधी लोगों के लिए भेष एक गुण है")
        missing = [p for p in pieces if p not in d]
        log(f"{spm_fp}: spm pieces={sp.get_piece_size()} vs dict entries={len(d)} "
            f"MATCH={sp.get_piece_size() == len(d)}")
        log(f"  sample encode={pieces}")
        log(f"  pieces missing from dict: {missing if missing else 'none'}")

    with safe_open(f"{SNAP}/model.safetensors", framework="pt") as f:
        names = list(f.keys())
        log(f"model.safetensors: {len(names)} tensors")
        for n in sorted(names):
            if "embed" in n or "lm_head" in n or "position" in n:
                log(f"  {n} {f.get_slice(n).get_shape()}")
    log("done — nothing read was modified")


if __name__ == "__main__":
    main()