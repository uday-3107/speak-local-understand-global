"""Module 5 gap filler: sentence embeddings + translation refinement.

Two missing Module-5 pieces, built on the cached NLLB-200 model:
  1. Multilingual sentence embeddings  — encode sentences in en/hi/te with
     NLLB encoder (mean-pooled hidden states, L2-normalized) and show
     cross-lingual cosine similarity.
  2. Translation refinement — a deterministic, rule-based post-processing
     layer (whitespace, punctuation, orthography, numbers) applied to raw
     NLLB output, with a before/after comparison.

Run from repo root (user runs):
    /opt/anaconda3/bin/python -m scripts.nlp_embeddings_refine

Writes: data/processed/nlp/embeddings_refine.json
"""
from __future__ import annotations

import json
import os
import re
import unicodedata

import numpy as np
import torch

LANG_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "te": "tel_Telu"}
OUT_DIR = "data/processed/nlp"
MODEL_ID = os.environ.get("SLUG_NLLB_MODEL", "facebook/nllb-200-distilled-600M")

MULTI_SPACE = re.compile(r"\s{2,}")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:।॥])")
NO_SPACE_AFTER_PUNCT = re.compile(r"(?<=[,;:!?।॥])(?=[\w\u0900-\u097F\u0C00-\u0C7F])")
MULTI_PUNCT = re.compile(r"([.!?।]{2,})")
SPACED_LETTERS = re.compile(r"\b([A-Za-z])\s+(?=[A-Za-z])")  # "h e l l o" -> latin only
NUM_GROUP = re.compile(r"(\d)\s?([.,])\s?(\d)")
STRAY_PERIODS = re.compile(r"\.\s*\.(\s*\.)*")


def log(msg: str) -> None:
    print(f"[module5-gap] {msg}")


# ------------------------- NLLB-IN-PLACE embeddings -------------------------

_LOADED: dict = {}


def _get_nllb() -> dict:
    if not _LOADED:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device)
        model.eval()
        _LOADED.update(tokenizer=tokenizer, model=model, device=device)
    return _LOADED


def sentence_embedding(text: str, lang: str) -> np.ndarray:
    """Mean-pooled encoder hidden state for one sentence, L2-normalized."""
    loaded = _get_nllb()
    tok, model, device = loaded["tokenizer"], loaded["model"], loaded["device"]
    tok.src_lang = LANG_CODES.get(lang, LANG_CODES["en"])
    inputs = tok(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        hidden = model.model.encoder(**inputs).last_hidden_state  # (1, T, D)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    vec = torch.nn.functional.normalize(vec, p=2, dim=1)
    return vec.squeeze(0).cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# --------------------------- Translation refinement --------------------------

def refine_text(text: str) -> str:
    """Rule-based refinement of raw MT output. Pure function (unit-testable)."""
    if not text:
        return text
    t = unicodedata.normalize("NFC", text)
    t = NUM_GROUP.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}", t)
    t = t.replace(" ", " ")  # nbsp
    t = SPACED_LETTERS.sub(lambda m: m.group(1), t)
    t = SPACE_BEFORE_PUNCT.sub(r"\1", t)
    t = NO_SPACE_AFTER_PUNCT.sub(" ", t)  # "है,और" -> "है, और"
    t = MULTI_SPACE.sub(" ", t).strip()
    t = MULTI_PUNCT.sub(lambda m: m.group(1)[0], t)
    t = t.replace("..", ".").replace(" .", ".")
    t = t.rstrip()
    if t and t[-1] not in ".!?।":
        t += "."
    return t


# ------------------------------- demo pipeline --------------------------------

def embedding_demo() -> dict:
    log("loading NLLB for embeddings…")
    pairs = [
        ("The teacher explained the concept clearly.", "en"),
        ("Education is important for everyone.", "en"),
        ("शिक्षा सभी के लिए महत्वपूर्ण है।", "hi"),
        ("वहाँ पर्याप्त ध्यान देना है।", "hi"),
        ("విద్య అందరికీ ముఖ్యమైనది.", "te"),
        ("విజ్ఞాన శాస్త్రం ఉపాధ్యాయుడు వివరించారు.", "te"),
    ]
    vectors = {}
    for text, lang in pairs:
        vectors[text] = sentence_embedding(text, lang)
    sims = []
    keys = list(vectors.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sims.append({
                "a": keys[i], "b": keys[j],
                "cosine_similarity": round(cosine_sim(vectors[keys[i]], vectors[keys[j]]), 4),
            })
    return {"pairs": pairs, "similarities": sims}


def refine_demo() -> dict:
    samples = [
        "यह अवधारणा सरल है   ,और महत्वपूर्ण  है .",
        "विद्या ప్రపంచం మారుతుంది .   ఇది విలువైనది.",
        "The   results are  accurate ,  clear,and useful.",
        "एक समय में एक विचार. .",
    ]
    out = []
    for s in samples:
        out.append({"raw": s, "refined": refine_text(s)})
    return {"samples": out}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    emb = embedding_demo()
    print("\n=== Multilingual sentence similarity (NLLB encoder) ===")
    for s in emb["similarities"]:
        print(f"  {s['cosine_similarity']:.3f}  {s['a'][:20]}… <-> {s['b'][:20]}…")

    print("\n=== Rule-based translation refinement (before/after) ===")
    ref = refine_demo()
    for r in ref["samples"]:
        print(f"  raw      : {r['raw']!r}")
        print(f"  refined  : {r['refined']!r}")

    result = {"embeddings": emb, "refinement": ref}
    with open(os.path.join(OUT_DIR, "embeddings_refine.json"), "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log("saved " + os.path.join(OUT_DIR, "embeddings_refine.json"))
    log("done")


if __name__ == "__main__":
    main()