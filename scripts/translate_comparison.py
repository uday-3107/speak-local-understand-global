"""Module 6: Offline Translation Engine — Rule-Based vs AI-Based comparison.

Implements two offline translation engines for en/hi/te and compares them on
IN22-Gen parallel data (human reference translations available):

  1. RuleBasedTranslator — deterministic dictionary + post-processing rules
     (word lookup, punctuation guards). Cheap, fully offline, limited
     coverage by design.
  2. AiTranslator — NLLB-200-distilled-600M seq2seq engine from the HF
     cache (the same model used by the live WebSocket pipeline).

Per direction (en-hi, hi-en, en-te, te-en) it reports:
  * BLEU-4 (nltk corpus BLEU, smoothing) vs the human reference
  * lexical coverage of the rule-based translator (OOV proxy)
  * mean latency per sentence

Run from repo root (user runs):
    /opt/anaconda3/bin/python -m scripts.translate_comparison --limit 6

Writes: data/processed/mt/translation_engine.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

PAIRS_DIR = "data/raw/in22_gen/pairs"
OUT_DIR = "data/processed/mt"

# (description, parquet file, source column, target column)
PARALLEL = {
    "en->hi": ("eng-hin.parquet", "source", "target"),
    "hi->en": ("hin-eng.parquet", "source", "target"),
    "en->te": ("eng-tel.parquet", "source", "target"),
    "te->en": ("tel-eng.parquet", "source", "target"),
}

# Small word-by-word dictionary for the rule-based engine (intentionally small).
WORD_DICT = {
    ("en", "hi"): {
        "the": "यह", "a": "एक", "of": "का", "and": "और", "is": "है",
        "in": "में", "for": "के लिए", "with": "के साथ", "are": "हैं",
        "on": "पर", "this": "यह", "that": "वह", "student": "छात्र",
        "teacher": "शिक्षक", "language": "भाषा", "book": "किताब",
        "school": "स्कूल", "water": "पानी", "education": "शिक्षा",
        "important": "महत्वपूर्ण", "right": "अधिकार", "person": "व्यक्ति",
        "time": "समय", "people": "लोग", "there": "वहाँ",
    },
    ("hi", "en"): {
        "यह": "this", "एक": "a", "का": "of", "और": "and", "है": "is",
        "में": "in", "के": "the", "छात्र": "student", "शिक्षक": "teacher",
        "भाषा": "language", "किताब": "book", "पानी": "water",
        "शिक्षा": "education", "समय": "time", "लोग": "people",
        "महत्वपूर्ण": "important",
    },
    ("en", "te"): {
        "the": "ఈ", "a": "ఒక", "of": "యొక్క", "and": "మరియు", "is": "ఉంది",
        "in": "లో", "for": "కోసం", "to": "కు", "are": "ఉన్నాయి",
        "on": "పై", "student": "విద్యార్థి", "teacher": "ఉపాధ్యాయుడు",
        "language": "భాష", "book": "పుస్తకం", "water": "నీరు",
        "education": "విద్య", "people": "ప్రజలు", "important": "ముఖ్యమైన",
        "this": "ఈ", "that": "ఆ",
    },
    ("te", "en"): {
        "ఈ": "this", "ఒక": "a", "యొక్క": "of", "మరియు": "and",
        "ఉంది": "is", "లో": "in", "కోసం": "for", "కు": "to",
        "విద్యార్థి": "student", "ఉపాధ్యాయుడు": "teacher", "భాష": "language",
        "పుస్తకం": "book", "విద్య": "education", "ప్రజలు": "people",
        "నీరు": "water", "ముఖ్యమైన": "important",
    },
}

_TOKEN_RE = re.compile(r"([^\w\u0900-\u097F\u0C00-\u0C7F']+)", re.UNICODE)


def log(msg: str) -> None:
    print(f"[module6] {msg}")


# ----------------------------- Rule engine -----------------------------

class RuleTranslator:
    name = "rule_based"

    def translate(self, text: str, src: str, tgt: str) -> str:
        table = WORD_DICT.get((src, tgt), {})
        parts = _TOKEN_RE.split(text)
        for i in range(0, len(parts), 2):
            word = parts[i].lower()
            parts[i] = table.get(word, parts[i])
        return "".join(parts).strip()

    def coverage(self, text: str, src: str, tgt: str) -> float:
        table = WORD_DICT.get((src, tgt), {})
        words = [p for p in _TOKEN_RE.split(text) if p and not
                 re.match(r"^[\W_]+$", p, re.UNICODE)]
        if not words:
            return 0.0
        return sum(1 for w in words for k in (w.lower(),)
                   if k in table) / len(words)


# ----------------------------- AI engine -----------------------------

class AiTranslator:
    name = "ai_nllb"

    def __init__(self) -> None:
        from backend.ml_models.nllb_service import NllbService

        self._svc = NllbService()

    def translate(self, text: str, src: str, tgt: str) -> str:
        out, _ = self._svc.translate(text, src, tgt)
        return out


def load_pairs(name: str, limit: int | None):
    fname, sc, tg = PARALLEL[name]
    df = pd.read_parquet(os.path.join(PAIRS_DIR, fname))
    if limit:
        df = df.head(limit)
    return df[sc].tolist(), df[tg].tolist()


def bleu4(reference: list[str], pred: list[str]) -> float:
    refs = [[r.split()] for r in reference]
    smooth = SmoothingFunction().method1
    return round(float(corpus_bleu(refs, [p.split() for p in pred],
                                   smoothing_function=smooth)), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6, help="sentences per direction")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rule = RuleTranslator()
    ai = AiTranslator()
    log("loading AI engine (NLLB)…")

    results = []
    for direction, (fname, sc, tg) in PARALLEL.items():
        log(f"evaluating {direction}…")
        src, tgt = direction.split("->")
        sources, reference = load_pairs(direction, args.limit)

        t0 = time.perf_counter()
        rb_pred = [rule.translate(s, src, tgt) for s in sources]
        rb_lat = (time.perf_counter() - t0) * 1000 / max(len(sources), 1)
        t0 = time.perf_counter()
        ai_pred = [ai.translate(s, src, tgt) for s in sources]
        ai_lat = (time.perf_counter() - t0) * 1000 / max(len(sources), 1)

        bleu_rb = bleu4(reference, rb_pred)
        bleu_ai = bleu4(reference, ai_pred)
        cov = sum(rule.coverage(s, src, tgt) for s in sources) / max(len(sources), 1)

        results.append({
            "direction": direction,
            "samples": len(sources),
            "bleu_rule": bleu_rb,
            "bleu_ai": bleu_ai,
            "rule_coverage": round(float(cov), 4),
            "latency_ms_rule": round(rb_lat, 1),
            "latency_ms_ai": round(ai_lat, 1),
        })
        log(f"{direction}: rule BLEU={bleu_rb} (cov {cov:.2f}) | AI BLEU={bleu_ai}")

    with open(os.path.join(OUT_DIR, "translation_engine.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Module 6: Rule-Based vs AI-Based translation ===")
    print("{:<10} {:>9} {:>9} {:>9} {:>11} {:>12}".format(
        "direction", "BLEU-rule", "BLEU-AI", "cov", "ms/rule", "ms/AI"))
    for r in results:
        print("{:<10} {:>9} {:>9} {:>9} {:>11} {:>12}".format(
            r["direction"], r["bleu_rule"], r["bleu_ai"], r["rule_coverage"],
            r["latency_ms_rule"], r["latency_ms_ai"]))
    print("\nSaved to:", os.path.join(OUT_DIR, "translation_engine.json"))
    log("done")


if __name__ == "__main__":
    main()