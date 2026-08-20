#!/usr/bin/env python3
"""Week 1 latency spike: faster-whisper STT + MarianMT/IndicTrans2 MT on local CPU/MPS.

Usage:
    python scripts/benchmark_latency.py --stage all
    python scripts/benchmark_latency.py --stage stt --whisper small
    python scripts/benchmark_latency.py --stage mt-marian
    python scripts/benchmark_latency.py --stage mt-indic
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

AUDIO_DIR = Path("data/processed/test_audio")

GROUND_TRUTH = {
    "hi_0.mp3": "हमने उसका जन्मदिन मनाया।",
    "hi_1.mp3": "साउथ दिल्ली नगर निगम सख्त, शॉपिंग मॉल के बाहर नहीं",
    "hi_2.mp3": "उत्तर कोरिया ने अमेरिका को दी हमले की धमकी",
    "hi_3.mp3": "अगले कमरे में अनेक रोमन मूर्तियाँ हैं।",
    "hi_4.mp3": "तुम ने टॉम को कहाँ भेज दिया?",
    "te_0.mp3": "ఆ మందహాసమని భావము",
    "te_1.mp3": "పల్లె పైరుగాలి పరిరంభణమ్ములు",
    "te_2.mp3": "ఆ అబ్బాయి చెడ్డవాడు.",
    "en_10004088536354799741.wav": "a tornado is a spinning column of very low-pressure air which",
    "en_10012216926115652402.wav": "the approach to obtaining information was different",
    "en_10035729252730569448.wav": "hsieh implied during the election that ma might flee the country",
}

DEVICE = "mps"


def load_whisper(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def bench_stt(whisper_size: str, lang: str | None, whisper_big: bool) -> None:
    model = load_whisper(whisper_size)
    label = f"{whisper_size}{' (forced lang)' if lang else ''}"
    print(f"\n=== STT: faster-whisper {label} (cpu/int8) ===")
    files = sorted(AUDIO_DIR.glob("*.mp3")) + sorted(AUDIO_DIR.glob("*.wav"))
    if lang == "te":
        files = [p for p in files if "te_" in p.name]
    elif lang == "hi":
        files = [p for p in files if "hi_" in p.name]
    elif lang == "en":
        files = [p for p in files if "en_" in p.name]
    print(f"{'file':<28}{'dur(s)':>7}{'lat(s)':>8}{'RTF':>7}  transcript / ground truth")
    for path in files:
        t0 = time.perf_counter()
        segments, info = model.transcribe(str(path), beam_size=5, language=lang)
        text = " ".join(s.text for s in segments).strip()
        lat = time.perf_counter() - t0
        gt = GROUND_TRUTH.get(path.name, "")
        print(f"{path.name:<28}{info.duration:7.1f}{lat:8.2f}{lat/info.duration:7.2f}  {text[:55]}")
        if gt:
            print(f"{'':<47}  gt: {gt[:55]}")


def load_marian():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    out = {}
    for pair, m in [("en-hi", "Helsinki-NLP/opus-mt-en-hi"), ("hi-en", "Helsinki-NLP/opus-mt-hi-en")]:
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForSeq2SeqLM.from_pretrained(m).to(DEVICE)
        out[pair] = (model, tok)
    return out


def bench_marian() -> None:
    import torch

    models = load_marian()
    print(f"\n=== MT: MarianMT (opus-mt) on {DEVICE.upper()} ===")
    samples = {
        "en-hi": "A uniform is often viewed as projecting a positive image of an organisation.",
        "en-hi": "We celebrated his birthday together with the whole family.",
        "hi-en": "हमने उसका जन्मदिन मनाया।",
        "hi-en": "उत्तर कोरिया ने अमेरिका को दी हमले की धमकी",
    }
    for pair, text in samples.items():
        model, tok = models[pair]
        inputs = tok(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=200)
            lat = time.perf_counter() - t0
        decoded = tok.batch_decode(out, skip_special_tokens=True)[0]
        print(f"{pair:<7} {lat:6.2f}s  {text[:38]} -> {decoded[:60]}")


def load_indic():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_id = "ai4bharat/indictrans2-en-indic-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn", tgt_lang="hin_Deva")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(DEVICE)
    return model, tokenizer


def bench_indic() -> None:
    import torch

    model, tokenizer = load_indic()
    print(f"\n=== MT: IndicTrans2 en-indic-1B on {DEVICE.upper()} ===")
    samples = [
        ("hin_Deva", "We celebrated his birthday together with the whole family."),
        ("hin_Deva", "A uniform is often viewed as projecting a positive image of an organisation."),
        ("tel_Telu", "A uniform is often viewed as projecting a positive image of an organisation."),
        ("tel_Telu", "The teacher explained the concept of photosynthesis to the students."),
    ]
    for tgt_lang, text in samples:
        tokenizer.tgt_lang = tgt_lang
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, num_beams=5, max_new_tokens=256)
        lat = time.perf_counter() - t0
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        print(f"{tgt_lang:<10} {lat:6.2f}s  {text[:38]} -> {decoded[:60]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 1 latency spike")
    ap.add_argument("--stage", choices=["all", "stt", "mt-marian", "mt-indic"], default="all")
    ap.add_argument("--whisper", default="small", help="whisper model size (tiny/base/small/medium)")
    ap.add_argument("--lang", default=None, help="force transcript language code, e.g. te")
    ap.add_argument("--whisper-big", action="store_true", help="use larger whisper, Telugu clips only")
    args = ap.parse_args()

    if args.stage in ("all", "stt"):
        bench_stt(args.whisper, args.lang, args.whisper_big)
    if args.stage in ("all", "mt-marian"):
        bench_marian()
    if args.stage in ("all", "mt-indic"):
        bench_indic()


if __name__ == "__main__":
    sys.exit(main())
