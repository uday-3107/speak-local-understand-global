"""Verify all 6 language pairs end-to-end with real models.

STT: whisper medium (en/hi), large-v3 (te)  MT: NLLB-200-600M
Usage: /opt/anaconda3/bin/python scripts/verify_pairs.py
"""
import os
import time

import av
import numpy as np

os.environ.setdefault("SLUG_WHISPER_MODEL", "medium")

from backend.ml_models import translate
from backend.ml_models.whisper_service import WhisperService


def to_16k_mono_float(path: str) -> np.ndarray:
    container = av.open(path)
    stream = container.streams.audio[0]
    frames = []
    resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
    for frame in container.decode(stream):
        for r in resampler.resample(frame):
            frames.append(r.to_ndarray().reshape(-1))
    return np.concatenate(frames).astype(np.float32)


CLIPS = {
    "en": "data/processed/test_audio/en_10004088536354799741.wav",
    "hi": "data/processed/test_audio/hi_0.mp3",
    "te": "data/processed/test_audio/te_0.mp3",
}

PAIRS = [
    ("en", "hi"),
    ("en", "te"),
    ("hi", "en"),
    ("hi", "te"),
    ("te", "en"),
    ("te", "hi"),
]


def main() -> None:
    stt = WhisperService()
    results = []
    for src, tgt in PAIRS:
        clip = CLIPS[src]
        print(f"\n=== {src} -> {tgt}  (clip: {os.path.basename(clip)}) ===")
        audio = to_16k_mono_float(clip)
        t0 = time.perf_counter()
        out = stt.transcribe_np(audio, language=src)
        stt_s = time.perf_counter() - t0
        if not out.text:
            print(f"  STT: <no speech> ({stt_s:.1f}s)")
            continue
        t0 = time.perf_counter()
        translated, mt_ms, model_used = translate(out.text, out.language or src, tgt)
        mt_s = time.perf_counter() - t0
        print(f"  STT [{stt_s:.1f}s, {out.model}]: {out.text!r}")
        print(f"  MT  [{mt_s:.1f}s, {model_used}]: {translated!r}")
        results.append((f"{src}->{tgt}", out.text, translated, stt_s, mt_s, out.model, model_used))

    print("\n\n=== SUMMARY ===")
    for pair, src_text, tgt_text, stt_s, mt_s, stt_m, mt_m in results:
        print(f"{pair:7s} STT={stt_m:11s} {stt_s:4.1f}s | MT={mt_m:5s} {mt_s:4.1f}s")


if __name__ == "__main__":
    main()
