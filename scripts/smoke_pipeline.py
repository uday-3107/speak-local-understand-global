"""Smoke test: real webm decode -> whisper STT -> NLLB MT, simulating the live WS pipeline.

Usage: .venv/bin/python scripts/smoke_pipeline.py [audio_path] [--src en|hi|te]
"""
import argparse
import os
import tempfile
import time

import av
import numpy as np

from backend.api.websocket import decode_webm_to_wav
from backend.ml_models import translate
from backend.ml_models.whisper_service import WhisperService


def to_webm(path: str) -> bytes:
    """Transcode any audio file to opus/webm (same codec MediaRecorder produces)."""
    container = av.open(path)
    stream = container.streams.audio[0]
    tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    tmp.close()
    out_container = av.open(tmp.name, "w", format="webm")
    out_stream = out_container.add_stream("libopus", rate=16000)
    out_stream.layout = "mono"
    for frame in container.decode(stream):
        frame.pts = None
        for packet in out_stream.encode(frame):
            out_container.mux(packet)
    for packet in out_stream.encode(None):
        out_container.mux(packet)
    out_container.close()
    with open(tmp.name, "rb") as fh:
        data = fh.read()
    os.unlink(tmp.name)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--src", default="en", choices=["en", "hi", "te"])
    parser.add_argument("--tgt", default="hi", choices=["en", "hi", "te"])
    args = parser.parse_args()

    print(f"transcoding {args.audio} -> webm/opus ...")
    webm = to_webm(args.audio)
    print(f"webm size: {len(webm) / 1024:.1f} KiB")

    print("decoding webm -> wav (16kHz mono PCM16) ...")
    wav = decode_webm_to_wav(webm)
    print(f"wav bytes: {len(wav)}")

    print(f"whisper({args.src}) ...")
    t0 = time.perf_counter()
    stt = WhisperService().transcribe_bytes(wav, language=args.src)
    print(f"  [{time.perf_counter() - t0:.2f}s] {stt.text!r}  (lang={stt.language}, model={stt.model})")

    print(f"translate {stt.language or args.src} -> {args.tgt} ...")
    t0 = time.perf_counter()
    translated, latency, model_used = translate(stt.text, stt.language or args.src, args.tgt)
    print(f"  [{time.perf_counter() - t0:.2f}s] {translated!r}  (model={model_used}, latency={latency}ms)")


if __name__ == "__main__":
    main()