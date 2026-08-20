"""Module 2: Data Preprocessing — build a clean multilingual speech dataset.

Pipeline stages (each logged):
  1. Load     — FLEURS test parquet + audio tar for en/hi/te; merge by filename
  2. Missing  — drop rows with no audio samples, NaN/empty transcription
  3. Text     — strip, collapse whitespace, unify quotes, drop non-letter tokens
  4. Audio    — spectral-gate noise reduction + peak normalization
  5. Features — MFCC (+ delta) and energy features per utterance
  6. Segment  — VAD-style split into segments at silence gaps > 0.35s
  7. Export   — clean parquet (features only) + normalized WAVs

Usage (from repo root):
    /opt/anaconda3/bin/python -m scripts.preprocess_dataset --limit 50
"""
from __future__ import annotations

import argparse
import io
import os
import re
import tarfile

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

RAW_DIR = "data/raw/fleurs"
OUT_DIR = "data/processed/clean"
CACHE_DIR = "data/cache/fleurs_audio"

LANGS = {
    "en": ("en_us", "English"),
    "hi": ("hi_in", "Hindi"),
    "te": ("te_in", "Telugu"),
}
SR = 16000

WHITESPACE = re.compile(r"\s+")
KEEP = re.compile(r"[\w\u0900-\u097F\u0C00-\u0C7F']")


def log(msg: str) -> None:
    print(f"[preprocess] {msg}")


def clean_text(text: str | None) -> str:
    """Text preprocessing: normalize whitespace + keep letters/numbers/indic scripts."""
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = WHITESPACE.sub(" ", text)
    text = " ".join("".join(c if KEEP.match(c) or c == " " else " " for c in text).split())
    return text


def read_mem_wav(raw: bytes, sr: int = SR) -> np.ndarray:
    data, file_sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    return np.asarray(data, dtype=np.float32)


def ensure_extracted(tar_path: str, cache_dir: str) -> bool:
    """Extract tar to cache once. Returns True when the cache is ready."""
    marker = os.path.join(cache_dir, ".done")
    if os.path.exists(marker):
        return True
    os.makedirs(cache_dir, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(cache_dir)
    except (tarfile.TarError, OSError) as e:
        log(f"extract failed for {tar_path}: {e}")
        return False
    open(marker, "w").close()
    return True


def load_audio(tar_path: str, member: str, cache_dir: str) -> np.ndarray | None:
    """Read one wav from the extracted cache (gzip tars can't seek per-file)."""
    if not ensure_extracted(tar_path, cache_dir):
        return None
    rel = member.replace("/", os.sep)
    wav_path = os.path.join(cache_dir, rel)
    if not os.path.exists(wav_path):
        return None
    try:
        x = read_mem_wav(open(wav_path, "rb").read())
        return x if x.size else None
    except (OSError, RuntimeError):
        return None


def reduce_noise(x: np.ndarray, n_fft: int = 1024) -> np.ndarray:
    """Spectral gating: subtract the per-band noise floor (first 0.2s) from the STFT."""
    if len(x) < SR // 5:
        return x
    stft = librosa.stft(x, n_fft=n_fft)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_floor = np.abs(librosa.stft(x[: SR // 5], n_fft=n_fft)).mean(axis=1, keepdims=True)
    if mag.shape[0] == noise_floor.shape[0]:
        mag = np.maximum(mag - noise_floor * 1.2, 0)
    return librosa.istft(mag * np.exp(1j * phase), length=len(x)).astype(np.float32)


def normalize_audio(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > 1e-6:
        x = x / peak * 0.98
    return x.astype(np.float32)


def frame_rms(x: np.ndarray, hop: int = 512) -> np.ndarray:
    frames = librosa.util.frame(x, frame_length=hop, hop_length=hop)
    return np.sqrt(np.mean(frames**2, axis=0))


def segment_utterance(x: np.ndarray, min_s: float = 1.0, gap_s: float = 0.35) -> list[np.ndarray]:
    """VAD-style segmentation: cut at silence gaps > gap_s; drop trailing silence."""
    hop = 512
    rms = frame_rms(x, hop)
    voiced = rms > 0.02
    segments = []
    start = None
    for i, v in enumerate(voiced):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if ((i - start) * hop / SR) >= 0.1:  # min segment ~0.1s
                segments.append(x[start * hop : i * hop])
            start = None
    if start is not None:
        segments.append(x[start * hop :])
    # pad tiny bursts into surroundings (avoid atomizing a sentence)
    merged = []
    for seg in segments:
        if merged and len(merged[-1]) / SR < min_s:
            merged[-1] = np.concatenate([merged[-1], seg])
        else:
            merged.append(seg)
    return [s for s in merged if len(s) >= SR * 0.3]


def mfcc_features(x: np.ndarray) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    return {
        "duration_s": round(len(x) / SR, 3),
        "mfcc_mean": round(float(mfcc.mean()), 4),
        "mfcc_std": round(float(mfcc.std()), 4),
        "mfcc_delta_mean": round(float(delta.mean()), 4),
        "mfcc_delta_std": round(float(delta.std()), 4),
        "rmse": round(float(np.sqrt(np.mean(x**2))), 4),
    }


def build_dataset(lang: str, split: str, limit: int | None) -> pd.DataFrame:
    tag, name = LANGS[lang]
    raw_dir = os.path.join(RAW_DIR, tag)
    df = pd.read_parquet(os.path.join(raw_dir, f"{split}.parquet"))
    if limit:
        df = df.head(limit)
    rows = []
    for row in df.itertuples():
        audio = load_audio(
            os.path.join(raw_dir, "audio", f"{split}.tar.gz"),
            f"{split}/{row.fileName}",
            os.path.join(CACHE_DIR, tag, split),
        )
        if audio is None:
            continue
        txt = clean_text(getattr(row, "transcription", ""))
        if not txt:
            continue
        rows.append(
            {
                "file": row.fileName,
                "id": str(row.id),
                "language": lang,
                "language_name": name,
                "gender": getattr(row, "gender", None),
                "source_text": txt,
                "audio": audio,
            }
        )
    return pd.DataFrame(rows)


def process_lang(lang: str, limit: int | None) -> int:
    split = "test"
    df = build_dataset(lang, split, limit)
    log(f"load {lang}: {len(df)} utterances")
    before = len(df)

    df = df[df["source_text"].str.len() >= 3]
    df = df.drop_duplicates(subset=["source_text", "language"])
    log(f"missing/dup cleanup: {before} -> {len(df)}")

    records = []
    seg_count = 0
    for row in df.itertuples():
        x = normalize_audio(reduce_noise(row.audio))
        for seg in segment_utterance(x):
            seg_count += 1
            feat = mfcc_features(seg)
            records.append(
                {
                    "file": row.file,
                    "id": row.id,
                    "language": row.language,
                    "language_name": row.language_name,
                    "gender": row.gender,
                    "source_text": row.source_text,
                    "segment_id": seg_count,
                    **feat,
                }
            )
            out = os.path.join(OUT_DIR, f"{lang}_{seg_count:06d}.wav")
            sf.write(out, seg, SR)
    log(f"segments: {seg_count} (from {len(df)} utterances)")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="cap utterances per language")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    summary = []
    all_rows: list[dict] = []
    for lang in LANGS:
        summary.append(f"{lang}: ...")
        all_rows.extend(process_lang(lang, args.limit))

    if not all_rows:
        log("no rows produced — check data/raw/fleurs/<lang>/audio/<split>.tar.gz")
        return

    meta = pd.DataFrame(all_rows).drop(columns=["file", "id"])
    meta.to_parquet(os.path.join(OUT_DIR, "clean.parquet"), index=False)
    log(
        f"wrote {len(meta)} segments to {OUT_DIR}/clean.parquet "
        f"(languages: {len(meta['language'].unique())})"
    )
    print("\nLanguage summary:")
    print(meta.groupby("language_name").agg({"source_text": "count"}).to_string())
    print("\nFirst rows:")
    print(meta.head(3).to_string())


if __name__ == "__main__":
    main()