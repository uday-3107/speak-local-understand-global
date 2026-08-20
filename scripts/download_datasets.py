#!/usr/bin/env python3
"""Download Common Voice, IN22-Gen, BPCC and FLEURS datasets into data/raw/.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --cv --fleurs
    python scripts/download_datasets.py --in22 --bpcc
    python scripts/download_datasets.py --dry-run

Layout on disk:
    data/raw/<dataset>/<config>/<split>.parquet
    data/raw/MANIFEST.json

Auth: IN22-Gen and BPCC are gated. Log in once with `huggingface-cli login`
and accept the dataset terms on the Hub, or export HF_TOKEN=<read token>.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from datasets import Audio, load_dataset
from huggingface_hub import get_token, list_repo_tree

COMMON_VOICE = "fixie-ai/common_voice_17_0"
IN22_GEN = "ai4bharat/IN22-Gen"
BPCC = "ai4bharat/BPCC"
FLEURS = "google/fleurs"

CV_LANGS = ["hi", "te"]
CV_SPLITS = ["train", "validation", "test"]
FLEURS_CONFIGS = ["hi_in", "te_in", "en_us"]
FLEURS_SPLITS = ["train", "dev", "test"]
TARGET_LANGS = ["eng", "hin", "tel"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gated_warning() -> None:
    if get_token():
        return
    log(
        "WARNING: no HF token found. IN22-Gen and BPCC are gated and will fail.\n"
        "         Run `huggingface-cli login` (or export HF_TOKEN) and accept the\n"
        "         dataset terms on https://huggingface.co/ai4bharat/IN22-Gen and\n"
        "         https://huggingface.co/ai4bharat/BPCC (click 'Agree and access')."
    )


def repo_size(repo_id: str) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for f in list_repo_tree(repo_id, repo_type="dataset", recursive=True):
        sizes[f.path] = getattr(f, "size", 0) or 0
    return sizes


BPCC_CONFIGS = {
    "bpcc-seed-latest": "bpcc-seed-latest",
    "nllb-filtered": "nllb_filtered",
    "samanantar-filtered": "samanantar_v0.3_filtered",
    "samanantar-v2": "samanantar_v2",
}
BPCC_TARGET_SPLITS = ["hin_Deva", "tel_Telu"]


def save_streaming(ds, out_path: Path, limit: int | None, extra_filter=None) -> int:
    if out_path.exists():
        log(f"  skip (already exists): {out_path}")
        return -1
    if extra_filter:
        ds = ds.filter(extra_filter)
    if limit:
        ds = ds.take(limit)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    ds.to_parquet(tmp)
    os.replace(tmp, out_path)
    return 0


def download_common_voice(data_dir: Path, langs: list[str], limit: int | None) -> None:
    out = data_dir / "common_voice"
    for lang in langs:
        for split in CV_SPLITS:
            dest = out / lang / f"{split}.parquet"
            log(f"Common Voice {lang}/{split}")
            ds = load_dataset(
                COMMON_VOICE, lang, split=split, streaming=True
            ).cast_column("audio", Audio(sampling_rate=48000, decode=False))
            save_streaming(ds, dest, limit)


def download_fleurs(data_dir: Path, configs: list[str], limit: int | None, with_audio: bool) -> None:
    out = data_dir / "fleurs"
    from huggingface_hub import hf_hub_download
    import pandas as pd

    for cfg in configs:
        for split in FLEURS_SPLITS:
            tsv_path = hf_hub_download(
                repo_id=FLEURS,
                filename=f"data/{cfg}/{split}.tsv",
                repo_type="dataset",
                token=get_token(),
            )
            df = pd.read_csv(tsv_path, sep="\t", header=None, names=["id", "fileName", "raw_transcription", "transcription", "chars", "num_samples", "gender"])
            if limit:
                df = df.head(limit)
            dest = out / cfg / f"{split}.parquet"
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(dest)
                log(f"FLEURS {cfg}/{split}: {len(df)} rows -> {dest}")
            else:
                log(f"  skip (already exists): {dest}")
            if with_audio:
                tar_path = hf_hub_download(
                    repo_id=FLEURS,
                    filename=f"data/{cfg}/audio/{split}.tar.gz",
                    repo_type="dataset",
                    token=get_token(),
                )
                audio_dir = out / cfg / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                dest_tar = audio_dir / f"{split}.tar.gz"
                if not dest_tar.exists():
                    import shutil
                    shutil.copy2(tar_path, dest_tar)
                    os.remove(tar_path)
                    log(f"FLEURS {cfg} audio {split}.tar.gz -> {dest_tar}")
                else:
                    log(f"  skip (already exists): {dest_tar}")


def download_in22(data_dir: Path, limit: int | None) -> None:
    import pandas as pd

    out = data_dir / "in22_gen"
    dest = out / "default" / "test.parquet"
    log("IN22-Gen default/test (full, 22 languages)")
    if not dest.exists():
        ds = load_dataset(IN22_GEN, split="test")
        df = ds.to_pandas()
        if limit:
            df = df.head(limit)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dest)
        log(f"  -> {dest} ({len(df)} rows)")
    else:
        log(f"  skip (already exists): {dest}")
        df = pd.read_parquet(dest)

    for src, tgt, name in [
        ("eng_Latn", "hin_Deva", "eng-hin"),
        ("hin_Deva", "eng_Latn", "hin-eng"),
        ("eng_Latn", "tel_Telu", "eng-tel"),
        ("tel_Telu", "eng_Latn", "tel-eng"),
    ]:
        pair_dest = out / "pairs" / f"{name}.parquet"
        if pair_dest.exists():
            log(f"  skip (already exists): {pair_dest}")
            continue
        pair = df[[src, tgt]].rename(columns={src: "source", tgt: "target"})
        pair = pair.dropna()
        pair_dest.parent.mkdir(parents=True, exist_ok=True)
        pair.to_parquet(pair_dest)
        log(f"  -> pairs/{name}.parquet ({len(pair)} rows)")


def download_bpcc(data_dir: Path, limit: int | None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out = data_dir / "bpcc"
    for cfg, subdir in BPCC_CONFIGS.items():
        for split in BPCC_TARGET_SPLITS:
            dest = out / cfg / f"{split}.parquet"
            if dest.exists():
                log(f"  skip (already exists): {dest}")
                continue
            tsv_path = hf_hub_download(
                repo_id=BPCC,
                filename=f"{subdir}/{split}.tsv",
                repo_type="dataset",
                token=get_token(),
            )
            log(f"BPCC {cfg}/{split}: converting {tsv_path}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".tmp")
            if tmp.exists():
                tmp.unlink()
            written = 0
            schema = pa.schema([pa.field("src", pa.string()), pa.field("tgt", pa.string())])
            with pq.ParquetWriter(tmp, schema) as writer:
                for chunk in pd.read_csv(tsv_path, sep="\t", chunksize=200_000, dtype=str, on_bad_lines="skip"):
                    if "src_lang" in chunk.columns:
                        chunk = chunk[chunk["src_lang"] == "eng_Latn"]
                    chunk = chunk[["src", "tgt"]].dropna()
                    if limit:
                        chunk = chunk.head(max(limit - written, 0))
                    if len(chunk) == 0:
                        if written >= (limit or 0):
                            break
                        continue
                    writer.write_table(pa.Table.from_pandas(chunk, preserve_index=False).cast(schema))
                    written += len(chunk)
                    if limit and written >= limit:
                        break
            os.replace(tmp, dest)
            log(f"  -> {dest} ({written} rows)")
            os.remove(tsv_path)


LICENSES = {
    "common_voice": "CC0 (public domain) — Common Voice, via fixie-ai mirror of v17.0",
    "fleurs": "CC-BY 4.0 — Google FLEURS",
    "in22_gen": "MIT — AI4Bharat IN22-Gen",
    "bpcc": "MIT — AI4Bharat BPCC",
}


def write_manifest(data_dir: Path) -> None:
    import pyarrow.parquet as pq

    manifest = []
    for f in sorted(data_dir.rglob("*.parquet")):
        rel = str(f.relative_to(data_dir))
        ds = rel.split("/")[0]
        try:
            rows = sum(
                pq.ParquetFile(part).metadata.num_rows for part in f.parent.glob(f.name)
            )
        except Exception:
            rows = 0
        manifest.append(
            {
                "path": rel,
                "rows": rows,
                "bytes": f.stat().st_size,
                "license": LICENSES.get(ds, ""),
                "download_ts": datetime.now(timezone.utc).isoformat(),
            }
        )
    (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))


def dry_run() -> None:
    gated_warning()
    for name, repo in [
        ("Common Voice (fixie mirror)", COMMON_VOICE),
        ("IN22-Gen", IN22_GEN),
        ("BPCC", BPCC),
        ("FLEURS", FLEURS),
    ]:
        try:
            sizes = repo_size(repo)
            total = sum(sizes.values())
            parq = {p: s for p, s in sizes.items() if p.endswith(".parquet")}
            log(
                f"{name}: {total / 1024**3:.2f} GB total, "
                f"{len(parq)} parquet files ({sum(parq.values()) / 1024**3:.2f} GB)"
            )
        except Exception as e:
            log(f"{name}: cannot inspect ({e})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download evaluation/training datasets into data/raw/")
    ap.add_argument("--all", action="store_true", help="download everything")
    ap.add_argument("--cv", action="store_true", help="Common Voice (STT corpus)")
    ap.add_argument("--in22", action="store_true", help="IN22-Gen (MT benchmark)")
    ap.add_argument("--bpcc", action="store_true", help="BPCC (MT training corpus)")
    ap.add_argument("--fleurs", action="store_true", help="FLEURS (eval)")
    ap.add_argument("--dry-run", action="store_true", help="report sizes, download nothing")
    ap.add_argument("--langs", nargs="+", default=CV_LANGS, help="Common Voice languages (default: hi te)")
    ap.add_argument("--configs", nargs="+", default=FLEURS_CONFIGS, help="FLEURS configs (default: hi_in te_in en_us)")
    ap.add_argument("--limit", type=int, default=None, help="cap rows per split (testing)")
    ap.add_argument("--with-audio", action="store_true", help="keep FLEURS audio column (large)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="output directory")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        dry_run()
        return

    want = args.all or (args.cv or args.in22 or args.bpcc or args.fleurs)
    if not want:
        ap.error("specify at least one dataset (or --all)")

    if args.in22 or args.bpcc:
        gated_warning()

    if args.cv or args.all:
        download_common_voice(data_dir, args.langs, args.limit)
    if args.fleurs or args.all:
        download_fleurs(data_dir, args.configs, args.limit, args.with_audio)
    if args.in22 or args.all:
        download_in22(data_dir, args.limit)
    if args.bpcc or args.all:
        download_bpcc(data_dir, args.limit)

    write_manifest(data_dir)
    log(f"done. data written to {data_dir} (see MANIFEST.json)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
