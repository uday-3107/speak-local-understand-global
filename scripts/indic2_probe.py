"""indic2_probe.py — ZERO-DOWNLOAD probe of the cached IndicTrans2 en-indic-1B.

Purpose: answer whether the already-downloaded model fixes hi->te / te->hi /
hi->en / te->en (the four directions that fail live), before ANY new model is
considered. This script ONLY measures; it changes nothing.

Guarantees:
  * Offline enforcement — HF_HUB_OFFLINE + TRANSFORMERS_OFFLINE set before any
    huggingface import: no network, no downloads, no cache writes.
  * Loads ONLY ai4bharat/indictrans2-en-indic-1B from the existing HF cache.
    No other model is loaded or touched.
  * Reads IN22-Gen pairs (data/raw/in22_gen/pairs/*.parquet) — hi<->te pairs
    are aligned via the shared English sentences in the eng-hin / eng-tel
    parquets (same IN22 test set, sentence-aligned across languages).
  * Prints cache size (du) BEFORE and AFTER so you can verify nothing was
    added or removed.
  * Writes exactly ONE file: data/processed/indic2_probe.json

Run from repo root (approval required before running):
    /opt/anaconda3/bin/python -m scripts.indic2_probe --limit 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd

from scripts.evaluate_system import exact_match_rate, latency_stats, script_consistency
from scripts.translate_comparison import bleu4, load_pairs

PAIRS_DIR = "data/raw/in22_gen/pairs"
OUT_FILE = "data/processed/indic2_probe.json"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface")


def log(msg: str) -> None:
    print(f"[probe] {msg}")


def cache_size() -> str:
    r = subprocess.run(["du", "-sh", CACHE_DIR], capture_output=True, text=True)
    return r.stdout.split()[0] if r.returncode == 0 else "n/a"


def load_hi_te_pairs(limit: int):
    """Align eng-hin + eng-tel parquets on the shared English sentence."""
    en_hi = pd.read_parquet(os.path.join(PAIRS_DIR, "eng-hin.parquet"))
    en_te = pd.read_parquet(os.path.join(PAIRS_DIR, "eng-tel.parquet"))
    merged = en_hi.merge(en_te, on="source", suffixes=("_hi", "_te"))
    return merged.head(limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=6, help="sentences per direction")
    args = parser.parse_args()

    before = cache_size()
    log(f"HF cache size BEFORE: {before}")

    from backend.ml_models.indic2_service import Indic2Service

    log("loading IndicTrans2 en-indic-1B from cache (offline, may take 1-3 min)…")
    t0 = time.perf_counter()
    svc = Indic2Service()
    log(f"model loaded in {time.perf_counter() - t0:.1f}s")

    paired = load_hi_te_pairs(args.limit)
    tests = {
        "hi->te": (paired["target_hi"].tolist(), paired["target_te"].tolist()),
        "te->hi": (paired["target_te"].tolist(), paired["target_hi"].tolist()),
    }
    hi_en_src, hi_en_ref = load_pairs("hi->en", args.limit)
    te_en_src, te_en_ref = load_pairs("te->en", args.limit)
    tests["hi->en"] = (hi_en_src, hi_en_ref)
    tests["te->en"] = (te_en_src, te_en_ref)

    results = []
    for direction, (sources, reference) in tests.items():
        src, tgt = direction.split("->")
        log(f"translating {len(sources)} sentences: {direction}…")
        preds, lat = [], []
        for s in sources:
            t1 = time.perf_counter()
            preds.append(svc.translate(s, src, tgt)[0])
            lat.append((time.perf_counter() - t1) * 1000)
        row = {
            "direction": direction,
            "samples": len(sources),
            "bleu4": bleu4(reference, preds),
            "exact_match": exact_match_rate(reference, preds),
            "script_consistency": script_consistency(preds, tgt),
            "latency_ms": latency_stats(lat),
            "sample": [{"src": s[:120], "ref": r[:120], "pred": p[:120]}
                       for s, r, p in zip(sources, reference, preds)][:2],
        }
        results.append(row)
        log(f"{direction}: BLEU={row['bleu4']} script_ok={row['script_consistency']} "
            f"lat mean={row['latency_ms']['mean']}ms")

    after = cache_size()
    log(f"HF cache size AFTER:  {after}")
    if before != after:
        log(f"WARNING: cache changed! {before} -> {after} (should NOT happen)")
    else:
        log("cache unchanged — nothing was downloaded or stored")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"before_cache": before, "after_cache": after, "results": results},
                  f, indent=2, ensure_ascii=False)
    log(f"saved {OUT_FILE}")

    print("\n=== IndicTrans2 en-indic-1B probe (cached, offline) ===")
    print(f"{'direction':<9} {'BLEU':>7} {'script':>7} {'lat_mean':>10}")
    for r in results:
        print(f"{r['direction']:<9} {r['bleu4']:>7} {r['script_consistency']:>7.2f} "
              f"{r['latency_ms']['mean']:>8.0f}ms")


if __name__ == "__main__":
    main()