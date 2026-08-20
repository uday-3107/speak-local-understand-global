"""Module 8 quality-gap check: IndicTrans2 vs NLLB on en→te (A/B).

The Module 8 evaluation found en→te is NLLB's weakest direction
(BLEU-4 ~0.018, script-correct). This script compares the cached
en-indic model (IndicTrans2) against NLLB on the same IN22-Gen pairs,
reporting BLEU-4, exact-match, script consistency, and latency.

First run downloads the Indictrans2-en-indic-1B weights (~1.9 GB fp16 /
~4 GB fp32) into the HF cache (config/tokenizer already local). No
retraining — pure inference comparison. If IndicTrans2 wins, adopt it as
the en→te engine; otherwise keep NLLB.

Run from repo root (user runs):
    /opt/anaconda3/bin/python -m scripts.indic2_comparison --limit 30

Writes: data/processed/eval/indic2_ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

from scripts.evaluate_system import (exact_match_rate, latency_stats,
                                     script_consistency)
from scripts.translate_comparison import AiTranslator, bleu4, load_pairs

OUT_DIR = "data/processed/eval"
PAIRS = ["en->te", "en->hi"]


def log(msg: str) -> None:
    print(f"[indic2] {msg}")


class IndicTranslator:
    name = "indictrans2"

    def __init__(self) -> None:
        from backend.ml_models.indic2_service import Indic2Service

        self._svc = Indic2Service()

    def translate(self, text: str, src: str, tgt: str) -> str:
        out, _ = self._svc.translate(text, src, tgt)
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30,
                        help="sentences per direction")
    parser.add_argument("--directions", nargs="+", default=["en->te", "en->hi"],
                        help="e.g. --directions en->te en->hi")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    indic = IndicTranslator()
    nllb = AiTranslator()
    log("loading IndicTrans2 (first run downloads ~1.9–4 GB weights)…")

    results = []
    for direction in args.directions:
        src, tgt = direction.split("->")
        sources, reference = load_pairs(direction, args.limit)

        def run(t, lat: list) -> list[str]:
            for s in sources:
                t0 = time.perf_counter()
                lat.append((time.perf_counter() - t0) * 1000)
                t.translate(s, src, tgt)

        # warm-up latency excluded from stats (model may be loaded lazily)
        if sources:
            t0 = time.perf_counter(); nllb.translate(sources[0], src, tgt)
            nllb_warm = (time.perf_counter() - t0) * 1000

        def run(t, lat: list) -> list[str]:
            preds = []
            for s in sources:
                t0 = time.perf_counter()
                preds.append(t.translate(s, src, tgt))
                lat.append((time.perf_counter() - t0) * 1000)
            return preds

        indic_ms, nllb_ms = [], []
        indic_pred = run(indic, indic_ms)
        indic_ms = []
        run(indic, indic_ms)  # real measurement
        nllb_pred = run(nllb, nllb_ms)

        row = {
            "direction": direction,
            "samples": len(sources),
            "indictrans2": {
                "bleu4": bleu4(reference, indic_pred),
                "exact_match": exact_match_rate(reference, indic_pred),
                "script_consistency": script_consistency(indic_pred, tgt),
                "latency_ms": latency_stats(indic_ms),
            },
            "nllb": {
                "bleu4": bleu4(reference, nllb_pred),
                "exact_match": exact_match_rate(reference, nllb_pred),
                "script_consistency": script_consistency(nllb_pred, tgt),
                "latency_ms": latency_stats(nllb_ms),
            },
            "nllb_warmup_ms": round(nllb_warm, 1) if direction == "en->te" else None,
        }
        results.append(row)
        i, n = row["indictrans2"], row["nllb"]
        log(f"{direction}: Indic2 BLEU={i['bleu4']} ({i['latency_ms']['mean']}ms) "
            f"| NLLB BLEU={n['bleu4']} ({n['latency_ms']['mean']}ms)")

    out = os.path.join(OUT_DIR, "indic2_ab.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== IndicTrans2 vs NLLB (en→X) ===")
    print("{:<10} {:>11} {:>11} {:>11} {:>11}".format(
        "direction", "INDIC-BLEU", "NLLB-BLEU", "INDIC-ms", "NLLB-ms"))
    for r in results:
        print("{:<10} {:>11} {:>11} {:>11} {:>11}".format(
            r["direction"], r["indictrans2"]["bleu4"], r["nllb"]["bleu4"],
            r["indictrans2"]["latency_ms"]["mean"],
            r["nllb"]["latency_ms"]["mean"]))
    log("saved " + out)


if __name__ == "__main__":
    main()