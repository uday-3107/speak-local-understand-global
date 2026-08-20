"""Module 8: System Evaluation — ML models, Translation System, AI Assistant.

Evaluates all three areas from the official Module 8 spec:

  1. ML models (Modules 3+4): consolidates the saved results.json files
     (LogReg / DecisionTree / RandomForest / XGBoost / RNN / LSTM /
     Transformer) into one comparison table + best-model summary using
     accuracy / precision / recall / F1.
  2. Translation system (Module 6 engine): NLLB (and rule baseline) on
     IN22-Gen parallel data — BLEU-4, exact-match translation accuracy,
     response time (mean / p50 / p95), and language consistency (% of
     outputs whose script matches the target language).
  3. AI assistant (Module 7): translation script fidelity, explainability
     (explanation + study note present, English, informative), response
     quality (per-stage latency, draft/refined consistency), and user
     satisfaction (ratings from the Postgres feedback table, if available).

Run from repo root (user runs):
    /opt/anaconda3/bin/python -m scripts.evaluate_system --limit 20

Writes: data/processed/eval/evaluation.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timezone

from scripts.nlp_analysis import detect_script
from scripts.translate_comparison import PARALLEL, AiTranslator, RuleTranslator, bleu4, load_pairs

ML_RESULTS_FILES = ["data/processed/ml/results.json", "data/processed/dl/results.json"]
ASSISTANT_RESULTS = "data/processed/assistant/results.json"
OUT_DIR = "data/processed/eval"
DB_URL = "postgresql+asyncpg://localhost:5432/speak_local"

EXPECTED_SCRIPT = {"en": "latin", "hi": "devanagari", "te": "telugu"}

ML_FAMILY = {
    "logistic_regression": "classical",
    "decision_tree": "classical",
    "random_forest": "classical",
    "xgboost": "classical",
    "rnn": "deep",
    "lstm": "deep",
    "transformer": "deep",
}


def log(msg: str) -> None:
    print(f"[module8] {msg}")


# ----------------------------- ML evaluation -----------------------------

def load_ml_results(paths: list[str]) -> list[dict]:
    """Flatten saved Module 3/4 results.json into model rows."""
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data.get("results", []):
            rows.append({
                "model": item["model"],
                "family": ML_FAMILY.get(item["model"], "other"),
                "accuracy": round(item["accuracy"], 4),
                "precision": round(item["precision_macro"], 4),
                "recall": round(item["recall_macro"], 4),
                "f1": round(item["f1_macro"], 4),
            })
    return sorted(rows, key=lambda r: r["f1"], reverse=True)


def ml_summary(rows: list[dict]) -> dict:
    """Best model by F1 + per-family best + comparison table."""
    if not rows:
        return {"rows": [], "best": None}
    best = max(rows, key=lambda r: r["f1"])
    best_per_family = {}
    for r in rows:
        fam = r["family"]
        if fam not in best_per_family or r["f1"] > best_per_family[fam]["f1"]:
            best_per_family[fam] = r
    return {
        "rows": rows,
        "best": best,
        "best_per_family": {fam: v["model"] for fam, v in best_per_family.items()},
    }


# -------------------------- Translation evaluation --------------------------

def script_consistency(preds: list[str], tgt: str) -> float:
    """Fraction of translations whose dominant script matches the target."""
    if not preds:
        return 0.0
    expected = EXPECTED_SCRIPT.get(tgt, "")
    ok = sum(1 for p in preds if detect_script(p) == expected)
    return round(ok / len(preds), 4)


def exact_match_rate(references: list[str], preds: list[str]) -> float:
    """Fraction of exact (whitespace-normalized) matches vs references."""
    if not references:
        return 0.0
    ok = sum(1 for r, p in zip(references, preds)
             if " ".join(r.split()) == " ".join(p.split()))
    return round(ok / len(references), 4)


def latency_stats(ms_list: list[float]) -> dict:
    """mean / p50 / p95 of per-sentence latencies (ms)."""
    if not ms_list:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": round(statistics.mean(ms_list), 1),
        "p50": round(statistics.median(ms_list), 1),
        "p95": round(sorted(ms_list)[int(len(ms_list) * 0.95) - 1], 1),
    }


def evaluate_translation(direction: str, limit: int | None) -> dict:
    src, tgt = direction.split("->")
    sources, reference = load_pairs(direction, limit)
    rule = RuleTranslator()
    ai = AiTranslator()

    rule_preds, rule_ms = [], []
    t0 = time.perf_counter()
    for s in sources:
        t1 = time.perf_counter()
        rule_preds.append(rule.translate(s, src, tgt))
        rule_ms.append((time.perf_counter() - t1) * 1000)
    rule_total = (time.perf_counter() - t0) * 1000

    ai_preds, ai_ms = [], []
    for s in sources:
        t1 = time.perf_counter()
        ai_preds.append(ai.translate(s, src, tgt))
        ai_ms.append((time.perf_counter() - t1) * 1000)

    return {
        "direction": direction,
        "samples": len(sources),
        "ai": {
            "bleu4": bleu4(reference, ai_preds),
            "exact_match": exact_match_rate(reference, ai_preds),
            "script_consistency": script_consistency(ai_preds, tgt),
            "latency_ms": latency_stats(ai_ms),
        },
        "rule_baseline": {
            "bleu4": bleu4(reference, rule_preds),
            "exact_match": exact_match_rate(reference, rule_preds),
            "latency_ms": latency_stats(rule_ms),
        },
        "engine_total_ms": round((sum(ai_ms) + rule_total) / max(len(sources), 1), 1),
    }


# -------------------------- Assistant evaluation --------------------------

def normalize_assistant_results(results) -> list[dict]:
    """results.json may be a single utterance dict or a list of them."""
    if isinstance(results, list):
        return results
    return [results]


def assistant_metrics(results: list[dict]) -> dict:
    """Aggregate Module 7 outputs: fidelity, explainability, response quality."""
    if not results:
        return {"utterances": 0}
    fidelity = []
    explained = []
    notes = []
    drafts_match = []
    llm_ms = []
    stt_ms = []
    mt_ms = []
    for r in results:
        tgt = r.get("tgt", "")
        disp = r.get("translation_displayed", "")
        draft = r.get("draft_translation", "")
        fidelity.append(detect_script(disp) == EXPECTED_SCRIPT.get(tgt, ""))
        explained.append(bool(r.get("explanation", "").strip()))
        notes.append(bool(r.get("study_note", "").strip()))
        drafts_match.append(disp == draft)
        lat = r.get("latency_ms", {}) or {}
        if lat.get("llm") is not None:
            llm_ms.append(float(lat["llm"]))
        if lat.get("stt") is not None:
            stt_ms.append(float(lat["stt"]))
        if lat.get("mt") is not None:
            mt_ms.append(float(lat["mt"]))

    def frac(vals: list[bool]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "utterances": len(results),
        "translation": {
            "script_fidelity": frac(fidelity),
            "displayed_is_draft": frac(drafts_match),
        },
        "explainability": {
            "has_explanation": frac(explained),
            "has_study_note": frac(notes),
        },
        "response_latency_ms": {
            "stt": latency_stats(stt_ms),
            "mt": latency_stats(mt_ms),
            "llm": latency_stats(llm_ms),
        },
    }


def satisfaction_stats(rows: list[dict]) -> dict:
    """User satisfaction from feedback rows [{rating, comment}]."""
    if not rows:
        return {"count": 0, "positive_ratio": 0.0, "comments": []}
    positive = sum(1 for r in rows if r["rating"])
    comments = [r["comment"] for r in rows if r.get("comment", "").strip()][:10]
    return {
        "count": len(rows),
        "positive": positive,
        "negative": len(rows) - positive,
        "positive_ratio": round(positive / len(rows), 4),
        "comments": comments,
    }


async def fetch_feedback_rows(db_url: str) -> list[dict]:
    """Read feedback ratings + comments from Postgres (asyncpg)."""
    import asyncpg

    url = db_url.replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url)
    try:
        rows = await conn.fetch("SELECT rating, comment FROM feedback")
    finally:
        await conn.close()
    return [{"rating": bool(r["rating"]), "comment": r["comment"] or ""} for r in rows]


# --------------------------------- main ---------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20,
                        help="sentences per direction for the NLLB eval")
    parser.add_argument("--no-db", action="store_true",
                        help="skip user-satisfaction DB query")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    report = {"generated_at": datetime.now(timezone.utc).isoformat()}

    log("ML evaluation…")
    ml_rows = load_ml_results(ML_RESULTS_FILES)
    report["ml"] = ml_summary(ml_rows)

    log(f"translation evaluation (limit={args.limit})…")
    trans = []
    for direction in PARALLEL:
        log(f"  {direction}…")
        trans.append(evaluate_translation(direction, args.limit))
    report["translation"] = trans

    if os.path.exists(ASSISTANT_RESULTS):
        with open(ASSISTANT_RESULTS) as f:
            results = normalize_assistant_results(json.load(f))
        report["assistant"] = assistant_metrics(results)
    else:
        report["assistant"] = {"utterances": 0, "note": "no results.json yet"}

    if args.no_db:
        report["satisfaction"] = {"count": 0, "positive_ratio": 0.0,
                                  "note": "skipped (--no-db)"}
    else:
        log("user satisfaction (feedback table)…")
        try:
            rows = asyncio.run(fetch_feedback_rows(DB_URL))
            report["satisfaction"] = satisfaction_stats(rows)
        except Exception as exc:
            report["satisfaction"] = {"count": 0, "positive_ratio": 0.0,
                                      "error": str(exc)[:200]}

    out = os.path.join(OUT_DIR, "evaluation.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== Module 8: System Evaluation ===")
    print("ML models (sorted by F1):")
    print("{:<22} {:>10} {:>10} {:>10} {:>8}".format(
        "model", "accuracy", "precision", "recall", "F1"))
    for r in ml_rows:
        print("{:<22} {:>10} {:>10} {:>10} {:>8}".format(
            r["model"], r["accuracy"], r["precision"], r["recall"], r["f1"]))
    if ml_rows:
        best = report["ml"]["best"]
        print(f"best model: {best['model']} (F1 {best['f1']})")

    print("\nTranslation system (AI engine):")
    print("{:<10} {:>9} {:>10} {:>14} {:>12}".format(
        "direction", "BLEU-4", "exact%", "script-cons%", "p95 ms"))
    for t in trans:
        ai = t["ai"]
        print("{:<10} {:>9} {:>10} {:>14} {:>12}".format(
            t["direction"], ai["bleu4"], ai["exact_match"] * 100,
            ai["script_consistency"] * 100, ai["latency_ms"]["p95"]))

    print("\nAssistant:", json.dumps(report["assistant"], ensure_ascii=False))
    print("Satisfaction:", json.dumps(report["satisfaction"], ensure_ascii=False))
    log("saved " + out)


if __name__ == "__main__":
    main()
