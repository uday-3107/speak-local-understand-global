"""Analytics endpoint — serves consolidated Module 8/9 metrics for the UI charts.

Reads the persisted evaluation artifacts (data/processed/{ml,dl,mt,eval}/results.json)
produced by the module scripts and reshapes them for Plotly visualizations in the
frontend (dashboard tab: model comparison, BLEU, latency).

Missing or unparseable files are skipped so the API stays up during development
(common in CI or fresh clones without scripts having run yet).
"""
import json
from pathlib import Path

from fastapi import APIRouter

ROOT = Path(__file__).resolve().parent.parent.parent  # backend/api -> repo root
PROCESSED = ROOT / "data" / "processed"

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _load(*parts: str) -> dict | None:
    path = PROCESSED.joinpath(*parts)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


ML_FAMILY = {
    "logistic_regression": "classical",
    "decision_tree": "classical",
    "random_forest": "classical",
    "xgboost": "classical",
    "rnn": "deep",
    "lstm": "deep",
    "transformer": "deep",
}


def collect_ml() -> list[dict]:
    """Flatten Module 3 + 4 results.json into one model-comparison table."""
    rows: list[dict] = []
    for results, family in [(_load("ml", "results.json"), None),
                            (_load("dl", "results.json"), None)]:
        if not results:
            continue
        for item in results.get("results", []):
            rows.append({
                "model": item["model"],
                "family": ML_FAMILY.get(item["model"], "other"),
                "accuracy": round(item["accuracy"], 4),
                "precision": round(item.get("precision_macro", 0.0), 4),
                "recall": round(item.get("recall_macro", 0.0), 4),
                "f1": round(item.get("f1_macro", 0.0), 4),
            })
    return sorted(rows, key=lambda r: r["f1"], reverse=True)


def collect_translation() -> list[dict]:
    """BLEU / script-consistency / latency per direction (rule vs AI)."""
    eval_data = _load("eval", "evaluation.json")
    if eval_data and eval_data.get("translation"):
        return [
            {
                "direction": t["direction"],
                "samples": t["samples"],
                "ai": t["ai"],
                "rule_baseline": t["rule_baseline"],
            }
            for t in eval_data["translation"]
        ]
    engine = _load("mt", "translation_engine.json")
    if engine:
        return [
            {
                "direction": t["direction"],
                "samples": t["samples"],
                "ai": {
                    "bleu4": t["bleu_ai"],
                    "script_consistency": None,
                    "latency_ms": {"mean": t["latency_ms_ai"]},
                },
                "rule_baseline": {
                    "bleu4": t["bleu_rule"],
                    "latency_ms": {"mean": t["latency_ms_rule"]},
                },
            }
            for t in engine
        ]
    return []


def collect_assistant() -> dict:
    eval_data = _load("eval", "evaluation.json") or {}
    assistant = eval_data.get("assistant") or {"utterances": 0}
    satisfaction = eval_data.get("satisfaction") or {
        "count": 0, "positive": 0, "negative": 0, "positive_ratio": 0.0,
    }
    return {
        "assistant": assistant,
        "satisfaction": satisfaction,
    }


@router.get("")
async def get_analytics() -> dict:
    sources = []
    for candidate in ["ml/results.json", "dl/results.json",
                      "mt/translation_engine.json", "eval/evaluation.json"]:
        if PROCESSED.joinpath(candidate).exists():
            sources.append(candidate)
    return {
        "generated_at": (
            (_load("eval", "evaluation.json") or {}).get("generated_at") or None
        ),
        "sources": sources,
        "ml": {"rows": collect_ml()},
        "translation": collect_translation(),
        **collect_assistant(),
    }