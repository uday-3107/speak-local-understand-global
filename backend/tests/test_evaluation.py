"""Module 8 tests: evaluation helpers (pure functions, no model load)."""
from scripts.evaluate_system import (assistant_metrics, exact_match_rate,
                                     latency_stats, load_ml_results,
                                     ml_summary, satisfaction_stats,
                                     script_consistency)


class TestML:
    def test_saves_only_existing_files(self):
        rows = load_ml_results(["nonexistent.json"])
        assert rows == []

    def test_consolidates_both_families(self):
        rows = load_ml_results(["data/processed/ml/results.json",
                                "data/processed/dl/results.json"])
        assert len(rows) == 7
        assert {r["model"] for r in rows} == {
            "logistic_regression", "decision_tree", "random_forest",
            "xgboost", "rnn", "lstm", "transformer"}

    def test_family_tagging(self):
        rows = load_ml_results(["data/processed/ml/results.json",
                                "data/processed/dl/results.json"])
        fam = {r["model"]: r["family"] for r in rows}
        assert fam["transformer"] == "deep"
        assert fam["xgboost"] == "classical"

    def test_ml_summary_best_f1(self):
        rows = [
            {"model": "a", "family": "deep", "accuracy": 0.9,
             "precision": 0.9, "recall": 0.8, "f1": 0.85},
            {"model": "b", "family": "classical", "accuracy": 0.6,
             "precision": 0.6, "recall": 0.6, "f1": 0.6},
        ]
        summ = ml_summary(rows)
        assert summ["best"]["model"] == "a"
        assert summ["best_per_family"]["deep"] == "a"
        assert summ["best_per_family"]["classical"] == "b"

    def test_ml_summary_empty(self):
        assert ml_summary([])["best"] is None


class TestTranslation:
    def test_script_consistency_all_good(self):
        assert script_consistency(["एक तूफान", "पानी है"], "hi") == 1.0

    def test_script_consistency_latin_to_telugu(self):
        assert script_consistency(["hello world", "the cat"], "te") == 0.0

    def test_script_consistency_mixed(self):
        preds = ["ఒక పుస్తకం", "some latin text"]
        assert script_consistency(preds, "te") == 0.5

    def test_script_consistency_empty(self):
        assert script_consistency([], "hi") == 0.0

    def test_exact_match_rate_perfect(self):
        assert exact_match_rate(["a b  c", "d"], ["a b c", "d"]) == 1.0

    def test_exact_match_rate_partial(self):
        assert exact_match_rate(["a b", "c d"], ["a b", "e f"]) == 0.5

    def test_latency_stats_profiles(self):
        s = latency_stats([100.0, 200.0, 300.0, 400.0, 500.0])
        assert s["mean"] == 300.0
        assert s["p50"] == 300.0
        assert s["p95"] == 400.0

    def test_latency_stats_empty(self):
        s = latency_stats([])
        assert s == {"mean": 0.0, "p50": 0.0, "p95": 0.0}


class TestAssistantMetrics:
    def test_metrics_empty(self):
        assert assistant_metrics([])["utterances"] == 0

    def test_full_utterance_scoring(self):
        results = [{
            "tgt": "hi",
            "translation_displayed": "एक तूफान बहुत कम दबाव वाली हवा है।",
            "draft_translation": "एक तूफान बहुत कम दबाव वाली हवा है।",
            "explanation": "A tornado is low pressure air.",
            "study_note": "Remember low pressure.",
            "latency_ms": {"stt": 1300, "mt": 2000, "llm": 10000},
        }]
        m = assistant_metrics(results)
        assert m["translation"]["script_fidelity"] == 1.0
        assert m["translation"]["displayed_is_draft"] == 1.0
        assert m["explainability"]["has_explanation"] == 1.0
        assert m["explainability"]["has_study_note"] == 1.0
        assert m["response_latency_ms"]["llm"]["mean"] == 10000.0

    def test_wrong_script_crosses_zero(self):
        results = [{
            "tgt": "hi",
            "translation_displayed": "tornado is a low pressure air",
            "draft_translation": "tornado is a low pressure air",
            "explanation": "", "study_note": "",
        }]
        m = assistant_metrics(results)
        assert m["translation"]["script_fidelity"] == 0.0
        assert m["explainability"]["has_explanation"] == 0.0


class TestSatisfaction:
    def test_no_rows(self):
        s = satisfaction_stats([])
        assert s["count"] == 0 and s["positive_ratio"] == 0.0

    def test_positive_ratio(self):
        rows = [{"rating": True}, {"rating": True}, {"rating": False}]
        s = satisfaction_stats(rows)
        assert s["count"] == 3
        assert s["positive"] == 2 and s["negative"] == 1
        assert abs(s["positive_ratio"] - 0.6667) < 0.01

    def test_comments_captured(self):
        s = satisfaction_stats([{"rating": True, "comment": "great"},
                                {"rating": False, "comment": ""}])
        assert s["comments"] == ["great"]