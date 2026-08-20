"""Analytics endpoint tests: reshape logic (pure, reads persisted artifacts)."""
import pytest

from backend.api.analytics import collect_ml, collect_translation, get_analytics


@pytest.fixture(scope="module")
def analytics():
    import asyncio

    return asyncio.run(get_analytics())


class TestMlComparison:
    def test_seven_models_present(self, analytics):
        rows = analytics["ml"]["rows"]
        assert {r["model"] for r in rows} == {
            "logistic_regression", "decision_tree", "random_forest",
            "xgboost", "rnn", "lstm", "transformer"}

    def test_family_tagging(self, analytics):
        fam = {r["model"]: r["family"] for r in analytics["ml"]["rows"]}
        assert fam["transformer"] == "deep"
        assert fam["xgboost"] == "classical"

    def test_sorted_by_f1_desc(self, analytics):
        rows = analytics["ml"]["rows"]
        f1s = [r["f1"] for r in rows]
        assert f1s == sorted(f1s, reverse=True)

    def test_metrics_in_range(self, analytics):
        for r in analytics["ml"]["rows"]:
            assert 0.0 <= r["accuracy"] <= 1.0
            assert 0.0 <= r["f1"] <= 1.0


class TestTranslation:
    def test_four_directions(self, analytics):
        dirs = {t["direction"] for t in analytics["translation"]}
        assert dirs == {"en->hi", "hi->en", "en->te", "te->en"}

    def test_engines_present(self, analytics):
        for t in analytics["translation"]:
            assert "ai" in t and "rule_baseline" in t
            assert "bleu4" in t["ai"]

    def test_latency_shape(self, analytics):
        for t in analytics["translation"]:
            assert "latency_ms" in t["ai"]
            assert "latency_ms" in t["rule_baseline"]


class TestArtifacts:
    def test_sources_reported(self, analytics):
        assert "eval/evaluation.json" in analytics["sources"]

    def test_satisfaction_present(self, analytics):
        assert "satisfaction" in analytics
        assert "positive_ratio" in analytics["satisfaction"]


def test_collect_ml_uses_real_files():
    rows = collect_ml()
    assert len(rows) == 7


def test_collect_translation_prefers_eval():
    rows = collect_translation()
    assert len(rows) == 4
    assert all(t["ai"]["script_consistency"] is not None for t in rows)