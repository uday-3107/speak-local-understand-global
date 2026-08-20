"""IndicTrans2 A/B benchmark tests (pure, no model load)."""
from scripts.indic2_benchmark import PAIRS, IndicTranslator


class TestIndic2Benchmark:
    def test_pairs_default(self):
        assert PAIRS == ["en->te", "en->hi"]

    def test_translator_name(self):
        assert IndicTranslator.name == "indictrans2"

    def test_translator_loads_lazily(self):
        svc = IndicTranslator()
        assert svc.name == "indictrans2"
