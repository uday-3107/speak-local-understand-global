"""Module 6 tests: rule-based engine correctness + BLEU helper (no model load)."""
from scripts.translate_comparison import RuleTranslator, bleu4, load_pairs


class TestRuleTranslator:
    def test_en_to_hi_known_words(self):
        r = RuleTranslator()
        assert r.translate("the teacher and student", "en", "hi") == "यह शिक्षक और छात्र"

    def test_en_to_te_known_words(self):
        r = RuleTranslator()
        assert r.translate("the teacher", "en", "te") == "ఈ ఉపాధ్యాయుడు"

    def test_hi_to_en_known_words(self):
        r = RuleTranslator()
        assert r.translate("शिक्षक और छात्र", "hi", "en") == "teacher and student"

    def test_unknown_word_passthrough(self):
        r = RuleTranslator()
        assert r.translate("quantum mechanics", "en", "hi") == "quantum mechanics"

    def test_coverage_fraction(self):
        r = RuleTranslator()
        assert round(r.coverage("the teacher is here", "en", "hi"), 2) == 0.75


class TestBleu:
    def test_perfect_match_is_one(self):
        assert bleu4(["this is a test"], ["this is a test"]) == 1.0

    def test_no_match_is_low(self):
        assert bleu4(["this is a test"], ["completely different words"]) < 0.5

    def test_orders_do_not_matter_for_ref(self):
        a = bleu4(["cat sat on mat"], ["the cat sat on the mat"])
        assert 0.0 < a <= 1.0