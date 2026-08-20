"""Module 5 gap tests: refinement rules + similarity math (no model loading)."""
import numpy as np

from scripts.nlp_embeddings_refine import (cosine_sim, refine_text,
                                           sentence_embedding)


class TestRefine:
    def test_collapses_spaces_and_strips(self):
        assert refine_text("यह   सरल है   ,  और महत्वपूर्ण है .") == "यह सरल है, और महत्वपूर्ण है."

    def test_no_space_before_punctuation(self):
        assert refine_text("accurate , clear .") == "accurate, clear."

    def test_space_after_comma_between_words(self):
        assert refine_text("है,और") == "है, और."

    def test_dedupes_repeated_punctuation(self):
        assert refine_text("wait . . .") == "wait."

    def test_adds_terminator(self):
        assert refine_text("hello world") == "hello world."

    def test_empty_string(self):
        assert refine_text("") == ""

    def test_nbsp_replaced(self):
        assert refine_text("good\u00a0morning") == "good morning."

    def test_devanagari_danda_preserved(self):
        assert refine_text("विद्या प्राप्त करें ।") == "विद्या प्राप्त करें।"

    def test_keeps_indic_text_intact(self):
        out = refine_text("शिक्षा सभी के लिए महत्वपूर्ण है")
        assert "शिक्षा" in out and "महत्वपूर्ण" in out


class TestSimilarity:
    def test_identical_vectors_unit_sim(self):
        v = np.ones(4) / 2
        assert cosine_sim(v, v) == 1.0

    def test_orthogonal_vectors_zero_sim(self):
        assert cosine_sim(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0

    def test_opposite_vectors_negative_sim(self):
        assert cosine_sim(np.array([1.0]), np.array([-1.0])) == -1.0