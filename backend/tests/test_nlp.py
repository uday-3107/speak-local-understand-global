"""Module 5 NLP tests: tokenization, script detection, stopwords, text-Lid."""
from collections import Counter

from scripts.nlp_analysis import (STOPWORDS, detect_script, language_stats,
                                  remove_stopwords, text_language_model,
                                  tokenize_words)


class TestTokenization:
    def test_tokenize_latin(self):
        assert tokenize_words("Hello, World!") == ["hello", "world"]

    def test_tokenize_hindi(self):
        assert tokenize_words("नमस्ते दुनिया") == ["नमस्ते", "दुनिया"]

    def test_tokenize_telugu(self):
        assert tokenize_words("నమస్తే ప్రపంచం") == ["నమస్తే", "ప్రపంచం"]


class TestScriptDetection:
    def test_latin(self):
        assert detect_script("this is english text") == "latin"

    def test_devanagari(self):
        assert detect_script("यह हिन्दी भाषा है") == "devanagari"

    def test_telugu(self):
        assert detect_script("ఇది తెలుగు భాష") == "telugu"


class TestStopwords:
    def test_remove_english_stopwords(self):
        words = tokenize_words("The cat and dog went to the park")
        filtered = remove_stopwords(words, "en")
        assert "the" not in filtered and "and" not in filtered
        assert "cat" in filtered and "park" in filtered

    def test_stopword_lists_present(self):
        for lang in ["en", "hi", "te"]:
            assert len(STOPWORDS[lang]) > 10


class TestStats:
    def test_language_stats_shape(self):
        texts = [
            "speech recognition turns audio into text in real time",
            "another utterance for the same test corpus here",
        ]
        s = language_stats(texts, "en")
        assert s["utterances"] == 2
        assert s["type_token_ratio"] > 0
        assert s["total_tokens"] > s["content_tokens"]
        assert isinstance(s["top_20_words"], list)

    def test_top_words_sorted_by_frequency(self):
        words = ["the"] * 5 + ["quick"] * 2 + ["brown", "fox", "fox"]
        freq = Counter(remove_stopwords((w for w in words), "en"))
        assert freq["fox"] == 2 and freq["quick"] == 2


def test_text_language_model_trains():
    import pandas as pd
    samples = {
        "en": "english words brown fox jumps over the lazy dog quickly",
        "hi": "वैज्ञानिकों ने अंतरिक्ष में नई ग्रह की खोज की हिंदी भाषा का उदाहरण",
        "te": "విజ్ఞానశాస్త్రజ్ఞులు ఒక చాలా ముఖ్యమైన తెలుగు వాక్యం ఇక్కడ రాసారు",
    }
    df = pd.DataFrame({
        "language": ["en", "hi", "te"] * 20,
        "source_text": [samples["en"], samples["hi"], samples["te"]] * 20,
    })
    res = text_language_model(df)
    assert res["accuracy"] == 1.0
    assert res["f1_macro"] == 1.0