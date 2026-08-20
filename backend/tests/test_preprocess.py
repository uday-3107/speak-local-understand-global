"""Tests for the Module 2 preprocessing pipeline (unit-level, no heavy libs)."""
import numpy as np
import pytest

from scripts.preprocess_dataset import clean_text, normalize_audio, reduce_noise, segment_utterance


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("  The   quick  brown fox   ") == "the quick brown fox"

    def test_drops_punctuation_keeps_letters(self):
        assert clean_text("Hello, world! (yes)") == "hello world yes"

    def test_keeps_indic_script(self):
        assert clean_text("नमस्ते दुनिया") == "नमस्ते दुनिया"

    def test_none_returns_empty(self):
        assert clean_text(None) == ""


class TestAudioProcessing:
    def test_normalize_scales_to_0_98_peak(self):
        x = np.random.uniform(-2, 2, size=16000).astype(np.float32)
        y = normalize_audio(x)
        assert abs(np.max(np.abs(y)) - 0.98) < 1e-3

    def test_normalize_silence_stays_zero(self):
        y = normalize_audio(np.zeros(8000, dtype=np.float32))
        assert np.all(y == 0)

    def test_reduce_noise_keeps_length(self):
        x = np.random.randn(16000).astype(np.float32) * 0.01
        y = reduce_noise(x)
        assert y.shape == x.shape
        assert np.isfinite(y).all()

    def test_reduce_noise_short_signal_untouched(self):
        x = np.random.randn(700).astype(np.float32)
        assert np.allclose(reduce_noise(x), x)


class TestSegmentation:
    def _signal(self, voiced_s, gap_s):
        sr = 16000
        parts = []
        idx = 0
        for v, g in zip(voiced_s, gap_s if gap_s else [0] * len(voiced_s)):
            parts.append(np.random.randn(int(v * sr)).astype(np.float32))
            if g:
                parts.append(np.zeros(int(g * sr), dtype=np.float32))
        return np.concatenate(parts)

    def test_splits_on_long_silence(self):
        x = self._signal([1.2, 1.2], [0.9, None])
        segs = segment_utterance(x)
        assert len(segs) >= 2

    def test_joins_tiny_burst_into_previous(self):
        x = self._signal([1.2, 0.4, 1.2], [0.1, 0.1, None])
        segs = segment_utterance(x)
        assert len(segs) == 2
        assert len(segs[0]) / 16000 >= 1.0  # 0.4s burst merged into 1.2s

    def test_returns_empty_for_silence(self):
        assert segment_utterance(np.zeros(16000 * 5, dtype=np.float32)) == []