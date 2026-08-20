"""Module 4 DL tests: model construction + shapes on synthetic MFCC frames."""
import os

import numpy as np
import torch

from scripts.train_module4_deep import (LSTMClassifier, RNNClassifier,
                                        TransformerClassifier, make_model)


def _batch(b=4, t=160, c=13):
    return torch.randn(b, t, c)


class TestShapes:
    def test_rnn_output_shape(self):
        m = RNNClassifier()
        out = m(_batch())
        assert out.shape == (4, 3)

    def test_lstm_output_shape(self):
        m = LSTMClassifier()
        out = m(_batch())
        assert out.shape == (4, 3)

    def test_transformer_output_shape(self):
        m = TransformerClassifier()
        out = m(_batch())
        assert out.shape == (4, 3)

    def test_transformer_varies_nhead_valid(self):
        m = TransformerClassifier(d_model=64, nhead=4)
        out = m(_batch())
        assert out.shape == (4, 3)


class TestMakeModel:
    def test_all_modes_construct(self):
        for mode in ["rnn", "lstm", "transformer"]:
            m = make_model(mode, torch.device("cpu"))
            assert hasattr(m, "forward")

    def test_forward_loss_is_finite(self):
        m = LSTMClassifier()
        logits = m(_batch())
        assert torch.isfinite(logits).all().item()

    def test_seed_path_naming(self):
        from scripts.train_module4_deep import seq_path
        assert seq_path("en", 1) == os.path.join("data", "processed", "clean", "en_000001.wav")