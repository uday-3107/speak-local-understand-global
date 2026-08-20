"""Module 3 ML tests: data loading + label pipeline (no heavy full fit)."""
import numpy as np
import joblib

from scripts.classify_language import FEATURES, load_data, score_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def test_load_data_has_three_languages_and_features():
    df = load_data()
    assert len(df) > 3000
    assert set(df["language"]) == {"en", "hi", "te"}
    assert set(FEATURES) <= set(df.columns)


def test_score_model_returns_metrics_dict():
    df = load_data()
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(df["language"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    m = LogisticRegression(max_iter=200).fit(X_train, y_train)
    out = score_model("lr", m, X_test, y_test)
    assert out["model"] == "lr"
    assert 0 <= out["accuracy"] <= 1
    assert 0 <= out["f1_macro"] <= 1
    assert len(out["confusion_matrix"]) == 3


def test_best_model_artifact_exists():
    import os
    json_path = "data/processed/ml/language_model.json"
    job_path = "data/processed/ml/language_model.joblib"
    exists = (os.path.exists(json_path) and os.path.getsize(json_path) > 1000) or \
        (os.path.exists(job_path) and os.path.getsize(job_path) > 1000)
    assert exists, "no persisted model artifact found"