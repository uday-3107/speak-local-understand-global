"""Module 3: ML language classification on clean.parquet audio features.

Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost to
classify speech-language (en/hi/te) from MFCC + energy features, reports
accuracy/precision/recall/F1 per model, and persists the best model.

Usage (from repo root, postgres OR server need not run):
    /opt/anaconda3/bin/python -m scripts.classify_language
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

CLEAN_PARQUET = "data/processed/clean/clean.parquet"
OUT_DIR = "data/processed/ml"

FEATURES = ["duration_s", "mfcc_mean", "mfcc_std",
            "mfcc_delta_mean", "mfcc_delta_std", "rmse"]

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=12),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "xgboost": XGBClassifier(n_estimators=200, max_depth=6,
                             learning_rate=0.1, random_state=42,
                             eval_metric="mlogloss"),
}


def log(msg: str) -> None:
    print(f"[module3] {msg}")


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_PARQUET)
    df = df.dropna(subset=FEATURES + ["language"])
    return df


def score_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds = model.predict(X_test)
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision_macro": round(precision_score(y_test, preds, average="macro"), 4),
        "recall_macro": round(recall_score(y_test, preds, average="macro"), 4),
        "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    log(f"dataset: {len(df)} segments, {len(df['language'].unique())} languages "
        f"({df['language'].value_counts().to_dict()})")

    X = df[FEATURES].to_numpy(dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["language"].to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    log(f"split: train {len(X_train)}, test {len(X_test)}")
    log(f"classes: {dict(zip(le.classes_.tolist(), le.transform(le.classes_).tolist()))}")

    results = []
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        res = score_model(name, model, X_test, y_test)
        results.append(res)
        log(f"{name}: acc={res['accuracy']} prec={res['precision_macro']} "
            f"rec={res['recall_macro']} f1={res['f1_macro']}")

    best = max(results, key=lambda r: r["f1_macro"])
    log(f"best: {best['model']} f1={best['f1_macro']}")

    best_model = MODELS[best["model"]]
    if isinstance(best_model, XGBClassifier):
        best_model.save_model(os.path.join(OUT_DIR, "language_model.json"))
    else:
        joblib.dump(best_model, os.path.join(OUT_DIR, "language_model.joblib"))
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.joblib"))
    np.save(os.path.join(OUT_DIR, "feature_names.npy"), np.array(FEATURES))
    classes = le.classes_.tolist()
    np.save(os.path.join(OUT_DIR, "classes.npy"), np.array(classes))

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"results": results, "best": best["model"],
                   "classes": classes, "features": FEATURES}, f, indent=2)

    print("\n=== Results (test set, n=%d) ===" % len(y_test))
    print(pd.DataFrame(results).to_string(index=False))
    print("\nBest model:", best["model"])
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()