"""Module 5: NLP tasks per language (English / Hindi / Telugu).

Applies classical NLP tasks to the Module 2 clean corpus (source_text):
  - script detection (Latin / Devanagari / Telugu) per utterance
  - word tokenization + stopword removal (per-language stopword lists)
  - vocabulary stats: token count, type-token ratio, mean sentence length
  - top-N most frequent words per language
  - text-level language identification: char n-gram TF-IDF + Logistic
    Regression trained on the corpus, with acc/prec/recall/F1 per language

Writes data/processed/nlp/results.json and prints a summary table.
Run from repo root (user runs):
    /opt/anaconda3/bin/python -m scripts.nlp_analysis
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

CLEAN_PARQUET = "data/processed/clean/clean.parquet"
OUT_DIR = "data/processed/nlp"
SEED = 42

WORD_RE = re.compile(r"[\w\u0900-\u097F\u0C00-\u0C7F']+")
SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")

STOPWORDS = {
    "en": {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for",
        "with", "as", "is", "are", "was", "were", "be", "been", "being", "it",
        "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
        "my", "your", "his", "her", "its", "our", "their", "not", "no", "yes",
        "have", "has", "had", "do", "does", "did", "will", "would", "can",
        "could", "should", "may", "might", "must", "from", "at", "by", "about",
    },
    "hi": {
        "है", "हैं", "और", "का", "की", "के", "में", "से", "को", "पर", "ने",
        "था", "थी", "थे", "हो", "था", "हम", "आप", "वह", "यह", "ये", "वे",
        "कि", "तो", "भी", "ही", "ना", "नहीं", "एक", "यहाँ", "वहाँ", "करना",
        "करता", "करते", "करती", "होने", "गया", "गई", "गए", "दिया", "दी",
        "जो", "जब", "जहाँ", "कब", "कहाँ", "क्या", "कौन", "क्यों", "अपना",
        "उनका", "उसका", "मेरा", "हमारा", "आपका", "लिए", "बाद", "पहले",
    },
    "te": {
        "ఒక", "మరియు", "కు", "కి", "లో", "పై", "నుండి", "తో", "కోసం", "గా",
        "ఉంది", "ఉన్న", "ఉన్నాడు", "ఉన్నది", "అతను", "ఆమె", "ఇది", "అది",
        "అవి", "ఇవి", "వారు", "మేము", "నేను", "మీరు", "అని", "అనే",
        "చేసే", "చేశారు", "జరిగింది", "వచ్చింది", "వచ్చాడు", "చేయడం",
        "వంటి", "మీద", "క్రింద", "అక్కడ", "ఇక్కడ", "ఎప్పుడు", "ఎక్కడ",
        "ఎందుకు", "ఎవరు", "ఏమి", "మా", "మీ", "వాటి", "వాటిని",
    },
}

SCRIPT_RANGES = {
    "latin": r"[\x00-\x7F]",
    "devanagari": r"[\u0900-\u097F]",
    "telugu": r"[\u0C00-\u0C7F]",
}
SCRIPT_PATTERNS = {k: re.compile(v) for k, v in SCRIPT_RANGES.items()}


def log(msg: str) -> None:
    print(f"[module5] {msg}")


def detect_script(text: str) -> str:
    """Classify utterance script by dominant char range."""
    counts = {name: len(p.findall(text)) for name, p in SCRIPT_PATTERNS.items()}
    if not any(counts.values()):
        return "other"
    return max(counts, key=counts.get)


def tokenize_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def remove_stopwords(words: list[str], lang: str) -> list[str]:
    sw = STOPWORDS.get(lang, set())
    return [w for w in words if w not in sw and len(w) > 1]


def sentence_stats(text: str) -> tuple[int, float]:
    sents = [s for s in SENT_SPLIT.split(text.strip()) if s]
    return len(sents), float(np.mean([len(s.split()) for s in sents])) if sents else 0.0


def language_stats(texts: list[str], lang: str) -> dict:
    all_words: list[str] = []
    n_sent = 0
    sent_lens: list[float] = []
    for t in texts:
        words = tokenize_words(t)
        all_words.extend(words)
        ns, mean = sentence_stats(t)
        n_sent += ns
        if ns:
            sent_lens.append(mean)
    content = remove_stopwords(all_words, lang)
    freq = Counter(content)
    return {
        "language": lang,
        "utterances": len(texts),
        "total_tokens": len(all_words),
        "content_tokens": len(content),
        "unique_tokens": len(set(all_words)),
        "type_token_ratio": round(len(set(all_words)) / len(all_words), 4) if all_words else 0.0,
        "total_sentences": n_sent,
        "mean_sentence_len": round(float(np.mean(sent_lens)), 3) if sent_lens else 0.0,
        "top_20_words": freq.most_common(20),
    }


def text_language_model(df: pd.DataFrame) -> dict:
    """Char n-gram TF-IDF + LogisticRegression for text language ID."""
    X, y = df["source_text"].tolist(), df["language"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          max_features=20000, lowercase=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=500, random_state=SEED)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    return {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision_macro": round(float(precision_score(y_test, preds, average="macro")), 4),
        "recall_macro": round(float(recall_score(y_test, preds, average="macro")), 4),
        "f1_macro": round(float(f1_score(y_test, preds, average="macro")), 4),
        "classification_report": classification_report(
            y_test, preds, labels=["en", "hi", "te"],
            target_names=["en", "hi", "te"], output_dict=True),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(CLEAN_PARQUET)
    df = df.dropna(subset=["source_text"])

    df["script"] = df["source_text"].map(detect_script)
    script_check = df.groupby("language")["script"].value_counts().unstack(fill_value=0)
    log("script detection (rows per language):")
    print(script_check.to_string())

    per_lang = {}
    for lang in ["en", "hi", "te"]:
        texts = df[df["language"] == lang]["source_text"].tolist()
        per_lang[lang] = language_stats(texts, lang)

    model_res = text_language_model(df)
    per_lang["_text_language_model"] = model_res

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(per_lang, f, indent=2, ensure_ascii=False)

    print("\n=== NLP stats per language ===")
    for lang in ["en", "hi", "te"]:
        s = per_lang[lang]
        print(f"{lang}: tokens={s['total_tokens']} unique={s['unique_tokens']} "
              f"ttr={s['type_token_ratio']} sentences={s['total_sentences']} "
              f"mean_slen={s['mean_sentence_len']}")
        print(f"   top words: {s['top_20_words'][:8]}")

    print("\n=== Text language identification (char n-gram + LogReg) ===")
    print(f"acc={model_res['accuracy']} prec={model_res['precision_macro']} "
          f"rec={model_res['recall_macro']} f1={model_res['f1_macro']} "
          f"(train {model_res['train_samples']} / test {model_res['test_samples']})")
    print("Saved to:", os.path.join(OUT_DIR, "results.json"))


if __name__ == "__main__":
    main()