"""
DAP391m Project - Feature Engineering (Sentiment Included)
==========================================================
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import spacy
from textstat import flesch_reading_ease

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

comparative_keywords = {
    "better",
    "worse",
    "more",
    "less",
    "than",
    "compared",
    "superior",
    "inferior",
    "greatest",
    "best",
}


def extract_advanced_features_batch(texts):
    results = []
    for doc in nlp.pipe(texts, batch_size=200):
        if not doc.text.strip():
            results.append((0.0, 0.0, 0, 0.0, 0))
            continue
        num_tokens = max(len(doc), 1)
        noun_count = sum(1 for token in doc if token.pos_ == "NOUN")
        adj_count = sum(1 for token in doc if token.pos_ == "ADJ")
        ner_count = len(doc.ents)

        try:
            readability = flesch_reading_ease(doc.text)
        except:
            readability = 0.0

        words_set = set(doc.text.lower().split())
        comparative_count = len(words_set.intersection(comparative_keywords))
        results.append(
            (
                noun_count / num_tokens,
                adj_count / num_tokens,
                ner_count,
                readability,
                comparative_count,
            )
        )
    return pd.DataFrame(
        results,
        columns=[
            "noun_ratio",
            "adj_ratio",
            "ner_count",
            "readability",
            "comparative_count",
        ],
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        return
    df = pd.read_csv(INPUT_CSV)

    print("Extracting advanced linguistic features (NER, POS, Readability)...")
    adv_feats_df = extract_advanced_features_batch(df["text"].astype(str))
    df = pd.concat([df, adv_feats_df], axis=1)

    # Save back to CSV so step 06 can use the features without recomputing
    df.to_csv(INPUT_CSV, index=False)

    # Target
    df["is_helpful"] = (df["helpful_vote"] >= 1).astype(int)

    # Features: Text + Meta (Rating, Length, Verified, Sentiment + Mới)
    X_text = df["processed_text"].astype(str)
    X_meta = df[
        [
            "rating",
            "review_length",
            "word_count",
            "sentence_count",
            "avg_word_length",
            "verified_purchase",
            "sentiment_score",
            "noun_ratio",
            "adj_ratio",
            "ner_count",
            "readability",
            "comparative_count",
        ]
    ].copy()
    X_meta["verified_purchase"] = X_meta["verified_purchase"].astype(int)

    y = df["is_helpful"]

    # Vectorizer: Use bigrams and more features for better context
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_tfidf_sparse = vectorizer.fit_transform(X_text)

    with open(ARTIFACT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    X_train_meta, X_test_meta, X_train_tfidf, X_test_tfidf, y_train, y_test = (
        train_test_split(
            X_meta, X_tfidf_sparse, y, test_size=0.2, random_state=42, stratify=y
        )
    )

    # Save
    X_train_meta.to_csv(OUT_DIR / "X_train_meta.csv", index=False)
    X_test_meta.to_csv(OUT_DIR / "X_test_meta.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    with open(OUT_DIR / "X_train_tfidf.pkl", "wb") as f:
        pickle.dump(X_train_tfidf, f)
    with open(OUT_DIR / "X_test_tfidf.pkl", "wb") as f:
        pickle.dump(X_test_tfidf, f)

    print("Feature engineering complete.")


if __name__ == "__main__":
    main()
