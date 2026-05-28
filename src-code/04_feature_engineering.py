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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_CSV.exists(): return
    df = pd.read_csv(INPUT_CSV)
    
    # Target
    df['is_helpful'] = (df['helpful_vote'] >= 1).astype(int)
    
    # Features: Text + Meta (Rating, Length, Verified, Sentiment)
    X_text = df['processed_text'].astype(str)
    X_meta = df[['rating', 'review_length', 'verified_purchase', 'sentiment_score']].copy()
    X_meta['verified_purchase'] = X_meta['verified_purchase'].astype(int)
    
    y = df['is_helpful']
    
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 1)) # Simplified for presentation
    X_tfidf_sparse = vectorizer.fit_transform(X_text)
    
    with open(ARTIFACT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    X_train_meta, X_test_meta, X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_meta, X_tfidf_sparse, y, test_size=0.2, random_state=42, stratify=y
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
