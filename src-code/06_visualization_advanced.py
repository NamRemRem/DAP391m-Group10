"""
DAP391m Project - Data Prep for Dashboard
=========================================
"""

import pandas as pd
import pickle
from pathlib import Path
from scipy.sparse import hstack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"

def main():
    # Load artifacts (Using RF as default for full data prep)
    with open(MODEL_DIR / "random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    df = pd.read_csv(INPUT_CSV)
    
    # Prepare features
    X_text = df['processed_text'].astype(str)
    X_tfidf = vectorizer.transform(X_text)
    X_meta = df[['rating', 'review_length', 'verified_purchase', 'sentiment_score']].copy()
    X_meta['verified_purchase'] = X_meta['verified_purchase'].astype(int)
    
    X_all = hstack([X_tfidf, X_meta.values])
    
    # Predictions
    df['helpfulness_score'] = model.predict_proba(X_all)[:, 1]
    df['predicted_helpful'] = model.predict(X_all)
    
    # Parse timestamp for line chart
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    
    # Save
    df.to_csv(PROJECT_ROOT / "Data" / "filtered" / "predictions.csv", index=False)
    print("Predictions for dashboard complete.")

if __name__ == "__main__":
    main()
