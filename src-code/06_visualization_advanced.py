"""
DAP391m Project - Visualization & Metrics
=========================================
Generates performance charts and full dataset predictions for the dashboard.
"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from scipy.sparse import hstack

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "filtered" / "processed"
MODEL_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "visualization_outputs"
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Load artifacts
    with open(MODEL_DIR / "helpfulness_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    
    # Load full processed data for dashboard
    df = pd.read_csv(INPUT_CSV)
    
    # Prepare features for all reviews
    print("Generating predictions for dashboard...")
    X_text = df['processed_text'].astype(str)
    X_tfidf = vectorizer.transform(X_text)
    X_meta = df[['rating', 'review_length', 'verified_purchase']].copy()
    X_meta['verified_purchase'] = X_meta['verified_purchase'].astype(int)
    
    X_all = hstack([X_tfidf, X_meta.values])
    
    # Predictions
    df['helpfulness_score'] = model.predict_proba(X_all)[:, 1]
    df['predicted_helpful'] = model.predict(X_all)
    
    # Save predictions
    df.to_csv(PROJECT_ROOT / "Data" / "filtered" / "predictions.csv", index=False)
    print(f"Predictions saved to {PROJECT_ROOT / 'Data' / 'filtered' / 'predictions.csv'}")

    # Metrics plots (using all data as a proxy for test performance for now)
    y_test = (df['helpful_vote'] >= 1).astype(int)
    y_pred = df['predicted_helpful']
    y_probs = df['helpfulness_score']
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Helpful', 'Helpful'], yticklabels=['Not Helpful', 'Helpful'])
    plt.title('Confusion Matrix')
    plt.savefig(OUT_DIR / "confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(OUT_DIR / "roc_curve.png")
    plt.close()

    print(f"Metrics plots saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
