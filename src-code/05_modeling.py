"""
DAP391m Project - Modeling (Logistic Regression & Random Forest)
================================================================
Trains two models for the student dashboard.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "filtered" / "processed"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "model_outputs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    X_train_meta = pd.read_csv(DATA_DIR / "X_train_meta.csv")
    X_test_meta = pd.read_csv(DATA_DIR / "X_test_meta.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

    with open(DATA_DIR / "X_train_tfidf.pkl", "rb") as f:
        X_train_tfidf = pickle.load(f)
    with open(DATA_DIR / "X_test_tfidf.pkl", "rb") as f:
        X_test_tfidf = pickle.load(f)

    X_train = hstack([X_train_tfidf, X_train_meta.values])
    X_test = hstack([X_test_tfidf, X_test_meta.values])

    model_stats = {}

    # 1. Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    model_stats["Logistic Regression"] = float(lr_acc)
    with open(OUT_DIR / "logistic_regression.pkl", "wb") as f:
        pickle.dump(lr, f)

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    model_stats["Random Forest"] = float(rf_acc)
    with open(OUT_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf, f)

    # Save Stats
    with open(OUT_DIR / "model_stats.json", "w") as f:
        json.dump(model_stats, f)

    print(f"Modeling complete. Stats: {model_stats}")


if __name__ == "__main__":
    main()
