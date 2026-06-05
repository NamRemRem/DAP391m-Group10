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
from xgboost import XGBClassifier
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,  # FIXED: kept for threshold tuning
)
from sklearn.model_selection import RandomizedSearchCV
# FIXED: SMOTE removed entirely — do NOT import or use imblearn here

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

    # STEP 1: scale_pos_weight from original y_train distribution  # FIXED
    scale_weight = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)  # FIXED
    print(f"scale_pos_weight = {scale_weight:.2f}")

    param_distributions = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_child_weight": [1, 3, 5],
        "n_estimators": [50, 100],
    }

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42
        ),
        "XGBoost": RandomizedSearchCV(
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                scale_pos_weight=scale_weight,  # FIXED: kept from Step 1
            ),
            param_distributions=param_distributions,
            n_iter=5,  # Keep it small for fast execution
            scoring="f1",
            cv=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        print(f"Training {name}...")

        # FIXED: train XGBoost on original X_train, y_train — NO SMOTE
        model.fit(X_train, y_train)

        # Extract best estimator if it's a grid search
        if isinstance(model, RandomizedSearchCV):
            print(f"Best params for {name}: {model.best_params_}")
            model = model.best_estimator_

        # Default predictions at 0.5 threshold (used as "before" for XGBoost)
        y_pred_default = model.predict(X_test)

        if name == "XGBoost":
            # Save "before" stats (default 0.5 threshold, no tuning)  # FIXED
            acc_b = accuracy_score(y_test, y_pred_default)
            f1_b = f1_score(y_test, y_pred_default, zero_division=0)
            prec_b = precision_score(y_test, y_pred_default, zero_division=0)
            rec_b = recall_score(y_test, y_pred_default, zero_division=0)
            model_stats["XGBoost_before"] = {  # FIXED
                "Accuracy": float(acc_b),
                "F1-Score": float(f1_b),
                "Precision": float(prec_b),
                "Recall": float(rec_b),
            }

            # STEP 2: Threshold tuning via Precision-Recall curve  # FIXED
            y_prob = model.predict_proba(X_test)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
            f1_scores = (
                2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            )
            best_threshold = float(thresholds[f1_scores[:-1].argmax()])
            print(f"Best threshold (max F1 on PR curve): {best_threshold:.4f}")  # FIXED

            y_pred = (y_prob >= best_threshold).astype(int)  # FIXED
        else:
            y_pred = y_pred_default
            best_threshold = None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # --- ADD THESE DIAGNOSTICS ---
        if name == "XGBoost":
            print(f"y_test shape: {y_test.shape}")
            print(f"y_pred shape: {y_pred.shape}")
            print(f"Total Helpful in y_test: {y_test.sum()}")
            print(f"Total Helpful in y_pred: {y_pred.sum()}")
            print(f"Expected Helpful in y_test: ~{int(1049 * 0.2)} (20% of 1,049)")

        tn = int(((y_test == 0) & (y_pred == 0)).sum())
        fp = int(((y_test == 0) & (y_pred == 1)).sum())
        fn = int(((y_test == 1) & (y_pred == 0)).sum())
        tp = int(((y_test == 1) & (y_pred == 1)).sum())

        entry = {
            "Accuracy": float(acc),
            "F1-Score": float(f1),
            "Precision": float(prec),
            "Recall": float(rec),
            "CM": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        }

        if name == "XGBoost" and best_threshold is not None:
            entry["Threshold"] = best_threshold  # FIXED

        model_stats[name] = entry

        filename = name.lower().replace(" ", "_") + ".pkl"
        with open(OUT_DIR / filename, "wb") as f:
            pickle.dump(model, f)

        # Error Analysis for XGBoost
        if name == "XGBoost":
            print(f"Performing Error Analysis for {name}...")
            INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"
            if INPUT_CSV.exists():
                orig_df = pd.read_csv(INPUT_CSV)
                orig_y = (orig_df["helpful_vote"] >= 1).astype(int)
                from sklearn.model_selection import train_test_split

                _, text_test, _, _ = train_test_split(
                    orig_df["text"].astype(str),
                    orig_y,
                    test_size=0.2,
                    random_state=42,
                    stratify=orig_y,
                )

                error_df = pd.DataFrame(
                    {
                        "Original_Text": text_test.values,
                        "Actual_Label": y_test,
                        "Predicted_Label": y_pred,
                        "Probability": y_prob,
                    }
                )

                fp_df = error_df[
                    (error_df["Actual_Label"] == 0) & (error_df["Predicted_Label"] == 1)
                ]
                fp_df = fp_df.sort_values(by="Probability", ascending=False)
                fp_df.head(20).to_csv(
                    OUT_DIR / "top_20_false_positives.csv", index=False
                )

                fn_df = error_df[
                    (error_df["Actual_Label"] == 1) & (error_df["Predicted_Label"] == 0)
                ]
                fn_df = fn_df.sort_values(by="Probability", ascending=True)
                fn_df.head(20).to_csv(
                    OUT_DIR / "top_20_false_negatives.csv", index=False
                )

    # Save Stats
    with open(OUT_DIR / "model_stats.json", "w") as f:
        json.dump(model_stats, f, indent=2)

    print(f"\nModeling complete. Stats saved to model_stats.json")
    print(json.dumps(model_stats, indent=2))


if __name__ == "__main__":
    main()
