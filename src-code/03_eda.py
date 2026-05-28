"""
DAP391m Project - EDA (Review Helpfulness)
==========================================
Analyzes relationships between features and helpfulness votes.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "eda_outputs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_CSV.exists():
        print("Input file not found.")
        return

    df = pd.read_csv(INPUT_CSV)

    # 1. Correlation between Length and Helpfulness
    plt.figure(figsize=(10, 6))
    # Using log scale for better visibility due to outliers
    sns.scatterplot(
        data=df[df["helpful_vote"] > 0], x="review_length", y="helpful_vote", alpha=0.3
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Review Length vs Helpful Votes (Log Scale)")
    plt.savefig(OUT_DIR / "length_vs_helpfulness.png")
    plt.close()

    # 2. Helpfulness vs Rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[df["helpful_vote"] > 0], x="rating", y="helpful_vote")
    plt.yscale("log")
    plt.title("Rating vs Helpful Votes")
    plt.savefig(OUT_DIR / "rating_vs_helpfulness.png")
    plt.close()

    # 3. Verified Purchase Impact
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="verified_purchase", y="helpful_vote")
    plt.title("Average Helpful Votes: Verified vs Non-Verified")
    plt.savefig(OUT_DIR / "verified_impact.png")
    plt.close()

    print(f"EDA plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
