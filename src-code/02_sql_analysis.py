"""
DAP391m Project - SQL-like Analysis (E-Shop Version)
===================================================
Performs business analysis using Pandas to mimic SQL queries.
Writes outputs to Data/filtered/sql_outputs/.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_data.csv"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "sql_outputs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    # 1. Monthly Category Trends
    monthly_trends = (
        df.groupby(["month", "page 1 (main category)"]).size().unstack().fillna(0)
    )
    monthly_trends.to_csv(OUT_DIR / "monthly_category_trends.csv")

    # 2. Top Countries by Session Volume
    top_countries = df.groupby("country").size().sort_values(ascending=False).head(10)
    top_countries.to_csv(OUT_DIR / "top_countries.csv")

    # 3. Average Price by Category
    avg_price = df.groupby("page 1 (main category)")["price"].mean()
    avg_price.to_csv(OUT_DIR / "avg_price_by_category.csv")

    print(f"SQL analysis outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
