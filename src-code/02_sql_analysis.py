"""
DAP391m Project - SQL Analysis (SQLite Integration)
===================================================
Executes SQL queries against a local database to generate analytical reports.
Matches the structure of the DAP reference repository.
"""

import pandas as pd
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"
SCHEMA_SQL = PROJECT_ROOT / "sql" / "schema.sql"
ANALYSIS_SQL = PROJECT_ROOT / "sql" / "analysis.sql"
OUT_DIR = PROJECT_ROOT / "Data" / "filtered" / "sql_outputs"
DB_PATH = PROJECT_ROOT / "Data" / "filtered" / "reviews.db"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_CSV.exists():
        print("Processed reviews CSV not found.")
        return

    # 1. Initialize Database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Run schema
    with open(SCHEMA_SQL, 'r') as f:
        cursor.executescript(f.read())
        
    # 2. Load Data into SQL
    df = pd.read_csv(INPUT_CSV)
    df.to_sql('reviews', conn, if_exists='append', index=False)
    
    # 3. Run Analytical Queries
    print("Executing SQL analysis queries...")
    with open(ANALYSIS_SQL, 'r') as f:
        queries = f.read().split(';')
        
    query_names = [
        "avg_helpful_by_rating",
        "sentiment_helpfulness_corr",
        "monthly_volume_trends",
        "top_10_helpful_reviews",
        "verified_impact",
        "length_vs_helpfulness"
    ]
    
    for i, query in enumerate(queries):
        query = query.strip()
        if not query: continue
        
        # Execute and save result
        res_df = pd.read_sql_query(query, conn)
        name = query_names[i] if i < len(query_names) else f"query_{i}"
        res_df.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"Saved {name}.csv")

    conn.close()
    print(f"SQL analysis results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
