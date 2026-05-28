"""
DAP391m Project - Ingestion & Cleaning (Amazon Review Helpfulness)
==================================================================
Loads compressed JSONL data, extracts relevant fields, and saves to CSV.
"""

import gzip
import json
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_GZ = PROJECT_ROOT / "Data" / "raw" / "All_Beauty.jsonl.gz"
OUTPUT_DIR = PROJECT_ROOT / "Data" / "filtered"
OUTPUT_CSV = OUTPUT_DIR / "clean_reviews.csv"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_GZ.exists():
        print(f"Error: {INPUT_GZ} not found.")
        sys.exit(1)
        
    print(f"Reading {INPUT_GZ}...")
    
    records = []
    # Using a limit for development speed, can be increased for full training
    LIMIT = 50000 
    
    with gzip.open(INPUT_GZ, 'rt', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            # Essential fields for helpfulness prediction
            record = {
                "rating": data.get("rating"),
                "title": data.get("title", ""),
                "text": data.get("text", ""),
                "helpful_vote": data.get("helpful_vote", 0),
                "verified_purchase": data.get("verified_purchase", False),
                "timestamp": data.get("timestamp"),
                "asin": data.get("asin", "")
            }
            records.append(record)
            if idx >= LIMIT:
                break
                
    df = pd.DataFrame(records)
    
    # Basic cleaning
    # Drop rows with empty text
    df = df[df['text'].str.strip().str.len() > 0].reset_index(drop=True)
    
    # Fill missing helpful votes with 0
    df['helpful_vote'] = df['helpful_vote'].fillna(0).astype(int)
    
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Ingestion complete. Saved {len(df)} records to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
