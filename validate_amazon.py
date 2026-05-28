import gzip
import json
from pathlib import Path
import pandas as pd

file_path = Path("Data/raw/All_Beauty.jsonl.gz")


def validate():
    if not file_path.exists():
        print(f"Error: {file_path} does not exist.")
        return

    print(f"Validating {file_path}...")

    records = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            records.append(json.loads(line))
            if idx >= 1000:  # Sample 1000 records for validation
                break

    df = pd.DataFrame(records)

    print("\nColumns found:", df.columns.tolist())

    checks = {
        "Has Text": "text" in df.columns,
        "Has Helpful Votes": "helpful_vote" in df.columns,
        "Has Rating": "rating" in df.columns,
        "Has Verified Purchase": "verified_purchase" in df.columns,
    }

    print("\nRequirement Checks:")
    for check, passed in checks.items():
        print(f"[{'X' if passed else ' '}] {check}")

    if checks["Has Helpful Votes"]:
        helpful_counts = df["helpful_vote"].value_counts()
        print("\nHelpful Vote Distribution (Sample 1000):")
        print(helpful_counts.head(5))

        non_zero_helpful = (df["helpful_vote"] > 0).sum()
        print(f"\nReviews with >0 helpful votes: {non_zero_helpful}/1000")


if __name__ == "__main__":
    validate()
