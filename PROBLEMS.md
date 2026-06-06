# Project Problems & Solutions

### 1. Imbalanced Data (Helpfulness Votes)
**Problem**: Most reviews have 0 helpful votes, making it hard for models to learn what makes a review helpful (Helpful class only ~1.7%).
**Solution**: Used a binary classification threshold (votes >= 1). Implemented **XGBoost, LinearSVC, and ComplementNB** utilizing class-imbalance weights (`scale_pos_weight`/`class_weight='balanced'`) to punish errors on the minority class. By extracting **advanced linguistic features** (Readability, POS, NER) and using robust PR-curve threshold tuning, we successfully improved Recall to ~73%.

### 2. Large Data Files
**Problem**: The Amazon raw dataset is too large for standard git uploads.
**Solution**: Used `.jsonl.gz` compressed format and created a robust `.gitignore` that excludes raw data while keeping processed aggregations for the dashboard.

### 3. GitHub Push Conflict
**Problem**: Branch protection on `main` prevented initial force pushes.
**Solution**: Used a `dev` branch for intermediate work and merged to `main` using `--allow-unrelated-histories` once the structure was stabilized.
