# Project Problems & Solutions

### 1. Imbalanced Data (Helpfulness Votes)
**Problem**: Most reviews have 0 helpful votes, making it hard for models to learn what makes a review helpful.
**Solution**: Used a binary classification approach with a threshold (votes >= 1) and balanced the classes using the `scale_pos_weight` parameter in XGBoost (initially) or by sampling/weighting in Logistic Regression.

### 2. Large Data Files
**Problem**: The Amazon raw dataset is too large for standard git uploads.
**Solution**: Used `.jsonl.gz` compressed format and created a robust `.gitignore` that excludes raw data while keeping processed aggregations for the dashboard.

### 3. GitHub Push Conflict
**Problem**: Branch protection on `main` prevented initial force pushes.
**Solution**: Used a `dev` branch for intermediate work and merged to `main` using `--allow-unrelated-histories` once the structure was stabilized.
