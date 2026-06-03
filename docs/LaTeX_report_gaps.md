# LaTeX Report Improvement Checklist
*A guide for the report partner to reach Springer-level quality (as per main.tex)*

## 1. Structure & Formalism
- [ ] Change `\documentclass{article}` to something more formal if allowed (e.g., `llncs` or a two-column layout).
- [ ] Add an **"Experimental Setup"** section with specific sub-sections for:
    - *Hardware*: Intel/AMD Specs.
    - *Software*: PyTorch, XGBoost, Streamlit, Gemini.

## 2. Missing Tables (Crucial for Grade)
- [ ] **Table: Comparison with Prior Works**: See `main.tex` lines 147-253. Compare our "Advanced Helpfulness Pipeline" with simple baseline EDA or basic ML works.
- [ ] **Table: Dataset & Artifact Inventory**: List number of records, number of features, and artifacts produced (Trained models, Log prompts, CSVs).
- [ ] **Table: Model Performance Snapshot**: Explicitly show Accuracy, F1, Precision, and Recall for all 3 models (LR, RF, XGB).

## 3. Missing Figures
- [ ] **System Architecture Diagram**: Create a diagram (Canva or Mermaid) showing Ingestion -> Preprocessing -> Modeling -> **AI Explainability via Gemini**.
- [ ] **Feature Heatmap**: Use a screenshot of the Heatmap from the Streamlit app.
- [ ] **XAI Case Study Screenshot**: Show a "Helpful" review prediction with its Gemini-generated explanation.

## 4. Discussion & Quality
- [ ] **Evaluation Metric Justification**: Explain why we used **F1-Score** and **Recall** due to the 1.7% class imbalance.
- [ ] **XAI Faithfulness**: Mention that we used Gemini to ground the predictions in human-readable insights.

## 5. References
- [ ] Ensure citations follow a standard format (e.g., `[1]`, `[2]`).
- [ ] Include references for:
    - Amazon Reviews 2023 Dataset.
    - XGBoost and RandomizedSearchCV.
    - VADER Sentiment Analysis.
    - Google Gemini / Large Language Models in XAI.
