# Reflection on AI Tool Usage – DAP391m Group 10

> **Project**: Review Helpfulness Prediction  
> **Author**: Group 10  
> **Date**: June 2026

---

## 1. Overview

Throughout this project, our team leveraged AI tools (GitHub Copilot, ChatGPT, and Google Gemini) to accelerate development across all pipeline stages — from data ingestion and cleaning, to feature engineering, modeling, and visualization. This document reflects on the accuracy, limitations, and responsible use of these tools.

---

## 2. Accuracy and Reliability

### What Worked Well
- **Data Preprocessing**: AI-generated cleaning functions (regex, punctuation removal) were accurate and ready to use with minimal modification. This saved approximately 2-3 hours of boilerplate coding.
- **Feature Engineering**: The `nlp.pipe()` batch processing suggestion for SpaCy was technically correct and significantly improved performance. The `textstat.flesch_reading_ease()` integration was also accurate.
- **Model Configuration**: The `scale_pos_weight` formula was mathematically correct and the resulting improvement in XGBoost Recall from 21% to ~60% validated the AI's recommendation.

### Where AI Made Mistakes / Required Verification
- **Library Version Conflicts**: ChatGPT initially suggested `use_label_encoder=False` for XGBoost, which caused a deprecation warning in newer XGBoost versions. We had to look up the official changelog to identify and remove this parameter.
- **SpaCy Model Download**: The suggested `subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])` pattern worked in development but required manual verification in CI/CD.
- **Gemini API**: The AI initially provided example code using an older `PaLM 2` API endpoint. We had to update it to the `google-genai` package with `gemini-2.5-flash` model.

---

## 3. Critical Assessment

| Aspect | Rating | Notes |
|---|---|---|
| Code Accuracy | 8/10 | Minor library compatibility issues |
| Conceptual Guidance | 9/10 | Solid advice on imbalanced classification |
| Time Savings | 9/10 | Significantly reduced boilerplate time |
| Reliability (requires verification) | 7/10 | Always needs testing before production use |

---

## 4. Responsible Use of AI

Our team followed these principles when using AI tools:

1. **Verify, don't blindly copy**: Every AI-generated code snippet was reviewed, tested, and adapted to our specific dataset and pipeline before being integrated.
2. **Use AI for acceleration, not substitution**: AI generated the skeleton; domain knowledge (understanding class imbalance, choosing F1 over Accuracy) came from the team.
3. **Cite AI assistance appropriately**: This log (`log_prompts.md`) documents every significant AI interaction transparently.
4. **Avoid data privacy issues**: We never sent raw customer review data to external AI APIs. Only anonymized, aggregated queries were used.
5. **Test AI-generated code rigorously**: We ran unit verifications on all AI-generated functions, particularly the feature extraction pipeline.

---

## 5. Faithfulness & Stability
- **Faithfulness**: We verified that AI-suggested feature engineering (NER, POS) directly contributed to the F1-score increase. We didn't just accept suggestions; we tested them against the baseline.
- **Stability**: Prompts were refined multiple times to ensure consistent code output. For example, the `clean_text` prompt was adjusted to specify "no special characters" to avoid edge cases in later stages.

## 6. Responsible AI Practices
- **Privacy**: We strictly used public datasets and never uploaded sensitive API keys or credentials to AI chat interfaces.
- **Transparency**: This project includes an "AI Insights" feature that uses Gemini to explain model decisions, ensuring the AI-human collaboration is transparent to the end-user.
- **Bias Mitigation**: We used AI to suggest sampling strategies (like `scale_pos_weight`) to mitigate the inherent bias in the imbalanced Amazon dataset.

## 7. Key Takeaway
AI tools significantly boosted our productivity... [Same as before]
