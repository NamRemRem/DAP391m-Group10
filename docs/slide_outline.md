# Slide Outline (DAP391m Final Presentation)
*Based on DAP391m_Slide_Presentation_Final_Sample.pptx*

## Slide 1: Title & Team
- **Title**: Predicting Review Helpfulness using NLP & XGBoost
- **Subtitle**: A Case Study on the Amazon Beauty Dataset
- **Team**: Group 10 - DAP391m
- **Members**: [User Name], [Partner Name]

## Slide 2: Introduction & Motivation
- **Context**: Growth of e-commerce and information overload.
- **Goal**: Help users find reliable feedback while filtering out superficial reviews.

## Slide 3: Problem Statement
- **The "So What?"**: Users ignore reviews if they can't distinguish "Verified/Detailed" from "Random/Emotional".
- **Metric gap**: Native "Helpful votes" are sparse and imbalanced.

## Slide 4: Related Work
| Project | Focus | Our Improvement |
| :--- | :--- | :--- |
| Standard EDA | Rating analysis | Advanced Linguistic Scoring (NER, POS) |
| Basic ML | Logistic Regression | Optimized XGBoost (scale_pos_weight) |
| Output | Static Report | **Interactive Dashboard + AI Explainer** |

## Slide 5: Methodology (The Pipeline)
- **Data**: Amazon Reviews 2023 (50k records).
- **Processing**: Lemmatization -> Feature Extraction (Readability, NER) -> Scaling.
- **Integration**: Cloud-based Explainability (Google Gemini).

## Slide 6: Model Selection & Results
- **Comparison**: Logistic Regression vs. Random Forest vs. XGBoost.
- **Key Metric**: F1-Score (60% Recall achieved despite 1.7% class imbalance).
- **Success**: XGBoost outperformed others in identifying sparse "Helpful" signals.

## Slide 7: Advanced Visualization (Dashboard)
- Screenshare/Screenshot of the **Streamlit Dashboard**.
- Show the **Heatmap** and **Feature Importance** charts.
- Highlight how Sentiment vs. Helpfulness is visualized (Violin Plot).

## Slide 8: Explainable AI (XAI)
- **The Solution**: Integrating Google Gemini for "Intelligent Q&A".
- **Screenshot**: Show the "AI Insights" explainability feature.
- **Benefit**: Transparent decision support for the customer.

## Slide 9: AI Audit & Reflection
- Overview of how AI tools (Gemini, Claude, Copilot) helped build the project.
- Focus on: **Faithfulness** and **Responsible AI** practices.

## Slide 10: Conclusion & Future Work
- **Summary**: Built an end-to-end predictive and explanatory system.
- **Future**: Deep learning (BERT) and cross-category testing.
