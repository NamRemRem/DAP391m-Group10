# AI Audit Log – DAP391m Group 10

| Phase / Action | Prompt Used | AI Tool & Response Summary | Human Correction / Reflection |
| :--- | :--- | :--- | :--- |
| **Data Collection** | "I have an Amazon Beauty reviews dataset in JSONL format... How should I define a 'helpful' review?" | **ChatGPT-4o**: Suggested binary threshold `helpful_vote >= 1` and handling imbalance. | Threshold strategy was effective. Imbalance warning was critical for model choice. |
| **Cleaning** | "Show me a Python function to load `.jsonl.gz` files and clean the `text` field..." | **GitHub Copilot**: Provided `clean_text()` using regex and `string.punctuation`. | Function worked out-of-the-box. Added NLTK lemmatization manually. |
| **Feature Eng.** | "I want to extract NLP features: POS ratios, NER count, Flesch score... How to do this for 50k texts?" | **Claude 3.5**: Recommended `nlp.pipe()` for batch processing in SpaCy. | `nlp.pipe()` reduced processing time significantly. Set `batch_size=200`. |
| **Modeling** | "Training XGBoost on imbalance data (1.7% pos). How to use `scale_pos_weight` and `RandomizedSearchCV`?" | **ChatGPT-4o**: Provided formula and F1-scoring strategy. | Recall rose from 21% to ~60% after applying `scale_pos_weight`. |
| **AI Integration** | "I want a 'Get AI Insights' feature where Gemini explains review helpfulness..." | **Google Gemini**: Provided `get_gemini_insights()` using `google-genai`. | Used `gemini-2.5-flash`. Improves transparency for predicting review quality. |
| **Visualization** | "Add advanced viz: correlation heatmap, XGBoost feature importance, and violin plots." | **Claude 3.5**: Suggested `plotly.express.imshow()` and `px.violin()`. | Violin plots showed helpful reviews are more neutral/descriptive than emotional. |
| **Pipeline Expansion** | "Add LinearSVC and MultinomialNB. TF-IDF causes negative values error with Naive Bayes." | **Gemini Adv**: Suggested `ComplementNB` with `np.abs()` preprocessing, and `CalibratedClassifierCV` for LinearSVC PR-tuning. | Successfully added 2 new robust models to our final 5-Model dashboard. |

---

## Faithfulness & Verification
- **Model Scores**: Verified against test set before/after AI-suggested tuning.
- **Linguistic Check**: Manually verified top features (Readability, Word Count) correlate with labels.
- **Code Audit**: All AI snippets were linted with `black` and `flake8`.
