# CLAUDE.md - Project Standards

## Project Overview
**Name**: Review Helpfulness Prediction
**Goal**: Identify helpful reviews using NLP and Machine Learning.

## Technical Stack
- **Languages**: Python 3.9+
- **ML Frameworks**: Scikit-Learn, XGBoost
- **NLP**: NLTK, VADER
- **UI**: Streamlit, Plotly
- **Data**: Pandas, NumPy

## Coding Style
- **Formatting**: Use `black` (checked via GitHub Actions).
- **Linting**: Follow `flake8` standards.
- **Type Hints**: Encouraged for complex functions.
- **Modularity**: Keep data stages separated in `src-code/XX_stage_name.py`.

## Useful Commands
- `streamlit run app/app.py`: Start the dashboard.
- `python -m pytest`: Run tests (if implemented).
- `black .`: Reformat all code.
