# Product Review Helpfulness Prediction

A machine learning project designed to identify and rank high-quality product reviews on e-commerce platforms. This dashboard helps surface the most useful content to customers, improving trust and sales.

## Features
- **Aggregate Dashboard**: View marketplace-wide review statistics.
- **Advanced NLP**: Lemmatization, SpaCy linguistic features (POS, NER), and Readability scoring (Flesch).
- **Flagship ML Classifier**: Optimized **XGBoost** with RandomizedSearchCV and class-weighting (scale\_pos\_weight).
- **Model Comparison**: Evaluation of Logistic Regression, Random Forest, and XGBoost using F1-Score and Recall.
- **Live Review Predictor**: Real-time helpfulness probability for custom text.
- **LaTeX Reporting**: Automated research paper generation in `report/paper.tex`.
- **CI/CD Integration**: Automated linting with Black and Flake8 via GitHub Actions.

## Project Structure
- `app/`: Streamlit web application.
- `src-code/`: Modular pipeline (Ingestion, Preprocessing, EDA, Modeling, Visualization).
- `Data/`: Filtered CSVs and trained model artifacts.
- `.github/`: CI workflows for code quality.

## Installation & Usage
1. **Clone the repo**:
   ```bash
   git clone https://github.com/NamRemRem/DAP391m-Group10.git
   cd DAP391m
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Pipeline**:
   ```bash
   # Execute scripts 01 to 06 in order
   python src-code/01_ingestion_cleaning.py
   ...
   ```
4. **Launch the Dashboard**:
   ```bash
   streamlit run app/app.py
   ```

## Dataset
This project uses the **Amazon Reviews 2023** (All Beauty category) dataset, which contains rich text data and helpfulness votes.

## License
MIT License
