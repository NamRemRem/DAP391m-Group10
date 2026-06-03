import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import string

# Page config
st.set_page_config(page_title="Review Helpfulness Project", layout="wide")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "filtered"
MODEL_DIR = DATA_DIR / "model_outputs"

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text

def load_data():
    if not (DATA_DIR / "predictions.csv").exists():
        return None
    return pd.read_csv(DATA_DIR / "predictions.csv")


def load_stats():
    if not (MODEL_DIR / "model_stats.json").exists():
        return {}
    with open(MODEL_DIR / "model_stats.json", "r") as f:
        return json.load(f)


def main():
    st.title("🛡️ Review Helpfulness Prediction Dashboard")
    st.markdown(
        "An enhanced project to analyze and predict the quality of marketplace reviews using NLP and Machine Learning."
    )

    df = load_data()
    stats = load_stats()
    if df is None:
        st.error("Please run the pipeline to generate data.")
        return

    # Sidebar
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.selectbox(
        "Choose Model", options=["XGBoost", "Random Forest", "Logistic Regression"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Info")
    st.sidebar.write("**Dataset**: Amazon All Beauty")
    st.sidebar.write("**Features**: Rating, Length, Linguistic (Word/Sentence), Sentiment, TF-IDF")

    # Row 1: High Level Stats
    st.subheader("📊 Model Performance & Data Overview")
    model_metrics = stats.get(model_choice, {})
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Accuracy", f"{model_metrics.get('Accuracy', 0):.2%}")
    col_b.metric("F1-Score", f"{model_metrics.get('F1-Score', 0):.2%}")
    col_c.metric("Precision", f"{model_metrics.get('Precision', 0):.2%}")
    col_d.metric("Recall", f"{model_metrics.get('Recall', 0):.2%}")

    st.markdown("---")

    # Row 2: Charts (Helpfulness & Ratings)
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Predicted Helpfulness Distribution")
        pie_data = df["predicted_helpful"].value_counts().reset_index()
        pie_data.columns = ["Status", "Count"]
        pie_data["Status"] = pie_data["Status"].map({1: "Helpful", 0: "Non-Helpful"})
        fig_pie = px.pie(
            pie_data,
            values="Count",
            names="Status",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.write("#### Rating Distribution")
        rating_counts = df["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Reviews"]
        fig_bar = px.bar(
            rating_counts,
            x="Rating",
            y="Reviews",
            color="Reviews",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Row 3: Live Prediction
    st.write("---")
    st.subheader("🔮 Live Review Helpfulness Predictor")
    with st.expander("Try it yourself!"):
        user_text = st.text_area("Enter a product review text:", "This product is amazing! It really helped my skin and smells great.")
        user_rating = st.slider("Select Rating", 1, 5, 5)
        
        if st.button("Predict Helpfulness"):
            # Load model and vectorizer
            model_file = model_choice.lower().replace(" ", "_") + ".pkl"
            with open(MODEL_DIR / model_file, "rb") as f:
                model = pickle.load(f)
            with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            
            # Preprocess
            from scipy.sparse import hstack
            import numpy as np
            import spacy
            from textstat import flesch_reading_ease
            
            cleaned = clean_text(user_text)
            tfidf_feat = vectorizer.transform([cleaned])
            
            sent_score = analyzer.polarity_scores(user_text)["compound"]
            word_count = len(user_text.split())
            sent_count = len(re.split(r'[.!?]+', user_text))
            avg_word_len = sum(len(w) for w in user_text.split()) / max(word_count, 1)
            
            try:
                nlp_mdl = spacy.load('en_core_web_sm')
                doc = nlp_mdl(user_text)
                num_tokens = max(len(doc), 1)
                noun_ratio = sum(1 for token in doc if token.pos_ == 'NOUN') / num_tokens
                adj_ratio = sum(1 for token in doc if token.pos_ == 'ADJ') / num_tokens
                ner_count = len(doc.ents)
                readability = flesch_reading_ease(user_text)
            except:
                noun_ratio = adj_ratio = ner_count = readability = 0.0

            comparative_keywords = {'better', 'worse', 'more', 'less', 'than', 'compared', 'superior', 'inferior', 'greatest', 'best'}
            comparative_count = len(set(user_text.lower().split()).intersection(comparative_keywords))
            
            meta_feat = np.array([[
                user_rating, 
                len(user_text), 
                word_count, 
                sent_count, 
                avg_word_len, 
                1, # Verified
                sent_score,
                noun_ratio, 
                adj_ratio, 
                ner_count, 
                readability, 
                comparative_count
            ]])
            
            X_input = hstack([tfidf_feat, meta_feat])
            
            prob = model.predict_proba(X_input)[0][1]
            pred = model.predict(X_input)[0]
            
            if pred == 1:
                st.success(f"High probability of being HELPFUL ({prob:.2%})")
            else:
                st.warning(f"Low probability of being helpful ({prob:.2%})")

    # Row 4: Example Predictions
    st.write("---")
    st.subheader("🔍 Sample Dataset Predictions")
    num_samples = st.slider("Number of samples to show", 5, 50, 10)
    samples = df.sample(num_samples)
    st.table(
        samples[
            [
                "text",
                "rating",
                "sentiment_score",
                "helpfulness_score",
                "predicted_helpful",
            ]
        ].head(num_samples)
    )


if __name__ == "__main__":
    main()
