import os
import re
import string

import numpy as np
import pandas as pd
import pickle
import json
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── Optional: Google Generative AI ─────────────────────────────────────────────
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Review Helpfulness Project",
    layout="wide",
    page_icon="🛡️",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data" / "filtered"
MODEL_DIR = DATA_DIR / "model_outputs"

analyzer = SentimentIntensityAnalyzer()

META_COLS = [
    "rating", "review_length", "word_count", "sentence_count", "avg_word_length",
    "verified_purchase", "sentiment_score", "noun_ratio", "adj_ratio",
    "ner_count", "readability", "comparative_count",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text


@st.cache_data
def load_data():
    path = DATA_DIR / "predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_stats():
    path = MODEL_DIR / "model_stats.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def get_gemini_insights(api_key: str, review_text: str, pred: int, prob: float) -> str:
    """
    Calls Google Gemini API using the new google-genai SDK.
    """
    api_key = api_key.strip()
    
    label = "HELPFUL" if pred == 1 else "NOT HELPFUL"
    prompt_text = (
        f"You are an expert in product review quality. A machine learning model predicted that "
        f"the following review is {label} with a probability of {prob:.1%}.\n\n"
        f"Review:\n\"{review_text}\"\n\n"
        f"Briefly explain in 3-4 bullet points why the model likely classified it as {label} "
        "based on its content, length, sentiment, and linguistic style. "
        "Keep your response short, professional, and actionable. Do not use emojis."
    )
    
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_text
        )
        if response.text:
            return response.text
        return "Error: Empty response from Gemini API."
    except Exception as e:
        return (
            f"Gemini API Error ({str(e)}).\n\n"
            "**Recommendation:**\n1. Go to [Google AI Studio](https://aistudio.google.com/).\n"
            "2. Ensure you have a valid API Key and the 'Generative Language API' is enabled."
        )


# ── Main Dashboard ─────────────────────────────────────────────────────────────
def main():
    st.title("Review Helpfulness Prediction Dashboard")
    st.markdown(
        "An enhanced project to analyze and predict the quality of marketplace reviews "
        "using **NLP**, **Machine Learning**, and **Generative AI**."
    )

    df = load_data()
    stats = load_stats()
    if df is None:
        st.error("Data not found. Please run the pipeline first (scripts 01–06).")
        return

    # ── Sidebar ────────────────────────────────────────────────────────────────
    st.sidebar.header("Settings")
    model_choice = st.sidebar.selectbox(
        "Choose Model", options=["XGBoost", "Random Forest", "Logistic Regression"]
    )
    st.sidebar.markdown("---")

    # Gemini API Key input
    st.sidebar.subheader("AI Insights (Gemini)")
    gemini_api_key = st.sidebar.text_input(
        "Gemini API Key", type="password",
        help="Enter your Google Gemini API key to enable AI-powered review explanations.",
        placeholder="AIza...",
    )
    if not GEMINI_AVAILABLE:
        st.sidebar.warning("`google-genai` not installed. Run: `pip install google-genai`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Info")
    st.sidebar.write("**Dataset**: Amazon All Beauty")
    st.sidebar.write("**Features**: Rating, Length, Linguistic (POS, NER), Sentiment, TF-IDF")
    st.sidebar.write("**Best Model**: XGBoost + scale_pos_weight")

    # ── Row 1: KPIs ───────────────────────────────────────────────────────────
    st.subheader("Model Performance")
    model_metrics = stats.get(model_choice, {})
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Accuracy",  f"{model_metrics.get('Accuracy', 0):.2%}")
    col_b.metric("F1-Score",  f"{model_metrics.get('F1-Score', 0):.2%}")
    col_c.metric("Precision", f"{model_metrics.get('Precision', 0):.2%}")
    col_d.metric("Recall",    f"{model_metrics.get('Recall', 0):.2%}")
    st.markdown("---")

    # ── Row 2: Helpfulness Distribution + Rating Distribution ─────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Predicted Helpfulness Distribution")
        pie_data = df["predicted_helpful"].value_counts().reset_index()
        pie_data.columns = ["Status", "Count"]
        pie_data["Status"] = pie_data["Status"].map({1: "Helpful", 0: "Non-Helpful"})
        fig_pie = px.pie(
            pie_data, values="Count", names="Status",
            color_discrete_sequence=["#2ecc71", "#e74c3c"], hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.write("#### Rating Distribution")
        rating_counts = df["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Reviews"]
        fig_bar = px.bar(
            rating_counts, x="Rating", y="Reviews",
            color="Reviews", color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 3: Advanced Visualizations ────────────────────────────────────────
    st.subheader("📈 Advanced Analysis")
    col3, col4 = st.columns(2)

    with col3:
        st.write("#### Feature Correlation Heatmap")
        heat_cols = [
            "rating", "word_count", "sentiment_score",
            "readability", "noun_ratio", "predicted_helpful",
        ]
        # Only include columns that exist in the dataframe
        heat_cols = [c for c in heat_cols if c in df.columns]
        if heat_cols:
            corr = df[heat_cols].corr().round(2)
            fig_heat = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Pearson Correlation",
            )
            fig_heat.update_layout(margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Correlation data not available in current predictions file.")

    with col4:
        st.write("#### Feature Importance (XGBoost)")
        xgb_path = MODEL_DIR / "xgboost.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                xgb_model = pickle.load(f)
            if hasattr(xgb_model, "feature_importances_"):
                # Meta features only (last N features of the combined matrix)
                importances = xgb_model.feature_importances_
                n_meta = len(META_COLS)
                meta_importances = importances[-n_meta:]
                imp_df = pd.DataFrame({
                    "Feature": META_COLS,
                    "Importance": meta_importances,
                }).sort_values("Importance", ascending=True)
                fig_imp = px.bar(
                    imp_df, x="Importance", y="Feature",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="Teal",
                    title="Meta-Feature Importances",
                )
                fig_imp.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importances not available for this model version.")
        else:
            st.info("XGBoost model not found. Run the pipeline first.")

    # ── Row 4: Sentiment vs Helpfulness ───────────────────────────────────────
    if "sentiment_score" in df.columns:
        st.markdown("---")
        st.write("#### Sentiment Score vs Helpfulness")
        fig_violin = px.violin(
            df, y="sentiment_score", x=df["predicted_helpful"].map({1: "Helpful", 0: "Non-Helpful"}),
            color=df["predicted_helpful"].map({1: "Helpful", 0: "Non-Helpful"}),
            box=True, points="outliers",
            color_discrete_map={"Helpful": "#2ecc71", "Non-Helpful": "#e74c3c"},
            labels={"x": "Predicted Label", "y": "VADER Sentiment Score"},
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    # ── Row 5: Live Prediction + AI Insights ──────────────────────────────────
    st.markdown("---")
    st.subheader("Live Review Helpfulness Predictor")
    with st.expander("Try it yourself!", expanded=True):
        user_text = st.text_area(
            "Enter a product review text:",
            "This product is amazing! It really helped my skin and smells great.",
        )
        user_rating = st.slider("Select Rating", 1, 5, 5)

        if st.button("Predict Helpfulness"):
            model_file = model_choice.lower().replace(" ", "_") + ".pkl"
            model_path = MODEL_DIR / model_file
            vec_path = MODEL_DIR / "tfidf_vectorizer.pkl"

            if not model_path.exists() or not vec_path.exists():
                st.error("Model files not found. Run the pipeline first.")
            else:
                from scipy.sparse import hstack

                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                with open(vec_path, "rb") as f:
                    vectorizer = pickle.load(f)

                cleaned = clean_text(user_text)
                tfidf_feat = vectorizer.transform([cleaned])

                sent_score = analyzer.polarity_scores(user_text)["compound"]
                word_count = len(user_text.split())
                sent_count = len(re.split(r"[.!?]+", user_text))
                avg_word_len = sum(len(w) for w in user_text.split()) / max(word_count, 1)

                try:
                    import spacy
                    from textstat import flesch_reading_ease
                    nlp_mdl = spacy.load("en_core_web_sm")
                    doc = nlp_mdl(user_text)
                    num_tokens = max(len(doc), 1)
                    noun_ratio = sum(1 for t in doc if t.pos_ == "NOUN") / num_tokens
                    adj_ratio = sum(1 for t in doc if t.pos_ == "ADJ") / num_tokens
                    ner_count = len(doc.ents)
                    readability = flesch_reading_ease(user_text)
                except Exception:
                    noun_ratio = adj_ratio = ner_count = readability = 0.0

                comparative_keywords = {
                    "better", "worse", "more", "less", "than", "compared",
                    "superior", "inferior", "greatest", "best",
                }
                comparative_count = len(
                    set(user_text.lower().split()).intersection(comparative_keywords)
                )

                meta_feat = np.array([[
                    user_rating, len(user_text), word_count, sent_count,
                    avg_word_len, 1, sent_score, noun_ratio, adj_ratio,
                    ner_count, readability, comparative_count,
                ]])

                X_input = hstack([tfidf_feat, meta_feat])
                prob = model.predict_proba(X_input)[0][1]
                pred = model.predict(X_input)[0]

                # ── Show Prediction Result ─────────────────────────────────
                st.markdown("##### Prediction Result")
                if pred == 1:
                    st.success(f"HIGH probability of being HELPFUL ({prob:.2%})")
                else:
                    st.warning(f"LOW probability of being helpful ({prob:.2%})")

                # ── Gemini AI Insights ─────────────────────────────────────
                st.markdown("##### AI Insights (Google Gemini)")
                if not gemini_api_key:
                    st.info(
                        "Enter your **Gemini API Key** in the sidebar to enable AI-powered explanations."
                    )
                elif not GEMINI_AVAILABLE:
                    st.error("Install `google-genai`: `pip install google-genai`")
                else:
                    with st.spinner("Generating AI insights..."):
                        try:
                            insights = get_gemini_insights(gemini_api_key, user_text, int(pred), float(prob))
                            st.markdown(insights)
                        except Exception as e:
                            st.error(f"Gemini API error: {e}")

    # ── Row 6: Sample Predictions ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Sample Dataset Predictions")
    num_samples = st.slider("Number of samples to show", 5, 50, 10)
    display_cols = [c for c in ["text", "rating", "sentiment_score", "helpfulness_score", "predicted_helpful"] if c in df.columns]
    st.table(df.sample(num_samples)[display_cols].head(num_samples))


if __name__ == "__main__":
    main()
