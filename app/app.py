import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(page_title="Review Helpfulness Project", layout="wide")

# Paths
DATA_DIR = Path("Data/filtered")
MODEL_DIR = Path("Data/filtered/model_outputs")


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
        "A simple student project to analyze and predict the quality of marketplace reviews."
    )

    df = load_data()
    stats = load_stats()
    if df is None:
        st.error("Please run the pipeline to generate data.")
        return

    # Sidebar
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.selectbox(
        "Choose Model", options=["Logistic Regression", "Random Forest"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Info")
    st.sidebar.write("**Dataset**: Amazon All Beauty")
    st.sidebar.write("**Features**: Rating, Length, Sentiment, TF-IDF")

    # Row 1: High Level Stats
    st.subheader("📊 Review Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", len(df))
    c2.metric("Helpful Reviews", len(df[df["predicted_helpful"] == 1]))
    c3.metric("Avg Sentiment", f"{df['sentiment_score'].mean():.2f}")

    # Model Accuracy
    acc = stats.get(model_choice, 0)
    c4.metric(f"{model_choice} Accuracy", f"{acc:.2%}")

    # Row 2: Charts (Helpfulness & Ratings)
    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Helpful vs. Non-Helpful")
        pie_data = df["predicted_helpful"].value_counts().reset_index()
        pie_data.columns = ["Status", "Count"]
        pie_data["Status"] = pie_data["Status"].map({1: "Helpful", 0: "Non-Helpful"})
        fig_pie = px.pie(
            pie_data,
            values="Count",
            names="Status",
            color_discrete_sequence=["#2ecc71", "#e74c3c"],
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

    # Row 3: Trends Over Time
    st.write("#### 📈 Review Trends Over Time")
    trend_data = df.groupby("date").size().reset_index(name="Daily Reviews")
    fig_line = px.line(
        trend_data, x="date", y="Daily Reviews", title="New Reviews Submitted Per Day"
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Row 4: Example Predictions
    st.write("---")
    st.subheader("🔍 Sample Predictions")
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
