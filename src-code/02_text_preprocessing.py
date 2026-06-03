"""
DAP391m Project - Text Preprocessing (Sentiment Added)
======================================================
Cleans review text and calculates sentiment scores using VADER.
"""

import pandas as pd
import re
import string
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

# Ensure NLTK resources
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))
negative_words = {
    "no",
    "not",
    "nor",
    "never",
    "none",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "can't",
    "couldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
}
filtered_stopwords = stop_words - negative_words

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_reviews.csv"
OUTPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"

analyzer = SentimentIntensityAnalyzer()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r"[^a-z0-9\s!?]", "", text)
    words = text.split()
    processed_words = []

    for word in words:
        if word not in filtered_stopwords:
            lemma = lemmatizer.lemmatize(word, pos="v")
            processed_words.append(lemma)

    return " ".join(processed_words)


def count_words(text):
    if not isinstance(text, str):
        return 0
    return len(text.split())


def avg_word_length(text):
    words = text.split()
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)


def count_sentences(text):
    if not isinstance(text, str):
        return 0
    return len(re.split(r"[.!?]+", text))


def main():
    if not INPUT_CSV.exists():
        return
    print("Preprocessing text and calculating advanced features...")
    df = pd.read_csv(INPUT_CSV)

    # Basic text features
    df["review_length"] = df["text"].astype(str).apply(len)
    df["word_count"] = df["text"].astype(str).apply(count_words)
    df["sentence_count"] = df["text"].astype(str).apply(count_sentences)
    df["avg_word_length"] = df["text"].astype(str).apply(avg_word_length)

    # Sentiment (Using raw text for VADER often works better)
    df["sentiment_score"] = (
        df["text"].astype(str).apply(lambda x: analyzer.polarity_scores(x)["compound"])
    )

    # Clean text for TF-IDF
    df["processed_text"] = df["text"].apply(clean_text)

    # Filter empty
    df = df[df["processed_text"].str.strip().str.len() > 0].reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Preprocessing complete. Saved {len(df)} records with linguistic features.")


if __name__ == "__main__":
    main()
