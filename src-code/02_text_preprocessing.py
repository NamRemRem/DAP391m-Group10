"""
DAP391m Project - Text Preprocessing (Sentiment Added)
======================================================
Cleans review text and calculates sentiment scores using VADER.
"""

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pathlib import Path

# Ensure NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "clean_reviews.csv"
OUTPUT_CSV = PROJECT_ROOT / "Data" / "filtered" / "processed_reviews.csv"

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def main():
    if not INPUT_CSV.exists(): return
    print("Preprocessing text and calculating sentiment...")
    df = pd.read_csv(INPUT_CSV)
    
    # Review length
    df['review_length'] = df['text'].astype(str).apply(len)
    
    # Sentiment (Using raw text for VADER often works better)
    df['sentiment_score'] = df['text'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    # Clean text for TF-IDF
    df['processed_text'] = df['text'].apply(clean_text)
    
    # Filter empty
    df = df[df['processed_text'].str.strip().str.len() > 0].reset_index(drop=True)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Preprocessing complete. Saved {len(df)} records.")

if __name__ == "__main__":
    main()
