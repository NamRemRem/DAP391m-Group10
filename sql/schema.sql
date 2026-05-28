-- schema.sql
-- Defines the structure for the product reviews database

DROP TABLE IF EXISTS reviews;

CREATE TABLE reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rating FLOAT,
    title TEXT,
    text TEXT,
    helpful_vote INTEGER DEFAULT 0,
    verified_purchase BOOLEAN,
    timestamp BIGINT,
    asin TEXT,
    review_length INTEGER,
    sentiment_score FLOAT,
    processed_text TEXT
);
