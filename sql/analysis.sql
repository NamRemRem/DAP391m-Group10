-- analysis.sql
-- 1. Average helpfulness score by rating
SELECT 
    rating, 
    COUNT(*) as total_reviews,
    ROUND(AVG(helpful_vote), 2) as avg_helpful_votes
FROM reviews
GROUP BY rating
ORDER BY rating DESC;

-- 2. Sentiment-Helpfulness correlation analysis
SELECT 
    CASE 
        WHEN sentiment_score > 0.5 THEN 'Highly Positive'
        WHEN sentiment_score > 0 THEN 'Positive'
        WHEN sentiment_score = 0 THEN 'Neutral'
        WHEN sentiment_score > -0.5 THEN 'Negative'
        ELSE 'Highly Negative'
    END as sentiment_category,
    COUNT(*) as review_count,
    ROUND(AVG(helpful_vote), 2) as avg_helpful_votes
FROM reviews
GROUP BY sentiment_category
ORDER BY avg_helpful_votes DESC;

-- 3. Monthly review volume trends
SELECT 
    strftime('%Y-%m', datetime(timestamp/1000, 'unixepoch')) as month,
    COUNT(*) as review_count
FROM reviews
GROUP BY month
ORDER BY month;

-- 4. Top 10 helpful reviews identified by vote count
SELECT 
    rating,
    helpful_vote,
    SUBSTR(text, 1, 100) || '...' as text_snippet,
    verified_purchase
FROM reviews
ORDER BY helpful_vote DESC
LIMIT 10;

-- 5. Impact of Verified Purchase on helpfulness
SELECT 
    verified_purchase,
    COUNT(*) as review_count,
    ROUND(AVG(helpful_vote), 2) as avg_helpful_votes
FROM reviews
GROUP BY verified_purchase;

-- 6. Review length vs. helpfulness distribution
SELECT 
    CASE 
        WHEN review_length < 100 THEN 'Short (<100)'
        WHEN review_length < 500 THEN 'Medium (100-500)'
        ELSE 'Long (>500)'
    END as length_category,
    COUNT(*) as review_count,
    ROUND(AVG(helpful_vote), 2) as avg_helpful_votes
FROM reviews
GROUP BY length_category
ORDER BY length_category;
