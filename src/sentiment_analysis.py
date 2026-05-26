# sentiment_analysis.py — compatibility shim. Import from data.sentiment instead.
from data.sentiment import (  # noqa: F401
    get_batch_sentiment_recommendations,
    get_sentiment_recommendation,
    SentimentAnalysisState,
    gather_sentiments,
    analyze_sentiment,
    BATCH_SIZE,
    MAX_CONCURRENT,
    MAX_RETRIES,
)
