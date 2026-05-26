# sentiments.py — compatibility shim. Import from data.news instead.
from data.news import (  # noqa: F401
    get_news_for_tickers_by_symbol,
    reddit_sentiments_for_tickers,
    _fetch_news_with_retry,
    _robinhood_news,
    _parse_yfinance_item,
    _parse_robinhood_item,
    MAX_NEWS_CONCURRENT,
)
