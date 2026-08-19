from __future__ import annotations

import datetime
from unittest.mock import patch

import pandas as pd


def test_symbols_due_for_outcome_backfill_skips_fully_resolved_rows() -> None:
    from portfolio.outcome_tracker import symbols_due_for_backfill

    today = datetime.date(2026, 8, 13)
    df = pd.DataFrame(
        [
            {
                "decision_date": "2026-05-01",
                "symbol": "DONE",
                "price": 10.0,
                "future_7d_return": 0.1,
                "future_30d_return": 0.2,
                "future_90d_return": 0.3,
            },
            {
                "decision_date": "2026-08-01",
                "symbol": "DUE7",
                "price": 20.0,
                "future_7d_return": None,
                "future_30d_return": None,
                "future_90d_return": None,
            },
            {
                "decision_date": "2026-08-10",
                "symbol": "TOO_NEW",
                "price": 30.0,
                "future_7d_return": None,
                "future_30d_return": None,
                "future_90d_return": None,
            },
            {
                "decision_date": "2026-07-01",
                "symbol": "NO_PRICE",
                "price": None,
                "future_7d_return": None,
                "future_30d_return": None,
                "future_90d_return": None,
            },
        ]
    )

    assert symbols_due_for_backfill(df, today=today) == ["DUE7"]


def test_news_concurrency_honors_bounded_environment_override(monkeypatch) -> None:
    from data.news import _news_concurrency

    monkeypatch.setenv("NEWS_CONCURRENCY", "8")
    assert _news_concurrency() == 8

    monkeypatch.setenv("NEWS_CONCURRENCY", "999")
    assert _news_concurrency() == 16

    monkeypatch.setenv("NEWS_CONCURRENCY", "not-a-number")
    assert _news_concurrency() == 3


def test_skip_account_data_avoids_login_and_preserves_account_snapshots() -> None:
    from cli.commands import cmd_fetch_data

    result = pd.DataFrame({"symbol": ["AAPL"]})
    with (
        patch("main.login") as login,
        patch("main._fetch_and_save_dividends") as dividends,
        patch("main._broker.get_holdings") as holdings,
        patch("main._maybe_fill_outcomes") as outcomes,
        patch("data.valuation.update_industry_valuations") as valuations,
        patch("data.market.get_data", return_value=result) as market_data,
    ):
        cmd_fetch_data(skip_account_data=True)

    login.assert_not_called()
    dividends.assert_not_called()
    holdings.assert_not_called()
    valuations.assert_called_once_with(verbose=True)
    market_data.assert_called_once_with(refresh=True)
    outcomes.assert_called_once_with()
