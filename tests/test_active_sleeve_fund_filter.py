from __future__ import annotations

import pandas as pd

from backtesting.data_loader import select_backtest_universe
from portfolio.manager import _exclude_pooled_vehicles_from_active_candidates


def test_live_active_candidate_filter_excludes_pooled_vehicles_but_keeps_adr_reit() -> None:
    df = pd.DataFrame({
        "symbol": ["AAPL", "BABU", "DNP", "EPD", "ING", "INVH"],
        "instrument_type": ["stock", "etp", "cef", "mlp", "adr", "reit"],
        "value_metric": [0.9, 0.95, 0.94, 0.93, 0.8, 0.7],
    })

    filtered, excluded = _exclude_pooled_vehicles_from_active_candidates(df)

    assert excluded == ["BABU", "DNP", "EPD"]
    assert filtered["symbol"].tolist() == ["AAPL", "ING", "INVH"]


def test_backtest_universe_excludes_pooled_vehicles_from_active_stock_pool() -> None:
    df = pd.DataFrame({
        "symbol": ["AAPL", "BABU", "DNP", "EPD", "ING", "INVH", "QQQ"],
        "instrument_type": ["stock", "etp", "cef", "mlp", "adr", "reit", "etp"],
        "volume": [1_000_000] * 7,
        "value_metric": [0.1, 0.99, 0.98, 0.97, 0.2, 0.3, 0.96],
        "sector": ["Technology", "Misc", "Misc", "Energy", "Finance", "Real Estate", "Misc"],
    })

    selected, bias = select_backtest_universe(
        df,
        mode="liquid_universe_full",
        universe_selection="liquid_all",
        max_symbols=10,
        min_volume=500_000,
        random_seed=42,
    )

    assert bias == "MEDIUM"
    assert selected["symbol"].tolist() == ["AAPL", "ING", "INVH"]
    assert not selected["instrument_type"].isin(["etp", "cef", "mlp", "etn"]).any()
