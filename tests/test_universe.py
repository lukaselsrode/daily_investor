from __future__ import annotations

from data.universe import (
    _fetch_robinhood_instrument_symbols,
    _is_robinhood_active_stock,
    _symbols_from_sources,
)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, pages: dict[str, dict]) -> None:
        self.pages = pages
        self.requested: list[str] = []

    def get(self, url: str, timeout: int = 30) -> _FakeResponse:
        assert timeout == 30
        self.requested.append(url)
        return _FakeResponse(self.pages[url])


def test_robinhood_active_stock_filter_excludes_funds_and_inactive_rows() -> None:
    base = {
        "symbol": "AAPL",
        "state": "active",
        "tradeable": True,
        "rhs_tradability": "tradable",
        "type": "stock",
    }

    assert _is_robinhood_active_stock(base)
    assert _is_robinhood_active_stock({**base, "symbol": "SONY", "type": "adr"})
    assert not _is_robinhood_active_stock({**base, "symbol": "SPY", "type": "etp"})
    assert not _is_robinhood_active_stock({**base, "state": "inactive"})
    assert not _is_robinhood_active_stock({**base, "tradeable": False})
    assert not _is_robinhood_active_stock({**base, "rhs_tradability": "untradable"})
    assert not _is_robinhood_active_stock({**base, "symbol": "TOO-LONG"})


def test_symbols_from_sources_survives_none_items() -> None:
    """robin_stocks returns [None] for retired market tags (e.g.
    'upcoming-earnings') — a non-dict item must count as invalid, not raise
    AttributeError and kill the whole universe refresh."""
    sources = [
        [{"symbol": "AAPL"}, None, {"symbol": "msft"}],  # lowercase → invalid
        None,                                            # whole source missing
        [None],                                          # the retired-tag shape
        [{"no_symbol_key": 1}, {"symbol": "TSLA"}],
    ]
    symbols, invalid = _symbols_from_sources(sources)
    assert symbols == {"AAPL", "TSLA"}
    assert invalid == 4


def test_fetch_robinhood_instrument_symbols_paginates_and_filters(monkeypatch) -> None:
    first_url = "https://api.robinhood.com/instruments/"
    second_url = "https://api.robinhood.com/instruments/?cursor=next"
    fake_session = _FakeSession(
        {
            first_url: {
                "next": second_url,
                "results": [
                    {
                        "symbol": "AAPL",
                        "state": "active",
                        "tradeable": True,
                        "rhs_tradability": "tradable",
                        "type": "stock",
                    },
                    {
                        "symbol": "SPY",
                        "state": "active",
                        "tradeable": True,
                        "rhs_tradability": "tradable",
                        "type": "etp",
                    },
                ],
            },
            second_url: {
                "next": None,
                "results": [
                    {
                        "symbol": "SONY",
                        "state": "active",
                        "tradeable": True,
                        "rhs_tradability": "tradable",
                        "type": "adr",
                    },
                    {
                        "symbol": "DEAD",
                        "state": "inactive",
                        "tradeable": True,
                        "rhs_tradability": "tradable",
                        "type": "stock",
                    },
                ],
            },
        }
    )

    monkeypatch.setattr("data.universe.requests.Session", lambda: fake_session)

    assert _fetch_robinhood_instrument_symbols(retries=1) == {"AAPL", "SONY"}
    assert fake_session.requested == [first_url, second_url]
