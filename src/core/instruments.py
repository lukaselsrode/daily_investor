"""
core/instruments.py — Instrument-type classification primitives.

Single source of truth for "is this a pooled fund (ETF/CEF/MLP/ETN) rather than
an individual operating company?" Lives in `core` so both the data layer
(market_structure / fundamentals) and domain layers (portfolio archetypes,
visualization factor map) can share ONE predicate without an import-boundary
violation or duplicated literals.

Robinhood ``instrument_type`` values treated as pooled funds. ADR / REIT remain
individual equities (they trade and behave like single-company stocks).
"""
from __future__ import annotations

# Robinhood instrument ``type`` values that denote a pooled fund / non-stock.
FUND_INSTRUMENT_TYPES: frozenset[str] = frozenset({"etp", "cef", "mlp", "etn"})

# Free-form ``asset_type`` / ``security_type`` values that denote a fund.
FUND_ASSET_VALUES: frozenset[str] = frozenset(
    {"etf", "fund", "etn", "index", "index_fund"}
)


def is_fund_instrument_type(instrument_type: object) -> bool:
    """True when a Robinhood ``instrument_type`` denotes a pooled fund.

    Case-insensitive; tolerant of None / non-string input (returns False).
    """
    if instrument_type is None:
        return False
    return str(instrument_type).strip().lower() in FUND_INSTRUMENT_TYPES


def is_fund_asset_value(asset_value: object) -> bool:
    """True when an ``asset_type`` / ``security_type`` value denotes a fund."""
    if asset_value is None:
        return False
    return str(asset_value).strip().lower() in FUND_ASSET_VALUES
