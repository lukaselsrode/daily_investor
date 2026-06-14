"""
research/pit_fundamentals.py — moved to data/pit_fundamentals.py (production home).

The implementation now lives in the `data/` layer (which owns FMP access) so production
and backtesting can use point-in-time fundamentals without depending on `research/`
(research stays read-only / offline-only). This thin re-export keeps existing research
and test imports working. Import from `data.pit_fundamentals` in new code.
"""
from __future__ import annotations

from data.pit_fundamentals import (  # noqa: F401
    dividend_yield_asof,
    fundamentals_asof,
)
