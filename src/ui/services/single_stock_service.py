"""
ui/services/single_stock_service.py — Thin wrapper for the single-stock analyzer.

UI components (and any CLI) call these instead of importing research.single_stock_analyzer
directly, per the service-layer convention. No business logic lives here — it orchestrates
load → analyze → return. Places NO orders and imports no broker/execution code.
"""
from __future__ import annotations


def analyze_single_stock(symbol: str, leverage_symbol: str | None = None, *,
                         allow_fetch: bool = True, include_social: bool = True,
                         include_news: bool = True, include_options: bool = True):
    """Run the read-only single-stock analysis. Returns a SingleStockAnalysis dataclass."""
    from research.single_stock_analyzer import analyze
    return analyze(symbol, leverage_symbol, allow_fetch=allow_fetch,
                   include_social=include_social, include_news=include_news,
                   include_options=include_options)


def position_targets(total_equity: float, common_pct: float, levered_pct: float,
                     cash_pct: float) -> dict:
    """Translate target sleeve percentages into dollar targets (hypothetical sizing only)."""
    from research.single_stock_analyzer import position_structure
    return position_structure(total_equity, common_pct, levered_pct, cash_pct)
