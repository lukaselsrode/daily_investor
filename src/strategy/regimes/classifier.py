"""strategy/regimes/classifier.py — the single source of truth for market-regime
classification, shared by the LIVE detector (strategy/regimes/detector.py) and the
BACKTEST (backtesting/simulator.py). Before this, live used a VIX-primary rule and the
backtest a SPY-vs-200DMA rule, so a regime meant different things when you tuned/validated
vs when you deployed. This function is VIX-primary (the live rule); the backtest now calls
it with historical ^VIX so its regime labels match live by construction.

Pure and stateless — confidence/notes/hysteresis live in the caller (detector.py)."""
from __future__ import annotations

from strategy.regimes.models import RegimeLabel


def classify_regime(
    spy_price: float | None,
    spy_ma200: float | None,
    vix: float | None = None,
    regime_config=None,
) -> RegimeLabel:
    """Classify the market regime from SPY, its 200DMA, and VIX (VIX-primary).

    Priority: VIX>=defensive -> defensive; VIX in [neutral, defensive) -> neutral;
    else SPY vs 200DMA decides bullish/neutral. When ``vix`` is None, falls back to the
    SPY-only branch (SPY>200DMA -> bullish else neutral) — matching the live detector's
    VIX-missing behavior. (The backtest's _detect_regime handles its OWN vix-None case with
    the legacy SPY-only-with-defensive rule, so VIX-less backtests stay byte-identical.)

    `regime_config` is a RegimeConfig (config/schema.py); defaults to the live singleton.
    """
    if regime_config is None:
        from config.manager import ConfigManager
        regime_config = ConfigManager.get().regime
    vix_def = regime_config.vix_defensive_threshold
    vix_neut = regime_config.vix_neutral_threshold

    spy_above_200 = (
        spy_price is not None and spy_ma200 is not None and spy_ma200 > 0
        and spy_price > spy_ma200
    )

    if vix is not None and vix >= vix_def:
        return "defensive"
    if vix is not None and vix >= vix_neut:
        return "neutral"
    if spy_above_200 and (vix is None or vix < vix_neut):
        return "bullish"
    # SPY at/below 200DMA with low/no VIX → neutral / corrective (live: never defensive
    # without a VIX spike).
    return "neutral"
