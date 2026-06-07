"""
strategy/regimes/detector.py — RegimeDetector: live and historical classification.

Classifies the market into three regimes:
  bullish   — SPY above 200DMA, VIX below neutral threshold
  neutral   — moderate volatility or SPY near/below 200DMA
  defensive — high volatility (VIX ≥ defensive threshold)

Design:
  - detect()              fetches live SPY + VIX and returns current RegimeState
  - detect_from_data()    pure computation from pre-fetched signals (testable)
  - classify_history()    downloads N days of history and replays classification
  - Confidence is calibrated by distance from thresholds, not a binary
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from .classifier import classify_regime
from .models import RegimeHistoryEntry, RegimeLabel, RegimeState

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Classify market regime from SPY price vs. 200DMA and VIX level.

    Usage:
        detector = RegimeDetector()
        state = detector.detect()
        print(state.regime, state.confidence, state.notes)

        history = detector.classify_history(days=365)
    """

    def __init__(self, config=None) -> None:
        if config is None:
            from config.manager import ConfigManager
            config = ConfigManager.get()
        self._cfg = config
        self._rc  = config.regime
        self._history: list[RegimeState] = []

    # ── Live detection ──────────────────────────────────────────────────────

    def detect(self) -> RegimeState:
        """Fetch live SPY + VIX data and classify the current regime."""
        spy_price: float | None = None
        spy_ma200: float | None = None
        vix_val:   float | None = None

        try:
            spy_df = yf.download(
                "SPY",
                period="210d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if not spy_df.empty:
                closes = spy_df["Close"].squeeze().dropna()
                spy_price = float(closes.iloc[-1])
                if len(closes) >= 200:
                    spy_ma200 = float(closes.rolling(200).mean().iloc[-1])
        except Exception as exc:
            logger.warning("SPY download failed for regime detection: %s", exc)

        try:
            vix_df = yf.download(
                "^VIX",
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if not vix_df.empty:
                vix_val = float(vix_df["Close"].squeeze().dropna().iloc[-1])
        except Exception as exc:
            logger.warning("VIX download failed for regime detection: %s", exc)

        return self.detect_from_data(
            spy_price=spy_price,
            spy_ma200=spy_ma200,
            vix=vix_val,
        )

    # ── Pure signal → regime mapping ────────────────────────────────────────

    def detect_from_data(
        self,
        spy_price: float | None,
        spy_ma200: float | None,
        vix: float | None,
    ) -> RegimeState:
        """
        Classify regime from pre-fetched signals.

        Confidence is calibrated by how far VIX is from the thresholds.
        """
        rc = self._rc
        vix_def  = rc.vix_defensive_threshold   # e.g. 30.0
        vix_neut = rc.vix_neutral_threshold      # e.g. 20.0
        notes: list[str] = []

        spy_vs_200dma: float | None = None
        spy_above_200 = False
        if spy_price is not None and spy_ma200 is not None and spy_ma200 > 0:
            spy_vs_200dma = round((spy_price / spy_ma200) - 1.0, 4)
            spy_above_200 = spy_vs_200dma > 0

        # --- Regime label: the SHARED classifier (same logic the backtest now uses) ---
        raw_regime: RegimeLabel = classify_regime(spy_price, spy_ma200, vix, rc)

        # --- Confidence + notes (live-only enrichment; conditions mirror classify_regime) ---
        if vix is not None and vix >= vix_def:
            excess = (vix - vix_def) / max(vix_def, 1.0)
            confidence = min(1.0, 0.65 + excess * 0.35)
            notes.append(f"VIX={vix:.1f} ≥ defensive threshold {vix_def}")
            if not spy_above_200:
                confidence = min(1.0, confidence + 0.05)
                notes.append("SPY below 200DMA reinforces defensive")

        elif vix is not None and vix >= vix_neut:
            # Confidence scales with VIX proximity to each boundary
            t = (vix - vix_neut) / max(vix_def - vix_neut, 1.0)
            confidence = 0.55 + t * 0.15
            notes.append(f"VIX={vix:.1f} in neutral band [{vix_neut:.0f}, {vix_def:.0f})")
            if not spy_above_200:
                confidence = min(1.0, confidence + 0.10)
                notes.append("SPY below 200DMA pushes toward defensive end of neutral")

        elif spy_above_200 and (vix is None or vix < vix_neut):
            confidence = 0.70
            if vix is not None and vix < 15.0:
                confidence = 0.90
                notes.append(f"Low VIX={vix:.1f} reinforces bullish")
            elif vix is not None:
                notes.append(f"VIX={vix:.1f} below neutral threshold")
            pct_str = f"{spy_vs_200dma:+.1%}" if spy_vs_200dma is not None else "N/A"
            notes.append(f"SPY {pct_str} above 200DMA")

        elif not spy_above_200 and (vix is None or vix < vix_neut):
            confidence = 0.55
            notes.append("SPY below 200DMA with low VIX → neutral / corrective")

        else:
            confidence = 0.50
            notes.append("Insufficient data — defaulting to neutral")

        prev: RegimeLabel | None = (
            self._history[-1].regime if self._history else None
        )

        state = RegimeState(
            regime=raw_regime,
            confidence=round(confidence, 3),
            vix=vix,
            spy_price=spy_price,
            spy_ma200=spy_ma200,
            spy_vs_200dma_pct=spy_vs_200dma,
            previous_regime=prev,
            transition_count=len(self._history),
            notes=notes,
        )
        self._history.append(state)
        return state

    # ── Historical replay ───────────────────────────────────────────────────

    def classify_history(self, days: int = 365) -> list[RegimeHistoryEntry]:
        """
        Download SPY + VIX history and classify each trading day.

        Returns a list of RegimeHistoryEntry sorted by date ascending.
        """
        lookback = days + 210  # extra buffer for 200DMA warm-up
        try:
            spy_df = yf.download(
                "SPY",
                period=f"{lookback}d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            vix_df = yf.download(
                "^VIX",
                period=f"{lookback}d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception as exc:
            logger.error("Failed to download history for regime replay: %s", exc)
            return []

        if spy_df.empty or vix_df.empty:
            return []

        spy_closes = spy_df["Close"].squeeze().rename("spy")
        vix_closes = vix_df["Close"].squeeze().rename("vix")
        spy_ma200  = spy_closes.rolling(200).mean().rename("ma200")

        combined = pd.DataFrame(
            {"spy": spy_closes, "vix": vix_closes, "ma200": spy_ma200}
        ).dropna().tail(days)

        result: list[RegimeHistoryEntry] = []
        # Use a fresh detector with no history to avoid contaminating self
        sub = RegimeDetector(config=self._cfg)
        for dt, row in combined.iterrows():
            state = sub.detect_from_data(
                spy_price=float(row["spy"]),
                spy_ma200=float(row["ma200"]),
                vix=float(row["vix"]),
            )
            date = dt.date() if hasattr(dt, "date") else dt
            result.append(
                RegimeHistoryEntry(
                    date=date,
                    regime=state.regime,
                    vix=float(row["vix"]),
                    spy_vs_200dma_pct=state.spy_vs_200dma_pct or 0.0,
                    confidence=state.confidence,
                )
            )

        return result


def get_current_regime() -> str:
    """Return the current market regime string: 'bullish', 'neutral', or 'defensive'."""
    return RegimeDetector().detect().regime
