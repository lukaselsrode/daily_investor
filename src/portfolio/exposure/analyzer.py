"""
portfolio/exposure/analyzer.py — ExposureAnalyzer: factor + sector + concentration diagnostics.

Computes:
  - Factor tilts vs. universe median (value, quality, income, momentum, composite)
  - Sector concentration
  - HHI (Herfindahl-Hirschman Index) for position concentration
  - Rolling exposure drift using the snapshot store

Input:
  portfolio — {symbol: {equity, quantity, sector, is_etf, ...}}
  universe_df — scored universe DataFrame (latest agg_data or snapshot)
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Factor columns that must be present in both PositionExposure and universe_df
_FACTOR_COLS = [
    "value_score",
    "quality_score",
    "income_score",
    "momentum_score",
    "value_metric",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class PositionExposure:
    """Per-position factor exposure and weight metadata."""

    symbol: str
    sector: str
    equity: float
    weight: float          # fraction of total portfolio equity
    value_score: float
    quality_score: float
    income_score: float
    momentum_score: float
    value_metric: float
    is_etf: bool = False


@dataclass
class ExposureReport:
    """Full portfolio exposure report produced by ExposureAnalyzer.analyze()."""

    timestamp: str
    total_equity: float
    etf_pct: float
    stock_pct: float
    cash_pct: float

    # Factor tilts — (weighted portfolio avg − universe median) / universe std
    value_tilt:     float = 0.0
    quality_tilt:   float = 0.0
    income_tilt:    float = 0.0
    momentum_tilt:  float = 0.0
    composite_tilt: float = 0.0

    # Sector allocation {sector_name: weight}
    sector_weights: dict[str, float] = field(default_factory=dict)

    # Concentration
    hhi:      float = 0.0   # Herfindahl-Hirschman Index (0=diversified, 1=concentrated)
    top5_pct: float = 0.0   # fraction of equity in top-5 holdings

    # Beta vs. SPY (computed separately; None if data unavailable)
    beta_spy: Optional[float] = None

    positions: list[PositionExposure] = field(default_factory=list)

    @property
    def n_positions(self) -> int:
        return len(self.positions)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ExposureAnalyzer:
    """
    Compute portfolio factor exposure, sector weights, and concentration.

    Usage:
        analyzer = ExposureAnalyzer()
        report   = analyzer.analyze(portfolio, universe_df, total_equity, cash)
        drift_df = analyzer.compute_rolling_drift(portfolio, days=90)
    """

    # ── Main entry point ────────────────────────────────────────────────────

    def analyze(
        self,
        portfolio: dict,
        universe_df: pd.DataFrame,
        total_equity: float = 0.0,
        cash: float = 0.0,
    ) -> ExposureReport:
        """
        Compute full exposure report for a live portfolio.

        portfolio keys: symbol
        portfolio values: dict with equity, sector, is_etf (minimum fields)
        """
        ts = datetime.datetime.utcnow().isoformat()

        if not portfolio:
            return ExposureReport(
                timestamp=ts,
                total_equity=total_equity,
                etf_pct=0.0,
                stock_pct=0.0,
                cash_pct=1.0 if total_equity > 0 else 0.0,
            )

        total_invested = sum(float(v.get("equity", 0)) for v in portfolio.values())
        etf_invested   = sum(
            float(v.get("equity", 0)) for v in portfolio.values() if v.get("is_etf")
        )
        stock_invested = total_invested - etf_invested

        if total_equity <= 0:
            total_equity = total_invested + cash

        etf_pct   = etf_invested   / total_equity if total_equity > 0 else 0.0
        stock_pct = stock_invested / total_equity if total_equity > 0 else 0.0
        cash_pct  = cash           / total_equity if total_equity > 0 else 0.0

        # --- Universe-level factor stats for normalization ---
        u_median: dict[str, float] = {}
        u_std:    dict[str, float] = {}
        universe_factor_lookup: dict[str, dict[str, float]] = {}

        if not universe_df.empty and "symbol" in universe_df.columns:
            u_idx = universe_df.set_index("symbol")
            for col in _FACTOR_COLS:
                if col not in u_idx.columns:
                    continue
                s = pd.to_numeric(u_idx[col], errors="coerce").dropna()
                if len(s) < 2:
                    continue
                u_median[col] = float(s.median())
                u_std[col]    = float(s.std()) if s.std() > 1e-9 else 1.0
                universe_factor_lookup[col] = pd.to_numeric(
                    u_idx[col], errors="coerce"
                ).to_dict()

        # --- Build per-position exposure list ---
        positions: list[PositionExposure] = []
        for sym, pos in portfolio.items():
            equity = float(pos.get("equity", 0))
            weight = equity / total_equity if total_equity > 0 else 0.0
            sector = str(pos.get("sector") or "Unknown")
            is_etf = bool(pos.get("is_etf", False))

            def _lookup(col: str) -> float:
                return float(universe_factor_lookup.get(col, {}).get(sym, 0.0))

            positions.append(
                PositionExposure(
                    symbol=sym,
                    sector=sector,
                    equity=equity,
                    weight=weight,
                    value_score=_lookup("value_score"),
                    quality_score=_lookup("quality_score"),
                    income_score=_lookup("income_score"),
                    momentum_score=_lookup("momentum_score"),
                    value_metric=_lookup("value_metric"),
                    is_etf=is_etf,
                )
            )

        # --- Factor tilts (weighted avg vs. universe median in std units) ---
        def _tilt(col: str, attr: str) -> float:
            if not positions or col not in u_median:
                return 0.0
            wtd = sum(p.weight * getattr(p, attr, 0.0) for p in positions)
            std = u_std.get(col, 1.0)
            return round((wtd - u_median[col]) / std, 3)

        # --- Sector weights ---
        sector_equity: dict[str, float] = {}
        for p in positions:
            sector_equity[p.sector] = sector_equity.get(p.sector, 0.0) + p.equity
        sector_weights = {
            s: round(e / total_equity, 4) if total_equity > 0 else 0.0
            for s, e in sorted(sector_equity.items(), key=lambda x: -x[1])
        }

        # --- Concentration metrics ---
        w_arr = np.array([p.weight for p in positions])
        hhi   = float(np.sum(w_arr ** 2))
        top5  = sorted(positions, key=lambda p: p.equity, reverse=True)[:5]
        top5_pct = sum(p.weight for p in top5)

        return ExposureReport(
            timestamp=ts,
            total_equity=round(total_equity, 2),
            etf_pct=round(etf_pct, 4),
            stock_pct=round(stock_pct, 4),
            cash_pct=round(cash_pct, 4),
            value_tilt=_tilt("value_score",    "value_score"),
            quality_tilt=_tilt("quality_score", "quality_score"),
            income_tilt=_tilt("income_score",   "income_score"),
            momentum_tilt=_tilt("momentum_score", "momentum_score"),
            composite_tilt=_tilt("value_metric", "value_metric"),
            sector_weights=sector_weights,
            hhi=round(hhi, 4),
            top5_pct=round(top5_pct, 4),
            positions=positions,
        )

    # ── Rolling drift ────────────────────────────────────────────────────────

    def compute_rolling_drift(
        self,
        portfolio: dict,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Track portfolio factor tilt drift over time using the snapshot store.

        Returns [date, value_tilt, quality_tilt, income_tilt, momentum_tilt].
        """
        try:
            from strategy.snapshots import load_snapshots
        except ImportError:
            logger.warning("strategy.snapshots not available — drift unavailable")
            return pd.DataFrame()

        end   = datetime.date.today()
        start = end - datetime.timedelta(days=days)

        try:
            snaps = load_snapshots(start=start, end=end)
        except Exception as exc:
            logger.warning("Could not load snapshots for drift: %s", exc)
            return pd.DataFrame()

        if snaps.empty or "snapshot_date" not in snaps.columns:
            return pd.DataFrame()

        rows: list[dict] = []
        for snap_date, grp in snaps.groupby("snapshot_date"):
            try:
                report = self.analyze(portfolio, grp.reset_index(drop=True), total_equity=1.0)
                rows.append({
                    "date":          snap_date,
                    "value_tilt":    report.value_tilt,
                    "quality_tilt":  report.quality_tilt,
                    "income_tilt":   report.income_tilt,
                    "momentum_tilt": report.momentum_tilt,
                })
            except Exception as exc:
                logger.debug("Drift computation failed for %s: %s", snap_date, exc)

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()
