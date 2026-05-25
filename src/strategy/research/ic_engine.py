"""
strategy/research/ic_engine.py — FactorResearchEngine: multi-horizon IC analytics.

Extends the single-horizon compute_forward_ic() in strategy/snapshots.py with:
  - Multi-horizon IC (5, 20, 60, 120, 252 days)
  - Pearson IC alongside Spearman
  - Hit rate, t-stat, cumulative IC
  - Monotonicity / decile spread
  - Factor decay curves (IC by horizon per factor)
  - Rolling IC and rolling ICIR
  - Regime-conditioned IC (bull / bear / high_vol / sideways)
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_FACTORS  = ["value_score", "quality_score", "income_score", "momentum_score", "value_metric"]
DEFAULT_HORIZONS = [5, 20, 60, 120, 252]


class FactorResearchEngine:
    """
    Multi-horizon IC computation and factor decay analytics.

    Reads from the parquet snapshot store in data/snapshots/.
    All methods return DataFrames ready for Streamlit display or Plotly charting.

    Usage:
        engine = FactorResearchEngine()
        ic_df  = engine.compute_multi_horizon_ic()
        summ   = engine.compute_ic_summary(ic_df)
        decay  = engine.compute_factor_decay()
    """

    def __init__(
        self,
        factors: Optional[list[str]] = None,
        horizons: Optional[list[int]] = None,
        min_overlap: int = 15,
        max_horizon_slop_pct: float = 0.5,
    ) -> None:
        self.factors               = factors  or DEFAULT_FACTORS
        self.horizons              = sorted(horizons or DEFAULT_HORIZONS)
        self.min_overlap           = min_overlap
        self.max_horizon_slop_pct  = max_horizon_slop_pct
        self._snap_cache: Optional[list[tuple[datetime.date, Path]]] = None

    # ── Snapshot access ──────────────────────────────────────────────────────

    def _get_snapshots(self) -> list[tuple[datetime.date, Path]]:
        if self._snap_cache is None:
            from strategy.snapshots import list_snapshots
            self._snap_cache = list_snapshots()
        return self._snap_cache

    def _load_dates_map(self) -> dict[datetime.date, pd.DataFrame]:
        """Load all available snapshots into a date → DataFrame map."""
        result: dict[datetime.date, pd.DataFrame] = {}
        for date, path in self._get_snapshots():
            try:
                result[date] = pd.read_parquet(path)
            except Exception as exc:
                logger.warning("Could not load snapshot %s: %s", path, exc)
        return result

    # ── Forward return computation ───────────────────────────────────────────

    def _forward_returns(
        self,
        df_t: pd.DataFrame,
        df_fwd: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute per-symbol forward return between two snapshot dates.

        Primary:  current_price ratio  (fwd / t - 1)
        Fallback: return_1m from forward snapshot when prices are sparse.

        Returns a pd.Series indexed by symbol with name "forward_return".
        """
        empty = pd.Series(dtype=float, name="forward_return")

        if "symbol" not in df_t.columns or "symbol" not in df_fwd.columns:
            return empty

        t_idx   = df_t.set_index("symbol")
        fwd_idx = df_fwd.set_index("symbol")

        common = t_idx.index.intersection(fwd_idx.index)
        if len(common) == 0:
            return empty

        if "current_price" in t_idx.columns and "current_price" in fwd_idx.columns:
            t_px   = pd.to_numeric(t_idx.loc[common, "current_price"],   errors="coerce")
            fwd_px = pd.to_numeric(fwd_idx.loc[common, "current_price"], errors="coerce")
            valid  = t_px.notna() & fwd_px.notna() & (t_px > 0)
            if valid.sum() >= self.min_overlap:
                return ((fwd_px / t_px) - 1.0)[valid].rename("forward_return")

        # Fallback: return_1m from the forward snapshot
        if "return_1m" in fwd_idx.columns:
            r1m = pd.to_numeric(fwd_idx.loc[common, "return_1m"], errors="coerce").dropna()
            if len(r1m) >= self.min_overlap:
                return r1m.rename("forward_return")

        return empty

    # ── Multi-horizon IC ─────────────────────────────────────────────────────

    def compute_multi_horizon_ic(
        self,
        factors: Optional[list[str]] = None,
        horizons: Optional[list[int]] = None,
        ic_type: str = "spearman",
    ) -> pd.DataFrame:
        """
        Compute IC for every (factor, horizon) pair across all snapshot dates.

        Columns: [date, factor, horizon_days, ic, n_stocks, p_value, forward_return_mean]
        """
        use_factors  = factors  or self.factors
        use_horizons = sorted(horizons or self.horizons)

        dates_map = self._load_dates_map()
        if len(dates_map) < 2:
            return pd.DataFrame(columns=[
                "date", "factor", "horizon_days", "ic",
                "n_stocks", "p_value", "forward_return_mean",
            ])

        sorted_dates = sorted(dates_map.keys())
        rows: list[dict] = []

        for h in use_horizons:
            min_days = max(1, int(h * (1.0 - self.max_horizon_slop_pct)))
            max_days = int(h * (1.0 + self.max_horizon_slop_pct))

            for i, t_date in enumerate(sorted_dates):
                # Find the nearest forward snapshot within the slop window
                fwd_date: Optional[datetime.date] = None
                best_diff = 9999
                for fwd in sorted_dates[i + 1:]:
                    diff = (fwd - t_date).days
                    if diff < min_days:
                        continue
                    if diff > max_days:
                        break
                    if abs(diff - h) < best_diff:
                        fwd_date = fwd
                        best_diff = abs(diff - h)

                if fwd_date is None:
                    continue

                df_t   = dates_map[t_date]
                df_fwd = dates_map[fwd_date]
                fr = self._forward_returns(df_t, df_fwd)

                if fr.empty or len(fr) < self.min_overlap:
                    continue

                for factor in use_factors:
                    if factor not in df_t.columns or "symbol" not in df_t.columns:
                        continue

                    fv = pd.to_numeric(
                        df_t.set_index("symbol")[factor], errors="coerce"
                    ).rename("factor_val")

                    merged = pd.DataFrame({"factor_val": fv, "forward_return": fr}).dropna()
                    if len(merged) < self.min_overlap:
                        continue

                    fv_arr = merged["factor_val"].values
                    rv_arr = merged["forward_return"].values

                    try:
                        if ic_type == "spearman":
                            ic_val, pv = stats.spearmanr(fv_arr, rv_arr)
                        else:
                            ic_val, pv = stats.pearsonr(fv_arr, rv_arr)
                    except Exception:
                        continue

                    if np.isnan(ic_val):
                        continue

                    rows.append({
                        "date":                t_date,
                        "factor":              factor,
                        "horizon_days":        h,
                        "ic":                  round(float(ic_val), 4),
                        "n_stocks":            len(merged),
                        "p_value":             round(float(pv), 4),
                        "forward_return_mean": round(float(rv_arr.mean()), 4),
                    })

        return pd.DataFrame(rows)

    # ── IC summary statistics ────────────────────────────────────────────────

    def compute_ic_summary(self, ic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate IC DataFrame by [factor, horizon_days] into summary statistics.

        Columns: [factor, horizon_days, mean_ic, icir, hit_rate, t_stat, n_periods, cumulative_ic]
        """
        if ic_df.empty:
            return pd.DataFrame()

        rows: list[dict] = []
        for (factor, horizon), grp in ic_df.groupby(["factor", "horizon_days"]):
            ic_vals = grp["ic"].dropna()
            if len(ic_vals) < 2:
                continue

            mean_ic  = float(ic_vals.mean())
            std_ic   = float(ic_vals.std())
            icir     = mean_ic / std_ic if std_ic > 1e-9 else 0.0
            hit_rate = float((ic_vals > 0).mean())
            t_stat   = (
                mean_ic / (std_ic / np.sqrt(len(ic_vals)))
                if std_ic > 1e-9 else 0.0
            )
            cum_ic = float(ic_vals.cumsum().iloc[-1])

            rows.append({
                "factor":        factor,
                "horizon_days":  horizon,
                "mean_ic":       round(mean_ic, 4),
                "icir":          round(icir, 3),
                "hit_rate":      round(hit_rate, 3),
                "t_stat":        round(t_stat, 3),
                "n_periods":     len(ic_vals),
                "cumulative_ic": round(cum_ic, 3),
            })

        return pd.DataFrame(rows)

    # ── Factor decay ─────────────────────────────────────────────────────────

    def compute_factor_decay(
        self,
        factors: Optional[list[str]] = None,
        ic_type: str = "spearman",
    ) -> pd.DataFrame:
        """
        IC mean vs. horizon — the factor decay curve.

        Returns [factor, horizon_days, mean_ic, icir, hit_rate].
        Plot horizon_days on x-axis, mean_ic on y-axis.
        """
        ic_df   = self.compute_multi_horizon_ic(factors=factors, ic_type=ic_type)
        summary = self.compute_ic_summary(ic_df)
        if summary.empty:
            return pd.DataFrame()
        return (
            summary[["factor", "horizon_days", "mean_ic", "icir", "hit_rate"]]
            .sort_values(["factor", "horizon_days"])
            .reset_index(drop=True)
        )

    # ── Decile spread (monotonicity) ─────────────────────────────────────────

    def compute_decile_spread(
        self,
        factor: str,
        horizon_days: int = 20,
        n_deciles: int = 10,
    ) -> pd.DataFrame:
        """
        Pool all snapshot-date cross-sections; rank stocks into deciles
        by factor score; compute mean forward return per decile.

        Columns: [decile, mean_forward_return, hit_rate, n_stocks]
        """
        dates_map = self._load_dates_map()
        sorted_dates = sorted(dates_map.keys())
        min_days = max(1, int(horizon_days * (1 - self.max_horizon_slop_pct)))
        max_days = int(horizon_days * (1 + self.max_horizon_slop_pct))

        pooled: list[pd.DataFrame] = []
        for i, t_date in enumerate(sorted_dates):
            fwd_date: Optional[datetime.date] = None
            best_diff = 9999
            for fwd in sorted_dates[i + 1:]:
                diff = (fwd - t_date).days
                if diff < min_days:
                    continue
                if diff > max_days:
                    break
                if abs(diff - horizon_days) < best_diff:
                    fwd_date = fwd
                    best_diff = abs(diff - horizon_days)

            if fwd_date is None:
                continue

            df_t   = dates_map[t_date]
            df_fwd = dates_map[fwd_date]

            if "symbol" not in df_t.columns or factor not in df_t.columns:
                continue

            fr = self._forward_returns(df_t, df_fwd)
            if fr.empty:
                continue

            fv = pd.to_numeric(
                df_t.set_index("symbol")[factor], errors="coerce"
            ).rename("factor_val")

            merged = pd.DataFrame({"factor_val": fv, "forward_return": fr}).dropna()
            if len(merged) >= n_deciles:
                pooled.append(merged)

        if not pooled:
            return pd.DataFrame()

        all_data = pd.concat(pooled)
        try:
            all_data["decile"] = pd.qcut(
                all_data["factor_val"], n_deciles, labels=False, duplicates="drop"
            ) + 1
        except ValueError:
            return pd.DataFrame()

        result = (
            all_data.groupby("decile")["forward_return"]
            .agg(
                mean_forward_return="mean",
                hit_rate=lambda x: float((x > 0).mean()),
                n_stocks="count",
            )
            .reset_index()
        )
        result["mean_forward_return"] = result["mean_forward_return"].round(4)
        result["hit_rate"] = result["hit_rate"].round(3)
        return result

    # ── Rolling ICIR ─────────────────────────────────────────────────────────

    def compute_rolling_icir(
        self,
        factor: str,
        horizon_days: int = 20,
        window: int = 12,
    ) -> pd.DataFrame:
        """
        Trailing `window`-period rolling ICIR over time.

        Columns: [date, ic, rolling_icir, cumulative_ic]
        """
        ic_df = self.compute_multi_horizon_ic(factors=[factor], horizons=[horizon_days])
        if ic_df.empty:
            return pd.DataFrame()

        grp = (
            ic_df[ic_df["factor"] == factor]
            .sort_values("date")
            .copy()
        )
        min_periods = max(2, window // 2)
        grp["rolling_icir"] = grp["ic"].rolling(window, min_periods=min_periods).apply(
            lambda x: x.mean() / x.std() if x.std() > 1e-9 else 0.0,
            raw=True,
        )
        grp["cumulative_ic"] = grp["ic"].cumsum()
        return grp[["date", "ic", "rolling_icir", "cumulative_ic"]].reset_index(drop=True)

    # ── Regime-conditioned IC ────────────────────────────────────────────────

    def compute_regime_conditioned_ic(
        self,
        factors: Optional[list[str]] = None,
        horizon_days: int = 20,
        ic_type: str = "spearman",
        lookback_days: int = 365,
        vix_high_vol: float = 25.0,
        bear_dma_threshold: float = -0.02,
    ) -> pd.DataFrame:
        """
        IC for each factor × market regime combination.

        Regime labels (priority order):
          high_vol  — VIX ≥ vix_high_vol (≥ 25 by default)
          bear      — SPY > vix_high_vol days below 200DMA
          bull      — SPY above 200DMA, VIX low
          sideways  — everything else (corrective / range-bound)

        Columns: [factor, regime, mean_ic, icir, hit_rate, t_stat, n_periods]
        """
        from strategy.regimes import RegimeDetector
        from strategy.regimes.models import RegimeHistoryEntry

        use_factors = [
            f for f in (factors or self.factors) if f != "value_metric"
        ]

        # ── Build date → regime map ──────────────────────────────────────────
        detector = RegimeDetector()
        history  = detector.classify_history(days=lookback_days)
        if not history:
            logger.warning("regime_conditioned_ic: no regime history available")
            return pd.DataFrame()

        def _label(entry: RegimeHistoryEntry) -> str:
            if entry.vix is not None and entry.vix >= vix_high_vol:
                return "high_vol"
            if entry.regime == "defensive":
                return "high_vol"
            if entry.spy_vs_200dma_pct < bear_dma_threshold:
                return "bear"
            if entry.regime == "bullish":
                return "bull"
            return "sideways"

        regime_by_date: dict[datetime.date, str] = {
            e.date: _label(e) for e in history
        }

        def _nearest_regime(d: datetime.date) -> Optional[str]:
            for delta in range(6):
                for sign in [0, -1, 1]:
                    candidate = d + datetime.timedelta(days=delta * sign)
                    if candidate in regime_by_date:
                        return regime_by_date[candidate]
            return None

        # ── Iterate snapshot pairs, collect IC per regime ────────────────────
        dates_map    = self._load_dates_map()
        sorted_dates = sorted(dates_map.keys())
        if len(sorted_dates) < 2:
            return pd.DataFrame()

        min_days = max(1, int(horizon_days * (1 - self.max_horizon_slop_pct)))
        max_days = int(horizon_days * (1 + self.max_horizon_slop_pct))

        bucket: dict[tuple[str, str], list[float]] = {}  # (regime, factor) → [ic]

        for i, t_date in enumerate(sorted_dates):
            regime = _nearest_regime(t_date)
            if regime is None:
                continue

            fwd_date: Optional[datetime.date] = None
            best_diff = 9999
            for fwd in sorted_dates[i + 1:]:
                diff = (fwd - t_date).days
                if diff < min_days:
                    continue
                if diff > max_days:
                    break
                if abs(diff - horizon_days) < best_diff:
                    fwd_date = fwd
                    best_diff = abs(diff - horizon_days)

            if fwd_date is None:
                continue

            df_t   = dates_map[t_date]
            df_fwd = dates_map[fwd_date]
            fr     = self._forward_returns(df_t, df_fwd)
            if fr.empty or len(fr) < self.min_overlap:
                continue

            for factor in use_factors:
                if factor not in df_t.columns or "symbol" not in df_t.columns:
                    continue

                fv = pd.to_numeric(
                    df_t.set_index("symbol")[factor], errors="coerce"
                ).rename("factor_val")

                merged = pd.DataFrame({"factor_val": fv, "forward_return": fr}).dropna()
                if len(merged) < self.min_overlap:
                    continue

                try:
                    if ic_type == "spearman":
                        ic_val, _ = stats.spearmanr(
                            merged["factor_val"].values, merged["forward_return"].values
                        )
                    else:
                        ic_val, _ = stats.pearsonr(
                            merged["factor_val"].values, merged["forward_return"].values
                        )
                except Exception:
                    continue

                if np.isnan(ic_val):
                    continue

                key = (regime, factor)
                bucket.setdefault(key, []).append(float(ic_val))

        if not bucket:
            return pd.DataFrame()

        # ── Aggregate per (regime, factor) ───────────────────────────────────
        rows: list[dict] = []
        for (regime, factor), ic_vals in bucket.items():
            arr      = np.array(ic_vals)
            mean_ic  = float(arr.mean())
            std_ic   = float(arr.std()) if len(arr) > 1 else 0.0
            icir     = mean_ic / std_ic if std_ic > 1e-9 else 0.0
            hit_rate = float((arr > 0).mean())
            t_stat   = (
                mean_ic / (std_ic / np.sqrt(len(arr)))
                if std_ic > 1e-9 and len(arr) > 1 else 0.0
            )
            rows.append({
                "factor":    factor,
                "regime":    regime,
                "mean_ic":   round(mean_ic, 4),
                "icir":      round(icir, 3),
                "hit_rate":  round(hit_rate, 3),
                "t_stat":    round(t_stat, 3),
                "n_periods": len(arr),
            })

        return pd.DataFrame(rows)

    # ── Conditional / interaction IC ─────────────────────────────────────────

    def compute_conditional_ic(
        self,
        horizon_days: int = 20,
        ic_type: str = "spearman",
    ) -> pd.DataFrame:
        """
        Compute IC summary for all conditional-momentum features vs baseline momentum_score.

        For each snapshot pair the interaction features are engineered on-the-fly from
        the base scores in the snapshot; no production columns are altered.

        Columns: [feature, label, group, description,
                  mean_ic, icir, hit_rate, t_stat, tail_ic, stability_score, n_periods]
        Rows are sorted: engineered features (by mean_ic desc) first, then baseline.
        """
        from strategy.factor_interactions import (
            add_interaction_features,
            INTERACTION_FEATURES,
            INTERACTION_FEATURE_NAMES,
        )

        dates_map    = self._load_dates_map()
        sorted_dates = sorted(dates_map.keys())
        if len(sorted_dates) < 2:
            return pd.DataFrame()

        min_days = max(1, int(horizon_days * (1 - self.max_horizon_slop_pct)))
        max_days = int(horizon_days * (1 + self.max_horizon_slop_pct))
        all_factors = ["momentum_score"] + INTERACTION_FEATURE_NAMES

        bucket: dict[str, list[float]] = {f: [] for f in all_factors}

        for i, t_date in enumerate(sorted_dates):
            fwd_date: Optional[datetime.date] = None
            best_diff = 9999
            for fwd in sorted_dates[i + 1:]:
                diff = (fwd - t_date).days
                if diff < min_days:
                    continue
                if diff > max_days:
                    break
                if abs(diff - horizon_days) < best_diff:
                    fwd_date = fwd
                    best_diff = abs(diff - horizon_days)

            if fwd_date is None:
                continue

            df_t   = dates_map[t_date].copy()
            df_fwd = dates_map[fwd_date]
            fr     = self._forward_returns(df_t, df_fwd)
            if fr.empty or len(fr) < self.min_overlap:
                continue

            add_interaction_features(df_t)

            for factor in all_factors:
                if factor not in df_t.columns or "symbol" not in df_t.columns:
                    continue
                fv = pd.to_numeric(
                    df_t.set_index("symbol")[factor], errors="coerce"
                ).rename("factor_val")
                merged = pd.DataFrame({"factor_val": fv, "forward_return": fr}).dropna()
                if len(merged) < self.min_overlap:
                    continue
                try:
                    if ic_type == "spearman":
                        ic_val, _ = stats.spearmanr(
                            merged["factor_val"].values, merged["forward_return"].values
                        )
                    else:
                        ic_val, _ = stats.pearsonr(
                            merged["factor_val"].values, merged["forward_return"].values
                        )
                except Exception:
                    continue
                if not np.isnan(ic_val):
                    bucket[factor].append(float(ic_val))

        feat_meta = {f["name"]: f for f in INTERACTION_FEATURES}
        rows: list[dict] = []
        for feature, ic_vals in bucket.items():
            if not ic_vals:
                continue
            arr      = np.array(ic_vals)
            mean_ic  = float(arr.mean())
            std_ic   = float(arr.std()) if len(arr) > 1 else 0.0
            icir     = mean_ic / std_ic if std_ic > 1e-9 else 0.0
            hit_rate = float((arr > 0).mean())
            t_stat   = (
                mean_ic / (std_ic / np.sqrt(len(arr)))
                if std_ic > 1e-9 and len(arr) > 1 else 0.0
            )
            tail_ic  = float(np.percentile(arr, 25)) if len(arr) >= 4 else mean_ic
            stability = float((arr > 0).mean())
            meta = feat_meta.get(feature, {})
            rows.append({
                "feature":         feature,
                "label":           meta.get("label", "Baseline Momentum"),
                "group":           meta.get("group", "baseline"),
                "description":     meta.get("description", "Raw momentum_score (benchmark)"),
                "mean_ic":         round(mean_ic, 4),
                "icir":            round(icir, 3),
                "hit_rate":        round(hit_rate, 3),
                "t_stat":          round(t_stat, 3),
                "tail_ic":         round(tail_ic, 4),
                "stability_score": round(stability, 3),
                "n_periods":       len(arr),
            })

        if not rows:
            return pd.DataFrame()

        result = pd.DataFrame(rows)
        baseline    = result[result["feature"] == "momentum_score"]
        engineered  = (
            result[result["feature"] != "momentum_score"]
            .sort_values("mean_ic", ascending=False)
        )
        return pd.concat([engineered, baseline], ignore_index=True)

    def compute_conditional_ic_timeseries(
        self,
        horizon_days: int = 20,
        ic_type: str = "spearman",
    ) -> pd.DataFrame:
        """
        Per-snapshot-date IC for each conditional feature and baseline momentum_score.

        Columns: [date, feature, ic]
        """
        from strategy.factor_interactions import (
            add_interaction_features,
            INTERACTION_FEATURE_NAMES,
        )

        dates_map    = self._load_dates_map()
        sorted_dates = sorted(dates_map.keys())
        if len(sorted_dates) < 2:
            return pd.DataFrame()

        min_days = max(1, int(horizon_days * (1 - self.max_horizon_slop_pct)))
        max_days = int(horizon_days * (1 + self.max_horizon_slop_pct))
        all_factors = ["momentum_score"] + INTERACTION_FEATURE_NAMES

        rows: list[dict] = []
        for i, t_date in enumerate(sorted_dates):
            fwd_date: Optional[datetime.date] = None
            best_diff = 9999
            for fwd in sorted_dates[i + 1:]:
                diff = (fwd - t_date).days
                if diff < min_days:
                    continue
                if diff > max_days:
                    break
                if abs(diff - horizon_days) < best_diff:
                    fwd_date = fwd
                    best_diff = abs(diff - horizon_days)

            if fwd_date is None:
                continue

            df_t   = dates_map[t_date].copy()
            df_fwd = dates_map[fwd_date]
            fr     = self._forward_returns(df_t, df_fwd)
            if fr.empty or len(fr) < self.min_overlap:
                continue

            add_interaction_features(df_t)

            for factor in all_factors:
                if factor not in df_t.columns or "symbol" not in df_t.columns:
                    continue
                fv = pd.to_numeric(
                    df_t.set_index("symbol")[factor], errors="coerce"
                ).rename("factor_val")
                merged = pd.DataFrame({"factor_val": fv, "forward_return": fr}).dropna()
                if len(merged) < self.min_overlap:
                    continue
                try:
                    if ic_type == "spearman":
                        ic_val, _ = stats.spearmanr(
                            merged["factor_val"].values, merged["forward_return"].values
                        )
                    else:
                        ic_val, _ = stats.pearsonr(
                            merged["factor_val"].values, merged["forward_return"].values
                        )
                except Exception:
                    continue
                if not np.isnan(ic_val):
                    rows.append({
                        "date":    t_date,
                        "feature": factor,
                        "ic":      round(float(ic_val), 4),
                    })

        return pd.DataFrame(rows)

    # ── Cumulative IC ─────────────────────────────────────────────────────────

    def compute_cumulative_ic(
        self,
        factors: Optional[list[str]] = None,
        horizon_days: int = 20,
        ic_type: str = "spearman",
    ) -> pd.DataFrame:
        """
        Cumulative IC over time for each factor at a given horizon.

        Columns: [date, factor, cumulative_ic]
        """
        use_factors = factors or self.factors
        ic_df = self.compute_multi_horizon_ic(
            factors=use_factors,
            horizons=[horizon_days],
            ic_type=ic_type,
        )
        if ic_df.empty:
            return pd.DataFrame()

        parts: list[pd.DataFrame] = []
        for factor, grp in ic_df.groupby("factor"):
            g = grp.sort_values("date").copy()
            g["cumulative_ic"] = g["ic"].cumsum()
            parts.append(g[["date", "factor", "cumulative_ic"]])

        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
