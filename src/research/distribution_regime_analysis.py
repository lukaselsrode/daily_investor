"""
research/distribution_regime_analysis.py — Distribution regime analysis engine.

Tests whether the bimodal/binomial value-score distribution under peer-relative
normalization contains predictive information, and whether alpha concentrates
in the tails rather than the full continuous ranking.

Key questions:
  1. Is the distribution genuinely bimodal?
  2. Do tails predict better than the middle?
  3. Is predictive power nonlinear / threshold-based?
  4. Do factor clusters correspond to future return regimes?
  5. Are factor interactions conditional (value only works inside high-quality bucket)?

All analyses are read-only. No config writes, no orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_SCORE_COLS = ["value_score", "quality_score", "income_score", "momentum_score", "value_metric"]
_RETURN_COLS = ["return_1m", "return_3m", "return_6m"]

# Standard percentile buckets for tail analysis
TAIL_BUCKETS: list[tuple[str, float, float]] = [
    ("top_1pct",  0.99, 1.00),
    ("top_5pct",  0.95, 1.00),
    ("top_10pct", 0.90, 1.00),
    ("mid_40_60", 0.40, 0.60),
    ("bot_10pct", 0.00, 0.10),
    ("bot_5pct",  0.00, 0.05),
    ("bot_1pct",  0.00, 0.01),
]


@dataclass
class BucketStats:
    label: str
    lo_pct: float
    hi_pct: float
    count: int
    mean_return: float
    median_return: float
    hit_rate: float
    volatility: float
    sharpe_proxy: float
    score_lo: float
    score_hi: float


@dataclass
class BimodalityResult:
    bimodality_coeff: float       # > 0.555 suggests bimodal (BC test)
    is_bimodal: bool
    skewness: float
    excess_kurtosis: float
    gmm_bic_k1: float             # GMM BIC with k=1 (unimodal)
    gmm_bic_k2: float             # GMM BIC with k=2 (bimodal)
    gmm_favors_bimodal: bool      # k=2 has lower BIC
    gmm_means: list[float]        # cluster means (k=2)
    gmm_weights: list[float]      # cluster mixing weights (k=2)
    separation_score: float       # |mean1-mean2| / pooled_std


class DistributionAnalyzer:
    """
    Read-only analysis of score distribution structure.

    Instantiate with the current agg_data DataFrame.
    All methods return DataFrames or structured results suitable for Streamlit display.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self._coerce_numerics()

    def _coerce_numerics(self) -> None:
        for col in _SCORE_COLS + _RETURN_COLS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

    # ── Distribution statistics ──────────────────────────────────────────────

    def distribution_stats(self, score_col: str = "value_metric") -> dict:
        """Return summary statistics for a score column."""
        s = self.df[score_col].dropna()
        if s.empty:
            return {}
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        result: dict = {
            "n": len(s),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "skew": round(float(s.skew()), 4),
            "kurt": round(float(float(s.kurtosis())), 4),
        }
        for p in percentiles:
            result[f"p{p}"] = round(float(s.quantile(p / 100)), 4)
        return result

    # ── Bimodality detection ─────────────────────────────────────────────────

    def test_bimodality(self, score_col: str = "value_metric") -> BimodalityResult:
        """
        Test whether the score distribution is bimodal via:
          - Bimodality coefficient (BC > 0.555)
          - Gaussian Mixture Model BIC comparison (k=1 vs k=2)
        """
        s = self.df[score_col].dropna().values
        if len(s) < 20:
            raise ValueError(f"Too few samples ({len(s)}) for bimodality test")

        skewness = float(stats.skew(s))
        kurtosis = float(stats.kurtosis(s))  # excess kurtosis

        # Bimodality coefficient: BC = (skew^2 + 1) / (kurt + 3)
        # BC > 0.555 suggests bimodality
        bc_den = kurtosis + 3
        bimodality_coeff = (skewness ** 2 + 1) / bc_den if abs(bc_den) > 1e-9 else 0.0

        gmm_bic_k1 = float("inf")
        gmm_bic_k2 = float("inf")
        gmm_means: list[float] = []
        gmm_weights: list[float] = []
        separation_score = 0.0

        try:
            from sklearn.mixture import GaussianMixture
            X = s.reshape(-1, 1)
            gm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
            gm2 = GaussianMixture(n_components=2, random_state=42).fit(X)
            gmm_bic_k1 = float(gm1.bic(X))
            gmm_bic_k2 = float(gm2.bic(X))

            order = np.argsort(gm2.means_.flatten())
            means_sorted = gm2.means_.flatten()[order]
            weights_sorted = gm2.weights_.flatten()[order]
            covs_sorted = gm2.covariances_.flatten()[order]

            gmm_means = [round(float(m), 4) for m in means_sorted]
            gmm_weights = [round(float(w), 4) for w in weights_sorted]

            pooled_std = float(np.sqrt(np.average(covs_sorted, weights=weights_sorted)))
            separation_score = abs(gmm_means[1] - gmm_means[0]) / max(pooled_std, 1e-9)
        except ImportError:
            logger.debug("sklearn not available — GMM bimodality skipped")
        except Exception as exc:
            logger.warning("GMM fit failed: %s", exc)

        return BimodalityResult(
            bimodality_coeff=round(bimodality_coeff, 4),
            is_bimodal=bimodality_coeff > 0.555,
            skewness=round(skewness, 4),
            excess_kurtosis=round(kurtosis, 4),
            gmm_bic_k1=round(gmm_bic_k1, 2),
            gmm_bic_k2=round(gmm_bic_k2, 2),
            gmm_favors_bimodal=gmm_bic_k2 < gmm_bic_k1,
            gmm_means=gmm_means,
            gmm_weights=gmm_weights,
            separation_score=round(separation_score, 3),
        )

    # ── Tail bucket analysis ─────────────────────────────────────────────────

    def compute_tail_buckets(
        self,
        score_col: str = "value_metric",
        return_col: str = "return_1m",
        buckets: list[tuple[str, float, float]] | None = None,
    ) -> list[BucketStats]:
        """Compute return statistics for each percentile bucket of score_col."""
        use_buckets = buckets or TAIL_BUCKETS
        df = self.df[[score_col, return_col]].dropna()
        if len(df) < 30:
            return []

        result: list[BucketStats] = []
        for label, lo_pct, hi_pct in use_buckets:
            score_lo = float(df[score_col].quantile(lo_pct))
            score_hi = float(df[score_col].quantile(hi_pct))

            if lo_pct == 0.0:
                mask = df[score_col] <= score_hi
            elif hi_pct == 1.0:
                mask = df[score_col] >= score_lo
            else:
                mask = (df[score_col] >= score_lo) & (df[score_col] <= score_hi)

            sub = df[mask]
            if sub.empty:
                continue

            rets = sub[return_col]
            vol = float(rets.std())
            mean_r = float(rets.mean())
            result.append(BucketStats(
                label=label,
                lo_pct=lo_pct,
                hi_pct=hi_pct,
                count=len(sub),
                mean_return=round(mean_r, 4),
                median_return=round(float(rets.median()), 4),
                hit_rate=round(float((rets > 0).mean()), 4),
                volatility=round(vol, 4),
                sharpe_proxy=round(mean_r / max(vol, 1e-9), 4),
                score_lo=round(score_lo, 4),
                score_hi=round(score_hi, 4),
            ))

        return result

    def buckets_to_df(self, buckets: list[BucketStats]) -> pd.DataFrame:
        if not buckets:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "bucket":        b.label,
                "range_pct":     f"{b.lo_pct*100:.0f}%–{b.hi_pct*100:.0f}%",
                "score_range":   f"[{b.score_lo}, {b.score_hi}]",
                "n":             b.count,
                "mean_return":   b.mean_return,
                "median_return": b.median_return,
                "hit_rate":      b.hit_rate,
                "volatility":    b.volatility,
                "sharpe_proxy":  b.sharpe_proxy,
            }
            for b in buckets
        ])

    # ── Monotonicity (decile spread) ─────────────────────────────────────────

    def compute_monotonicity(
        self,
        score_col: str = "value_metric",
        return_col: str = "return_1m",
        n_deciles: int = 10,
    ) -> pd.DataFrame:
        """
        Compute mean return per score decile to test monotonicity.
        Returns DataFrame; attrs['kendall_tau'] and attrs['kendall_p'] contain the
        Kendall tau between decile rank and mean return.
        """
        df = self.df[[score_col, return_col]].dropna()
        if len(df) < n_deciles * 5:
            return pd.DataFrame()

        try:
            df = df.copy()
            df["decile"] = pd.qcut(df[score_col], q=n_deciles, labels=False, duplicates="drop") + 1
        except ValueError:
            return pd.DataFrame()

        agg = (
            df.groupby("decile")[return_col]
            .agg(
                mean_return="mean",
                median_return="median",
                hit_rate=lambda x: float((x > 0).mean()),
                volatility="std",
                count="count",
            )
        ).round(4)
        agg["sharpe_proxy"] = (
            agg["mean_return"] / agg["volatility"].replace(0, float("nan"))
        ).round(4)

        if len(agg) >= 3:
            tau, pval = stats.kendalltau(agg.index.values, agg["mean_return"].values)
            agg.attrs["kendall_tau"] = round(float(tau), 4)
            agg.attrs["kendall_p"] = round(float(pval), 4)

        return agg.reset_index()

    # ── Local IC (nonlinear predictive power) ─────────────────────────────────

    def compute_local_ic(
        self,
        score_col: str = "value_metric",
        return_col: str = "return_1m",
        window_pct: float = 0.20,
        step_pct: float = 0.05,
    ) -> pd.DataFrame:
        """
        Compute IC in sliding percentile windows over the sorted score distribution.

        High local IC in tails + low IC in center = threshold-based alpha structure.
        Columns: [center_pct, center_score, local_ic, p_value, n]
        """
        df = self.df[[score_col, return_col]].dropna().sort_values(score_col).reset_index(drop=True)
        n = len(df)
        if n < 40:
            return pd.DataFrame()

        window_size = max(20, int(n * window_pct))
        step_size = max(1, int(n * step_pct))
        rows: list[dict] = []

        for start in range(0, n - window_size + 1, step_size):
            window = df.iloc[start: start + window_size]
            try:
                ic, pv = stats.spearmanr(window[score_col].values, window[return_col].values)
            except Exception:
                continue
            if np.isnan(ic):
                continue
            rows.append({
                "center_pct":   round((start + window_size / 2) / n, 3),
                "center_score": round(float(window[score_col].median()), 4),
                "local_ic":     round(float(ic), 4),
                "p_value":      round(float(pv), 4),
                "n":            len(window),
            })

        return pd.DataFrame(rows)

    # ── Regime clustering ─────────────────────────────────────────────────────

    def compute_clusters(
        self,
        features: list[str] | None = None,
        n_clusters: int = 2,
        method: str = "gmm",
        return_col: str = "return_1m",
    ) -> pd.DataFrame:
        """
        Cluster stocks by factor scores and analyze return statistics per cluster.

        method: 'gmm' | 'kmeans'
        Returns cluster summary DataFrame with mean factor scores and return stats.
        """
        use_features = [c for c in (features or ["value_score", "quality_score", "momentum_score"])
                        if c in self.df.columns]
        if not use_features:
            return pd.DataFrame()

        cols = use_features + ([return_col] if return_col in self.df.columns else [])
        cluster_df = self.df[cols].dropna()
        if len(cluster_df) < n_clusters * 10:
            return pd.DataFrame()

        X = cluster_df[use_features].values
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)

        labels: np.ndarray | None = None
        try:
            if method == "gmm":
                from sklearn.mixture import GaussianMixture
                labels = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(X_std)
            else:
                from sklearn.cluster import KMeans
                labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_std)
        except ImportError:
            # Median-split fallback for k=2
            if n_clusters == 2:
                med = float(np.median(X_std[:, 0]))
                labels = (X_std[:, 0] >= med).astype(int)
        except Exception as exc:
            logger.warning("Clustering failed: %s", exc)
            return pd.DataFrame()

        if labels is None:
            return pd.DataFrame()

        cluster_df = cluster_df.copy()
        cluster_df["cluster"] = labels

        rows: list[dict] = []
        for c in sorted(cluster_df["cluster"].unique()):
            sub = cluster_df[cluster_df["cluster"] == c]
            row: dict = {"cluster": int(c), "count": len(sub)}
            for f in use_features:
                row[f"mean_{f}"] = round(float(sub[f].mean()), 4)
            if return_col in sub.columns:
                rets = sub[return_col]
                row["mean_return"] = round(float(rets.mean()), 4)
                row["hit_rate"] = round(float((rets > 0).mean()), 4)
                row["volatility"] = round(float(rets.std()), 4)
                row["sharpe_proxy"] = round(float(rets.mean()) / max(float(rets.std()), 1e-9), 4)
            rows.append(row)

        return pd.DataFrame(rows).sort_values("mean_return", ascending=False).reset_index(drop=True)

    def compute_cluster_labels(
        self,
        features: list[str] | None = None,
        n_clusters: int = 2,
        method: str = "gmm",
    ) -> pd.Series:
        """Return a Series of cluster labels indexed by the original DataFrame's index."""
        use_features = [c for c in (features or ["value_score", "quality_score", "momentum_score"])
                        if c in self.df.columns]
        if not use_features:
            return pd.Series(dtype=int)

        sub = self.df[use_features].dropna()
        X = sub.values
        X_std = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-9)

        try:
            if method == "gmm":
                from sklearn.mixture import GaussianMixture
                labels = GaussianMixture(n_components=n_clusters, random_state=42).fit_predict(X_std)
            else:
                from sklearn.cluster import KMeans
                labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X_std)
        except ImportError:
            labels = (X_std[:, 0] >= np.median(X_std[:, 0])).astype(int)
        except Exception:
            return pd.Series(dtype=int)

        return pd.Series(labels, index=sub.index, name="cluster")

    # ── Conditional alpha ──────────────────────────────────────────────────────

    def compute_conditional_ic(
        self,
        primary_factor: str,
        conditioning_factor: str,
        return_col: str = "return_1m",
        n_quartiles: int = 4,
    ) -> pd.DataFrame:
        """
        IC of primary_factor within each quartile of conditioning_factor.

        Reveals hidden conditional alpha: factor A may only work inside
        specific levels of factor B.
        Columns: [quartile, cond_label, n_stocks, ic, p_value, significant]
        """
        needed = [primary_factor, conditioning_factor, return_col]
        if not all(c in self.df.columns for c in needed):
            return pd.DataFrame()

        df = self.df[needed].dropna()
        if len(df) < n_quartiles * 15:
            return pd.DataFrame()

        try:
            df = df.copy()
            df["cond_q"] = pd.qcut(df[conditioning_factor], q=n_quartiles, labels=False,
                                   duplicates="drop") + 1
        except ValueError:
            return pd.DataFrame()

        rows: list[dict] = []
        for q in range(1, n_quartiles + 1):
            sub = df[df["cond_q"] == q]
            if len(sub) < 10:
                continue
            lo = round(float(df[conditioning_factor].quantile((q - 1) / n_quartiles)), 4)
            hi = round(float(df[conditioning_factor].quantile(q / n_quartiles)), 4)
            try:
                ic, pv = stats.spearmanr(sub[primary_factor].values, sub[return_col].values)
            except Exception:
                continue
            if np.isnan(ic):
                continue
            rows.append({
                "quartile":    q,
                "cond_label":  f"Q{q} [{lo:.3f}, {hi:.3f}]",
                "n_stocks":    len(sub),
                "ic":          round(float(ic), 4),
                "p_value":     round(float(pv), 4),
                "significant": bool(pv < 0.05),
            })

        return pd.DataFrame(rows)

    def compute_interaction_matrix(
        self,
        score_cols: list[str] | None = None,
        return_col: str = "return_1m",
        n_quartiles: int = 4,
    ) -> pd.DataFrame:
        """
        Build a matrix of conditional ICs: rows = primary factor, columns = conditioning factor.
        Entry [row, col] = mean IC of row factor when col factor is in its top quartile.
        """
        use_cols = [c for c in (score_cols or ["value_score", "quality_score",
                                               "income_score", "momentum_score"])
                    if c in self.df.columns]
        if len(use_cols) < 2:
            return pd.DataFrame()

        result: dict[str, dict[str, float]] = {}
        for primary in use_cols:
            result[primary] = {}
            for cond in use_cols:
                if primary == cond:
                    result[primary][cond] = float("nan")
                    continue
                cic = self.compute_conditional_ic(primary, cond, return_col, n_quartiles)
                if cic.empty:
                    result[primary][cond] = float("nan")
                    continue
                top_q = cic[cic["quartile"] == cic["quartile"].max()]
                result[primary][cond] = round(float(top_q["ic"].mean()), 4) if not top_q.empty else float("nan")

        return pd.DataFrame(result).T

    # ── Threshold simulation ───────────────────────────────────────────────────

    def simulate_threshold_modes(
        self,
        score_col: str = "value_metric",
        return_col: str = "return_1m",
        thresholds: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Simulate threshold-gated selection vs full-universe ranking.
        Shows how restricting to high-confidence names affects return statistics.
        """
        use_thresholds = thresholds or [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        df = self.df[[score_col, return_col]].dropna()
        if df.empty:
            return pd.DataFrame()

        total = len(df)
        rows: list[dict] = []

        # Baseline: all names
        rets_all = df[return_col]
        rows.append({
            "mode":           "all (baseline)",
            "threshold":      None,
            "n_selected":     total,
            "pct_universe":   1.0,
            "mean_return":    round(float(rets_all.mean()), 4),
            "median_return":  round(float(rets_all.median()), 4),
            "hit_rate":       round(float((rets_all > 0).mean()), 4),
            "volatility":     round(float(rets_all.std()), 4),
            "sharpe_proxy":   round(float(rets_all.mean()) / max(float(rets_all.std()), 1e-9), 4),
        })

        for thresh in use_thresholds:
            above = df[df[score_col] >= thresh]
            if above.empty or len(above) < 5:
                continue
            rets = above[return_col]
            vol = float(rets.std())
            rows.append({
                "mode":           f"score ≥ {thresh}",
                "threshold":      thresh,
                "n_selected":     len(above),
                "pct_universe":   round(len(above) / total, 3),
                "mean_return":    round(float(rets.mean()), 4),
                "median_return":  round(float(rets.median()), 4),
                "hit_rate":       round(float((rets > 0).mean()), 4),
                "volatility":     round(vol, 4),
                "sharpe_proxy":   round(float(rets.mean()) / max(vol, 1e-9), 4),
            })

        return pd.DataFrame(rows)

    # ── Factor confidence engine ───────────────────────────────────────────────

    @staticmethod
    def compute_factor_confidence(ic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive a composite confidence score (0–1) per factor from historical IC data.

        Combines ICIR, hit rate, and directional consistency.
        Higher confidence = factor is consistently predictive = weight it more.
        Columns: [factor, mean_ic, icir, hit_rate, n_periods, confidence, suggested_weight_adj]
        """
        if ic_df.empty or "factor" not in ic_df.columns or "ic" not in ic_df.columns:
            return pd.DataFrame()

        rows: list[dict] = []
        for factor, grp in ic_df.groupby("factor"):
            ics = grp["ic"].dropna()
            if len(ics) < 2:
                continue
            mean_ic = float(ics.mean())
            std_ic = float(ics.std())
            icir = mean_ic / std_ic if std_ic > 1e-9 else 0.0
            hit_rate = float((ics > 0).mean())

            # Confidence components (each 0-1):
            # ICIR score: normalize so ICIR=0 → 0.5, ICIR>0.5 → high, ICIR<-0.5 → low
            icir_score = min(1.0, max(0.0, (icir + 1.0) / 2.0))
            # Hit rate: already 0-1
            hit_score = hit_rate
            # Directional: positive mean IC rewarded, negative penalized
            sign_score = min(1.0, max(0.0, (mean_ic + 0.15) / 0.30))

            confidence = 0.40 * icir_score + 0.35 * hit_score + 0.25 * sign_score

            # Suggested weight direction: +/- relative to equal weight
            weight_adj = round((confidence - 0.5) * 0.2, 3)  # up to ±10% adjustment

            rows.append({
                "factor":            factor,
                "mean_ic":           round(mean_ic, 4),
                "icir":              round(icir, 3),
                "hit_rate":          round(hit_rate, 3),
                "n_periods":         len(ics),
                "confidence":        round(confidence, 3),
                "weight_adj":        weight_adj,
            })

        return pd.DataFrame(rows).sort_values("confidence", ascending=False).reset_index(drop=True)

    # ── Distribution evolution (multi-snapshot) ────────────────────────────────

    @staticmethod
    def compute_distribution_evolution(score_col: str = "value_metric") -> pd.DataFrame:
        """
        Track distribution shape (mean, std, skew, kurtosis, bimodality coeff)
        across all available snapshot files.

        Requires the strategy.snapshots module to list available snapshots.
        Returns DataFrame indexed by snapshot date.
        """
        try:
            from strategy.snapshots import list_snapshots
            snapshots = list_snapshots()
        except Exception:
            return pd.DataFrame()

        rows: list[dict] = []
        for snap_date, snap_path in snapshots:
            try:
                df = pd.read_parquet(snap_path)
                if score_col not in df.columns:
                    continue
                s = pd.to_numeric(df[score_col], errors="coerce").dropna()
                if len(s) < 20:
                    continue
                skew_val = float(s.skew())
                kurt_val = float(s.kurtosis())
                bc_den = kurt_val + 3
                bc = (skew_val ** 2 + 1) / bc_den if abs(bc_den) > 1e-9 else 0.0
                rows.append({
                    "date":               snap_date,
                    "n":                  len(s),
                    "mean":               round(float(s.mean()), 4),
                    "std":                round(float(s.std()), 4),
                    "skew":               round(skew_val, 4),
                    "excess_kurtosis":    round(kurt_val, 4),
                    "bimodality_coeff":   round(bc, 4),
                    "p10":                round(float(s.quantile(0.10)), 4),
                    "p90":                round(float(s.quantile(0.90)), 4),
                    "tail_spread":        round(float(s.quantile(0.90) - s.quantile(0.10)), 4),
                })
            except Exception as exc:
                logger.debug("Skipping snapshot %s: %s", snap_path, exc)

        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True) if rows else pd.DataFrame()
