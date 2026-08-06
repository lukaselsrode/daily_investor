"""
scripts/recalibrate_peer3_thresholds.py — percentile-match score thresholds old → peer-3.

The peer-3 engine changes the DISTRIBUTIONS of quality_score, income_score and
value_metric; every absolute threshold tuned against the peer-2 scales silently
shifts meaning. This script maps each threshold to the value that preserves its
UNIVERSE PERCENTILE: q = ECDF_old(T_old) on the old-score distribution of the
reference snapshot, then T_new = Quantile_new(q) on the peer-3 rescored copy.

Two reference variants:
  neutral — regime-neutral rescore (matches the snapshot-rescore convention;
            used for exit-side thresholds)
  tilted  — today's-regime tilt (matches the live entry path; used for
            entry-side thresholds)

POPULATION RULE (learned 2026-08-06): each threshold must be mapped on the
population its gate actually filters. ENTRY thresholds run AFTER the manager's
min_liquidity_volume pre-filter, so they map on the LIQUIDITY-ELIGIBLE subset —
full-universe mapping starved the eligible pool 225 -> 36 when peer-3 stopped
burying illiquid ADRs inside quality. EXIT floors judge held (liquid) positions;
archetype floors judge candidates post-gate — same rule.

Safety checks printed at the end:
  1. entry-gate pass count old vs new within ±10%
  2. top-12 candidate Jaccard overlap
  3. holdings sell-floor dry check — no position newly breaches the proposed
     sell floors purely from recalibration

RESEARCH-ONLY: prints a proposed YAML diff; NEVER writes config (apply via
`daily-investor config migrate-scoring` / the gated UI writer per AGENTS.md).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

SNAPSHOT = "data/snapshots/2026_08_05_00_13.parquet"
ASOF = "2026-08-05"
TOP_N = 12


def _ecdf_map(old: pd.Series, new: pd.Series, threshold: float) -> tuple[float, float]:
    """(percentile of threshold under old, matching quantile under new)."""
    old = pd.to_numeric(old, errors="coerce").dropna()
    new = pd.to_numeric(new, errors="coerce").dropna()
    q = float((old <= threshold).mean())
    return q, float(new.quantile(q))


def main() -> None:
    from data.fundamental_features import add_fundamental_features
    from strategy.scoring.composite import compute_metric
    from util import SCORE_WEIGHTS, SCORING_PARAMS

    df = pd.read_parquet(SNAPSHOT)
    old = df[[
        "symbol", "value_score", "quality_score", "income_score",
        "momentum_score", "value_metric", "dividend_yield",
    ]].copy()

    add_fundamental_features(df, ASOF)
    neutral = df.copy()
    compute_metric(neutral, SCORE_WEIGHTS, SCORING_PARAMS)              # regime-neutral

    tilted = df.copy()
    regime = None
    try:
        from strategy.regimes.detector import get_current_regime
        regime = get_current_regime()
    except Exception as exc:
        print(f"WARNING: regime detection failed ({exc}) — tilted variant = neutral")
    compute_metric(tilted, SCORE_WEIGHTS, SCORING_PARAMS, regime)
    print(f"reference: {SNAPSHOT} ({len(df)} rows), regime={regime}")

    payers = pd.to_numeric(old["dividend_yield"], errors="coerce").fillna(0) > 0

    # Entry gates map on the liquidity-eligible population (see POPULATION RULE):
    # share ADV floor AND (when configured) the dollar-volume floor.
    from util import RISK_LIMITS
    liq_mask = pd.to_numeric(df["volume"], errors="coerce").fillna(0) >= \
        RISK_LIMITS["min_liquidity_volume"]
    _min_dv = RISK_LIMITS.get("min_dollar_volume", 0.0)
    if _min_dv > 0 and "dollar_vol_21d" in df.columns:
        liq_mask &= pd.to_numeric(df["dollar_vol_21d"], errors="coerce").fillna(0) >= _min_dv
    old_liq, tilted_liq = old[liq_mask.values], tilted[liq_mask.values]

    # (label, config path, current value, old series, new series)
    rows = [
        ("metric_threshold (exit anchor)", "metric_threshold", 0.8947,
         old["value_metric"], neutral["value_metric"]),
        ("entry_threshold_override (LIQUID pop)", "candidate_selection.entry_threshold_override", 0.75,
         old_liq["value_metric"], tilted_liq["value_metric"]),
        ("fallback_thresholds[1] (LIQUID pop)", "candidate_selection.fallback_thresholds[1]", 0.72,
         old_liq["value_metric"], tilted_liq["value_metric"]),
        ("fallback_thresholds[2] (LIQUID pop)", "candidate_selection.fallback_thresholds[2]", 0.70,
         old_liq["value_metric"], tilted_liq["value_metric"]),
        ("absolute_score_floor (unused, LIQUID pop)", "candidate_selection.absolute_score_floor", 0.45,
         old_liq["value_metric"], tilted_liq["value_metric"]),
        ("cs.min_quality_score (LIQUID pop)", "candidate_selection.min_quality_score", 0.38,
         old_liq["quality_score"], tilted_liq["quality_score"]),
        ("sell_weak_value_below", "sell_rules.sell_weak_value_below", -0.0852,
         old["value_metric"], neutral["value_metric"]),
        ("sell_low_quality_below", "sell_rules.sell_low_quality_below", -0.25,
         old["quality_score"], neutral["quality_score"]),
        ("ed.hard_exit_score_below", "exit_decision.hard_exit_score_below", -0.35,
         old["value_metric"], neutral["value_metric"]),
        ("ed.strong_quality_review_floor", "exit_decision.strong_quality_review_floor", 0.70,
         old["quality_score"], neutral["quality_score"]),
        ("ed.trim_score_below", "exit_decision.trim_score_below", 0.74,
         old["value_metric"], neutral["value_metric"]),
        ("arch.defensive_income.min_income_score", "archetype_classifier.defensive_income.min_income_score", 0.30,
         old["income_score"], neutral["income_score"]),
        ("arch.defensive_income.min_income_score (payers)", "(payers-only view of the same)", 0.30,
         old.loc[payers, "income_score"], neutral.loc[payers, "income_score"]),
        ("arch.defensive_income.min_quality_score", "archetype_classifier.defensive_income.min_quality_score", 0.40,
         old["quality_score"], neutral["quality_score"]),
    ]

    print("\n=== Percentile-matched threshold proposals ===")
    print(f"{'threshold':50s} {'old':>8s} {'pctile':>8s} {'proposed':>9s}")
    proposals: dict[str, float] = {}
    for label, path, cur, old_s, new_s in rows:
        q, t_new = _ecdf_map(old_s, new_s, cur)
        proposals[path] = round(t_new, 4)
        print(f"{label:50s} {cur:8.4f} {q:8.1%} {t_new:9.4f}")

    print("\nNo-op (multiplier-anchored, inherit automatically): "
          "harvest_only_if_value_metric_below_multiplier, take_profit_value_floor_multiplier")

    # ── Safety check 1: entry-gate pass counts ────────────────────────────────
    entry_old, entry_new = 0.75, proposals["candidate_selection.entry_threshold_override"]
    n_old = int((pd.to_numeric(old["value_metric"], errors="coerce") >= entry_old).sum())
    n_new = int((pd.to_numeric(tilted["value_metric"], errors="coerce") >= entry_new).sum())
    drift = (n_new - n_old) / max(n_old, 1)
    print(f"\n[check 1] entry-gate pass count: old {n_old} -> new {n_new} ({drift:+.1%}) "
          f"{'OK' if abs(drift) <= 0.10 else 'REVIEW'}")

    # ── Safety check 2: top-N Jaccard ─────────────────────────────────────────
    top_old = set(old.nlargest(TOP_N, "value_metric")["symbol"])
    top_new = set(tilted.nlargest(TOP_N, "value_metric")["symbol"])
    jac = len(top_old & top_new) / len(top_old | top_new)
    print(f"[check 2] top-{TOP_N} Jaccard overlap old vs new: {jac:.2f}")
    print(f"          entering: {sorted(top_new - top_old)}")
    print(f"          leaving:  {sorted(top_old - top_new)}")

    # ── Safety check 3: holdings sell-floor dry check ─────────────────────────
    try:
        import glob
        hp = sorted(glob.glob("data/holdings_*.csv"))
        holdings = pd.read_csv(hp[-1]) if hp else pd.DataFrame()
        # Sell floors apply to the ACTIVE sleeve only — exclude the ETF core.
        from util import _app as _cfg
        etfs = set(_cfg.get("etfs", [])) | set(_cfg.get("harvest", {}).get("harvest_etfs", []))
        etf_types = {"etp", "cef"}
        inst = df.set_index("symbol").get("instrument_type")
        held = [
            s for s in holdings.get("symbol", pd.Series(dtype=str))
            if isinstance(s, str) and s not in etfs
            and (inst is None or s not in inst.index or str(inst.get(s)) not in etf_types)
        ]
        if held:
            merged = neutral[neutral["symbol"].isin(held)][
                ["symbol", "value_metric", "quality_score"]
            ]
            vw = proposals["sell_rules.sell_weak_value_below"]
            lq = proposals["sell_rules.sell_low_quality_below"]
            breach = merged[
                (merged["value_metric"] < vw) | (merged["quality_score"] < lq)
            ]
            old_h = old[old["symbol"].isin(held)]
            old_breach = old_h[
                (old_h["value_metric"] < -0.0852) | (old_h["quality_score"] < -0.25)
            ]
            new_only = set(breach["symbol"]) - set(old_breach["symbol"])
            print(f"[check 3] holdings below sell floors: old {len(old_breach)}, "
                  f"new {len(breach)}, NEWLY breaching: {sorted(new_only) or 'none'} "
                  f"{'OK' if not new_only else 'REVIEW'}")
        else:
            print("[check 3] no holdings CSV found — skipped")
    except Exception as exc:
        print(f"[check 3] failed: {exc}")

    print("\n--- proposed YAML values (apply via the gated writer, NEVER by hand) ---")
    for path, val in proposals.items():
        if path.startswith("("):
            continue
        print(f"{path}: {val}")


if __name__ == "__main__":
    main()
