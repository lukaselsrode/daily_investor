"""
scripts/apply_peer3_config.py — one-shot surgical peer-3 cutover for cfg/config.yaml.

Exact-string block replacements (the apply_etf_allocation_params precedent): the rest
of the file — including every comment — is left byte-for-byte untouched. Creates
cfg/config.yaml.pre_peer3.bak first and REFUSES to run twice (idempotent by marker).

What it applies (validated 2026-08-05):
  - scoring: quality_components → the 7 fundamentals components; quality_liquidity /
    quality_analyst / quality_checklist removed; quality_fundamentals + income_inputs
    added; factors.quality anchor_blend 0.7 → 0.0 (checklist anchor retired).
  - Percentile-matched thresholds (scripts/recalibrate_peer3_thresholds.py output).
  - scoring.momentum_residual_blend added to tuning.frozen_parameters (sim-only slot).
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

CONFIG = Path("cfg/config.yaml")
MARKER = "peer-3 cutover 2026-08-05"

REPLACEMENTS: list[tuple[str, str]] = [
    # ── exit-ladder anchor ────────────────────────────────────────────────────
    (
        "metric_threshold: 0.8947\n",
        "# 2026-08-05 peer-3 cutover: 0.8947 -> 0.716, percentile-matched (99.5th pct\n"
        "# of the universe composite) to the peer-3 score distribution.\n"
        "metric_threshold: 0.716\n",
    ),
    # ── sell rules ────────────────────────────────────────────────────────────
    (
        "  sell_weak_value_below: -0.0852\n",
        "  # 2026-08-05 peer-3 cutover: -0.0852 -> -0.048 (percentile-matched, 28th pct).\n"
        "  sell_weak_value_below: -0.048\n",
    ),
    (
        "  sell_low_quality_below: -0.25\n",
        "  # 2026-08-05 peer-3 cutover: -0.25 -> -0.288 (percentile-matched, 3.8th pct\n"
        "  # of the fundamentals-based quality distribution).\n"
        "  sell_low_quality_below: -0.288\n",
    ),
    # ── exit decision floors ──────────────────────────────────────────────────
    (
        "  hard_exit_score_below: -0.35\n",
        "  # 2026-08-05 peer-3 cutover: -0.35 -> -0.149 (percentile-matched; still\n"
        "  # below sell_weak_value_below -0.048, preserving the ladder ordering).\n"
        "  hard_exit_score_below: -0.149\n",
    ),
    (
        "  review_score_below: 0.45\n",
        "  # 2026-08-05 peer-3 cutover: 0.45 -> 0.236 (percentile-matched, 69th pct).\n"
        "  review_score_below: 0.236\n",
    ),
    (
        "  strong_quality_review_floor: 0.7\n",
        "  # 2026-08-05 peer-3 cutover: 0.7 -> 0.463 (percentile-matched, 77th pct).\n"
        "  strong_quality_review_floor: 0.463\n",
    ),
    (
        "  trim_score_below: 0.74\n",
        "  # 2026-08-05 peer-3 cutover: 0.74 -> 0.515 (percentile-matched, 94th pct).\n"
        "  trim_score_below: 0.515\n",
    ),
    # ── candidate selection ───────────────────────────────────────────────────
    (
        "  absolute_score_floor: 0.45\n",
        "  absolute_score_floor: 0.339  # peer-3 percentile-matched (floor unused: use_absolute_score_floor false)\n",
    ),
    (
        "  min_quality_score: 0.38\n",
        "  # 2026-08-05 peer-3 cutover: 0.38 -> 0.135 (percentile-matched, 47th pct of\n"
        "  # the fundamentals-based quality distribution).\n"
        "  min_quality_score: 0.135\n",
    ),
    (
        "  entry_threshold_override: 0.75\n"
        "  fallback_thresholds:\n"
        "  - 0.75\n"
        "  - 0.72\n"
        "  - 0.7\n",
        "  # 2026-08-05 peer-3 cutover: 0.75/0.72/0.70 -> 0.623/0.596/0.575 —\n"
        "  # percentile-matched on the regime-tilted (entry-path) distribution;\n"
        "  # entry-gate pass count preserved within +10%.\n"
        "  entry_threshold_override: 0.623\n"
        "  fallback_thresholds:\n"
        "  - 0.623\n"
        "  - 0.596\n"
        "  - 0.575\n",
    ),
    # ── archetype classifier floors ───────────────────────────────────────────
    (
        "    min_income_score: 0.30\n"
        "    min_quality_score: 0.40\n",
        "    # 2026-08-05 peer-3 cutover: income 0.30 -> 0.05 (percentile mapping is\n"
        "    # degenerate at the non-payer point mass; any positive floor expresses the\n"
        "    # old gate's 'is a ranked payer' semantic), quality 0.40 -> 0.163 (50th pct).\n"
        "    min_income_score: 0.05\n"
        "    min_quality_score: 0.163\n",
    ),
    # ── scoring: quality blocks ───────────────────────────────────────────────
    (
        "  # ── peer-2 quality liquidity (2026-07-03) ─────────────────────────────────\n"
        "  # Quality's liquidity input moved from raw share-ADV to multi-horizon DOLLAR\n"
        "  # volume (close×volume from the FMP price cache) + consistency (low CV of daily\n"
        "  # dollar volume = stable institutional participation). Share volume favored\n"
        "  # cheap low-priced names and double-counted the hard 500k-share gates.\n"
        "  quality_liquidity:\n"
        "    horizon_weights:\n"
        "      dv_5d: 0.20\n"
        "      dv_21d: 0.30\n"
        "      dv_63d: 0.50\n"
        "    # Components whose input column is (near-)absent in a frame are dropped and\n"
        "    # the remaining component weights renormalize (old snapshot vintages).\n"
        "    min_coverage: 0.30\n"
        "  quality_analyst:\n"
        "    min_num_ratings: 5\n"
        "  # Peer-quality component weights (config-fixed, NOT tuner slots — 9 correlated\n"
        "  # DOF on a ~160-snapshot substrate is pure overfit surface). dollar_volume +\n"
        "  # volume_consistency together carry the 0.20 mass share-ADV held in peer-1.\n"
        "  quality_components:\n"
        "    dollar_volume: 0.12\n"
        "    volume_consistency: 0.08\n"
        "    has_positive_pe: 0.20\n"
        "    has_positive_pb: 0.10\n"
        "    no_distress_pe: 0.20\n"
        "    healthy_yield: 0.10\n"
        "    position_52w: 0.05\n"
        "    market_cap: 0.05\n"
        "    analyst_conviction: 0.05\n",
        "  # ── peer-3 quality (2026-08-05) ───────────────────────────────────────────\n"
        "  # Quality rebuilt on real statement fundamentals (data.fundamental_features,\n"
        "  # PIT from the FMP cache). Dividend terms moved out (income owns dividends —\n"
        "  # they drove corr(quality, income) 0.53), PE/PB terms moved out (value owns\n"
        "  # valuation + distress), liquidity/size moved out (the reliability layer and\n"
        "  # hard volume gates own tradability), position_52w moved out (momentum owns\n"
        "  # price geometry), analyst_conviction removed (never cleared its coverage\n"
        "  # gate). Weights config-fixed, NOT tuner slots (overfit surface).\n"
        "  quality_fundamentals:\n"
        "    # A component whose column is (near-)absent in a frame drops and the\n"
        "    # remaining weights renormalize (old snapshot vintages).\n"
        "    min_coverage: 0.30\n"
        "  quality_components:\n"
        "    roe_ttm: 0.22\n"
        "    fcf_to_assets: 0.20\n"
        "    neg_accruals: 0.15\n"
        "    gross_margin_ttm: 0.12\n"
        "    low_leverage: 0.12\n"
        "    share_count_shrink_yoy: 0.10\n"
        "    gm_trend_yoy: 0.09\n",
    ),
    # ── factors.quality: retire the checklist anchor ──────────────────────────
    (
        "    quality:\n"
        "      enabled: true\n"
        "      peer_relative: true\n"
        "      use_legacy_checklist_fallback: true\n"
        "      anchor_blend: 0.7\n",
        "    quality:\n"
        "      enabled: true\n"
        "      peer_relative: true\n"
        "      # 2026-08-05 peer-3: the legacy checklist anchor (0.7 weight on a 4-input\n"
        "      # PE/PB/volume/dividend checklist) is retired — it carried the dividend\n"
        "      # and valuation overlap the peer-3 rework removes.\n"
        "      anchor_blend: 0.0\n",
    ),
    # ── quality_checklist → income_inputs ─────────────────────────────────────
    (
        "  quality_checklist:\n"
        "    income_score_cap: 1.5\n"
        "    yield_trap_threshold: 0.1\n"
        "    distress_pe_max: 5.0\n"
        "    quality_volume_high: 1000000\n"
        "    quality_volume_low: 100000\n"
        "    quality_dividend_min: 0.02\n"
        "    quality_dividend_max: 0.06\n"
        "    quality_weight_has_positive_pe: 0.5\n"
        "    quality_weight_distress_pe: -0.4\n"
        "    quality_weight_has_positive_pb: 0.2\n"
        "    quality_weight_high_volume: 0.3\n"
        "    quality_weight_low_volume: -0.3\n"
        "    quality_weight_yield_trap: -0.6\n"
        "    quality_weight_healthy_dividend: 0.2\n",
        "  # ── peer-3 income (2026-08-05) ────────────────────────────────────────────\n"
        "  # Income de-saturated: the max(peer_rank, yield/threshold) cap that pinned\n"
        "  # every yield >= 4.5% at 1.500 (305 names) is gone; yield is weighted with\n"
        "  # FMP payout sustainability (coverage/growth/streak). Yield-trap semantics\n"
        "  # unchanged (moved here from the retired quality_checklist block).\n"
        "  income_inputs:\n"
        "    yield_trap_threshold: 0.1\n"
        "    weights:\n"
        "      dividend_yield: 0.50\n"
        "      div_fcf_coverage: 0.20\n"
        "      div_growth: 0.15\n"
        "      div_streak: 0.15\n",
    ),
    # ── freeze the sim-only residual-momentum slot ────────────────────────────
    (
        "  # frozen_parameters intentionally empty: presets define the tunable surface\n"
        "  # per-run, and OOS validation gates catch overfitting. A no-preset tune now\n"
        "  # optimizes the full base param space (slots 0-15); archetype/regime/cs/sizing\n"
        "  # tail slots remain frozen-by-default until their preset unfreezes them.\n"
        "  frozen_parameters: []\n",
        "  # frozen_parameters near-empty: presets define the tunable surface per-run,\n"
        "  # and OOS validation gates catch overfitting. momentum_residual_blend is\n"
        "  # PERMANENTLY frozen: slot 49 is implemented only in the simulator — a tuned\n"
        "  # nonzero value would change sim results but never live scoring.\n"
        "  frozen_parameters:\n"
        "  - scoring.momentum_residual_blend\n",
    ),
]


def main() -> None:
    text = CONFIG.read_text()
    if MARKER in text:
        print("config.yaml already carries the peer-3 cutover — nothing to do")
        return

    bak = CONFIG.with_suffix(".yaml.pre_peer3.bak")
    if not bak.exists():
        shutil.copy2(CONFIG, bak)
        print(f"backup: {bak}")

    for old, new in REPLACEMENTS:
        count = text.count(old)
        if count != 1:
            print(f"ABORT: expected exactly 1 occurrence, found {count}:\n---\n{old[:200]}...\n---")
            sys.exit(1)
        text = text.replace(old, new)

    text = f"# {MARKER} — scoring model peer-3 + percentile-matched thresholds applied.\n" + text
    CONFIG.write_text(text)
    print(f"applied {len(REPLACEMENTS)} surgical replacements to {CONFIG}")


if __name__ == "__main__":
    main()
