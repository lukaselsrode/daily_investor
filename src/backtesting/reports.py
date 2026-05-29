"""
backtesting/reports.py — Human-readable report formatting for BacktestReport.
"""

from __future__ import annotations

from util import CANDIDATE_SELECTION_PARAMS

from .types import BacktestReport


def print_backtest_report(report: BacktestReport) -> None:
    """Print a formatted BacktestReport to stdout."""
    r  = report
    tr = r.train_result
    print(f"\n{'=' * 64}")
    print(f"BACKTEST REPORT  [{r.mode}  sel={r.universe_selection}  bias={r.lookahead_bias_level}]")
    print(f"{'=' * 64}")
    print(f"  Universe: {r.n_symbols} symbols, {r.n_days} trading days")

    pc = r.peer_config or {}
    blend = pc.get("blend", {}) or {}
    engine = r.scoring_engine_version or "peer-1"
    print(
        f"  Scoring engine: {engine}"
        f" (industry={blend.get('industry_relative', 0.0):.2f}"
        f" / sector={blend.get('sector_relative', 0.0):.2f}"
        f" / market={blend.get('market_relative', 0.0):.2f}"
        f", min_group={pc.get('min_group_size', '?')},"
        f" method={pc.get('method', 'percentile')})"
    )

    pool = tr.pool_diagnostics
    if pool is not None:
        print(f"\n  CANDIDATE POOL  (selection mode: {CANDIDATE_SELECTION_PARAMS['mode']})")
        print(f"    Candidates:    {pool.n_candidates}  (cutoff={pool.score_cutoff:.3f})")
        print(
            f"    Avg factors:   quality={pool.avg_quality:.2f}  momentum={pool.avg_momentum:.2f}  "
            f"income={pool.avg_income:.2f}  value={pool.avg_value:.2f}"
        )
        v3_extra = getattr(pool, "n_peer_relative_rank_excluded", 0) or 0
        if (
            pool.n_floor_excluded
            or pool.n_quality_gate_excluded
            or pool.n_momentum_gate_excluded
            or pool.n_income_trap_excluded
            or v3_extra
        ):
            extra = f"  peer_rank={v3_extra}" if v3_extra else ""
            print(
                f"    Gates excluded: floor={pool.n_floor_excluded}  "
                f"quality={pool.n_quality_gate_excluded}  "
                f"momentum={pool.n_momentum_gate_excluded}  "
                f"income_trap={pool.n_income_trap_excluded}"
                + extra
            )
        if pool.excluded_high_income_low_momentum:
            print(f"    Income-trap excluded: {', '.join(pool.excluded_high_income_low_momentum)}")
        peer_fb = getattr(pool, "peer_fallback_usage", None)
        if peer_fb:
            top_fb = sorted(peer_fb.items(), key=lambda kv: -kv[1])[:4]
            print("    Peer fallbacks: " + "  ".join(f"{k}={v}" for k, v in top_fb))
        if pool.sector_counts:
            top = sorted(pool.sector_counts.items(), key=lambda x: -x[1])[:6]
            print("    Sectors:       " + "  ".join(f"{s}={c}" for s, c in top))

    print("\n  TRAIN WINDOW")
    print(f"    Return (TWR):    {tr.total_return:+.2%}  (bench TWR {r.train_benchmark_twr:+.2%})")
    print(f"    Benchmark (buy-hold):  {r.benchmark_return:+.2%}")
    print(f"    Excess return:   {r.excess_return:+.2%}")
    print(f"    Sharpe:          {tr.sharpe:+.3f}  (benchmark {r.benchmark_sharpe:+.3f})")
    print(f"    Calmar:          {tr.calmar:+.3f}")
    print(f"    Max drawdown:    {tr.max_drawdown:.2%}  (benchmark {r.benchmark_max_drawdown:.2%})")
    print(
        f"    Final value:     ${tr.final_value:,.2f}  "
        f"contributions=${tr.net_contributions:,.2f}  profit=${tr.profit:,.2f}"
    )
    print(f"    Trades:          {tr.trades_made}  sells={tr.sells_made}  skipped={tr.skipped_buys}")
    print(f"    Stopouts:        {tr.stopout_count}  cooldown skips={tr.cooldown_skips}")
    print(f"    Cap reductions:  {tr.cap_reductions}")
    print(f"    Avg positions:   {tr.average_positions:.1f}  max={tr.max_positions}")
    print(f"    Avg cash %:      {tr.average_cash_pct:.1%}")
    print(f"    Friction cost:   ${tr.friction_cost:.2f}  turnover={tr.turnover_estimate:.4f}")
    if tr.regime_days:
        rd       = tr.regime_days
        total_rd = max(sum(rd.values()), 1)
        print(
            f"    Regime days:     bullish={rd['bullish']} ({rd['bullish']/total_rd:.0%})  "
            f"neutral={rd['neutral']} ({rd['neutral']/total_rd:.0%})  "
            f"defensive={rd['defensive']} ({rd['defensive']/total_rd:.0%})"
        )
    if r.validation_result:
        vr = r.validation_result
        print("\n  VALIDATION WINDOW")
        print(f"    Return (TWR):    {vr.total_return:+.2%}  (bench TWR {r.val_benchmark_twr:+.2%})")
        print(f"    Benchmark:       {r.validation_benchmark_return:+.2%}")
        print(f"    Sharpe:          {vr.sharpe:+.3f}")
        print(f"    Calmar:          {vr.calmar:+.3f}")
        print(f"    Max drawdown:    {vr.max_drawdown:.2%}")
        print(f"    Stopouts:        {vr.stopout_count}  cooldown skips={vr.cooldown_skips}")
        if vr.regime_days:
            rd       = vr.regime_days
            total_rd = max(sum(rd.values()), 1)
            print(
                f"    Regime days:     bullish={rd['bullish']} ({rd['bullish']/total_rd:.0%})  "
                f"neutral={rd['neutral']} ({rd['neutral']/total_rd:.0%})  "
                f"defensive={rd['defensive']} ({rd['defensive']/total_rd:.0%})"
            )
    if r.notes:
        print("\n  NOTES")
        for n in r.notes:
            print(f"    • {n}")
    print("=" * 64)


def print_comparison_report(comparison: dict) -> None:
    """Print a formatted A/B/C candidate-selection-mode comparison table."""
    bench  = comparison.get("_benchmark_return", 0.0)
    n_days = comparison.get("_n_days", "?")
    print(f"\n{'=' * 72}")
    print(f"CANDIDATE SELECTION MODE COMPARISON  ({n_days}d train)  bench={bench:+.1%}")
    print(f"{'=' * 72}")
    hdr = f"  {'Mode':<22}  {'Return':>8}  {'Sharpe':>7}  {'DD':>7}  {'Trades':>7}  {'Cands':>6}  {'AvgMom':>7}"
    print(hdr)
    print("  " + "-" * 68)
    labels = {"A_absolute": "A absolute", "B_percentile": "B percentile", "C_percentile_gates": "C +factor gates"}
    for key, name in labels.items():
        if key not in comparison:
            continue
        s = comparison[key]["sim"]
        p = comparison[key]["pool"]
        print(
            f"  {name:<22}  {s.total_return:>+8.2%}  {s.sharpe:>+7.3f}  "
            f"{s.max_drawdown:>7.2%}  {s.trades_made:>7d}  {p.n_candidates:>6d}  {p.avg_momentum:>+7.2f}"
        )
    print("=" * 72)
