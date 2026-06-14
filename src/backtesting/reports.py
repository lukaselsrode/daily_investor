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
        peer_rank_excluded = getattr(pool, "n_peer_relative_rank_excluded", 0) or 0
        if (
            pool.n_floor_excluded
            or pool.n_quality_gate_excluded
            or pool.n_momentum_gate_excluded
            or pool.n_income_trap_excluded
            or peer_rank_excluded
        ):
            extra = f"  peer_rank={peer_rank_excluded}" if peer_rank_excluded else ""
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
    _ct = getattr(r.train_result, "contribution_timing", None)
    if _ct:
        print("\n  CONTRIBUTION TIMING")
        print(
            f"    Contributed: ${_ct['total_contributed']:,.0f} over {_ct['weeks']} weeks"
            f"  (avg ${_ct['avg_weekly']:,.0f}, min ${_ct['min_weekly']:,.0f},"
            f" max ${_ct['max_weekly']:,.0f})"
        )
        print(
            f"    Weeks above base: {_ct['pct_weeks_above_base']:.0%}"
            f"   below base: {_ct['pct_weeks_below_base']:.0%}"
            f"   avg dip score: {_ct['avg_dip_score']:.2f}"
            f"   carry-forward end: ${_ct['final_carry_forward']:,.0f}"
        )
        _rows = _ct.get("schedule", [])[-8:]
        if _rows:
            print("    last weeks:  day | dip  | mult  | contrib | reasons")
            for _row in _rows:
                _dip = f"{_row['dip_score']:.2f}" if _row["dip_score"] == _row["dip_score"] else " n/a"
                print(
                    f"      {_row['day']:>10} | {_dip} | {_row['multiplier']:.2f}x |"
                    f" ${_row['contribution']:>6,.0f} | {','.join(_row['reason_codes'][:3])}"
                )
    if r.notes:
        print("\n  NOTES")
        for n in r.notes:
            print(f"    • {n}")
    print("=" * 64)


def _fmt_pct(x, places: int = 2) -> str:
    return "n/a" if x is None else f"{x:+.{places}%}"


def format_etf_sleeve_diagnostics(sim, label: str = "", current_weights: dict | None = None) -> str:
    """ETF SLEEVE DIAGNOSTICS block: sleeve/active/total return, ETF excess vs SPY,
    avg ETF allocation, ETF turnover, and current-vs-proposed final-weight diff.
    `sim` is a SimResult; current_weights is the incumbent's final weights for the diff."""
    lines = [f"\n  ETF SLEEVE DIAGNOSTICS{(' — ' + label) if label else ''}"]
    lines.append(f"    ETF sleeve return:     {_fmt_pct(getattr(sim, 'etf_sleeve_return', None))}")
    lines.append(f"    Active sleeve return:  {_fmt_pct(getattr(sim, 'active_total_return', None))}")
    lines.append(f"    Total return:          {_fmt_pct(getattr(sim, 'total_return', None))}")
    lines.append(f"    ETF excess vs SPY:     {_fmt_pct(getattr(sim, 'etf_excess_return', None))}")
    lines.append(f"    Active excess vs SPY:  {_fmt_pct(getattr(sim, 'active_excess_return', None))}")
    _alloc = getattr(sim, "etf_allocation_avg", None)
    lines.append(f"    Avg ETF allocation:    {('n/a' if _alloc is None else f'{_alloc:.1%}')}")
    _to = getattr(sim, "etf_turnover", None)
    lines.append(f"    ETF turnover (1-way):  {('n/a' if _to is None else f'{_to:.2f}x sleeve')}")
    fw = getattr(sim, "etf_final_weights", None) or {}
    if fw:
        lines.append("    Final ETF weights (proposed vs current):")
        for etf in sorted(fw, key=lambda k: -fw[k]):
            cur = (current_weights or {}).get(etf)
            cur_s = "—" if cur is None else f"{cur:.1%}"
            lines.append(f"      {etf:<6} {fw[etf]:>6.1%}   (cur {cur_s})")
    return "\n".join(lines)


def format_etf_regime_table(rows: list[dict]) -> str:
    """Regime allocation table. Each row: regime, weights(dict), etf_return, spy_return,
    excess, max_dd, etf_turnover."""
    out = [
        "\n  ETF ALLOCATION BY REGIME",
        f"    {'regime':<10} {'ETF return':>11} {'SPY':>9} {'excess':>9} {'maxDD':>8} {'turnover':>9}",
        "    " + "-" * 60,
    ]
    for r in rows:
        _to = r.get("etf_turnover")
        _to_s = "n/a" if _to is None else f"{_to:.2f}x"
        out.append(
            f"    {r.get('regime',''):<10} {_fmt_pct(r.get('etf_return')):>11} "
            f"{_fmt_pct(r.get('spy_return')):>9} {_fmt_pct(r.get('excess')):>9} "
            f"{_fmt_pct(r.get('max_dd')):>8} {_to_s:>9}"
        )
        w = r.get("weights") or {}
        if w:
            out.append("      weights: " + ", ".join(
                f"{k} {v:.0%}" for k, v in sorted(w.items(), key=lambda kv: -kv[1]) if v > 0.005
            ))
    return "\n".join(out)


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
