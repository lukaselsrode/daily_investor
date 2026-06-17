"""
main.py — Daily investment strategy entry point.

Responsibilities:
  - Robinhood login
  - Fund top-up
  - Orchestrate PortfolioManager.rebalance()
"""

import datetime
import logging
import os
import sys

from dotenv import load_dotenv

from data.market import get_data as generate_daily_undervalued_stocks
from execution.robinhood import RobinhoodBroker
from portfolio.harvest import HarvestManager
from portfolio.manager import PortfolioManager
from portfolio.risk import RiskManager
from strategy.regimes.detector import get_current_regime
from util import (
    AUTO_APPROVE,
    CONFIDENCE_THRESHOLD,
    DATA_DIRECTORY,
    DIVIDEND_PARAMS,
    METRIC_THRESHOLD,
    USE_SENTIMENT_ANALYSIS,
    WEEKLY_INVESTMENT,
    store_data_as_csv,
    update_industry_valuations,
)

load_dotenv()

_broker  = RobinhoodBroker()
_risk    = RiskManager()
_harvest = HarvestManager()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("investment_bot.log"),
    ],
)
logger = logging.getLogger("investment_bot")

_DIVIDEND_HISTORY_CSV = os.path.join(DATA_DIRECTORY, "dividend_history.csv")

_HOLDINGS_SCHEMA = [
    "symbol", "name", "quantity", "average_buy_price", "equity",
    "percent_change", "equity_change", "percentage", "current_price",
    "type", "pe_ratio", "id",
]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def login() -> None:
    _broker.login(os.getenv("RB_ACCT"), os.getenv("RB_CREDS"), os.getenv("RB_MFA_SECRET"))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def confirm(prompt: str) -> bool:
    if AUTO_APPROVE:
        logger.info(f"AUTO-APPROVED: {prompt}")
        return True
    return input(f"{prompt} [y/n] ").strip().lower() in ("y", "yes")


def save_holdings_csv(holdings: dict) -> None:
    if not holdings:
        return
    rows = [[symbol] + [d.get(k, "") for k in _HOLDINGS_SCHEMA[1:]]
            for symbol, d in holdings.items()]
    try:
        store_data_as_csv("holdings", _HOLDINGS_SCHEMA, rows)
        logger.info(f"Saved holdings CSV: {len(rows)} positions")
    except Exception as e:
        logger.warning(f"Could not save holdings CSV: {e}")


def add_funds_to_account(target_amount: float | None = None) -> None:
    """Deposit up to the weekly target. `target_amount` overrides WEEKLY_INVESTMENT
    when the contribution-timing overlay recommends a different amount this week —
    the existing confirmation prompt is unchanged (no new automatic money movement)."""
    target = WEEKLY_INVESTMENT if target_amount is None else float(target_amount)
    available = _broker.get_cash()
    if available >= target:
        logger.info(f"Sufficient cash (${available:,.2f} ≥ ${target:,.2f}) — no deposit needed")
        return
    needed = target - available
    resp = input(
        f"Cash ${available:,.2f} < target ${target:,.2f}. Deposit ${needed:,.2f}? [y/n] "
    ).strip().lower()
    if resp not in ("y", "yes"):
        return
    _broker.add_funds(needed)


_CONTRIBUTION_LOG_CSV = os.path.join(DATA_DIRECTORY, "contribution_timing_log.csv")


def _contribution_timing_recommendation() -> float | None:
    """Compute, display, and persist this week's contribution recommendation.
    Returns the adjusted weekly amount, or None when the overlay is disabled or
    the signal could not be computed (caller falls back to WEEKLY_INVESTMENT)."""
    from util import CONTRIBUTION_TIMING_PARAMS as _ct
    if not _ct.get("enabled", False):
        return None
    try:
        import yfinance as yf

        from portfolio.contribution_timing import (
            decide_contribution,
            format_live_panel,
            load_live_state,
            record_live_decision,
        )
        hist = yf.download(
            _ct.get("benchmark_symbol", "SPY"), period="400d",
            auto_adjust=True, progress=False,
        )
        closes = hist["Close"]
        if hasattr(closes, "squeeze"):
            closes = closes.squeeze()
        closes = closes.dropna().to_numpy(dtype=float)
        if len(closes) == 0:
            logger.warning("Contribution timing: no benchmark history — using base amount")
            return None
        try:
            from strategy.regimes.detector import get_current_regime
            regime = get_current_regime()
        except Exception:
            regime = None
        state = load_live_state(_CONTRIBUTION_LOG_CSV, _ct)
        decision = decide_contribution(closes, _ct, state, regime=regime)
        logger.info("\n%s", format_live_panel(decision, _ct))
        record_live_decision(_CONTRIBUTION_LOG_CSV, decision)
        return decision.adjusted_amount
    except Exception as exc:
        logger.warning("Contribution timing recommendation failed (using base amount): %s", exc)
        return None


def _fetch_and_save_dividends() -> None:
    if not DIVIDEND_PARAMS.get("enabled") or not DIVIDEND_PARAMS.get("track_income"):
        return
    try:
        df = _broker.get_dividends()
        if df.empty:
            logger.info("No dividend history returned")
            return
        df.to_csv(_DIVIDEND_HISTORY_CSV, index=False)
        paid = df[df["state"] == "paid"]["amount"].sum()
        logger.info(f"Dividend history saved: {len(df)} records | total paid=${paid:.2f}")
    except Exception as e:
        logger.warning(f"Could not fetch/save dividends: {e}")


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

def _maybe_fill_outcomes() -> None:
    """Backfill realized return outcomes for past decisions if config flag is set."""
    import yaml

    from core.paths import CONFIG_FILE
    try:
        with open(CONFIG_FILE) as f:
            cfg = yaml.safe_load(f) or {}
        if not cfg.get("outcome_tracking", {}).get("fill_returns_on_run", False):
            return
    except Exception:
        return

    try:
        import datetime

        import yfinance as yf

        from portfolio.outcome_tracker import fill_future_returns, load_outcomes

        df = load_outcomes()
        if df.empty:
            return

        sym_col = "symbol" if "symbol" in df.columns else "ticker"
        symbols = [s for s in df[sym_col].dropna().unique() if str(s).strip()]
        if not symbols:
            return

        fetch_syms = sorted(set(symbols) | {"SPY"})
        # yfinance uses "-" for share classes where Robinhood uses "."
        # (PBR.A -> PBR-A); fetch under the yf alias, key results back to ours.
        yf_alias = {s: s.replace(".", "-") for s in fetch_syms}
        today  = datetime.date.today()
        start  = (today - datetime.timedelta(days=125)).isoformat()
        hist   = yf.download(sorted(set(yf_alias.values())), start=start, auto_adjust=True, progress=False)
        close  = hist["Close"] if "Close" in hist.columns else hist

        current_prices: dict[str, float] = {}
        spy_price_history: dict[str, float] = {}
        spy_current_price: float | None = None

        for sym in fetch_syms:
            alias = yf_alias[sym]
            col = close[alias] if alias in close.columns else None
            if col is not None and not col.dropna().empty:
                current_prices[sym] = float(col.dropna().iloc[-1])

        if "SPY" in close.columns:
            spy_series = close["SPY"].dropna()
            spy_current_price = float(spy_series.iloc[-1])
            for ts, px in spy_series.items():
                spy_price_history[str(ts)[:10]] = float(px)

        n = fill_future_returns(
            current_prices=current_prices,
            spy_current_price=spy_current_price,
            spy_price_history=spy_price_history,
        )
        if n:
            logger.info("Outcome backfill: %d cells updated", n)
    except Exception as exc:
        logger.warning("Outcome backfill failed (non-fatal): %s", exc)


def run_daily_strat() -> None:
    logger.info(f"=== Daily Investment Strategy {datetime.datetime.now():%Y-%m-%d %H:%M} ===")
    if USE_SENTIMENT_ANALYSIS:
        logger.info(f"Sentiment ON | METRIC_THRESHOLD={METRIC_THRESHOLD} | CONFIDENCE={CONFIDENCE_THRESHOLD}%")

    if not AUTO_APPROVE and not confirm("Generate new picks and run strategy?"):
        logger.info("Cancelled")
        return

    try:
        # Contribution-timing overlay: weekly recommendation panel (and the
        # deposit target when a deposit is part of this run). None → flat base.
        recommended_contribution = _contribution_timing_recommendation()

        skip_data = "--skip-data" in sys.argv
        if not skip_data:
            update_industry_valuations(verbose=True)
            add_funds_to_account(target_amount=recommended_contribution)
            _fetch_and_save_dividends()
            refresh = AUTO_APPROVE or confirm("Generate fresh data? (takes several minutes)")
        else:
            logger.info("--skip-data: using existing CSVs")
            refresh = False
        df = generate_daily_undervalued_stocks(refresh=refresh)
    except Exception as e:
        logger.error(f"Strategy setup failed: {e}")
        if not AUTO_APPROVE:
            input("Press Enter to exit...")
        return

    cash = _broker.get_cash()
    if cash <= 0:
        logger.error("No funds available — aborting strategy")
        if not AUTO_APPROVE:
            input("Press Enter to exit...")
        return
    if cash < WEEKLY_INVESTMENT:
        logger.info(f"Proceeding with ${cash:,.2f} available (below ${WEEKLY_INVESTMENT:,.2f} weekly target)")

    regime = get_current_regime()
    pm = PortfolioManager(_broker, _risk, _harvest)
    pm.rebalance(df, regime)

    _maybe_fill_outcomes()


# ---------------------------------------------------------------------------
# Tuner CLI
# ---------------------------------------------------------------------------

def _run_tuner_cli(n_days: int, objective: str) -> None:
    from tuning.reports import print_config_diff
    from tuning.tuner import run_tuner
    try:
        best_params, best_result = run_tuner(
            n_days=n_days,
            objective=objective,
            starting_capital=10_000.0,
        )
        print_config_diff(best_params, best_result)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)


def _run_stability_scan_cli(
    windows: "list[int] | None" = None,
    mode: "str | None" = None,
    output_dir: "str | None" = None,
) -> None:
    """CLI entry point for --stability-scan. RESEARCH / DIAGNOSTIC ONLY — never writes config.yaml."""
    from tuning.stability import run_stability_scan
    try:
        run_stability_scan(windows=windows, mode=mode, output_dir=output_dir)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)


def _run_auto_tune_cli(
    n_days: int,
    mode: "str | None" = None,
    apply: bool = False,
    force_apply: bool = False,
    llm_review: bool = False,
) -> None:
    from tuning.reports import _diff_table
    from tuning.tuner import run_auto_tune
    try:
        avg_params, sharpe_result, calmar_result, avg_result, sharpe_params, calmar_params = run_auto_tune(
            n_days=n_days,
            starting_capital=10_000.0,
            mode=mode,
            apply=apply,
            force_apply=force_apply,
            llm_review=llm_review,
        )
        _diff_table(
            avg_params,
            label=f"mean of Sharpe + Calmar over {n_days}d",
            sharpe_ref=sharpe_result,
            calmar_ref=calmar_result,
            sharpe_params=sharpe_params,
            calmar_params=calmar_params,
        )
        print(
            f"\nAveraged result:  ret={avg_result.total_return:+.1%}  "
            f"sharpe={avg_result.sharpe:+.3f}  "
            f"calmar={avg_result.calmar:+.3f}  "
            f"trades={avg_result.trades_made}"
        )
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _raise_fd_limit() -> None:
    """Raise the open-file-descriptor soft limit to avoid EMFILE during bulk yfinance downloads."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 4096)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.debug(f"Raised RLIMIT_NOFILE: {soft} → {target} (hard={hard})")
    except Exception as e:
        logger.debug(f"Could not raise fd limit: {e}")


def main() -> None:
    _raise_fd_limit()
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        print("""
daily_investor — Robinhood systematic investment bot
  Default (no flags): login and run the live trading strategy.

USAGE
  python main.py [OPTIONS]

DATA OPTIONS
  --skip-data          Reuse existing agg_data.csv instead of fetching fresh fundamentals.
                       Useful for repeated tuning runs without re-downloading data.
  --skip-fetch-news    Refresh all other data but reuse the latest cached news scrape
                       (skips the slowest stage, any age). 0DTE options sentiment is
                       unaffected. NEWS_FORCE_REFETCH=1 overrides the default 8h reuse.

BACKTEST / TUNING OPTIONS
  --tune DAYS          Single-objective back-simulation over DAYS trading days.
                       Prints a suggested config diff — does NOT write config.yaml.
                       Requires --objective (default: sharpe).
                       Example: python main.py --tune 120 --objective calmar

  --auto-tune [DAYS]   Dual-objective run (Sharpe + Calmar averaged), with a 70/30
                       train/validation split and out-of-sample validation gates.
                       Writes config.yaml only if --apply or --force-apply is passed
                       AND validation gates pass.
                       Default: 90 trading days.
                       Example: python main.py --auto-tune 180 --apply

  --objective METRIC   Optimization target for --tune: sharpe (default) or calmar.

  --mode MODE          Backtest universe selection. Controls lookahead-bias level.
                       All modes span the FULL liquid universe (max_symbols=0) — breadth is the edge:
                         liquid_universe_full   — full liquid universe, deterministic   [MEDIUM bias, default]
                         current_universe_stress_test  — full universe ranked by current score [HIGH bias, not predictive]
                         walk_forward_price_only_test  — full universe, volume filter only      [LOW bias]

  --stability-scan     Run the optimizer across all configured windows and objectives, then
                       generate stability heatmaps, CSV summaries, and a human-readable
                       robustness report. RESEARCH/DIAGNOSTIC ONLY — never writes config.yaml.
                       Output: reports/stability/ (configurable via stability.output_dir in config)
                       Optional: --mode MODE, --output-dir PATH
                       Requires matplotlib for heatmaps: pip install matplotlib
                       Relevant config keys: stability.windows, stability.objectives,
                         stability.unstable_cv_threshold, stability.unstable_spread_threshold

CONFIG WRITE OPTIONS (auto-tune only)
  --apply              Write config.yaml if out-of-sample validation gates pass.
  --force-apply        Write config.yaml unconditionally (bypasses validation — use with care).
  --llm-review         After optimization, send all three candidates (sharpe-opt, calmar-opt,
                       averaged) to Claude for a second-opinion review. The model, whether its
                       adjustments are applied, and the top-N reviewed are set in config.yaml:
                         backtest.llm_review_model   (default: claude-sonnet-4-6)
                         backtest.llm_review_apply   (default: false — review is advisory only)
                         backtest.llm_review_enabled (default: false — set true to always review)
                       Requires ANTHROPIC_API_KEY env var.

LIVE TRADING OPTIONS
  --op-mode MODE       Override auto_approve and use_sentiment_analysis for this run only.
                       Does NOT write config.yaml — takes effect immediately, resets on next run.
                         safe          auto_approve=false  use_sentiment=true   (manual confirm every trade)
                         automated     auto_approve=true   use_sentiment=true   (fully hands-off)
                         no-sentiment  auto_approve=false  use_sentiment=false  (value_metric only, no Claude)
                       Example: python main.py --op-mode safe

OTHER
  -h, --help           Show this message and exit.

VALIDATION GATES (auto-tune)
  Tuned params are only written if the validation window (held-out 30%) satisfies:
    • excess return  ≥ min_validation_excess_return   (cfg: backtest.min_validation_excess_return)
    • max drawdown   ≥ max_validation_drawdown         (cfg: backtest.max_validation_drawdown)
    • Sharpe ratio   ≥ min_validation_sharpe           (cfg: backtest.min_validation_sharpe)
""")
        return

    if "--auto-tune" in args:
        idx = args.index("--auto-tune")
        n_days = 90
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            n_days = int(args[idx + 1])
        mode = None
        if "--mode" in args:
            mi = args.index("--mode")
            if mi + 1 < len(args):
                mode = args[mi + 1]
        apply = "--apply" in args
        force_apply = "--force-apply" in args
        llm_review = "--llm-review" in args
        _run_auto_tune_cli(n_days, mode=mode, apply=apply, force_apply=force_apply, llm_review=llm_review)
        return

    if "--tune" in args:
        idx = args.index("--tune")
        try:
            n_days = int(args[idx + 1])
        except (IndexError, ValueError):
            print("--tune requires an integer argument, e.g. --tune 90")
            sys.exit(1)
        objective = "sharpe"
        if "--objective" in args:
            oi = args.index("--objective")
            try:
                objective = args[oi + 1].lower()
                if objective not in ("sharpe", "calmar"):
                    raise ValueError
            except (IndexError, ValueError):
                print("--objective must be 'sharpe' or 'calmar'")
                sys.exit(1)
        _run_tuner_cli(n_days, objective)
        return

    if "--stability-scan" in args:
        mode = None
        if "--mode" in args:
            mi = args.index("--mode")
            if mi + 1 < len(args):
                mode = args[mi + 1]
        out_dir = None
        if "--output-dir" in args:
            oi = args.index("--output-dir")
            if oi + 1 < len(args):
                out_dir = args[oi + 1]
        _run_stability_scan_cli(mode=mode, output_dir=out_dir)
        return

    if "--op-mode" in args:
        oi = args.index("--op-mode")
        if oi + 1 >= len(args):
            print("--op-mode requires an argument: safe | automated | no-sentiment")
            sys.exit(1)
        _apply_op_mode(args[oi + 1])

    login()
    run_daily_strat()


_OP_MODES = {
    "safe":         {"auto_approve": False, "use_sentiment_analysis": True},
    "automated":    {"auto_approve": True,  "use_sentiment_analysis": True},
    "no-sentiment": {"auto_approve": False, "use_sentiment_analysis": False},
}


def _apply_op_mode(mode: str) -> None:
    """Override AUTO_APPROVE / USE_SENTIMENT_ANALYSIS for this run only."""
    global AUTO_APPROVE, USE_SENTIMENT_ANALYSIS
    if mode not in _OP_MODES:
        print(f"Unknown --op-mode '{mode}'. Valid options: {', '.join(_OP_MODES)}")
        sys.exit(1)
    settings = _OP_MODES[mode]
    AUTO_APPROVE           = settings["auto_approve"]
    USE_SENTIMENT_ANALYSIS = settings["use_sentiment_analysis"]
    logger.info(
        f"Op-mode '{mode}': auto_approve={AUTO_APPROVE}  "
        f"use_sentiment_analysis={USE_SENTIMENT_ANALYSIS}"
    )


if __name__ == "__main__":
    main()
