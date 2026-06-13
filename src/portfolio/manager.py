"""
portfolio/manager.py — PortfolioManager: orchestrates the full rebalance cycle.

Coordinates:
  - SellDecisionEngine  (which positions to exit)
  - RiskManager         (which buys are safe)
  - HarvestManager      (where to route take-profit proceeds)
  - BrokerAdapter       (how to place orders)
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from typing import TYPE_CHECKING

import pandas as pd

from core.instruments import is_fund_instrument_type
from portfolio.position_archetypes import classify_archetype
from portfolio.sell_engine import evaluate_sell_candidate
from portfolio.sleeve_tracker import (
    log_cash_sweep,
    log_exit_proceeds,
    log_harvest_proceeds,
    log_trim_proceeds,
)
from strategy.regimes.detector import get_current_regime
from util import (
    ARCHETYPE_PARAMS,
    AUTO_APPROVE,
    CANDIDATE_ROTATION_PARAMS,
    CANDIDATE_SELECTION_PARAMS,
    CONFIDENCE_THRESHOLD,
    CONTRARIAN_PENALTY_PARAMS,
    DATA_DIRECTORY,
    DIVIDEND_PARAMS,
    ETF_RISK_PARAMS,
    ETFS,
    EXIT_DECISION_PARAMS,
    HARVEST_PARAMS,
    INDEX_PCT,
    MAX_ITERATIONS,
    METRIC_KEYS,
    METRIC_THRESHOLD,
    REBALANCE_PARAMS,
    REGIME_PARAMS,
    RELIABILITY_PARAMS,
    RISK_LIMITS,
    SELL_SENTIMENT_OVERRIDE_CONFIDENCE,
    USE_SENTIMENT_ANALYSIS,
    read_data_as_pd,
    safe_float,
    store_data_as_csv,
)

if TYPE_CHECKING:
    from execution.base import BrokerAdapter
    from portfolio.harvest import HarvestManager
    from portfolio.risk import RiskManager

logger = logging.getLogger(__name__)


def _exclude_pooled_vehicles_from_active_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove pooled vehicles from active-sleeve candidate DataFrames.

    Active buys are intended to be single-company alpha bets. ETF/CEF/MLP/ETN
    wrappers are handled by ETF/index/harvest sleeves, not the stock picker.
    ADR and REIT remain eligible by design.
    """
    if df.empty or "instrument_type" not in df.columns:
        return df.copy(), []
    fund_mask = df["instrument_type"].map(is_fund_instrument_type).fillna(False)
    excluded = df.loc[fund_mask, "symbol"].astype(str).tolist()
    if not excluded:
        return df.copy(), []
    return df.loc[~fund_mask].copy(), excluded


_HOLDINGS_SCHEMA = [
    "symbol", "name", "quantity", "average_buy_price", "equity",
    "percent_change", "equity_change", "percentage", "current_price",
    "type", "pe_ratio", "id",
]


class PortfolioManager:
    """
    Orchestrates the sell → buy → sweep rebalance cycle.

    Constructor accepts broker, risk, and harvest as explicit dependencies
    so they can be swapped (e.g. PaperBroker in tests).
    """

    _PEAK_PRICES_CSV = os.path.join(DATA_DIRECTORY, "peak_prices.csv")
    _BUY_HISTORY_CSV  = os.path.join(DATA_DIRECTORY, "buy_history.csv")
    _SELL_HISTORY_CSV = os.path.join(DATA_DIRECTORY, "sell_history.csv")

    def __init__(
        self,
        broker: BrokerAdapter,
        risk: RiskManager,
        harvest: HarvestManager,
        auto_approve: bool | None = None,
        use_sentiment: bool | None = None,
    ) -> None:
        self._broker = broker
        self._risk = risk
        self._harvest = harvest
        self._auto_approve   = AUTO_APPROVE        if auto_approve  is None else auto_approve
        self._use_sentiment  = USE_SENTIMENT_ANALYSIS if use_sentiment is None else use_sentiment
        # Set per-run by rebalance(); guards the end-of-run ETF sweep so a Claude
        # API outage never routes the undeployed active sleeve into ETFs.
        self._sentiment_failed = False
        # Per-run sentiment verdict cache keyed by (action, symbol). Iterations
        # within one run previously re-queried Claude for the same candidates a
        # minute apart — duplicate spend AND nondeterministic flips (a BUY 80%
        # became HOLD 65% between iterations on identical data).
        self._sentiment_run_cache: dict[tuple[str, str], dict] = {}

    # ------------------------------------------------------------------
    # Confirmation gate
    # ------------------------------------------------------------------

    def _confirm(self, prompt: str) -> bool:
        if self._auto_approve:
            logger.info(f"AUTO-APPROVED: {prompt}")
            return True
        return input(f"{prompt} [y/n] ").strip().lower() in ("y", "yes")

    # ------------------------------------------------------------------
    # Holdings persistence
    # ------------------------------------------------------------------

    def _save_holdings_csv(self, holdings: dict) -> None:
        if not holdings:
            return
        rows = [[symbol] + [d.get(k, "") for k in _HOLDINGS_SCHEMA[1:]]
                for symbol, d in holdings.items()]
        try:
            store_data_as_csv("holdings", _HOLDINGS_SCHEMA, rows)
            logger.info(f"Saved holdings CSV: {len(rows)} positions")
        except Exception as e:
            logger.warning(f"Could not save holdings CSV: {e}")

    # ------------------------------------------------------------------
    # History tracking
    # ------------------------------------------------------------------

    def _load_peak_prices(self) -> dict[str, float]:
        try:
            df = pd.read_csv(self._PEAK_PRICES_CSV)
            if "symbol" in df.columns and "peak_price" in df.columns:
                return dict(zip(df["symbol"], df["peak_price"].astype(float)))
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not load peak prices: {e}")
        return {}

    def _save_peak_prices(self, peaks: dict[str, float]) -> None:
        try:
            pd.DataFrame(
                [{"symbol": sym, "peak_price": price} for sym, price in peaks.items()]
            ).to_csv(self._PEAK_PRICES_CSV, index=False)
        except Exception as e:
            logger.warning(f"Could not save peak prices: {e}")

    def _load_buy_history(self) -> dict[str, datetime.date]:
        try:
            df = pd.read_csv(self._BUY_HISTORY_CSV, parse_dates=["bought_date"])
            if "symbol" in df.columns and "bought_date" in df.columns:
                latest = df.sort_values("bought_date").groupby("symbol").last().reset_index()
                return {row["symbol"]: row["bought_date"].date() for _, row in latest.iterrows()}
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not load buy history: {e}")
        return {}

    def _record_buy(self, symbol: str) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        row = pd.DataFrame([{"symbol": symbol, "bought_date": today}])
        try:
            if os.path.exists(self._BUY_HISTORY_CSV):
                row.to_csv(self._BUY_HISTORY_CSV, mode="a", header=False, index=False)
            else:
                row.to_csv(self._BUY_HISTORY_CSV, index=False)
        except Exception as e:
            logger.warning(f"Could not record buy history for {symbol}: {e}")

    def _load_sell_history(self) -> dict[str, tuple[datetime.date, bool]]:
        """Latest sell per symbol → (sell_date, was_loss). Mirrors _load_buy_history;
        feeds the live sell/stopout buy-cooldown (backtest parity — the simulator
        enforces cooldown_days_after_sell/stopout, live previously did not, so a
        stop-loss exit could be rebought minutes later in the same run)."""
        try:
            df = pd.read_csv(self._SELL_HISTORY_CSV, parse_dates=["sell_date"])
            if "symbol" in df.columns and "sell_date" in df.columns:
                latest = df.sort_values("sell_date").groupby("symbol").last().reset_index()
                return {
                    row["symbol"]: (row["sell_date"].date(), bool(row.get("was_loss", False)))
                    for _, row in latest.iterrows()
                }
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not load sell history: {e}")
        return {}

    def _record_sell_event(self, symbol: str, was_loss: bool) -> None:
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        row = pd.DataFrame([{"symbol": symbol, "sell_date": today, "was_loss": was_loss}])
        try:
            if os.path.exists(self._SELL_HISTORY_CSV):
                row.to_csv(self._SELL_HISTORY_CSV, mode="a", header=False, index=False)
            else:
                row.to_csv(self._SELL_HISTORY_CSV, index=False)
        except Exception as e:
            logger.warning(f"Could not record sell history for {symbol}: {e}")

    def _check_wash_sale_risk(self, symbol: str) -> str | None:
        if not DIVIDEND_PARAMS.get("wash_sale_warning"):
            return None
        try:
            df = pd.read_csv(self._SELL_HISTORY_CSV, parse_dates=["sell_date"])
            recent_loss = df[
                (df["symbol"] == symbol) &
                (df["was_loss"].astype(bool)) &
                ((datetime.date.today() - df["sell_date"].dt.date).apply(lambda d: d.days) <= 30)
            ]
            if not recent_loss.empty:
                sell_date = recent_loss["sell_date"].max().strftime("%Y-%m-%d")
                return f"wash-sale risk: {symbol} sold at a loss on {sell_date} (<30d ago)"
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.debug(f"Wash-sale check failed for {symbol}: {e}")
        return None

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _load_news_for_symbol(self, symbol: str, news_df: pd.DataFrame | None) -> dict:
        try:
            if news_df is not None and not news_df.empty:
                rows = news_df[news_df["symbol"] == symbol]["news"]
                if not rows.empty:
                    raw = rows.iloc[0] if len(rows) == 1 else rows.tolist()
                    if isinstance(raw, str):
                        raw = json.loads(raw)
                    return {symbol: raw}
        except Exception as e:
            logger.debug(f"News load failed for {symbol}: {e}")
        return {}

    def _build_stocks_data(self, candidates: pd.DataFrame, action: str) -> list[dict]:
        try:
            agg_df = read_data_as_pd("agg_data")
        except Exception:
            agg_df = None
        try:
            news_df = read_data_as_pd("news")
        except Exception:
            news_df = None

        agg_index = (
            agg_df.set_index("symbol")
            if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns
            else None
        )

        stocks_data = []
        for _, row in candidates.iterrows():
            symbol = row["symbol"]
            if agg_index is not None and symbol in agg_index.index:
                agg_row = agg_index.loc[symbol]
                if isinstance(agg_row, pd.DataFrame):
                    agg_row = agg_row.iloc[0]
                fundamentals = {k: agg_row.get(k) for k in METRIC_KEYS}
            else:
                fundamentals = {k: row.get(k) for k in METRIC_KEYS}

            stocks_data.append({
                "symbol":              symbol,
                "action":              action,
                "fundamental_metrics": fundamentals,
                "news_sentiment":      self._load_news_for_symbol(symbol, news_df),
            })
        return stocks_data

    # ------------------------------------------------------------------
    # Order helpers
    # ------------------------------------------------------------------

    def _update_local_exposures_after_buy(
        self,
        symbol: str,
        allocation: float,
        agg_df: pd.DataFrame | None,
        sector_exposure: dict[str, float],
    ) -> None:
        if agg_df is None or agg_df.empty or "symbol" not in agg_df.columns:
            return
        row = agg_df[agg_df["symbol"] == symbol]
        if row.empty:
            return
        sector = str(row.iloc[0].get("sector") or "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + allocation

    @staticmethod
    def _candidate_sector(
        symbol: str, row: pd.Series, agg_df: pd.DataFrame | None,
    ) -> str:
        """Resolve a candidate's sector the SAME way the running sector_exposure dict
        is keyed (agg_df lookup, 'Unknown' fallback) so the buy-time sector cap reads
        the same bucket the dict accumulates into. Prefers the candidate row when it
        already carries a sector, else falls back to an agg_df lookup."""
        _s = row.get("sector") if hasattr(row, "get") else None
        if _s is not None and str(_s).strip():
            return str(_s)
        if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
            _r = agg_df[agg_df["symbol"] == symbol]
            if not _r.empty:
                return str(_r.iloc[0].get("sector") or "Unknown")
        return "Unknown"

    def _process_whole_share_queue(
        self,
        queue: list[tuple[str, float]],
        agg_df: pd.DataFrame | None,
        sector_exposure: dict[str, float],
        purchased: list[str],
        failed: list[str],
        skipped: list[str],
        holdings: dict | None = None,
        portfolio_value: float = 0.0,
    ) -> None:
        max_ws = RISK_LIMITS["max_whole_share_buys_per_run"]
        max_ws_mult = RISK_LIMITS["max_whole_share_allocation_multiplier"]
        ws_count = 0

        for idx, (symbol, alloc) in enumerate(queue):
            if ws_count >= max_ws:
                logger.info(f"Whole-share limit ({max_ws}) reached — remaining queue skipped")
                skipped.extend(sym for sym, _ in queue[idx:])
                break

            current_price: float | None = None
            if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
                row = agg_df[agg_df["symbol"] == symbol]
                if not row.empty:
                    current_price = safe_float(row.iloc[0].get("current_price"))

            if not current_price or current_price <= 0:
                logger.warning(f"{symbol}: no price available for whole-share fallback — failing")
                failed.append(symbol)
                continue

            max_alloc = alloc * max_ws_mult
            if current_price > max_alloc:
                logger.warning(
                    f"{symbol}: price ${current_price:.2f} > {max_ws_mult}× original alloc "
                    f"${max_alloc:.2f} — skipping whole-share fallback"
                )
                skipped.append(symbol)
                continue

            if holdings is not None:
                available_cash = self._broker.get_cash()
                _d = self._risk.can_buy(
                    symbol, current_price, holdings, agg_df, portfolio_value, available_cash, sector_exposure
                )
                if not _d.approved:
                    logger.info(f"Whole-share {symbol} blocked by risk re-check: {_d.reason}")
                    skipped.append(symbol)
                    continue

            if self._broker.buy_whole(symbol, 1).success:
                purchased.append(symbol)
                ws_count += 1
                self._record_buy(symbol)
                self._update_local_exposures_after_buy(symbol, current_price, agg_df, sector_exposure)
            else:
                failed.append(symbol)

    # ------------------------------------------------------------------
    # Decision logging
    # ------------------------------------------------------------------

    def _log_candidate(
        self,
        symbol: str,
        row: pd.Series,
        state: str,
        selected: bool,
        skip_reason: str,
        sentiment: dict | None,
        risk_ok: bool,
        risk_reason: str,
        proposed_alloc: float,
        final_alloc: float,
        regime: str,
        rank: int,
        agg_df,
    ) -> None:
        try:
            from portfolio.decision_logger import log_candidate_decision
            log_candidate_decision(
                symbol=symbol,
                row=row,
                decision_state=state,
                selected_bool=selected,
                skipped_bool=not selected,
                skip_reason=skip_reason,
                sentiment_result_dict=sentiment,
                risk_check_passed=risk_ok,
                risk_check_fail_reason=risk_reason,
                proposed_allocation=proposed_alloc,
                final_allocation=final_alloc,
                regime=regime,
                candidate_rank=rank,
                agg_df=agg_df,
            )
        except Exception as exc:
            logger.debug("Candidate log failed for %s: %s", symbol, exc)

    def _log_all_holding_decisions(
        self,
        all_evaluated: dict,
        sold: set,
        held_on_sentiment: set,
        regime: str,
        bc_df,
        agg_df,
    ) -> None:
        try:
            from portfolio.decision_logger import log_holding_decision
        except Exception as exc:
            logger.debug("decision_logger not available: %s", exc)
            return

        for symbol, info in all_evaluated.items():
            try:
                bc_row: dict | None = None
                if bc_df is not None and not bc_df.empty and "symbol" in bc_df.columns:
                    match = bc_df[bc_df["symbol"] == symbol]
                    if not match.empty:
                        bc_row = match.iloc[0].to_dict()

                log_holding_decision(
                    symbol=symbol,
                    holding=info["data"],
                    metrics_row=info["metrics_row"],
                    raw_decision=info["decision"],
                    executed=(symbol in sold),
                    order_id=None,
                    regime=regime,
                    buy_context_row=bc_row,
                    agg_df=agg_df,
                    soft_sell_held=(symbol in held_on_sentiment),
                    archetype_result=info.get("archetype_result"),
                )
            except Exception as exc:
                logger.debug("Outcome log failed for %s: %s", symbol, exc)

    # ------------------------------------------------------------------
    # Sell helpers
    # ------------------------------------------------------------------

    def _get_pending_sell_symbols(self) -> set[str]:
        try:
            pending: set[str] = set()
            for order in self._broker.get_open_orders():
                if order.get("side") != "sell":
                    continue
                # partially_filled still has the remaining shares committed.
                if order.get("state") not in ("confirmed", "queued", "unconfirmed", "partially_filled"):
                    continue
                # Raw order payloads carry an instrument URL, NOT a symbol key —
                # order.get("symbol") is always None, which silently emptied this
                # guard and let later iterations re-attempt every queued sell
                # ("Not enough shares to sell"). Resolve the URL when needed.
                sym = order.get("symbol")
                if not sym and order.get("instrument"):
                    resolver = getattr(self._broker, "resolve_instrument_symbol", None)
                    if resolver is not None:
                        sym = resolver(order["instrument"])
                if sym:
                    pending.add(sym)
            return pending
        except Exception as exc:
            logger.warning(f"Could not fetch open sell orders: {exc}")
            return set()

    def _etf_ma_sell_check(self, holdings: dict, sold: list[str]) -> None:
        etf_risk = ETF_RISK_PARAMS
        if not etf_risk.get("enabled") or not etf_risk.get("use_ma_filter"):
            return
        if get_current_regime() != "defensive":
            return

        ma_period = etf_risk.get("ma_period", 200)
        try:
            import yfinance as _yf
            for symbol, data in holdings.items():
                if symbol not in ETFS:
                    continue
                qty = float(data.get("quantity", 0))
                if qty <= 0:
                    continue
                hist = _yf.Ticker(symbol).history(period="1y")["Close"]
                if len(hist) < ma_period:
                    continue
                price = float(hist.iloc[-1])
                ma    = float(hist.rolling(ma_period).mean().iloc[-1])
                if price < ma:
                    logger.info(
                        f"ETF MA filter: {symbol} ${price:.2f} < {ma_period}d MA ${ma:.2f} "
                        f"(defensive regime) — selling"
                    )
                    if self._confirm(f"Sell {qty} shares of {symbol}?") and \
                            self._broker.sell(symbol, qty).success:
                        sold.append(symbol)
        except Exception as e:
            logger.warning(f"ETF MA filter check failed: {e}")

    # ------------------------------------------------------------------
    # Sell cycle
    # ------------------------------------------------------------------

    def sell_cycle(self) -> tuple[list[str], list[str]]:
        """
        Evaluate all holdings for sell conditions.

        Hard sells (stop-loss, yield-trap, quality floor) execute immediately.
        Soft sells (take-profit, weak value) are optionally held by sentiment.
        Trim exits (trim_exit) execute a partial position reduction.
        Sentiment can only override soft sells — never hard sells.
        ETF positions are protected from stock stop-loss logic; they are only
        exited by the ETF MA filter (defensive regime only).

        Returns (sold_full, trimmed_partial) — symbols with full vs partial exits.
        """
        sold: list[str] = []
        trimmed: list[str] = []
        # Resolved once per cycle: defensive regime tightens the stop-loss floor
        # inside SellDecisionEngine.evaluate (regime.defensive.stop_loss_tighten).
        try:
            _sell_regime = get_current_regime()
        except Exception:
            _sell_regime = None

        try:
            holdings = self._broker.get_holdings()
            self._broker.enrich_holdings_created_at(holdings)
            if getattr(self._broker, "is_live", True):
                self._save_holdings_csv(holdings)
        except Exception as e:
            logger.error(f"Could not fetch holdings: {e}")
            return sold, trimmed

        try:
            agg_df = read_data_as_pd("agg_data")
        except Exception:
            agg_df = None

        peaks = self._load_peak_prices()
        for symbol, data in holdings.items():
            if symbol in ETFS or float(data.get("quantity", 0)) <= 0:
                continue
            current_p = safe_float(data.get("price"))
            if current_p and current_p > 0:
                if symbol not in peaks or peaks[symbol] <= 0:
                    avg_buy = safe_float(data.get("average_buy_price"), 0.0) or 0.0
                    peaks[symbol] = max(avg_buy, current_p)
                else:
                    peaks[symbol] = max(peaks[symbol], current_p)
        self._save_peak_prices(peaks)

        # Opportunity-cost stall tracking: maintain the per-symbol "last progress"
        # store so SellDecisionEngine can cull dead money (max hold WITHOUT progress).
        # Only active when the feature is enabled — otherwise no store is touched
        # (behavior-preserving). First-seen symbols are seeded to today below, so the
        # feature never retroactively culls a long-standing holding on its first run.
        _oc_cfg     = EXIT_DECISION_PARAMS.get("opportunity_cost", {}) or {}
        _oc_enabled = bool(_oc_cfg.get("enabled", False))
        _oc_today   = datetime.date.today()
        _last_progress: dict[str, datetime.date] = {}
        if _oc_enabled:
            from portfolio.progress_tracker import load_last_progress
            _last_progress = load_last_progress()

        # Consecutive-weak-evaluation streaks for the archetype
        # `thesis_exit_requires_confirmation` switch. Rewritten each run (below) from the
        # symbols still signalling weak, so sold / recovered names drop out.
        _arch_on = ARCHETYPE_PARAMS.get("enabled", False)
        _weak_prev: dict[str, int] = {}
        _weak_next: dict[str, int] = {}
        if _arch_on:
            from portfolio.thesis_confirm_tracker import load_weak_streak
            _weak_prev = load_weak_streak()

        pending_sells = self._get_pending_sell_symbols()
        if pending_sells:
            logger.info(
                f"Skipping {len(pending_sells)} symbol(s) with existing open sell orders: "
                f"{sorted(pending_sells)}"
            )

        try:
            from portfolio.buy_context import load_buy_context
            _bc_df = load_buy_context()
        except Exception:
            _bc_df = None

        scanned    = 0
        hard_sells: dict[str, dict] = {}
        soft_sells: dict[str, dict] = {}
        _all_evaluated: dict[str, dict] = {}

        # Market-structure enrichment (maintenance_ratio, analyst_buy_pct, etc.) is read
        # from the agg_data block as columns (ETL layer merges it). No live
        # load_market_structure() call at decision time — absent columns degrade
        # gracefully to fundamentals-only classification.
        from data.market_structure import MARKET_STRUCTURE_DF_COLS

        for symbol, data in holdings.items():
            if symbol in ETFS:
                continue
            if float(data.get("quantity", 0)) <= 0:
                continue
            if symbol in pending_sells:
                continue

            scanned += 1

            metrics_row = None
            if agg_df is not None and not agg_df.empty and "symbol" in agg_df.columns:
                row = agg_df[agg_df["symbol"] == symbol]
                if not row.empty:
                    metrics_row = row.iloc[0]

            # Derive archetype policy for this position
            _arch_policy = None
            _arch_result = None
            if ARCHETYPE_PARAMS.get("enabled", False):
                try:
                    _signals: dict = {"symbol": symbol}
                    if metrics_row is not None:
                        for _k in ("quality_score", "momentum_score", "value_score",
                                   "income_score", "value_metric", "yield_trap_flag",
                                   "sector", "industry", "buy_to_sell_ratio",
                                   *MARKET_STRUCTURE_DF_COLS):
                            _v = metrics_row.get(_k)
                            if _v is not None and not (isinstance(_v, float) and pd.isna(_v)):
                                _signals[_k] = _v
                    _arch_result = classify_archetype(_signals, ARCHETYPE_PARAMS)
                    _arch_policy = _arch_result.policy
                    logger.debug(
                        "%s archetype=%s confidence=%.0f%% drivers=%s",
                        symbol, _arch_result.archetype,
                        _arch_result.confidence * 100,
                        "; ".join(_arch_result.drivers[:2]),
                    )
                except Exception as _exc:
                    logger.debug("Archetype classification failed for %s: %s", symbol, _exc)

            # Update the stall clock: reset to today if the position made progress
            # (fresh high or strong momentum) or is first-seen; else it accrues.
            _stall_days: int | None = None
            if _oc_enabled:
                from portfolio.exit_analysis import is_progress
                _mom = safe_float(metrics_row.get("momentum_score")) if metrics_row is not None else None
                _progressing = is_progress(
                    safe_float(data.get("price")), peaks.get(symbol), _mom,
                    float(_oc_cfg.get("reclaim_band", 0.03)),
                    float(_oc_cfg.get("progress_momentum_floor", 0.10)),
                )
                if _progressing or symbol not in _last_progress:
                    _last_progress[symbol] = _oc_today
                _stall_days = (_oc_today - _last_progress[symbol]).days

            decision = evaluate_sell_candidate(
                symbol, data, metrics_row,
                peak_price=peaks.get(symbol),
                archetype_policy=_arch_policy,
                stall_days=_stall_days,
                weak_streak=_weak_prev.get(symbol, 0),
                regime=_sell_regime,
            )

            # Carry forward the thesis-weak streak; symbols not in a weak streak (sold,
            # recovered, or never weak) simply don't get re-added and so drop from the store.
            _wk_next = decision.get("weak_streak_next")
            if _wk_next:
                _weak_next[symbol] = int(_wk_next)

            _all_evaluated[symbol] = {
                "data": data,
                "metrics_row": metrics_row,
                "decision": decision,
                "archetype_result": _arch_result,
            }

            if not decision["should_sell"]:
                continue

            if decision["severity"] == "hard":
                hard_sells[symbol] = decision
            else:
                soft_sells[symbol] = decision

        if _oc_enabled:
            from portfolio.progress_tracker import save_last_progress
            save_last_progress(_last_progress)

        if _arch_on:
            from portfolio.thesis_confirm_tracker import save_weak_streak
            save_weak_streak(_weak_next)

        logger.info(
            f"Sell scan: {scanned} holdings scanned | "
            f"{len(hard_sells)} hard | {len(soft_sells)} soft | "
            f"{scanned - len(hard_sells) - len(soft_sells)} no-action"
        )

        self._etf_ma_sell_check(holdings, sold)

        for symbol, decision in hard_sells.items():
            pct = decision.get("percent_change")
            pct_str = f" | P/L={pct:.1%}" if pct is not None else ""
            logger.info(f"HARD SELL {symbol} | {decision['reason']}{pct_str}")
            quantity = float(holdings[symbol].get("quantity", 0))
            if self._confirm(f"Sell {quantity} shares of {symbol}?") and \
                    self._broker.sell(symbol, quantity).success:
                sold.append(symbol)
                self._record_sell_event(symbol, was_loss=(pct is not None and pct < 0))
                log_exit_proceeds(
                    symbol, safe_float(holdings[symbol].get("equity"), 0.0) or 0.0,
                    etf_pct=0.0, notes=decision.get("reason"),
                )

        held_on_sentiment: set[str] = set()
        sentiment_results: dict[str, dict] = {}

        if soft_sells:
            if self._use_sentiment:
                soft_df = pd.DataFrame([
                    {"symbol": sym, **{k: None for k in METRIC_KEYS}}
                    for sym in soft_sells
                ])
                stocks_data = self._build_stocks_data(soft_df, action="sell")
                try:
                    from data.sentiment import get_batch_sentiment_recommendations
                    sentiment_results = get_batch_sentiment_recommendations(
                        stocks_data, action="sell", regime=get_current_regime(),
                    )
                except Exception:
                    logger.error("Batch sentiment failed for soft sells — executing all", exc_info=True)

                for sym, result in sentiment_results.items():
                    action     = result.get("action")
                    sentiment  = result.get("sentiment")
                    confidence = result["confidence"]

                    bullish_hold = (
                        action == "HOLD"
                        and sentiment == "bullish"
                        and confidence >= CONFIDENCE_THRESHOLD
                    )
                    high_conf_hold = (
                        action == "HOLD"
                        and confidence >= SELL_SENTIMENT_OVERRIDE_CONFIDENCE
                    )

                    if bullish_hold or high_conf_hold:
                        logger.info(
                            f"HOLD {sym} — sentiment overrides soft sell "
                            f"({confidence}% {sentiment}): {result['reasoning']}"
                        )
                        held_on_sentiment.add(sym)
                    else:
                        reason = (
                            f"confidence {confidence}% below threshold"
                            if action == "HOLD"
                            else f"action={action}"
                        )
                        logger.info(
                            f"SOFT SELL proceeds {sym} — {reason} "
                            f"sentiment={sentiment} ({confidence}%): {result['reasoning']}"
                        )

            harvest_proceeds = 0.0
            trim_proceeds    = 0.0

            for symbol, decision in soft_sells.items():
                if symbol in held_on_sentiment:
                    continue

                pct      = decision.get("percent_change")
                pct_str  = f" | P/L={pct:.1%}" if pct is not None else ""
                quantity = float(holdings[symbol].get("quantity", 0))
                exit_type = decision.get("exit_type")

                if exit_type == "trim_exit":
                    # --- Partial exit: sell trim_fraction of position ---
                    frac      = decision.get("trim_fraction") or EXIT_DECISION_PARAMS["trim_fraction"]
                    sell_qty  = round(quantity * frac, 6)
                    remain    = quantity - sell_qty
                    equity    = safe_float(holdings[symbol].get("equity"), 0.0) or 0.0
                    logger.info(
                        f"TRIM {symbol} | selling {sell_qty:.4f} / {quantity:.4f} shares "
                        f"({frac:.0%}) | {decision['reason']}{pct_str}"
                    )
                    if sell_qty > 0 and self._confirm(
                        f"Trim {sell_qty:.4f} shares of {symbol} "
                        f"({frac:.0%}, keeping {remain:.4f})?"
                    ) and self._broker.sell(symbol, sell_qty).success:
                        trimmed.append(symbol)
                        trim_proceeds += equity * frac
                        self._record_sell_event(symbol, was_loss=False)
                        log_trim_proceeds(
                            symbol, equity * frac,
                            etf_pct=EXIT_DECISION_PARAMS["trim_to_etfs_pct"],
                            notes=decision.get("reason"),
                        )
                        logger.info(
                            f"TRIM executed {symbol}: sold {sell_qty:.4f} "
                            f"shares, {remain:.4f} remain | proceeds ${equity * frac:.2f}"
                        )
                else:
                    # --- Full exit ---
                    logger.info(f"SOFT SELL {symbol} | {decision['reason']}{pct_str}")
                    if self._confirm(f"Sell {quantity} shares of {symbol}?") and \
                            self._broker.sell(symbol, quantity).success:
                        sold.append(symbol)
                        self._record_sell_event(symbol, was_loss=(pct is not None and pct < 0))
                        _equity = safe_float(holdings[symbol].get("equity"), 0.0) or 0.0
                        if exit_type == "harvest_exit":
                            harvest_proceeds += _equity
                            log_harvest_proceeds(
                                symbol, _equity,
                                etf_pct=HARVEST_PARAMS["harvest_to_etfs_pct"],
                                notes=decision.get("reason"),
                            )
                        else:
                            log_exit_proceeds(
                                symbol, _equity, etf_pct=0.0, notes=decision.get("reason"),
                            )

            # Route harvest proceeds (take-profit full exits)
            if harvest_proceeds >= HARVEST_PARAMS["min_harvest_amount"]:
                self._harvest.route_proceeds(harvest_proceeds, self._broker)

            # Route trim proceeds (partial exits)
            if trim_proceeds >= HARVEST_PARAMS["min_harvest_amount"]:
                trim_to_etfs = EXIT_DECISION_PARAMS["trim_to_etfs_pct"]
                etf_portion  = trim_proceeds * trim_to_etfs
                logger.info(
                    f"TRIM proceeds routing: ${trim_proceeds:.2f} total | "
                    f"${etf_portion:.2f} ({trim_to_etfs:.0%}) → ETFs | "
                    f"${trim_proceeds - etf_portion:.2f} retained as active reserve"
                )
                if etf_portion >= HARVEST_PARAMS["min_harvest_amount"]:
                    self._harvest.route_proceeds(etf_portion, self._broker)

        if sold:
            for sym in sold:
                peaks.pop(sym, None)
            self._save_peak_prices(peaks)

        self._log_all_holding_decisions(
            all_evaluated=_all_evaluated,
            sold=set(sold) | set(trimmed),
            held_on_sentiment=held_on_sentiment,
            regime=get_current_regime(),
            bc_df=_bc_df,
            agg_df=agg_df,
        )

        logger.info(
            f"Sell summary: {scanned} scanned | "
            f"{len(hard_sells)} hard | {len(soft_sells)} soft candidates | "
            f"{len(held_on_sentiment)} held on sentiment | "
            f"{len(sold)} full exits | {len(trimmed)} trims | "
            f"{len(hard_sells) + len(soft_sells) - len(sold) - len(trimmed)} skipped/no-action"
        )
        return sold, trimmed

    # ------------------------------------------------------------------
    # Buy cycle
    # ------------------------------------------------------------------

    def _compute_contribution_split(
        self,
        total_cash: float,
        effective_index_pct: float,
    ) -> tuple[float, float]:
        """
        Split `total_cash` between ETF and active sleeves.

        When `proportional_deficit_routing` is enabled:
          - If the ETF sleeve is within ±drift_tolerance_pct of target → normal proportional split.
          - If ETF is overweight beyond tolerance → route ALL cash to active.
          - If ETF is underweight beyond tolerance → route ALL cash to ETF.

        This allows a legacy ETF-heavy portfolio to converge toward target weight via
        contributions without force-selling ETF lots (no tax event, no turnover).
        Falls back to proportional split on any error.
        """
        rb = REBALANCE_PARAMS
        if not rb.get("proportional_deficit_routing"):
            etf_amt = total_cash * effective_index_pct
            return etf_amt, total_cash - etf_amt

        drift_tol = float(rb.get("drift_tolerance_pct", 0.03))
        try:
            holdings         = self._broker.get_holdings()
            portfolio_equity = self._broker.get_portfolio_value()
            if portfolio_equity > 0:
                current_etf = sum(
                    safe_float(holdings.get(etf, {}).get("equity")) or 0.0
                    for etf in ETFS
                )
                current_etf_pct = current_etf / portfolio_equity
                drift = current_etf_pct - effective_index_pct

                if drift > drift_tol:
                    # ETF overweight — all new cash goes to active sleeve.
                    logger.info(
                        f"Deficit routing: ETF {current_etf_pct:.1%} is +{drift:.1%} above "
                        f"target {effective_index_pct:.0%} (tolerance ±{drift_tol:.0%}) "
                        f"— routing all ${total_cash:.2f} to active sleeve"
                    )
                    return 0.0, total_cash

                if drift < -drift_tol:
                    # ETF underweight — fill only the deficit, remainder to active.
                    deficit = max(0.0, portfolio_equity * effective_index_pct - current_etf)
                    etf_amt = min(total_cash, deficit)
                    logger.info(
                        f"Deficit routing: ETF {current_etf_pct:.1%} is {drift:.1%} below "
                        f"target {effective_index_pct:.0%} (tolerance ±{drift_tol:.0%}) "
                        f"— filling deficit ${deficit:.2f} with ${etf_amt:.2f}, "
                        f"${total_cash - etf_amt:.2f} to active sleeve"
                    )
                    return etf_amt, total_cash - etf_amt

                logger.info(
                    f"Deficit routing: ETF {current_etf_pct:.1%} within ±{drift_tol:.0%} "
                    f"tolerance — normal {effective_index_pct:.0%}/{1-effective_index_pct:.0%} split"
                )
        except Exception as exc:
            logger.warning(f"Deficit routing check failed ({exc}) — proportional fallback")

        etf_amt = total_cash * effective_index_pct
        return etf_amt, total_cash - etf_amt

    @staticmethod
    def _resolve_regime_params(regime: str) -> tuple[float, int]:
        """Return (effective_index_pct, effective_max_buys) for the given regime."""
        rp = REGIME_PARAMS
        index_pct = INDEX_PCT
        max_buys  = RISK_LIMITS["max_buys_per_rebalance"]
        if regime in ("defensive", "neutral"):
            ovr = rp.get(regime, {})
            if ovr.get("index_pct_override") is not None:
                index_pct = float(ovr["index_pct_override"])
            if ovr.get("max_buys_override") is not None:
                max_buys = int(ovr["max_buys_override"])
        return index_pct, max_buys

    def buy_cycle(
        self,
        df: pd.DataFrame,
        is_first_iteration: bool = True,
        regime: str = "bullish",
        effective_index_pct: float | None = None,
    ) -> tuple[list, list, list]:
        """
        Execute buy orders. Returns (purchased, skipped, failed).

        ETF buys only happen on the first iteration.  On subsequent iterations
        `stock_amount` is the full available cash because the ETF sleeve was
        already funded in iteration 1.
        """
        _eff_idx, effective_max_buys = self._resolve_regime_params(regime)
        if effective_index_pct is None:
            effective_index_pct = _eff_idx

        if regime == "defensive":
            logger.info(
                f"DEFENSIVE regime: index_pct={effective_index_pct:.0%}  max_buys={effective_max_buys}"
            )
        elif regime == "neutral":
            logger.info(f"NEUTRAL regime: index_pct={effective_index_pct:.0%}")

        total_cash = self._broker.get_cash()

        if is_first_iteration:
            # First pass: split cash between ETF sleeve and active sleeve.
            # Uses deficit-aware routing when proportional_deficit_routing is enabled.
            etf_amount, stock_amount = self._compute_contribution_split(
                total_cash, effective_index_pct
            )
            logger.info(
                f"Iter-1 allocation — ETFs ${etf_amount:.2f} ({etf_amount/max(total_cash,1e-9):.0%}), "
                f"stocks ${stock_amount:.2f} ({stock_amount/max(total_cash,1e-9):.0%}) "
                f"| cash=${total_cash:.2f}"
            )
            if etf_amount > 0 and \
                    (self._auto_approve or self._confirm(f"Buy ETFs (${etf_amount:,.2f})?")):
                per_etf = etf_amount / max(len(ETFS), 1)
                if per_etf < RISK_LIMITS["min_order_amount"]:
                    logger.info(
                        f"Per-ETF amount ${per_etf:.2f} < min_order "
                        f"${RISK_LIMITS['min_order_amount']:.2f} — skipping ETF buys"
                    )
                else:
                    for etf in ETFS:
                        try:
                            if self._auto_approve or self._confirm(f"Buy ${per_etf:,.2f} of {etf}?"):
                                result = self._broker.buy_fractional(etf, per_etf)
                                logger.info(f"ETF {etf}: {result.state}")
                        except Exception as e:
                            logger.error(f"ETF buy failed for {etf}: {e}")
        else:
            # Subsequent iterations: ETF sleeve already funded; all cash is for stocks.
            stock_amount = total_cash
            logger.info(
                f"Iter-N allocation — all ${total_cash:.2f} available for stocks "
                f"(ETFs already funded this run)"
            )

        if regime == "defensive":
            logger.info(
                "Defensive regime — skipping individual stock buys "
                "(remaining cash swept to ETFs at end)"
            )
            return [], [], []

        if df.empty or stock_amount <= 0:
            logger.warning("No stock picks or no funds for stocks")
            return [], [], []

        df["value_metric"] = pd.to_numeric(df["value_metric"], errors="coerce").fillna(0.0)
        logger.info(f"value_metric dtype: {df['value_metric'].dtype}")
        logger.info(f"value_metric max: {df['value_metric'].max()}")
        logger.info(
            "Top value_metric rows:\n%s",
            df[["symbol", "value_metric"]]
            .sort_values("value_metric", ascending=False)
            .head(10)
            .to_string(index=False),
        )
        # --- Constant gates (not threshold-dependent) ---
        df_eligible = df.copy()

        # Active sleeve is for single-company alpha bets. Pooled vehicles (ETF/CEF/MLP/ETN)
        # belong in the ETF/index/harvest sleeves; letting them through here spends
        # alpha budget on wrappers/bonds/index products and breaks live-vs-backtest fidelity.
        if "instrument_type" in df_eligible.columns:
            df_eligible, _fund_syms = _exclude_pooled_vehicles_from_active_candidates(df_eligible)
            if _fund_syms:
                logger.info(
                    "Active-sleeve fund filter: excluded %d pooled vehicles: %s",
                    len(_fund_syms), _fund_syms[:15],
                )
                if df_eligible.empty:
                    logger.warning("No candidates remain after active-sleeve fund filter")
                    return [], df["symbol"].tolist(), []
        # Liquidity pre-filter: the min_liquidity_volume gate is deterministic, so
        # apply it BEFORE candidates consume sentiment-batch slots (RiskManager.can_buy
        # still re-checks per-order as defense-in-depth). Previously illiquid names
        # earned BUY verdicts from Claude and were then skipped every time.
        if "volume" in df_eligible.columns:
            _min_vol = RISK_LIMITS["min_liquidity_volume"]
            _vols = pd.to_numeric(df_eligible["volume"], errors="coerce").fillna(0.0)
            _illiquid = df_eligible.loc[_vols < _min_vol, "symbol"].tolist()
            if _illiquid:
                df_eligible = df_eligible[_vols >= _min_vol].copy()
                logger.info(
                    "Liquidity pre-filter: excluded %d symbols below %s ADV (e.g. %s)",
                    len(_illiquid), f"{_min_vol:,.0f}", _illiquid[:8],
                )
                if df_eligible.empty:
                    logger.warning("No candidates remain after liquidity pre-filter")
                    return [], df["symbol"].tolist(), []
        if "strategy_bucket" in df_eligible.columns and CONTRARIAN_PENALTY_PARAMS["enabled"]:
            contrarian_mask = df_eligible["strategy_bucket"] == "contrarian_watchlist"
            contrarian_syms = df_eligible.loc[contrarian_mask, "symbol"].tolist()
            if contrarian_syms:
                _sm = CONTRARIAN_PENALTY_PARAMS["score_multiplier"]
                df_eligible.loc[contrarian_mask, "value_metric"] = (
                    df_eligible.loc[contrarian_mask, "value_metric"] * _sm
                )
                logger.info(
                    f"Contrarian soft penalty ({_sm}×, pos cap "
                    f"{CONTRARIAN_PENALTY_PARAMS['max_position_multiplier']}×): "
                    f"{len(contrarian_syms)} symbols (e.g. {contrarian_syms[:8]})"
                )

        if RELIABILITY_PARAMS["enabled"] and "reliability_score" in df_eligible.columns:
            min_rel = RELIABILITY_PARAMS["min_reliability_score"]
            rel_scores = pd.to_numeric(df_eligible["reliability_score"], errors="coerce").fillna(0.0)
            low_rel = df_eligible[rel_scores < min_rel]["symbol"].tolist()
            df_eligible = df_eligible[rel_scores >= min_rel].copy()
            if low_rel:
                logger.info(
                    f"Reliability gate ({min_rel:.2f}): excluded {len(low_rel)} "
                    f"low-quality candidates: {low_rel[:10]}"
                )

        if df_eligible.empty:
            logger.warning("No candidates remain after reliability gate")
            return [], df["symbol"].tolist(), []

        # --- Cooldown setup + exemption state ---
        cooldown_days    = CANDIDATE_ROTATION_PARAMS["buy_cooldown_days"]
        add_above        = CANDIDATE_ROTATION_PARAMS.get("allow_add_to_existing_if_score_above")
        exempt_underweight = CANDIDATE_ROTATION_PARAMS.get("cooldown_exempt_if_active_underweight", False)

        # Fetch holdings early for cooldown exemptions (reused below to avoid double API call)
        _early_holdings: dict = {}
        held_symbols: set = set()
        if cooldown_days > 0 and (add_above is not None or exempt_underweight):
            try:
                _early_holdings = self._broker.get_holdings()
                held_symbols = set(_early_holdings.keys())
            except Exception:
                pass

        active_is_underweight = False
        if exempt_underweight and held_symbols:
            try:
                _pv   = self._broker.get_portfolio_value()
                _cash = self._broker.get_cash()
                _etf_val = sum(
                    safe_float(_early_holdings.get(e, {}).get("equity")) or 0.0 for e in ETFS
                )
                _active_pct = (_pv - _cash - _etf_val) / max(_pv, 1.0)
                active_is_underweight = _active_pct < (1.0 - effective_index_pct)
                if active_is_underweight:
                    logger.info(
                        f"Active sleeve underweight ({_active_pct:.1%} vs "
                        f"{1.0 - effective_index_pct:.0%} target) — cooldown exemptions active"
                    )
            except Exception:
                pass

        buy_history  = self._load_buy_history() if cooldown_days > 0 else {}
        _today       = datetime.date.today()

        # Sell-side cooldowns mirror the backtest (cooldown_days_after_sell /
        # _stopout) so live and sim share exit-then-rebuy discipline. Loss exits
        # (stop-loss / trailing-stop territory) get the longer stopout window.
        # These are UNCONDITIONAL — no underweight/high-conviction exemptions: a
        # name we just stopped out of must not be rebought the same run because
        # the sleeve happens to be underweight.
        from util import BACKTEST_PARAMS as _bp
        _sell_cd    = int(_bp.get("cooldown_days_after_sell", 0))
        _stopout_cd = int(_bp.get("cooldown_days_after_stopout", 0))
        sell_history = self._load_sell_history() if (_sell_cd > 0 or _stopout_cd > 0) else {}

        def _apply_cooldown(frame: pd.DataFrame) -> tuple[pd.DataFrame, list]:
            if "symbol" not in frame.columns:
                return frame, []
            cooled_out = []
            sell_blocked = []
            for sym in frame["symbol"].tolist():
                sold = sell_history.get(sym)
                if sold is not None:
                    _cd = _stopout_cd if sold[1] else _sell_cd
                    if _cd > 0 and (_today - sold[0]).days < _cd:
                        sell_blocked.append(sym)
                        continue
                if cooldown_days <= 0 or sym not in buy_history:
                    continue
                if (_today - buy_history[sym]).days >= cooldown_days:
                    continue
                score = float(frame.loc[frame["symbol"] == sym, "value_metric"].iloc[0])
                if add_above is not None and score >= add_above and sym in held_symbols:
                    continue  # high-conviction add-to-existing
                if active_is_underweight and sym in held_symbols:
                    continue  # sleeve underweight — allow topping up held names
                cooled_out.append(sym)
            if sell_blocked:
                logger.info(
                    f"Sell cooldown ({_sell_cd}d sell / {_stopout_cd}d stopout): "
                    f"excluded {len(sell_blocked)} recently-sold symbols: {sell_blocked[:10]}"
                )
            blocked = sell_blocked + cooled_out
            if blocked:
                frame = frame[~frame["symbol"].isin(blocked)].copy()
            return frame, cooled_out

        # --- Fallback threshold ladder ---
        _cs            = CANDIDATE_SELECTION_PARAMS
        # Entry gate decoupled from METRIC_THRESHOLD: metric_threshold anchors the
        # EXIT ladder (trim / take-profit floor scaling) and is unreachable as an
        # entry gate under peer-percentile scoring. entry_threshold_override is the
        # explicit live entry rung; null falls back to METRIC_THRESHOLD (legacy).
        _entry_override = _cs.get("entry_threshold_override")
        _primary_thr    = float(_entry_override) if _entry_override is not None else METRIC_THRESHOLD
        _raw_fallbacks = _cs.get("fallback_thresholds", [])
        _thresholds    = [float(t) for t in _raw_fallbacks] if _raw_fallbacks else [_primary_thr]
        if not _thresholds or _thresholds[0] != _primary_thr:
            _thresholds = [_primary_thr] + _thresholds
        _min_post_cd   = int(_cs.get("min_post_cooldown_candidates", 1))

        candidates = pd.DataFrame()
        for _fi, _thr in enumerate(_thresholds):
            _tier = df_eligible[df_eligible["value_metric"] >= _thr].copy()
            if _fi == 0:
                logger.info(
                    f"Pre-filter: {len(df)} → {len(_tier)} stocks (value_metric ≥ {_thr})"
                )
            else:
                logger.info(
                    f"Fallback threshold {_thr}: {len(_tier)} candidates "
                    f"(was starved at {_thresholds[_fi - 1]})"
                )

            if _tier.empty:
                if _fi < len(_thresholds) - 1:
                    continue
                logger.warning(f"No stocks pass value_metric ≥ {_thr} (fallbacks exhausted)")
                return [], df["symbol"].tolist(), []

            _tier, _cooled = _apply_cooldown(_tier)
            if _cooled:
                logger.info(
                    f"Buy cooldown ({cooldown_days}d): excluded {len(_cooled)} "
                    f"recently-purchased symbols: {_cooled[:10]}"
                )

            if _tier.empty:
                if _fi < len(_thresholds) - 1:
                    logger.info(
                        f"All candidates cooled out at threshold {_thr} — "
                        f"stepping down to {_thresholds[_fi + 1]}"
                    )
                    continue
                logger.warning("No candidates remain after buy cooldown filter (fallbacks exhausted)")
                return [], df["symbol"].tolist(), []

            if len(_tier) >= _min_post_cd or _fi == len(_thresholds) - 1:
                candidates = _tier
                break

            logger.info(
                f"Only {len(_tier)} candidate(s) post-cooldown at threshold {_thr} "
                f"(min={_min_post_cd}) — stepping down to {_thresholds[_fi + 1]}"
            )

        if candidates.empty:
            logger.warning("No candidates remain after all filters")
            return [], df["symbol"].tolist(), []

        jitter_pct = CANDIDATE_ROTATION_PARAMS["score_jitter_pct"]
        if jitter_pct > 0 and "value_metric" in candidates.columns:
            import numpy as _np
            score_range = candidates["value_metric"].max() - candidates["value_metric"].min()
            if score_range > 0:
                noise = _np.random.uniform(
                    -jitter_pct * score_range, jitter_pct * score_range, len(candidates)
                )
                candidates = candidates.copy()
                candidates["_jittered"] = candidates["value_metric"] + noise
            else:
                candidates["_jittered"] = candidates["value_metric"]
        else:
            candidates["_jittered"] = candidates["value_metric"]

        max_sc = RISK_LIMITS["max_sentiment_candidates"]
        if len(candidates) > max_sc:
            sort_cols = ["_jittered"] + (
                ["buy_to_sell_ratio"] if "buy_to_sell_ratio" in candidates.columns else []
            )
            candidates = (
                candidates
                .sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
                .head(max_sc)
                .drop(columns=["_jittered"])
                .copy()
            )
            logger.info(
                f"Candidates capped to {max_sc} (jitter={jitter_pct:.0%}, "
                f"cooldown={cooldown_days}d)"
            )
        else:
            candidates = candidates.drop(columns=["_jittered"]).copy()

        holdings        = _early_holdings if _early_holdings else self._broker.get_holdings()
        self._save_holdings_csv(holdings)
        portfolio_value = self._broker.get_portfolio_value()
        try:
            agg_df = read_data_as_pd("agg_data")
        except Exception:
            agg_df = None

        # Per-candidate archetype policy (used for enabled/min_score gate + sizing caps).
        # Built once and reused below. Empty dict when archetype management disabled.
        _buy_arch_policies: dict = {}
        if ARCHETYPE_PARAMS.get("enabled", False):
            # Market-structure enrichment is read from the agg_data block (ETL layer
            # merges it as columns). No live load_market_structure() call here — if the
            # columns are absent (stale block), classification degrades gracefully to
            # fundamentals-only, same as before enrichment existed.
            from data.market_structure import MARKET_STRUCTURE_DF_COLS
            _to_drop: list[str] = []
            for _i_idx, _crow in candidates.iterrows():
                _sym_c = _crow["symbol"]
                _signals_c: dict = {"symbol": _sym_c}
                for _k in ("quality_score", "momentum_score", "value_score",
                           "income_score", "value_metric", "yield_trap_flag",
                           "sector", "industry", "buy_to_sell_ratio",
                           *MARKET_STRUCTURE_DF_COLS):
                    _v = _crow.get(_k)
                    if _v is not None and not (isinstance(_v, float) and pd.isna(_v)):
                        _signals_c[_k] = _v
                try:
                    _ar = classify_archetype(_signals_c, ARCHETYPE_PARAMS)
                    _policy_c = _ar.policy
                except Exception:
                    _policy_c = None
                if _policy_c is None:
                    continue
                _buy_arch_policies[_sym_c] = _policy_c
                # Skip new buys when archetype is disabled (existing holdings still managed by sell side)
                if not _policy_c.enabled:
                    _to_drop.append(_sym_c)
                    continue
                # Hard score gate
                _vm = float(_crow.get("value_metric") or 0.0)
                if _policy_c.min_score_to_buy is not None and _vm < float(_policy_c.min_score_to_buy):
                    _to_drop.append(_sym_c)
                    continue
                # Score multiplier — applied in-place to value_metric so the existing
                # ranking + budget-share logic naturally accounts for it.
                if _policy_c.score_multiplier != 1.0:
                    candidates.loc[candidates["symbol"] == _sym_c, "value_metric"] = (
                        _vm * float(_policy_c.score_multiplier)
                    )
            if _to_drop:
                logger.info(
                    f"Archetype buy-filter dropped {len(_to_drop)} candidates "
                    f"(disabled/below-min-score): {_to_drop[:10]}"
                )
                candidates = candidates[~candidates["symbol"].isin(_to_drop)].copy()
                if candidates.empty:
                    logger.warning("No candidates remain after archetype buy-filter")
                    return [], df["symbol"].tolist(), []

        # Allocation pre-check: if even the BEST-scored candidate cannot clear
        # min_order at this budget, no buy can execute — skip the entire sentiment
        # batch (a full Claude batch was previously spent on iterations that were
        # guaranteed to buy nothing, e.g. $162 budget x 12% floor = $19 allocs).
        if not candidates.empty:
            _tv_pre = float(candidates["value_metric"].sum())
            _floor_pre = stock_amount * RISK_LIMITS["min_candidate_allocation_pct"]
            _best_alloc = max(
                max((float(v) / _tv_pre) * stock_amount if _tv_pre > 0 else 0.0, _floor_pre)
                for v in candidates["value_metric"]
            )
            _best_alloc = min(_best_alloc, stock_amount)
            if _best_alloc < RISK_LIMITS["min_order_amount"]:
                logger.info(
                    f"Best possible allocation ${_best_alloc:.2f} < min order "
                    f"${RISK_LIMITS['min_order_amount']:.2f} at budget ${stock_amount:.2f} "
                    f"— skipping buy phase (and its sentiment batch)"
                )
                return [], candidates["symbol"].tolist(), []

        sentiment_results: dict[str, dict] = {}
        if self._use_sentiment:
            stocks_data = self._build_stocks_data(candidates, action="buy")
            # Per-run cache: a symbol already judged this run keeps its verdict —
            # iterations re-querying identical data wasted tokens and produced
            # nondeterministic flips (BUY 80% -> HOLD 65% a minute apart).
            _cache = getattr(self, "_sentiment_run_cache", None)
            if _cache is None:
                _cache = self._sentiment_run_cache = {}
            cached_results = {
                item["symbol"]: _cache[("buy", item["symbol"])]
                for item in stocks_data if ("buy", item["symbol"]) in _cache
            }
            to_query = [item for item in stocks_data if ("buy", item["symbol"]) not in _cache]
            if cached_results:
                logger.info(
                    f"Sentiment cache: reusing {len(cached_results)} verdict(s) from "
                    f"earlier this run: {sorted(cached_results)[:10]}"
                )
            fresh_results: dict[str, dict] = {}
            if to_query:
                logger.info(f"Running batch sentiment on {len(to_query)} candidates...")
                try:
                    from data.sentiment import (
                        get_batch_sentiment_recommendations,
                        is_api_error_sentinel,
                    )
                    fresh_results = get_batch_sentiment_recommendations(
                        to_query, action="buy", regime=regime,
                    )
                except Exception:
                    logger.error(
                        "Batch sentiment failed — active sleeve undeployed this run; "
                        "leftover cash will be HELD (not swept to ETFs)",
                        exc_info=True,
                    )
                    self._sentiment_failed = True
                    return [], candidates["symbol"].tolist(), []
                # The sentiment layer swallows API outages and returns HOLD sentinels
                # instead of raising. If EVERY freshly-queried symbol came back as an
                # API-error sentinel, treat it exactly like the exception path above
                # (partial failures keep normal per-symbol gating).
                if all(
                    is_api_error_sentinel(fresh_results.get(item["symbol"]))
                    for item in to_query
                ):
                    logger.error(
                        "Batch sentiment returned API-error sentinels for all candidates — "
                        "active sleeve undeployed this run; leftover cash will be HELD "
                        "(not swept to ETFs)"
                    )
                    self._sentiment_failed = True
                    return [], candidates["symbol"].tolist(), []
                # Cache real verdicts only — sentinels must be retried next iteration.
                for _sym_f, _res_f in fresh_results.items():
                    if not is_api_error_sentinel(_res_f):
                        _cache[("buy", _sym_f)] = _res_f
            sentiment_results = {**cached_results, **fresh_results}

        sector_exposure = (
            self._risk.get_sector_exposure(holdings, agg_df) if portfolio_value > 0 else {}
        )
        # Active-sleeve target value. Concentration caps must bind the ACTIVE SLEEVE,
        # not total PV: a 60%/40% cluster/sector cap measured against total PV is
        # ~12×/8× too loose on a ~5%-of-PV sleeve and never binds (and is inert at
        # cold start). Denominating against (1-index_pct)*PV makes them constrain the
        # stock book itself. Mirrored in backtesting/simulator._do_buy.
        active_sleeve_value = max((1.0 - effective_index_pct) * portfolio_value, 0.0)

        purchased: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        whole_share_queue: list[tuple[str, float]] = []
        allow_ws_fallback = RISK_LIMITS["allow_whole_share_fallback"]
        total_value  = candidates["value_metric"].sum()
        buys_made    = 0
        _cand_rank   = 0
        stock_deployed = 0.0
        # Running deployed-to-archetype tally (for max_active_weight sleeve caps)
        _arch_deployed_this_pass: dict[str, float] = {}

        for _, row in candidates.iterrows():
            _cand_rank += 1

            if buys_made >= effective_max_buys:
                logger.info(
                    f"Reached max_buys limit ({effective_max_buys}) for regime={regime} — stopping"
                )
                remaining_syms = [
                    r["symbol"] for _, r in candidates.iterrows()
                    if r["symbol"] not in purchased + skipped + failed
                ]
                for _sym in remaining_syms:
                    self._log_candidate(
                        _sym, row, "SKIP", False, "max_buys_limit",
                        None, True, "", 0.0, 0.0, regime, _cand_rank, agg_df,
                    )
                skipped.extend(remaining_syms)
                break

            symbol = row["symbol"]
            _sent_result = sentiment_results.get(symbol) if sentiment_results else None
            remaining_stock = max(0.0, stock_amount - stock_deployed)
            _raw_alloc = (row["value_metric"] / total_value) * remaining_stock if total_value else 0
            _min_floor = stock_amount * RISK_LIMITS["min_candidate_allocation_pct"]
            alloc = min(max(_raw_alloc, _min_floor), remaining_stock)

            if sentiment_results:
                _sent: dict = sentiment_results.get(
                    symbol,
                    {"action": "HOLD", "sentiment": "neutral", "confidence": 0.0, "reasoning": "No result"},
                )
                logger.info(
                    f"{'='*60}\nBUY {symbol} | action={_sent.get('action')} "
                    f"sentiment={_sent.get('sentiment')} {_sent['confidence']:.1f}% | "
                    f"{_sent['reasoning']}\n{'='*60}"
                )
                if _sent.get("action") != "BUY" or _sent["confidence"] < CONFIDENCE_THRESHOLD:
                    logger.info(f"Skipping {symbol}")
                    self._log_candidate(
                        symbol, row, "SKIP", False, "sentiment_gate",
                        _sent_result, True, "", alloc, 0.0, regime, _cand_rank, agg_df,
                    )
                    skipped.append(symbol)
                    continue

            cash = max(0.0, stock_amount - stock_deployed)
            if cash < RISK_LIMITS["min_order_amount"]:
                logger.info(
                    f"Stock budget ${cash:.2f} below min order "
                    f"${RISK_LIMITS['min_order_amount']:.2f} — exiting buy loop"
                )
                break

            alloc = (row["value_metric"] / total_value) * cash if total_value else 0
            min_alloc_floor = stock_amount * RISK_LIMITS["min_candidate_allocation_pct"]
            alloc = min(max(alloc, min_alloc_floor), cash)

            if (
                row.get("strategy_bucket") == "contrarian_watchlist"
                and CONTRARIAN_PENALTY_PARAMS["enabled"]
            ):
                _ctr_cap = (
                    (stock_amount / max(len(candidates), 1))
                    * CONTRARIAN_PENALTY_PARAMS["max_position_multiplier"]
                )
                alloc = min(alloc, _ctr_cap)

            # Per-archetype position cap (max_position_multiplier × max_single_position_pct)
            # and sleeve cap (max_active_weight × portfolio_value). Defaults are no-ops.
            _arch_pol = _buy_arch_policies.get(symbol) if _buy_arch_policies else None
            if _arch_pol is not None and portfolio_value > 0:
                if _arch_pol.max_position_multiplier != 1.0:
                    _max_pos_dollars = (
                        RISK_LIMITS["max_single_position_pct"]
                        * portfolio_value
                        * float(_arch_pol.max_position_multiplier)
                    )
                    _cur_pos_val = safe_float(holdings.get(symbol, {}).get("equity")) or 0.0
                    _room_pos = max(0.0, _max_pos_dollars - _cur_pos_val)
                    if alloc > _room_pos:
                        alloc = _room_pos
                if _arch_pol.max_active_weight is not None:
                    _label = _arch_pol.archetype
                    # Current sleeve exposure for this archetype (sum of held-position equity)
                    _sleeve_val = 0.0
                    if held_symbols:
                        for _sym2 in held_symbols:
                            _pol2 = _buy_arch_policies.get(_sym2)
                            if _pol2 is not None and _pol2.archetype == _label:
                                _sleeve_val += safe_float(holdings.get(_sym2, {}).get("equity")) or 0.0
                    _sleeve_cap_dollars = portfolio_value * float(_arch_pol.max_active_weight)
                    _running = _arch_deployed_this_pass.get(_label, 0.0)
                    _room_sleeve = max(0.0, _sleeve_cap_dollars - _sleeve_val - _running)
                    if alloc > _room_sleeve:
                        alloc = _room_sleeve

            if alloc < RISK_LIMITS["min_order_amount"]:
                logger.info(
                    f"Skipping {symbol}: archetype cap reduced alloc below min_order"
                )
                self._log_candidate(
                    symbol, row, "SKIP", False, "archetype_cap",
                    _sent_result, True, "", alloc, 0.0, regime, _cand_rank, agg_df,
                )
                skipped.append(symbol)
                continue

            # ── Cluster concentration enforcement (config-gated) ──────────────
            _cluster_decision = None
            _cluster_id_for_log: str | None = None
            _cluster_cw = 0.0
            _cluster_pw = 0.0
            _cluster_limit = 0.0
            _cluster_block_reason = ""
            from util import CONCENTRATION_LIMIT_PARAMS as _CLP
            if (
                _CLP["enabled"] and not _CLP["warn_only"]
                and _CLP["apply_to"]["active_sleeve"]
                and getattr(self, "_cluster_report", None) is not None
            ):
                from portfolio.exposure.cluster_enforcement import (
                    cluster_buy_decision,
                    current_cluster_weight,
                )
                _cluster_lookup = self._cluster_report.universe_cluster_lookup
                _cluster_id_for_log = _cluster_lookup.get(symbol)
                if _cluster_id_for_log is not None and active_sleeve_value > 0:
                    # Weight is measured against the active-sleeve target, not total PV.
                    _cluster_cw = current_cluster_weight(
                        holdings, _cluster_lookup, active_sleeve_value, _cluster_id_for_log,
                    )
                    _cluster_limit = float(_CLP["max_cluster_weight"])
                    _cluster_decision, _new_alloc, _cluster_block_reason = cluster_buy_decision(
                        symbol=symbol,
                        cluster_id=_cluster_id_for_log,
                        current_weight=_cluster_cw,
                        alloc=alloc,
                        portfolio_value=active_sleeve_value,
                        cluster_limit=_cluster_limit,
                        enforcement_cfg=_CLP["enforcement"],
                        min_order_amount=RISK_LIMITS["min_order_amount"],
                        # The underweight exemption is NOT applied to the per-cluster cap:
                        # building the sleeve toward target is fine, but it must not be
                        # built up concentrated in one cluster (the cold-start gap).
                        is_underweight=False,
                    )
                    _cluster_pw = _cluster_cw + (_new_alloc / active_sleeve_value)
                    if _cluster_decision == "blocked":
                        logger.info(
                            "Skipping %s: %s", symbol, _cluster_block_reason,
                        )
                        self._log_candidate(
                            symbol, row, "SKIP", False, f"cluster_cap: {_cluster_block_reason}",
                            _sent_result, True, "", alloc, 0.0, regime, _cand_rank, agg_df,
                        )
                        skipped.append(symbol)
                        continue
                    if _cluster_decision == "downsized":
                        logger.info(
                            "%s: cluster_cap downsized $%.2f → $%.2f",
                            symbol, alloc, _new_alloc,
                        )
                        alloc = _new_alloc

            # ── GICS sector concentration cap (active-sleeve relative) ────────
            # max_sector_weight (concentration_limits) was previously enforced only
            # in warning logs. Enforce it here against the active-sleeve target so a
            # single sector (shipping/banks/REITs) can't dominate the stock book.
            # "Unknown"/blank sectors are skipped: the running sector_exposure dict
            # buckets the ETF sleeve under "Unknown", so capping it is meaningless
            # (the cluster cap + can_buy's max_sector_pct still apply to those names).
            if (
                _CLP["enabled"] and not _CLP["warn_only"]
                and _CLP["apply_to"]["active_sleeve"]
                and active_sleeve_value > 0 and alloc > 0
            ):
                _sec = self._candidate_sector(symbol, row, agg_df)
                _max_sec_w = float(_CLP["max_sector_weight"])
                if _sec and _sec != "Unknown" and _max_sec_w > 0:
                    # Same cap math as the cluster cap — reuse the tested pure function
                    # with the sector label as the "cluster" and the sector weight cap.
                    from portfolio.exposure.cluster_enforcement import cluster_buy_decision
                    _cur_sec_w = sector_exposure.get(_sec, 0.0) / active_sleeve_value
                    _sec_decision, _sec_alloc, _sec_reason = cluster_buy_decision(
                        symbol=symbol,
                        cluster_id=_sec,
                        current_weight=_cur_sec_w,
                        alloc=alloc,
                        portfolio_value=active_sleeve_value,
                        cluster_limit=_max_sec_w,
                        enforcement_cfg=_CLP["enforcement"],
                        min_order_amount=RISK_LIMITS["min_order_amount"],
                        is_underweight=False,
                    )
                    if _sec_decision == "blocked":
                        logger.info(
                            "Skipping %s: sector_cap %r would exceed %.0f%% of active "
                            "sleeve (%s)", symbol, _sec, _max_sec_w * 100, _sec_reason,
                        )
                        self._log_candidate(
                            symbol, row, "SKIP", False, f"sector_cap: {_sec}",
                            _sent_result, True, "", alloc, 0.0, regime, _cand_rank, agg_df,
                        )
                        skipped.append(symbol)
                        continue
                    if _sec_decision == "downsized":
                        logger.info(
                            "%s: sector_cap %r downsized $%.2f → $%.2f "
                            "(≤ %.0f%% of active sleeve)",
                            symbol, _sec, alloc, _sec_alloc, _max_sec_w * 100,
                        )
                        alloc = _sec_alloc

            n_remaining = len(candidates) - (_cand_rank - 1)
            _buy_dec = self._risk.can_buy(
                symbol, alloc, holdings, agg_df, portfolio_value, cash, sector_exposure,
                n_remaining_candidates=n_remaining,
            )
            ok, reason, adj_alloc = (
                _buy_dec.approved, _buy_dec.reason, _buy_dec.adjusted_allocation
            )
            if not ok:
                logger.info(f"Skipping {symbol}: {reason}")
                self._log_candidate(
                    symbol, row, "SKIP", False, reason,
                    _sent_result, False, reason, alloc, 0.0, regime, _cand_rank, agg_df,
                )
                skipped.append(symbol)
                continue

            wash_warning = self._check_wash_sale_risk(symbol)
            if wash_warning:
                if DIVIDEND_PARAMS.get("block_rebuy_on_wash_sale_risk"):
                    logger.warning(f"Blocking buy {symbol} — {wash_warning}")
                    self._log_candidate(
                        symbol, row, "SKIP", False, f"wash_sale: {wash_warning}",
                        _sent_result, True, wash_warning, alloc, 0.0, regime, _cand_rank, agg_df,
                    )
                    skipped.append(symbol)
                    continue
                logger.warning(f"TAX WARNING — {wash_warning}")

            try:
                if self._auto_approve or self._confirm(
                    f"Buy ${adj_alloc:,.2f} of {symbol}? "
                    f"({row['value_metric'] / total_value:.1%} of stock budget)"
                ):
                    _frac_result = self._broker.buy_fractional(symbol, adj_alloc)
                    ok_frac, detail = _frac_result.success, _frac_result.detail
                    if ok_frac:
                        purchased.append(symbol)
                        buys_made += 1
                        stock_deployed += adj_alloc
                        if _arch_pol is not None:
                            _arch_deployed_this_pass[_arch_pol.archetype] = (
                                _arch_deployed_this_pass.get(_arch_pol.archetype, 0.0) + adj_alloc
                            )
                        self._record_buy(symbol)
                        self._update_local_exposures_after_buy(
                            symbol, adj_alloc, agg_df, sector_exposure
                        )
                        self._log_candidate(
                            symbol, row, "BUY", True, "",
                            _sent_result, True, "", alloc, adj_alloc, regime, _cand_rank, agg_df,
                        )
                        time.sleep(0.5)
                    elif allow_ws_fallback:
                        logger.warning(
                            f"{symbol}: fractional failed ({detail}) — queued for whole-share fallback"
                        )
                        whole_share_queue.append((symbol, adj_alloc))
                    else:
                        logger.warning(f"{symbol}: fractional failed ({detail}) — skipping")
                        self._log_candidate(
                            symbol, row, "SKIP", False, f"fractional_failed: {detail}",
                            _sent_result, True, detail, alloc, 0.0, regime, _cand_rank, agg_df,
                        )
                        failed.append(symbol)
            except Exception as e:
                logger.error(f"Order failed for {symbol}: {e}")
                self._log_candidate(
                    symbol, row, "SKIP", False, f"order_exception: {e}",
                    _sent_result, True, str(e), alloc, 0.0, regime, _cand_rank, agg_df,
                )
                failed.append(symbol)

        if whole_share_queue:
            ws_purchased_before = len(purchased)
            self._process_whole_share_queue(
                whole_share_queue, agg_df, sector_exposure, purchased, failed, skipped,
                holdings, portfolio_value,
            )
            for sym in purchased[ws_purchased_before:]:
                ws_alloc = next((a for s, a in whole_share_queue if s == sym), 0.0)
                stock_deployed += ws_alloc

        logger.info(
            f"Buy summary: {len(purchased)} bought, {len(skipped)} skipped, {len(failed)} failed"
        )
        logger.info(f"Cash remaining: ${self._broker.get_cash():,.2f}")
        return purchased, skipped, failed

    # ------------------------------------------------------------------
    # Rebalance loop
    # ------------------------------------------------------------------

    def _compute_etf_sweep_amount(self, remaining_cash: float, effective_index_pct: float) -> float:
        """
        Compute how much of `remaining_cash` should be swept to ETFs.

        Uses target-allocation accounting: only fills the gap between the current
        ETF sleeve value and the target (portfolio_equity * effective_index_pct).
        This prevents double-buying ETFs when the sleeve is already at or above target.
        """
        if not ETFS or remaining_cash <= 0:
            return 0.0
        try:
            holdings = self._broker.get_holdings()
            portfolio_equity = self._broker.get_portfolio_value()
            if portfolio_equity <= 0:
                return remaining_cash * effective_index_pct

            current_etf_equity = sum(
                safe_float(holdings.get(etf, {}).get("equity")) or 0.0
                for etf in ETFS
            )
            target_etf = portfolio_equity * effective_index_pct
            deficit = max(0.0, target_etf - current_etf_equity)
            sweep = min(remaining_cash, deficit)
            logger.info(
                f"ETF allocation check — portfolio ${portfolio_equity:.2f} | "
                f"target ETF {effective_index_pct:.0%} = ${target_etf:.2f} | "
                f"current ETF ${current_etf_equity:.2f} | "
                f"deficit ${deficit:.2f} | sweep ${sweep:.2f}"
            )
            return sweep
        except Exception as exc:
            logger.warning(f"ETF sweep computation failed ({exc}) — using proportional fallback")
            return remaining_cash * effective_index_pct

    def rebalance(self, df: pd.DataFrame, regime: str = "bullish") -> None:
        """
        Run the full iteration loop: sell → buy → repeat until cash exhausted,
        then sweep any ETF-sleeve deficit into ETFs.

        ETF buys happen exactly once (iteration 1).  Sell proceeds in later
        iterations are fully available for stock buys.  The end-of-run sweep
        only fills the gap between current ETF value and the target weight —
        it never over-allocates to ETFs.
        """
        effective_index_pct, _ = self._resolve_regime_params(regime)

        # Reset per-run sentiment-failure flag. When sentiment is enabled but the
        # Claude API fails, the active sleeve is left undeployed; we must NOT sweep
        # that active-sleeve cash into ETFs (it would silently violate index_pct).
        self._sentiment_failed = False
        self._sentiment_run_cache = {}

        # Snapshot portfolio state at run start for diagnostics.
        try:
            _start_pv    = self._broker.get_portfolio_value()
            _start_cash  = self._broker.get_cash()
            _start_etf   = sum(
                safe_float(self._broker.get_holdings().get(etf, {}).get("equity")) or 0.0
                for etf in ETFS
            )
            _start_active = _start_pv - _start_cash - _start_etf
            logger.info(
                f"\n{'='*60}\n"
                f"PORTFOLIO SNAPSHOT (run start)\n"
                f"  Portfolio equity : ${_start_pv:,.2f}\n"
                f"  Cash             : ${_start_cash:,.2f}\n"
                f"  ETF sleeve       : ${_start_etf:,.2f}  "
                f"({_start_etf / max(_start_pv, 1):.1%} actual vs "
                f"{effective_index_pct:.0%} target)\n"
                f"  Active sleeve    : ${_start_active:,.2f}  "
                f"({_start_active / max(_start_pv, 1):.1%} actual vs "
                f"{1 - effective_index_pct:.0%} target)\n"
                f"  Regime           : {regime}\n"
                f"{'='*60}"
            )
        except Exception:
            _start_pv = 0.0

        # ── Concentration diagnostic (logs warnings; enforcement happens in buy_cycle) ──
        self._cluster_report = None  # populated below; consumed by buy_cycle
        try:
            from util import CONCENTRATION_LIMIT_PARAMS as _clp
            if _clp["enabled"]:
                from portfolio.exposure.cluster_concentration import run_concentration_check
                _agg_for_conc = read_data_as_pd("agg_data")
                _hold_for_conc = read_data_as_pd("holdings")
                _conc = run_concentration_check(
                    holdings_df=_hold_for_conc,
                    agg_df=_agg_for_conc,
                    etfs=ETFS,
                )
                if _conc is not None:
                    self._cluster_report = _conc
                    if _conc.has_violations:
                        logger.warning(
                            "=== CONCENTRATION WARNINGS ===\n%s",
                            "\n".join(_conc.summary_lines()),
                        )
                        _conc.log_warnings()
                    else:
                        logger.info("Concentration check: no violations")
        except Exception as _ce:
            logger.debug("Concentration check skipped: %s", _ce)

        permanently_skipped: set[str] = set()

        for iteration in range(1, MAX_ITERATIONS + 1):
            logger.info(
                f"\n{'='*60}\n"
                f"ITERATION {iteration}/{MAX_ITERATIONS} | "
                f"skipped so far: {len(permanently_skipped)}\n"
                f"{'='*60}"
            )

            if permanently_skipped:
                df = df[~df["symbol"].isin(permanently_skipped)].copy()

            if df.empty:
                logger.info("No remaining candidates — exiting")
                break

            made_buys = made_sells = False

            self._broker.clear_orders_cache()

            try:
                logger.info("=== SELL PHASE ===")
                sold, trimmed = self.sell_cycle()
                if sold or trimmed:
                    made_sells = True
            except Exception as e:
                logger.error(f"Sell phase error (iter {iteration}): {e}")
                sold, trimmed = [], []

            cash = self._broker.get_cash()
            if cash < RISK_LIMITS["min_order_amount"]:
                logger.info(
                    f"Cash ${cash:.2f} below min order "
                    f"${RISK_LIMITS['min_order_amount']:.2f} — skipping buy phase"
                )
                if not made_sells:
                    logger.info("No sells either — exiting loop")
                    break
                continue

            try:
                logger.info("=== BUY PHASE ===")
                purchased, skipped, failed = self.buy_cycle(
                    df,
                    is_first_iteration=(iteration == 1),
                    regime=regime,
                    effective_index_pct=effective_index_pct,
                )
                if purchased:
                    made_buys = True
                permanently_skipped.update(skipped)
                permanently_skipped.update(failed)
            except Exception as e:
                logger.error(f"Buy phase error (iter {iteration}): {e}")

            if not made_buys and not made_sells:
                logger.info("No activity this iteration — exiting loop")
                break

        # End-of-run ETF sweep: only fill the ETF-sleeve deficit.
        # Refresh the open-orders cache first so get_cash() subtracts buys placed
        # in the final iteration (the cache is otherwise only cleared at the start
        # of each iteration, so the sweep would double-count committed cash).
        self._broker.clear_orders_cache()
        remaining = self._broker.get_cash()
        if self._sentiment_failed:
            # Sentiment was enabled but the Claude API failed, so the active sleeve
            # was never deployed. Hold the leftover cash for the next run instead of
            # sweeping it into ETFs — otherwise an API outage silently routes the
            # active allocation to the index sleeve, violating index_pct.
            logger.warning(
                f"Sentiment failed this run — retaining ${remaining:,.2f} as cash "
                f"(ETF sweep skipped; active sleeve will be retried next run)"
            )
        elif remaining > 0 and ETFS:
            sweep_amount = self._compute_etf_sweep_amount(remaining, effective_index_pct)
            if sweep_amount < RISK_LIMITS["min_order_amount"]:
                logger.info(
                    f"ETF sweep skipped — deficit ${sweep_amount:.2f} < "
                    f"min_order ${RISK_LIMITS['min_order_amount']:.2f} "
                    f"(ETFs at or near target)"
                )
            else:
                per_etf = sweep_amount / len(ETFS)
                logger.info(
                    f"=== CASH SWEEP: ${sweep_amount:,.2f} → ETFs "
                    f"(${per_etf:.2f} each, ${remaining - sweep_amount:.2f} retained) ==="
                )
                _swept = 0.0
                for etf in ETFS:
                    try:
                        result = self._broker.buy_fractional(etf, per_etf)
                        if result.success:
                            _swept += per_etf
                            logger.info(f"Sweep {etf}: {result.state}")
                        else:
                            logger.warning(f"Sweep {etf}: order failed — {result.detail}")
                    except Exception as e:
                        logger.error(f"Sweep {etf} failed: {e}")
                if _swept > 0:
                    log_cash_sweep(_swept, notes="end-of-run ETF deficit sweep")

        # Run-end summary.
        try:
            _end_pv   = self._broker.get_portfolio_value()
            _end_cash = self._broker.get_cash()
            _end_etf  = sum(
                safe_float(self._broker.get_holdings().get(etf, {}).get("equity")) or 0.0
                for etf in ETFS
            )
            _end_active = _end_pv - _end_cash - _end_etf
            logger.info(
                f"\n{'='*60}\n"
                f"STRATEGY COMPLETE\n"
                f"  Portfolio equity : ${_end_pv:,.2f}\n"
                f"  Cash remaining   : ${_end_cash:,.2f}\n"
                f"  ETF sleeve       : ${_end_etf:,.2f}  ({_end_etf / max(_end_pv, 1):.1%})\n"
                f"  Active sleeve    : ${_end_active:,.2f}  ({_end_active / max(_end_pv, 1):.1%})\n"
                f"  Total skipped    : {len(permanently_skipped)}\n"
                f"{'='*60}"
            )
        except Exception:
            logger.info(
                f"\n{'='*60}\n"
                f"STRATEGY COMPLETE\n"
                f"  Final cash: ${self._broker.get_cash():,.2f}\n"
                f"  Total skipped: {len(permanently_skipped)}\n"
                f"{'='*60}"
            )
