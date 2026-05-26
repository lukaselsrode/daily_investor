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
from typing import TYPE_CHECKING, Optional

import pandas as pd

from portfolio.sell_engine import evaluate_sell_candidate
from strategy.regimes.detector import get_current_regime
from util import (
    AUTO_APPROVE,
    CANDIDATE_ROTATION_PARAMS,
    CONFIDENCE_THRESHOLD,
    DATA_DIRECTORY,
    DIVIDEND_PARAMS,
    ETF_RISK_PARAMS,
    ETFS,
    HARVEST_PARAMS,
    INDEX_PCT,
    MAX_ITERATIONS,
    METRIC_KEYS,
    METRIC_THRESHOLD,
    REGIME_PARAMS,
    RELIABILITY_PARAMS,
    RISK_LIMITS,
    SELL_SENTIMENT_OVERRIDE_CONFIDENCE,
    USE_SENTIMENT_ANALYSIS,
    WEEKLY_INVESTMENT,
    read_data_as_pd,
    safe_float,
    store_data_as_csv,
)

if TYPE_CHECKING:
    from execution.base import BrokerAdapter
    from portfolio.risk import RiskManager
    from portfolio.harvest import HarvestManager

logger = logging.getLogger(__name__)

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
        broker: "BrokerAdapter",
        risk: "RiskManager",
        harvest: "HarvestManager",
        auto_approve: Optional[bool] = None,
        use_sentiment: Optional[bool] = None,
    ) -> None:
        self._broker = broker
        self._risk = risk
        self._harvest = harvest
        self._auto_approve   = AUTO_APPROVE        if auto_approve  is None else auto_approve
        self._use_sentiment  = USE_SENTIMENT_ANALYSIS if use_sentiment is None else use_sentiment

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

    def _check_wash_sale_risk(self, symbol: str) -> Optional[str]:
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

    def _load_news_for_symbol(self, symbol: str, news_df: Optional[pd.DataFrame]) -> dict:
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
        agg_df: Optional[pd.DataFrame],
        sector_exposure: dict[str, float],
    ) -> None:
        if agg_df is None or agg_df.empty or "symbol" not in agg_df.columns:
            return
        row = agg_df[agg_df["symbol"] == symbol]
        if row.empty:
            return
        sector = str(row.iloc[0].get("sector") or "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + allocation

    def _process_whole_share_queue(
        self,
        queue: list[tuple[str, float]],
        agg_df: Optional[pd.DataFrame],
        sector_exposure: dict[str, float],
        purchased: list[str],
        failed: list[str],
        skipped: list[str],
        holdings: Optional[dict] = None,
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

            current_price: Optional[float] = None
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
        sentiment: Optional[dict],
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
                bc_row: Optional[dict] = None
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
                if order.get("state") not in ("confirmed", "queued", "unconfirmed"):
                    continue
                sym = order.get("symbol")
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

    def sell_cycle(self) -> list[str]:
        """
        Evaluate all holdings for sell conditions.

        Hard sells (stop-loss, yield-trap, quality floor) execute immediately.
        Soft sells (take-profit, weak value) are optionally held by sentiment.
        Sentiment can only override soft sells — never hard sells.
        ETF positions are protected from stock stop-loss logic; they are only
        exited by the ETF MA filter (defensive regime only).
        """
        sold: list[str] = []

        try:
            holdings = self._broker.get_holdings()
            self._broker.enrich_holdings_created_at(holdings)
            self._save_holdings_csv(holdings)
        except Exception as e:
            logger.error(f"Could not fetch holdings: {e}")
            return sold

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

            decision = evaluate_sell_candidate(symbol, data, metrics_row, peak_price=peaks.get(symbol))

            _all_evaluated[symbol] = {
                "data": data,
                "metrics_row": metrics_row,
                "decision": decision,
            }

            if not decision["should_sell"]:
                continue

            if decision["severity"] == "hard":
                hard_sells[symbol] = decision
            else:
                soft_sells[symbol] = decision

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
                    sentiment_results = get_batch_sentiment_recommendations(stocks_data, action="sell")
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
            for symbol, decision in soft_sells.items():
                if symbol in held_on_sentiment:
                    continue
                pct = decision.get("percent_change")
                pct_str = f" | P/L={pct:.1%}" if pct is not None else ""
                logger.info(f"SOFT SELL {symbol} | {decision['reason']}{pct_str}")
                quantity = float(holdings[symbol].get("quantity", 0))
                if self._confirm(f"Sell {quantity} shares of {symbol}?") and \
                        self._broker.sell(symbol, quantity).success:
                    sold.append(symbol)
                    self._record_sell_event(symbol, was_loss=(pct is not None and pct < 0))
                    if decision.get("exit_type") == "harvest_exit":
                        harvest_proceeds += safe_float(holdings[symbol].get("equity"), 0.0) or 0.0

            if harvest_proceeds >= HARVEST_PARAMS["min_harvest_amount"]:
                self._harvest.route_proceeds(harvest_proceeds, self._broker)

        if sold:
            for sym in sold:
                peaks.pop(sym, None)
            self._save_peak_prices(peaks)

        self._log_all_holding_decisions(
            all_evaluated=_all_evaluated,
            sold=set(sold),
            held_on_sentiment=held_on_sentiment,
            regime=get_current_regime(),
            bc_df=_bc_df,
            agg_df=agg_df,
        )

        logger.info(
            f"Sell summary: {scanned} scanned | "
            f"{len(hard_sells)} hard | {len(soft_sells)} soft candidates | "
            f"{len(held_on_sentiment)} held on sentiment | "
            f"{len(sold)} executed | "
            f"{len(hard_sells) + len(soft_sells) - len(sold)} skipped/no-action"
        )
        return sold

    # ------------------------------------------------------------------
    # Buy cycle
    # ------------------------------------------------------------------

    def buy_cycle(
        self,
        df: pd.DataFrame,
        is_first_iteration: bool = True,
        regime: str = "bullish",
    ) -> tuple[list, list, list]:
        """
        Execute buy orders. Returns (purchased, skipped, failed).
        regime controls index_pct and max_buys overrides.
        """
        rp = REGIME_PARAMS
        effective_index_pct = INDEX_PCT
        effective_max_buys  = RISK_LIMITS["max_buys_per_rebalance"]

        if regime == "defensive":
            ovr = rp.get("defensive", {})
            if ovr.get("index_pct_override") is not None:
                effective_index_pct = float(ovr["index_pct_override"])
            if ovr.get("max_buys_override") is not None:
                effective_max_buys = int(ovr["max_buys_override"])
            logger.info(
                f"DEFENSIVE regime: index_pct={effective_index_pct:.0%}  max_buys={effective_max_buys}"
            )
        elif regime == "neutral":
            ovr = rp.get("neutral", {})
            if ovr.get("index_pct_override") is not None:
                effective_index_pct = float(ovr["index_pct_override"])
            if ovr.get("max_buys_override") is not None:
                effective_max_buys = int(ovr["max_buys_override"])
            logger.info(f"NEUTRAL regime: index_pct={effective_index_pct:.0%}")

        total_cash   = self._broker.get_cash()
        etf_amount   = total_cash * effective_index_pct
        stock_amount = total_cash - etf_amount
        logger.info(
            f"Allocating ${etf_amount:.2f} to ETFs, ${stock_amount:.2f} to stocks (regime={regime})"
        )

        if is_first_iteration and etf_amount > 0 and \
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
        candidates = df[df["value_metric"] >= METRIC_THRESHOLD].copy()
        logger.info(
            f"Pre-filter: {len(df)} → {len(candidates)} stocks "
            f"(value_metric ≥ {METRIC_THRESHOLD})"
        )

        if candidates.empty:
            logger.warning(f"No stocks pass value_metric ≥ {METRIC_THRESHOLD}")
            return [], df["symbol"].tolist(), []

        if "strategy_bucket" in candidates.columns:
            contrarian_mask = candidates["strategy_bucket"] == "contrarian_watchlist"
            contrarian_syms = candidates.loc[contrarian_mask, "symbol"].tolist()
            candidates = candidates[~contrarian_mask].copy()
            if contrarian_syms:
                logger.info(f"Contrarian watchlist excluded from buys: {contrarian_syms}")

        if candidates.empty:
            logger.warning("No candidates remain after contrarian exclusion")
            return [], df["symbol"].tolist(), []

        if RELIABILITY_PARAMS["enabled"] and "reliability_score" in candidates.columns:
            min_rel = RELIABILITY_PARAMS["min_reliability_score"]
            rel_scores = pd.to_numeric(candidates["reliability_score"], errors="coerce").fillna(0.0)
            low_rel = candidates[rel_scores < min_rel]["symbol"].tolist()
            candidates = candidates[rel_scores >= min_rel].copy()
            if low_rel:
                logger.info(
                    f"Reliability gate ({min_rel:.2f}): excluded {len(low_rel)} "
                    f"low-quality candidates: {low_rel[:10]}"
                )

        if candidates.empty:
            logger.warning("No candidates remain after reliability gate")
            return [], df["symbol"].tolist(), []

        cooldown_days = CANDIDATE_ROTATION_PARAMS["buy_cooldown_days"]
        if cooldown_days > 0 and "symbol" in candidates.columns:
            buy_history = self._load_buy_history()
            today = datetime.date.today()
            cooled_out = [
                sym for sym in candidates["symbol"]
                if sym in buy_history and (today - buy_history[sym]).days < cooldown_days
            ]
            if cooled_out:
                candidates = candidates[~candidates["symbol"].isin(cooled_out)].copy()
                logger.info(
                    f"Buy cooldown ({cooldown_days}d): excluded {len(cooled_out)} "
                    f"recently-purchased symbols: {cooled_out[:10]}"
                )

        if candidates.empty:
            logger.warning("No candidates remain after buy cooldown filter")
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

        holdings        = self._broker.get_holdings()
        self._save_holdings_csv(holdings)
        portfolio_value = self._broker.get_portfolio_value()
        try:
            agg_df = read_data_as_pd("agg_data")
        except Exception:
            agg_df = None

        sentiment_results: dict[str, dict] = {}
        if self._use_sentiment:
            logger.info(f"Running batch sentiment on {len(candidates)} candidates...")
            stocks_data = self._build_stocks_data(candidates, action="buy")
            try:
                from data.sentiment import get_batch_sentiment_recommendations
                sentiment_results = get_batch_sentiment_recommendations(stocks_data, action="buy")
            except Exception:
                logger.error("Batch sentiment failed — all candidates skipped", exc_info=True)
                return [], candidates["symbol"].tolist(), []

        sector_exposure = (
            self._risk.get_sector_exposure(holdings, agg_df) if portfolio_value > 0 else {}
        )

        purchased, skipped, failed = [], [], []
        whole_share_queue: list[tuple[str, float]] = []
        allow_ws_fallback = RISK_LIMITS["allow_whole_share_fallback"]
        total_value  = candidates["value_metric"].sum()
        buys_made    = 0
        _cand_rank   = 0
        stock_deployed = 0.0

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
                result = sentiment_results.get(
                    symbol,
                    {"action": "HOLD", "sentiment": "neutral", "confidence": 0.0, "reasoning": "No result"},
                )
                logger.info(
                    f"{'='*60}\nBUY {symbol} | action={result.get('action')} "
                    f"sentiment={result.get('sentiment')} {result['confidence']:.1f}% | "
                    f"{result['reasoning']}\n{'='*60}"
                )
                if result.get("action") != "BUY" or result["confidence"] < CONFIDENCE_THRESHOLD:
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

    def rebalance(self, df: pd.DataFrame, regime: str = "bullish") -> None:
        """
        Run the full iteration loop: sell → buy → repeat until cash exhausted,
        then sweep remaining cash into ETFs.
        """
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
                sold = self.sell_cycle()
                if sold:
                    made_sells = True
            except Exception as e:
                logger.error(f"Sell phase error (iter {iteration}): {e}")

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
                    df, is_first_iteration=(iteration == 1), regime=regime
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

        remaining = self._broker.get_cash()
        if remaining > 0 and ETFS:
            per_etf = remaining / len(ETFS)
            if per_etf < RISK_LIMITS["min_order_amount"]:
                logger.info(
                    f"Sweep per-ETF ${per_etf:.2f} < min_order "
                    f"${RISK_LIMITS['min_order_amount']:.2f} — skipping sweep"
                )
            else:
                logger.info(f"=== CASH SWEEP: ${remaining:,.2f} → ETFs (${per_etf:.2f} each) ===")
                for etf in ETFS:
                    try:
                        result = self._broker.buy_fractional(etf, per_etf)
                        if result.success:
                            logger.info(f"Sweep {etf}: {result.state}")
                        else:
                            logger.warning(f"Sweep {etf}: order failed — {result.detail}")
                    except Exception as e:
                        logger.error(f"Sweep {etf} failed: {e}")

        logger.info(
            f"\n{'='*60}\n"
            f"STRATEGY COMPLETE\n"
            f"Final cash: ${self._broker.get_cash():,.2f}\n"
            f"Total skipped: {len(permanently_skipped)}\n"
            f"{'='*60}"
        )
