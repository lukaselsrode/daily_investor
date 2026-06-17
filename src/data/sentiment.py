"""
data/sentiment.py — Batch async + single-stock Claude sentiment analysis.

Two paths:
  - Batch async (buy candidates): get_batch_sentiment_recommendations()
  - Single-stock LangGraph (sell hold-check, backward compat): get_sentiment_recommendation()
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import random
import re
from typing import Literal

import anthropic
import robin_stocks.robinhood as rb
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

load_dotenv()

# These imports follow load_dotenv() intentionally so env vars (ANTHROPIC_API_KEY,
# etc.) are loaded before the imported modules read them at import time.
from core.utils import run_async  # noqa: E402
from data.cache import read_data_as_pd  # noqa: E402
from util import CONFIDENCE_THRESHOLD, METRIC_KEYS  # noqa: E402

logger = logging.getLogger("investment_bot")

# ---------------------------------------------------------------------------
# Batch config
# ---------------------------------------------------------------------------

BATCH_SIZE     = 6
MAX_CONCURRENT = 5
MAX_RETRIES    = 5

# Model id is config-driven (app.sentiment_model) with an env override
# (SENTIMENT_MODEL) so the final guard can be pointed at a newer Claude without
# a code change. Falls back to the previous hardcoded default.
def _resolve_model() -> str:
    env = os.getenv("SENTIMENT_MODEL")
    if env:
        return env
    try:
        from util import SENTIMENT_MODEL as _SM
        if _SM:
            return str(_SM)
    except Exception:
        pass
    return "claude-opus-4-8"

_CLAUDE_MODEL = _resolve_model()

# ---------------------------------------------------------------------------
# Regime-aware guard framing
# ---------------------------------------------------------------------------
# The sentiment layer is a FINAL GUARD, not a backtested factor. It cannot be
# validated historically (no point-in-time news corpus), so we keep its logic
# conservative and transparent. The one principled enhancement: tell the analyst
# what market regime we're in, so the guard leans defensive in fear tapes and
# permissive in confirmed bull tapes — mirroring the regime-conditional scoring
# (momentum tilt in bull, caution in defensive) without inventing new dials.

_REGIME_GUIDANCE = {
    "bullish": (
        "MARKET REGIME: BULLISH (SPY above 200DMA, low VIX). Trend is up. "
        "Give momentum and growth names the benefit of the doubt; do not veto a "
        "buy purely for being extended. Only flag clear, specific red flags."
    ),
    "neutral": (
        "MARKET REGIME: NEUTRAL (mixed signals). Balance upside and downside; "
        "weight company-specific evidence over market direction."
    ),
    "defensive": (
        "MARKET REGIME: DEFENSIVE / FEAR (SPY below 200DMA or elevated VIX). "
        "Capital preservation first. Be more skeptical of new buys, weight "
        "downside risks and deteriorating theses more heavily, and require a "
        "clearly bullish, well-supported case before overriding a sell."
    ),
}


def _regime_prefix(regime: str | None) -> str:
    """Return a leading regime-guidance block for the system prompt, or ''."""
    if not regime:
        return ""
    g = _REGIME_GUIDANCE.get(str(regime).lower())
    return (g + "\n\n") if g else ""

# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class SentimentAnalysisState(TypedDict):
    symbol: str
    action: Literal["buy", "sell"]
    news_sentiment: dict
    reddit_sentiment: dict
    position_info: dict
    fundamental_metrics: dict
    analysis: str
    sentiment_action: Literal["BUY", "HOLD", "SELL"]
    sentiment: Literal["bullish", "neutral", "bearish"]
    confidence: float
    reasoning: str
    skip_analysis: bool


# ---------------------------------------------------------------------------
# Client initialisation
# ---------------------------------------------------------------------------

def _make_langchain_model() -> ChatAnthropic | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — sentiment analysis disabled")
        return None
    # Opus 4.8: temperature is rejected (400); adaptive thinking aids the guard's
    # judgment. max_tokens must be set explicitly (langchain default 1024 is too
    # tight once thinking is on) — give the short answer headroom past thinking.
    return ChatAnthropic(
        model=_CLAUDE_MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
    )


def _make_async_client() -> anthropic.AsyncAnthropic | None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    return anthropic.AsyncAnthropic()


_lc_model     = _make_langchain_model()
_async_client = _make_async_client()

# ---------------------------------------------------------------------------
# Shared prompt helpers
# ---------------------------------------------------------------------------

_SCORING_GUIDE = """\
Scoring guide:
- VALUE_SCORE measures PE/PB cheapness relative to sector thresholds; higher is better.
- PE_COMP/PB_COMP > 1 indicates cheaper than sector threshold (positive signal).
- INCOME_SCORE rewards reasonable dividend yield.
- QUALITY_SCORE is a basic quality/liquidity proxy; higher is better.
- MOMENTUM_SCORE measures 52-week price-location and 1-month direction:
    position_52w < 0.15 may indicate a falling-knife risk (-0.4 base).
    position_52w 0.35–0.75 is healthy middle range (+0.3 base).
    position_52w 0.75–0.95 shows strong momentum (+0.5 base).
    position_52w > 0.95 may indicate extension near 52-week highs (+0.2 base).
    1M Return >= +5% with position_52w < 0.40 adds +0.15 (recovering from lows).
    1M Return <= -10% with position_52w < 0.40 adds -0.20 (falling knife warning).
- YIELD_TRAP_FLAG=True → dividend likely caused by price collapse (major warning).
- FINAL_VALUE_METRIC combines value + quality + income + momentum; higher is better.
- Buy/Sell Ratio > 1 supports buying; < 1 is a warning sign."""


def _format_news(news_data: dict, symbol: str) -> str:
    """Render up to 3 items for the active-sleeve LLM. News and social (Reddit/X) items are
    rendered UNIFORMLY as articles — title, source/provenance (publisher + api_source), date,
    link, summary/raw text — so the model reads them as one list and judges them together.

    We deliberately do NOT surface a precomputed social bullish/bearish label or net score here:
    aggregation and sentiment judgement are the LLM's job, and injecting a separate social score
    would treat social differently, bias the model, and double-count it against the news. Raw
    engagement counts (upvotes/comments) are factual provenance — not a verdict — and may show."""
    lines = []
    for a in news_data.get(symbol, [])[:3]:
        if not isinstance(a, dict):
            continue
        title   = a.get("title", "")
        pub     = a.get("publisher", "")
        date    = a.get("formatted_date", "")
        summary = a.get("summary", "")
        if title and pub:
            src  = a.get("api_source", "")
            link = a.get("link", "")
            head = f"  Publisher: {pub}  Date: {date}"
            if src:
                head += f"  Source: {src}"
            lines += [f"• {title[:100]}", head]
            # Factual provenance only (counts, not a sentiment verdict). Present on social items.
            eng = a.get("engagement")
            if isinstance(eng, dict) and ("score" in eng or "num_comments" in eng):
                lines.append(
                    f"  Engagement: {eng.get('score', 0)} upvotes, "
                    f"{eng.get('num_comments', 0)} comments")
            if link:
                lines.append(f"  Link: {link}")
            lines.append(f"  Summary: {summary[:200]}")
    return "\n".join(lines) if lines else "No news articles found"


def _forward_signals_text(symbol: str) -> str:
    """Forward-looking catalyst signals the backward quant factors can't see — earnings surprise +
    analyst consensus, from FMP (cache-first, best-effort). Lets the LLM weigh whether a beaten-down
    name (negative momentum) actually has the market pricing in an inflection. Never raises."""
    try:
        from data import fmp_client as fmp
        parts: list[str] = []
        earn = fmp.statement(symbol, "earnings", allow_fetch=True)
        if earn is not None and "epsActual" in earn.columns:
            rep = earn[earn["epsActual"].notna()]
            if len(rep):
                r = rep.iloc[0]
                act, est = r.get("epsActual"), r.get("epsEstimated")
                if act is not None and est:
                    parts.append(f"last earnings {r.get('date')}: EPS {act} vs est {est} "
                                 f"({(float(act) - float(est)) / abs(float(est)) * 100:+.0f}% surprise)")
        grades = fmp.statement(symbol, "grades-consensus", allow_fetch=True)
        if grades is not None and len(grades):
            g = grades.iloc[0]
            buy = int(g.get("strongBuy", 0) or 0) + int(g.get("buy", 0) or 0)
            sell = int(g.get("sell", 0) or 0) + int(g.get("strongSell", 0) or 0)
            parts.append(f"analyst consensus: {buy} buy / {int(g.get('hold', 0) or 0)} hold / {sell} sell")
        return "; ".join(parts) if parts else "no forward signals available"
    except Exception:
        return "forward signals unavailable"


def _valuation_block(symbol: str, f: dict, news_text: str) -> str:
    return (
        f"STOCK: {symbol}\n"
        f"FUNDAMENTAL METRICS:\n"
        f"  PE={f.get('pe_ratio','N/A')}  PB={f.get('pb_ratio','N/A')}\n"
        f"  Dividend Yield={f.get('dividend_yield','N/A')}  Volume={f.get('volume','N/A')}\n"
        f"  Current Price={f.get('current_price','N/A')}\n"
        f"  52W Low={f.get('low_52w','N/A')}  52W High={f.get('high_52w','N/A')}\n"
        f"  52W Position={f.get('position_52w','N/A')}  1M Return={f.get('return_1m','N/A')}\n"
        f"  Industry={f.get('industry','N/A')}  Sector={f.get('sector','N/A')}\n\n"
        f"FACTOR SCORES:\n"
        f"  VALUE_SCORE={f.get('value_score','N/A')}  INCOME_SCORE={f.get('income_score','N/A')}\n"
        f"  QUALITY_SCORE={f.get('quality_score','N/A')}  MOMENTUM_SCORE={f.get('momentum_score','N/A')}\n"
        f"  PE_COMP={f.get('pe_comp','N/A')}  PB_COMP={f.get('pb_comp','N/A')}\n"
        f"  YIELD_TRAP_FLAG={f.get('yield_trap_flag','N/A')}\n"
        f"  FINAL_VALUE_METRIC={f.get('value_metric','N/A')}\n\n"
        f"ANALYST: Buy/Sell Ratio={f.get('buy_to_sell_ratio','N/A')}\n"
        f"FORWARD SIGNALS (catalysts the momentum factor can't see): {_forward_signals_text(symbol)}\n\n"
        f"NEWS:\n{news_text}"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(text: str) -> dict:
    result = {
        "action":    "HOLD",
        "sentiment": "neutral",
        "confidence": 0.0,
        "reasoning": "Could not parse response",
    }
    for line in text.strip().split("\n"):
        u = line.strip().upper()
        raw = line.strip()
        if u.startswith("ACTION:"):
            val = u.split(":", 1)[1].strip()
            if val in ("BUY", "HOLD", "SELL"):
                result["action"] = val
        elif u.startswith("SENTIMENT:"):
            val = raw.split(":", 1)[1].strip().lower()
            if val in ("bullish", "neutral", "bearish"):
                result["sentiment"] = val
        elif u.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(u.split(":", 1)[1].strip().replace("%", ""))
            except ValueError:
                pass
        elif u.startswith("REASONING:"):
            result["reasoning"] = raw.split(":", 1)[1].strip()
    return result


# ---------------------------------------------------------------------------
# Batch async path
# ---------------------------------------------------------------------------

_BUY_SYSTEM = (
    "You are a financial analyst screening buy candidates. "
    "For each stock, decide if it is worth buying now.\n"
    "Reply for EVERY stock in this EXACT format, one block per stock, nothing else:\n\n"
    "STOCK: <SYMBOL>\n"
    "ACTION: <BUY|HOLD|SELL>\n"
    "SENTIMENT: <bullish|neutral|bearish>\n"
    "CONFIDENCE: <0-100>%\n"
    "REASONING: <one sentence>\n"
    "---\n\n"
    "ACTION meanings: BUY = proceed with purchase, HOLD = skip for now, SELL = avoid/not a buy candidate.\n"
)

_SELL_SYSTEM = (
    "You are a financial analyst reviewing positions that have triggered soft sell conditions. "
    "For each, decide whether the position should be HELD (override the sell) or SOLD (confirm the sell).\n"
    "Reply for EVERY stock in this EXACT format, one block per stock, nothing else:\n\n"
    "STOCK: <SYMBOL>\n"
    "ACTION: <BUY|HOLD|SELL>\n"
    "SENTIMENT: <bullish|neutral|bearish>\n"
    "CONFIDENCE: <0-100>%\n"
    "REASONING: <one sentence>\n"
    "---\n\n"
    "ACTION meanings: HOLD = override the sell (you see a clear bullish recovery case), "
    "SELL = confirm the sell, BUY = rare strong reversal signal.\n"
    "IMPORTANT: Only return HOLD if SENTIMENT is bullish. "
    "A bearish or deteriorating thesis must result in SELL.\n"
)


def _build_batch_prompt(batch: list[dict], action: str, regime: str | None = None) -> tuple[str, str]:
    system = _regime_prefix(regime) + (_BUY_SYSTEM if action == "buy" else _SELL_SYSTEM) + _SCORING_GUIDE
    blocks = [
        _valuation_block(
            item["symbol"],
            item.get("fundamental_metrics", {}),
            _format_news(item.get("news_sentiment", {}), item["symbol"]),
        )
        for item in batch
    ]
    context = "buy candidates" if action == "buy" else "soft sell candidates"
    user = f"Analyze these {len(batch)} {context}:\n\n" + "\n\n---\n\n".join(blocks)
    return system, user


def _api_error_sentinel() -> dict:
    """Synthetic HOLD result emitted when the Claude API fails after all retries.

    The "api_error" marker lets callers distinguish a real HOLD from an outage:
    if EVERY result in a batch is this sentinel, the caller must treat the run
    as a sentiment failure (hold cash) rather than "all candidates rejected".
    """
    return {
        "action": "HOLD", "sentiment": "neutral",
        "confidence": 0.0, "reasoning": "API error after retries",
        "api_error": True,
    }


def is_api_error_sentinel(result: dict | None) -> bool:
    """True when `result` is a synthetic API-error sentinel, not a real Claude verdict."""
    return bool(result) and bool(result.get("api_error", False))


def _parse_batch_response(raw: str, batch: list[dict]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for block in re.split(r"\n---+\n?", raw.strip()):
        block = block.strip()
        if not block:
            continue
        sym_m = re.search(r"STOCK:\s*([A-Z.\-]+)", block, re.IGNORECASE)
        if not sym_m:
            continue
        symbol = sym_m.group(1).upper()
        results[symbol] = _parse_response(block)

    for item in batch:
        sym = item["symbol"]
        if sym not in results:
            logger.warning(f"No result parsed for {sym} — defaulting HOLD/neutral")
            results[sym] = {
                "action": "HOLD", "sentiment": "neutral",
                "confidence": 0.0, "reasoning": "Missing from Claude response",
            }

    return results


async def _call_batch_async(
    batch: list[dict],
    action: str,
    semaphore: asyncio.Semaphore,
    regime: str | None = None,
) -> dict[str, dict]:
    if not _async_client:
        return {}

    system_prompt, user_prompt = _build_batch_prompt(batch, action, regime)

    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                response = await _async_client.messages.create(
                    model=_CLAUDE_MODEL,
                    max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                # Adaptive thinking emits a thinking block first — scan for the
                # text block rather than assuming content[0].
                if response.stop_reason == "max_tokens":
                    logger.warning("Batch sentiment hit max_tokens — answer may be truncated")
                text = next((b.text for b in response.content if b.type == "text"), "")
                return _parse_batch_response(text, batch)

            except anthropic.RateLimitError:
                wait = (2 ** attempt) * (1 + random.random())
                logger.warning(f"Rate-limited (attempt {attempt+1}/{MAX_RETRIES}), sleeping {wait:.1f}s")
                await asyncio.sleep(wait)

            except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
                wait = (2 ** attempt) * (1 + random.random())
                logger.warning(f"Transient error: {exc} — sleeping {wait:.1f}s")
                await asyncio.sleep(wait)

            except Exception as exc:
                logger.error(f"Unrecoverable batch error: {exc}")
                break

    return {item["symbol"]: _api_error_sentinel() for item in batch}


async def _run_all_batches(stocks_data: list[dict], action: str, regime: str | None = None) -> dict[str, dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    chunks = [stocks_data[i: i + BATCH_SIZE] for i in range(0, len(stocks_data), BATCH_SIZE)]
    logger.info(f"Dispatching {len(chunks)} batch(es) for {len(stocks_data)} stocks (regime={regime or 'n/a'})")
    chunk_results = await asyncio.gather(*[_call_batch_async(c, action, semaphore, regime) for c in chunks])
    merged: dict[str, dict] = {}
    for r in chunk_results:
        merged.update(r)
    return merged


def get_batch_sentiment_recommendations(
    stocks_data: list[dict],
    action: str = "buy",
    regime: str | None = None,
) -> dict[str, dict]:
    """Analyze multiple stocks in one async round-trip.

    `regime` ("bullish"|"neutral"|"defensive") prepends regime-specific guidance
    to the analyst system prompt: the final guard leans defensive in fear tapes
    and permissive in confirmed-bull tapes. None = regime-neutral (legacy behavior).
    """
    if not _async_client:
        logger.warning("Async client unavailable — falling back to per-stock analysis")
        return {item["symbol"]: get_sentiment_recommendation(item["symbol"], action) for item in stocks_data}

    return run_async(_run_all_batches(stocks_data, action, regime))


# ---------------------------------------------------------------------------
# Single-stock LangGraph path
# ---------------------------------------------------------------------------

def _load_fundamentals(symbol: str) -> tuple[dict, bool]:
    try:
        agg_df = read_data_as_pd("agg_data")
        if agg_df is not None and not agg_df.empty:
            row = agg_df[agg_df["symbol"] == symbol]
            if not row.empty:
                r = row.iloc[0]
                return {k: r.get(k) for k in METRIC_KEYS}, True
    except Exception as e:
        logger.error(f"Error loading fundamentals for {symbol}: {e}")
    return {k: None for k in METRIC_KEYS}, False


def _load_news(symbol: str) -> tuple[dict, bool]:
    try:
        news_df = read_data_as_pd("news")
        if news_df is not None and not news_df.empty:
            rows = news_df[news_df["symbol"] == symbol]["news"]
            if not rows.empty:
                raw = rows.iloc[0] if len(rows) == 1 else rows.tolist()
                if isinstance(raw, str):
                    raw = json.loads(raw)
                return {symbol: raw}, True
    except Exception as e:
        logger.error(f"Error loading news for {symbol}: {e}")
    return {}, False


def _process_position(positions: list, symbol: str) -> dict:
    pos = next((p for p in positions if p.get("symbol") == symbol), None)
    if not pos:
        return {"has_position": False}
    try:
        try:
            price = float(rb.stocks.get_latest_price(symbol)[0])
        except Exception:
            price = 0.0
        qty = float(pos.get("quantity", 0))
        avg = float(pos.get("average_buy_price", 0))
        return {
            "has_position": qty > 0,
            "quantity": qty,
            "average_buy_price": avg,
            "current_price": price,
            "current_value": qty * price,
            "cost_basis": qty * avg,
            "unrealized_pl": (price - avg) * qty,
            "unrealized_pl_pct": ((price / avg) - 1) * 100 if avg else 0,
            "days_held": (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.datetime.fromisoformat(pos.get("created_at", "").replace("Z", "+00:00"))
            ).days,
        }
    except Exception as e:
        return {"has_position": False, "error": str(e)}


def gather_sentiments(state: SentimentAnalysisState) -> dict:
    symbol = state["symbol"]
    logger.info(f"Gathering sentiments for {symbol}...")

    try:
        positions = rb.get_all_positions()
    except Exception:
        positions = []

    fundamentals, has_fundamentals = _load_fundamentals(symbol)
    news_data, has_news = _load_news(symbol)

    has_meaningful = (
        has_fundamentals
        and fundamentals.get("pe_ratio") is not None
        and fundamentals.get("pb_ratio") is not None
    )

    if not has_news and not has_meaningful:
        logger.warning(f"No valid data for {symbol} — skipping Claude call")
        return {
            "sentiment_action": "HOLD",
            "sentiment": "neutral",
            "confidence": 0.0,
            "reasoning": "No valid news or fundamental data available",
            "skip_analysis": True,
        }

    return {
        "news_sentiment":      news_data,
        "reddit_sentiment":    {},
        "position_info":       _process_position(positions, symbol),
        "fundamental_metrics": fundamentals,
        "skip_analysis":       False,
    }


def analyze_sentiment(state: SentimentAnalysisState) -> dict:
    symbol = state["symbol"]
    action = state["action"]

    if not _lc_model:
        return {"recommendation": "NEUTRAL", "confidence": 0.0, "reasoning": "Model not initialised"}

    news_text = _format_news(state["news_sentiment"], symbol)
    block = _valuation_block(symbol, state["fundamental_metrics"], news_text)

    system = (_BUY_SYSTEM if action == "buy" else _SELL_SYSTEM) + _SCORING_GUIDE
    context = "buy candidate" if action == "buy" else "soft sell candidate"
    user = (
        f"Analyze the following {context} — {symbol}:\n\n{block}\n\n"
        "Respond EXACTLY:\n"
        "ACTION: [BUY|HOLD|SELL]\n"
        "SENTIMENT: [bullish|neutral|bearish]\n"
        "CONFIDENCE: [0-100]%\n"
        "REASONING: [2-3 sentences with specific metric values]"
    )

    try:
        response = _lc_model.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        # With adaptive thinking, .content is a list of blocks; .text concatenates
        # only the text blocks (skips thinking) → keep _parse_response on a string.
        text = response.text
        result = _parse_response(text)
        result["analysis"] = text
        if result["confidence"] < CONFIDENCE_THRESHOLD:
            return {
                "sentiment_action": "HOLD",
                "sentiment": "neutral",
                "confidence": result["confidence"],
                "reasoning": f"Confidence {result['confidence']}% below threshold {CONFIDENCE_THRESHOLD}%",
            }
        return {"sentiment_action": result["action"], **result}
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {symbol}: {e}")
        return {
            "sentiment_action": "HOLD", "sentiment": "neutral",
            "confidence": 0.0, "reasoning": f"Analysis error: {e}",
        }


def _route_after_gather(state: SentimentAnalysisState) -> str:
    return END if state.get("skip_analysis", False) else "analyze_sentiment"


def _build_workflow() -> object:
    wf = StateGraph(SentimentAnalysisState)
    wf.add_node("gather_sentiments", gather_sentiments)
    wf.add_node("analyze_sentiment", analyze_sentiment)
    wf.add_conditional_edges(
        "gather_sentiments",
        _route_after_gather,
        {"analyze_sentiment": "analyze_sentiment", END: END},
    )
    wf.add_edge("analyze_sentiment", END)
    wf.set_entry_point("gather_sentiments")
    return wf.compile()


_workflow = _build_workflow()


def get_sentiment_recommendation(symbol: str, action: str) -> dict:
    """Single-stock path — used for sell hold-checks and backward compatibility."""
    initial: SentimentAnalysisState = {
        "symbol": symbol,
        "action": action,
        "news_sentiment": {},
        "reddit_sentiment": {},
        "position_info": {},
        "fundamental_metrics": {},
        "analysis": "",
        "sentiment_action": "HOLD",
        "sentiment": "neutral",
        "confidence": 0.0,
        "reasoning": "",
        "skip_analysis": False,
    }
    try:
        final = _workflow.invoke(initial)
        return {
            "action":    final.get("sentiment_action", "HOLD"),
            "sentiment": final.get("sentiment", "neutral"),
            "confidence": final["confidence"],
            "reasoning":  final["reasoning"],
        }
    except Exception as e:
        logger.error(f"Workflow error for {symbol}: {e}")
        return {"action": "HOLD", "sentiment": "neutral", "confidence": 0.0, "reasoning": f"Workflow error: {e}"}


