"""
data/classification_arbiter.py — FMP cross-validation of sector classification.

A borderline-only second-source layer on top of the normalization map + manual
CLASSIFICATION_OVERRIDES. For each candidate whose Robinhood sector disagrees
MATERIALLY with FMP's (GICS-mapped) sector — i.e. the implied PE benchmark swings
beyond a threshold — Claude adjudicates which sector best fits the business. The
verdict is PERSISTED (data/classification_adjudications.json) so it is asked once
and scoring stays stable run-to-run.

Fail-closed: any error (no API key, network, parse) leaves classifications untouched
and never breaks the data pipeline. Manual overrides always win (those symbols are
skipped). Applied entirely BEFORE scoring, so a verdict flows into the PE/PB
benchmark, peer ranking, concentration, and persistence — like a manual override.
"""
from __future__ import annotations

import json
import logging
import os

from core.paths import DATA_DIRECTORY
from data import fmp_client
from data.valuation import (
    _fmp_to_benchmark_key,
    _resolve_sector_key,
    benchmark_pe,
)

logger = logging.getLogger(__name__)

ADJUDICATIONS_PATH = os.path.join(DATA_DIRECTORY, "classification_adjudications.json")


# ---------------------------------------------------------------------------
# Verdict cache (persisted, keyed by the exact disagreement inputs)
# ---------------------------------------------------------------------------

def _verdict_key(symbol: str, rh_sector: str, fmp_sector: str) -> str:
    return f"{symbol}|{rh_sector}|{fmp_sector}"


def load_verdicts() -> dict:
    try:
        with open(ADJUDICATIONS_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _save_verdicts(verdicts: dict) -> None:
    try:
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        with open(ADJUDICATIONS_PATH, "w") as fh:
            json.dump(verdicts, fh, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning("Could not persist classification adjudications: %s", exc)


# ---------------------------------------------------------------------------
# Material-disagreement detection (sector-level PE swing)
# ---------------------------------------------------------------------------

def _detect_material(symbol: str, rh_sector: str, rh_industry: str,
                     profile: dict, swing_threshold: float) -> dict | None:
    """Return a disagreement record if FMP's mapped sector differs from Robinhood's
    AND the benchmark PE swings by more than swing_threshold, else None."""
    # Pooled vehicles (ETFs/funds) are excluded from active stock scoring and their
    # "sector" is meaningless — never cross-validate them.
    if profile.get("isEtf") or profile.get("isFund"):
        return None

    fmp_sector = str(profile.get("sector") or "").strip()
    fmp_industry = str(profile.get("industry") or "").strip()
    if not fmp_sector:
        return None

    rh_key = _resolve_sector_key(rh_sector)
    fmp_key = _fmp_to_benchmark_key(fmp_sector, fmp_industry)
    if not rh_key or not fmp_key or rh_key == fmp_key:
        return None

    rh_pe, fmp_pe = benchmark_pe(rh_key), benchmark_pe(fmp_key)
    if not rh_pe or not fmp_pe:
        return None
    swing = abs(fmp_pe - rh_pe) / rh_pe
    if swing <= swing_threshold:
        return None

    return {
        "symbol": symbol,
        "rh_sector": rh_sector, "rh_industry": rh_industry, "rh_key": rh_key,
        "fmp_sector": fmp_sector, "fmp_industry": fmp_industry, "fmp_key": fmp_key,
        "rh_pe": rh_pe, "fmp_pe": fmp_pe, "swing": round(swing, 3),
        "company_name": str(profile.get("companyName") or symbol),
        "description": str(profile.get("description") or "")[:1500],
    }


# ---------------------------------------------------------------------------
# LLM adjudication
# ---------------------------------------------------------------------------

def _adjudicate(disc: dict, model: str) -> dict | None:
    """Ask Claude which of the two candidate sectors best fits the business.
    Returns {"choice": "rh"|"fmp", "applied_sector": <key>, "reasoning": str} or None."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — classification adjudication skipped")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            "You are classifying a public company into ONE sector for valuation "
            "benchmarking. Two data sources disagree.\n\n"
            f"Company: {disc['company_name']} ({disc['symbol']})\n"
            f"Business: {disc['description']}\n\n"
            f'Option A (Robinhood): "{disc["rh_sector"]}" — industry "{disc["rh_industry"]}"\n'
            f'Option B (FMP):       "{disc["fmp_sector"]}" — industry "{disc["fmp_industry"]}"\n\n'
            "Which sector best reflects how this business actually makes money and how the "
            "market values it? Reply with ONLY a JSON object:\n"
            '{"choice": "A" or "B", "reasoning": "<one sentence>"}'
        )
        resp = client.messages.create(
            model=model, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            getattr(b, "text", "") for b in resp.content
            if getattr(b, "type", "") == "text"
        )
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        parsed = json.loads(text[start:end + 1])
        choice = str(parsed.get("choice", "")).strip().upper()
        if choice not in ("A", "B"):
            return None
        # Option A → keep Robinhood (already resolves); Option B → apply FMP's mapped
        # benchmark KEY (a valid TRBC sector that resolves), never the raw GICS name.
        applied = disc["rh_key"] if choice == "A" else disc["fmp_key"]
        if _resolve_sector_key(applied) is None:
            return None
        return {
            "choice": "rh" if choice == "A" else "fmp",
            "applied_sector": applied,
            "reasoning": str(parsed.get("reasoning", ""))[:300],
        }
    except Exception as exc:
        logger.warning("Adjudication failed for %s: %s", disc["symbol"], exc)
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def cross_validate(fundamentals: dict[str, dict], *, allow_fetch: bool) -> dict:
    """Cross-validate sector classification against FMP for borderline cases and
    apply LLM-adjudicated verdicts to `fundamentals` IN PLACE (before scoring).

    Returns a summary dict {checked, flagged, applied}. Fail-closed throughout.
    """
    from util import CLASSIFICATION_OVERRIDES, CROSS_VALIDATION_PARAMS

    params = CROSS_VALIDATION_PARAMS
    if not params.get("enabled"):
        return {"checked": 0, "flagged": 0, "applied": 0}

    swing_threshold = float(params.get("swing_threshold", 0.20))
    fetch_cap = int(params.get("profile_fetch_per_run", 200))
    model = str(params.get("model") or "") or _resolved_sentiment_model()

    verdicts = load_verdicts()
    manual = set(CLASSIFICATION_OVERRIDES)
    fetched = checked = flagged = applied = 0

    for symbol, data in fundamentals.items():
        if str(symbol).upper() in manual:
            continue  # human override is authoritative
        rh_sector = str(data.get("sector") or "").strip()
        if not rh_sector:
            continue

        # Cached profile first; spend a live fetch only within the per-run cap.
        profile = fmp_client.company_profile(symbol, allow_fetch=False)
        if profile is None and allow_fetch and fetched < fetch_cap:
            fetched += 1
            try:
                profile = fmp_client.company_profile(symbol, allow_fetch=True)
            except fmp_client.FMPNetworkError:
                profile = None
        if not profile:
            continue

        checked += 1
        disc = _detect_material(
            symbol, rh_sector, str(data.get("industry") or ""), profile, swing_threshold,
        )
        if not disc:
            continue
        flagged += 1

        vkey = _verdict_key(symbol, disc["rh_sector"], disc["fmp_sector"])
        verdict = verdicts.get(vkey)
        if verdict is None:
            verdict = _adjudicate(disc, model)
            if verdict is None:
                continue
            verdict = {**verdict, **{k: disc[k] for k in (
                "rh_sector", "rh_industry", "fmp_sector", "fmp_industry",
                "rh_key", "fmp_key", "swing", "company_name",
            )}}
            verdicts[vkey] = verdict
            _save_verdicts(verdicts)

        applied_sector = verdict.get("applied_sector")
        if applied_sector and applied_sector != rh_sector:
            logger.info(
                "Cross-val reclassify: %s '%s' → '%s' (FMP '%s', PE %.0f→%.0f, swing %.0f%%): %s",
                symbol, rh_sector, applied_sector, disc["fmp_sector"],
                disc["rh_pe"], disc["fmp_pe"], disc["swing"] * 100,
                verdict.get("reasoning", ""),
            )
            data["sector"] = applied_sector
            applied += 1

    if flagged:
        logger.info(
            "Cross-validation: checked %d profiles, %d material disagreements, "
            "%d reclassified (%d new FMP fetches this run).",
            checked, flagged, applied, fetched,
        )
    return {"checked": checked, "flagged": flagged, "applied": applied}


def _resolved_sentiment_model() -> str:
    try:
        from data.sentiment import _resolve_model
        return _resolve_model()
    except Exception:
        return "claude-opus-4-8"
