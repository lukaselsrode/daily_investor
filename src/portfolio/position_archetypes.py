"""
portfolio/position_archetypes.py — Position archetype classifier and policy engine.

Classifies holdings into behavioral archetypes based on broker risk signals,
factor scores, analyst data, and business description. Each archetype carries
a management policy (trim/harvest/stop thresholds) that is applied ONLY to
position management — never to entry scoring or factor weights.

Archetypes:
  quality_compounder     — durable platform/cloud/mega-cap; hold through volatility
  legacy_turnaround      — old-economy / patent / restructuring; take gains quickly
  speculative_momentum   — high beta, low quality, momentum-driven; tight stops
  value_recovery         — undervalued cyclical / contrarian; moderate thresholds
  defensive_income       — yield / utility / REIT; income-focused management
  core_default           — fallback when signals are insufficient

Classifier is signal-weight transparent: each archetype gets an additive score
from documented evidence rules. Highest score wins. Confidence = winner / total.

Usage:
    from portfolio.position_archetypes import classify_archetype, get_archetype_policy

    result = classify_archetype({
        "quality_score": 0.7,
        "momentum_score": 0.4,
        "market_cap": 2_000_000_000_000,
        "maintenance_ratio": 0.25,
        "buy_to_sell_ratio": 6.2,
        "sector": "Technology Services",
        "description": "...platform...cloud...",
    })
    # result.archetype == "quality_compounder"
    # result.confidence == 0.87

    policy = get_archetype_policy(result.archetype)
    # policy.harvest_profit_threshold == 0.50
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARCHETYPE_LABELS: frozenset[str] = frozenset({
    "quality_compounder",
    "legacy_turnaround",
    "speculative_momentum",
    "value_recovery",
    "defensive_income",
    "core_default",
})

_COMPOUNDER_TERMS: frozenset[str] = frozenset({
    "platform", "ecosystem", "cloud", "advertising", "marketplace",
    "subscription", "operating leverage", "scale", "market share", "software",
    "saas", "data center", "network effect", "recurring revenue", "ai",
    "artificial intelligence", "machine learning", "digital transformation",
})

_LEGACY_TERMS: frozenset[str] = frozenset({
    "licensing", "patent", "restructuring", "turnaround", "legacy",
    "hardware", "declining", "handset", "telecom equipment", "royalty",
    "divestiture", "spin-off", "formerly", "transition", "pivot",
    "monetization of intellectual property", "patent portfolio",
})

_DEFENSIVE_TERMS: frozenset[str] = frozenset({
    "utility", "regulated", "dividend", "reit", "pipeline",
    "telecom services", "consumer staples", "real estate", "infrastructure",
    "natural gas", "electric", "water", "transmission",
})

_DEFENSIVE_SECTORS: frozenset[str] = frozenset({
    "Utilities",
    "Real Estate",
    "Consumer Non-Durables",
    "Consumer Staples",
    "Finance",
})

_DEFENSIVE_INDUSTRIES: frozenset[str] = frozenset({
    "Electric Utilities",
    "Gas Utilities",
    "Multi-Utilities",
    "Water Utilities",
    "Real Estate Investment Trusts",
    "Real Estate (Operations & Services)",
})

_MEGA_CAP  = 100_000_000_000   # $100B
_LARGE_CAP = 10_000_000_000    # $10B
_MID_CAP   = 2_000_000_000     # $2B
_SMALL_CAP = 500_000_000       # $500M

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchetypePolicy:
    """Management thresholds derived from archetype classification."""
    archetype: str
    trim_profit_threshold: float
    harvest_profit_threshold: float
    trailing_stop_pct: float
    minimum_hold_days: int
    thesis_exit_requires_confirmation: bool
    allow_deeper_drawdown: bool


@dataclass
class ArchetypeResult:
    """Full classification result with evidence trail."""
    archetype: str
    confidence: float                           # 0.0–1.0
    scores: dict[str, float]                    # raw score per archetype
    drivers: list[str]                          # human-readable evidence
    policy: ArchetypePolicy


# ---------------------------------------------------------------------------
# Default policy table (used when no config overrides are present)
# ---------------------------------------------------------------------------

_DEFAULT_POLICIES: dict[str, dict] = {
    "quality_compounder": {
        "trim_profit_threshold":            0.35,
        "harvest_profit_threshold":         0.50,
        "trailing_stop_pct":               -0.12,
        "minimum_hold_days":                30,
        "thesis_exit_requires_confirmation":True,
        "allow_deeper_drawdown":            True,
    },
    "legacy_turnaround": {
        "trim_profit_threshold":            0.12,
        "harvest_profit_threshold":         0.22,
        "trailing_stop_pct":               -0.06,
        "minimum_hold_days":                5,
        "thesis_exit_requires_confirmation":False,
        "allow_deeper_drawdown":            False,
    },
    "speculative_momentum": {
        "trim_profit_threshold":            0.10,
        "harvest_profit_threshold":         0.20,
        "trailing_stop_pct":               -0.06,
        "minimum_hold_days":                5,
        "thesis_exit_requires_confirmation":False,
        "allow_deeper_drawdown":            False,
    },
    "value_recovery": {
        "trim_profit_threshold":            0.15,
        "harvest_profit_threshold":         0.30,
        "trailing_stop_pct":               -0.08,
        "minimum_hold_days":                10,
        "thesis_exit_requires_confirmation":False,
        "allow_deeper_drawdown":            False,
    },
    "defensive_income": {
        "trim_profit_threshold":            0.12,
        "harvest_profit_threshold":         0.25,
        "trailing_stop_pct":               -0.08,
        "minimum_hold_days":                15,
        "thesis_exit_requires_confirmation":False,
        "allow_deeper_drawdown":            False,
    },
    "core_default": {
        "trim_profit_threshold":            0.20,
        "harvest_profit_threshold":         0.30,
        "trailing_stop_pct":               -0.08,
        "minimum_hold_days":                10,
        "thesis_exit_requires_confirmation":False,
        "allow_deeper_drawdown":            False,
    },
}


def get_archetype_policy(archetype: str, cfg: dict | None = None) -> ArchetypePolicy:
    """
    Return management policy for *archetype*, with optional config override.

    cfg should be the parsed archetype_management sub-dict from config.yaml:
        cfg = yaml_cfg.get("archetype_management", {})
    """
    if archetype not in _DEFAULT_POLICIES:
        archetype = "core_default"

    defaults = _DEFAULT_POLICIES[archetype]
    overrides: dict = {}
    if cfg and isinstance(cfg, dict) and cfg.get("enabled", True):
        overrides = cfg.get(archetype, {}) or {}

    def _get(key: str):
        return overrides.get(key, defaults[key])

    return ArchetypePolicy(
        archetype=archetype,
        trim_profit_threshold=float(_get("trim_profit_threshold")),
        harvest_profit_threshold=float(_get("harvest_profit_threshold")),
        trailing_stop_pct=float(_get("trailing_stop_pct")),
        minimum_hold_days=int(_get("minimum_hold_days")),
        thesis_exit_requires_confirmation=bool(_get("thesis_exit_requires_confirmation")),
        allow_deeper_drawdown=bool(_get("allow_deeper_drawdown")),
    )


# ---------------------------------------------------------------------------
# Description feature extraction
# ---------------------------------------------------------------------------

def _desc_features(description: str | None) -> dict[str, bool]:
    """Return {compounder, legacy, defensive} term-hit flags from company description."""
    if not description or not isinstance(description, str):
        return {"compounder": False, "legacy": False, "defensive": False}
    text = description.lower()
    return {
        "compounder":  any(t in text for t in _COMPOUNDER_TERMS),
        "legacy":      any(t in text for t in _LEGACY_TERMS),
        "defensive":   any(t in text for t in _DEFENSIVE_TERMS),
    }


def _desc_matched_terms(description: str | None) -> dict[str, list[str]]:
    """Return lists of matched terms per category — used for driver explanations."""
    if not description or not isinstance(description, str):
        return {"compounder": [], "legacy": [], "defensive": []}
    text = description.lower()
    return {
        "compounder":  [t for t in _COMPOUNDER_TERMS  if t in text][:4],
        "legacy":      [t for t in _LEGACY_TERMS      if t in text][:4],
        "defensive":   [t for t in _DEFENSIVE_TERMS   if t in text][:4],
    }


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _sf(signals: dict, key: str, default=None):
    """Safe-float lookup from signals dict."""
    v = signals.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _analyst_buy_pct(signals: dict) -> float | None:
    """
    Derive analyst buy percentage from available signals.
    Prefers analyst_buy_pct if present; falls back to buy_to_sell_ratio heuristic.

    The buy_to_sell_ratio heuristic is conservative — holds are typically more
    numerous than buys+sells combined, so we apply a substantial denominator penalty.
    Calibrated against known examples:
      BB  ratio≈2.0 → ~0.30  (actual 2/9 = 22%)
      NOK ratio≈2.8 → ~0.40  (actual 17/30 = 57% — hold count is high)
      GOOG ratio=62  → ~0.85  (actual 62/72 = 86%)
      AMZN ratio=68  → ~0.88  (actual 68/71 = 96%)
    """
    direct = _sf(signals, "analyst_buy_pct")
    if direct is not None:
        return min(max(direct, 0.0), 1.0)

    ratio = _sf(signals, "buy_to_sell_ratio")
    if ratio is None:
        return None
    # Conservative lookup: buys / (buys + holds + sells)
    # With many more holds than buys+sells in practice, buy_pct is much lower than
    # buy_to_sell_ratio alone suggests.
    if ratio <= 0.5:
        return 0.15
    if ratio <= 1.5:
        return 0.28
    if ratio <= 3.0:
        return 0.40
    if ratio <= 6.0:
        return 0.57
    if ratio <= 12.0:
        return 0.72
    if ratio <= 20.0:
        return 0.82
    return 0.88


# ---------------------------------------------------------------------------
# Per-archetype scorecards
# ---------------------------------------------------------------------------

def _score_quality_compounder(signals: dict, desc_feats: dict, desc_terms: dict) -> tuple[float, list[str]]:
    score = 0.0
    drivers: list[str] = []

    maint = _sf(signals, "maintenance_ratio")
    day_trade = _sf(signals, "day_trade_ratio")
    market_cap = _sf(signals, "market_cap")
    quality = _sf(signals, "quality_score", 0.0)
    buy_pct = _analyst_buy_pct(signals)
    employees = _sf(signals, "num_employees")

    # Market cap — strongest size signal
    if market_cap is not None:
        if market_cap >= _MEGA_CAP:
            score += 0.30
            drivers.append(f"mega-cap (${market_cap/1e9:.0f}B)")
        elif market_cap >= _LARGE_CAP:
            score += 0.15
            drivers.append(f"large-cap (${market_cap/1e9:.1f}B)")
        elif market_cap < _SMALL_CAP:
            score -= 0.10
            drivers.append(f"small-cap (${market_cap/1e6:.0f}M) — weak compounder signal")

    # Margin/risk ratios — strict boundary: only 0.25 standard margin gets full credit
    if maint is not None:
        if maint <= 0.25:
            score += 0.25
            drivers.append(f"maintenance_ratio={maint:.2f} — institution-trusted")
        elif maint <= 0.27:
            score += 0.08
            drivers.append(f"maintenance_ratio={maint:.2f} — low-risk margin profile")
        elif maint >= 1.0:
            score -= 0.30
            drivers.append(f"maintenance_ratio={maint:.2f} — speculative flag (−)")

    if day_trade is not None and day_trade <= 0.25:
        score += 0.08
        drivers.append(f"day_trade_ratio={day_trade:.2f} — normal")

    # Analyst consensus
    if buy_pct is not None:
        if buy_pct > 0.80:
            score += 0.22
            drivers.append(f"analyst buy%={buy_pct:.0%} — very strong consensus")
        elif buy_pct > 0.65:
            score += 0.14
            drivers.append(f"analyst buy%={buy_pct:.0%} — strong consensus")
        elif buy_pct < 0.40:
            score -= 0.20
            drivers.append(f"analyst buy%={buy_pct:.0%} — weak consensus (−)")

    # Quality score
    if quality >= 0.60:
        score += 0.15
        drivers.append(f"quality_score={quality:.3f} — high")
    elif quality >= 0.35:
        score += 0.07
    elif quality < 0.10:
        score -= 0.10
        drivers.append(f"quality_score={quality:.3f} — low (−)")

    # Employee count → scaled platform proxy
    if employees is not None:
        if employees >= 50_000:
            score += 0.10
            drivers.append(f"employees={employees:,} — scaled organization")
        elif employees < 2_000:
            score -= 0.05

    # Description terms
    if desc_feats["compounder"]:
        score += 0.12
        terms = desc_terms["compounder"]
        drivers.append(f"description: {', '.join(terms)}")
    if desc_feats["legacy"]:
        score -= 0.12
        drivers.append("description: legacy/patent/restructuring language (−)")

    return max(score, 0.0), drivers


def _score_legacy_turnaround(signals: dict, desc_feats: dict, desc_terms: dict) -> tuple[float, list[str]]:
    score = 0.0
    drivers: list[str] = []

    maint = _sf(signals, "maintenance_ratio")
    day_trade = _sf(signals, "day_trade_ratio")
    market_cap = _sf(signals, "market_cap")
    _quality = _sf(signals, "quality_score", 0.0)
    momentum = _sf(signals, "momentum_score", 0.0)
    buy_pct = _analyst_buy_pct(signals)
    inst_type = str(signals.get("instrument_type", "") or "").lower()
    country = str(signals.get("country", "") or "").upper()
    _sector = str(signals.get("sector", "") or "")
    _industry = str(signals.get("industry", "") or "")

    # Margin/risk ratios — strongest discriminators
    if maint is not None:
        if maint >= 1.0:
            score += 0.38
            drivers.append(f"maintenance_ratio={maint:.2f} — strong speculative/risk flag")
        elif maint > 0.40:
            score += 0.20
            drivers.append(f"maintenance_ratio={maint:.2f} — elevated risk ratio")
        elif maint > 0.27:
            score += 0.12
            drivers.append(f"maintenance_ratio={maint:.2f} — above-standard margin requirement")

    if day_trade is not None and day_trade > 0.25:
        score += 0.12
        drivers.append(f"day_trade_ratio={day_trade:.2f} — elevated day-trade requirement")

    # Market cap
    if market_cap is not None:
        if market_cap < _MID_CAP:
            score += 0.15
            drivers.append(f"small/mid cap (${market_cap/1e6:.0f}M)")
        elif market_cap < _LARGE_CAP:
            score += 0.08
        elif market_cap >= _MEGA_CAP:
            score -= 0.30
            drivers.append("mega-cap disqualifies legacy archetype (−)")

    # Analyst consensus
    if buy_pct is not None:
        if buy_pct < 0.35:
            score += 0.18
            drivers.append(f"analyst buy%={buy_pct:.0%} — weak analyst conviction")
        elif buy_pct < 0.55:
            score += 0.10
            drivers.append(f"analyst buy%={buy_pct:.0%} — moderate analyst support")
        elif buy_pct > 0.80:
            score -= 0.20
            drivers.append(f"analyst buy%={buy_pct:.0%} — too strong for legacy archetype (−)")

    # Legacy description terms
    if desc_feats["legacy"]:
        score += 0.18
        terms = desc_terms["legacy"]
        drivers.append(f"description: {', '.join(terms)}")

    # Strong recent momentum on a legacy name = rally pattern
    if momentum is not None and momentum > 0.30:
        score += 0.10
        drivers.append(f"momentum_score={momentum:.3f} — strong rally pattern on legacy name")

    # ADR + non-US combined with other risk factors
    if inst_type == "adr":
        score += 0.06
        drivers.append("instrument_type=ADR — foreign-listed")
    if country not in ("US", ""):
        if maint is not None and maint > 0.30:
            score += 0.05
            drivers.append(f"non-US country={country} + elevated margin ratio")

    # Compounder description terms reduce legacy score
    if desc_feats["compounder"]:
        score -= 0.12

    return max(score, 0.0), drivers


def _score_speculative_momentum(signals: dict, desc_feats: dict, desc_terms: dict) -> tuple[float, list[str]]:
    score = 0.0
    drivers: list[str] = []

    maint = _sf(signals, "maintenance_ratio")
    day_trade = _sf(signals, "day_trade_ratio")
    market_cap = _sf(signals, "market_cap")
    quality = _sf(signals, "quality_score", 0.0)
    momentum = _sf(signals, "momentum_score", 0.0)
    income = _sf(signals, "income_score", 0.0)
    yield_trap = signals.get("yield_trap_flag", False)

    # Momentum is the central signal
    if momentum is not None:
        if momentum > 0.60:
            score += 0.28
            drivers.append(f"momentum_score={momentum:.3f} — very strong")
        elif momentum > 0.35:
            score += 0.18
            drivers.append(f"momentum_score={momentum:.3f} — strong")
        elif momentum < 0.0:
            score -= 0.15
            drivers.append(f"momentum_score={momentum:.3f} — negative (−)")

    # High margin/risk ratios
    if maint is not None:
        if maint >= 1.0:
            score += 0.25
            drivers.append(f"maintenance_ratio={maint:.2f} — high risk flag")
        elif maint > 0.40:
            score += 0.15
    if day_trade is not None and day_trade > 0.40:
        score += 0.12
        drivers.append(f"day_trade_ratio={day_trade:.2f} — high speculative flag")

    # Low quality = speculative
    if quality is not None:
        if quality < 0.10:
            score += 0.20
            drivers.append(f"quality_score={quality:.3f} — very low quality")
        elif quality < 0.25:
            score += 0.10
            drivers.append(f"quality_score={quality:.3f} — low quality")
        elif quality > 0.60:
            score -= 0.25
            drivers.append(f"quality_score={quality:.3f} — too high for speculative archetype (−)")

    # Small cap
    if market_cap is not None:
        if market_cap < _SMALL_CAP:
            score += 0.15
            drivers.append(f"small-cap (${market_cap/1e6:.0f}M) — high speculative risk")
        elif market_cap >= _MEGA_CAP:
            score -= 0.25

    # No income / no dividend
    if income is not None and income <= 0.05 and not yield_trap:
        score += 0.08
        drivers.append("no income/dividend — pure price return play")

    # Compounder terms disqualify
    if desc_feats["compounder"]:
        score -= 0.15

    return max(score, 0.0), drivers


def _score_value_recovery(signals: dict, desc_feats: dict, desc_terms: dict) -> tuple[float, list[str]]:
    score = 0.0
    drivers: list[str] = []

    value = _sf(signals, "value_score", 0.0)
    quality = _sf(signals, "quality_score", 0.0)
    momentum = _sf(signals, "momentum_score", 0.0)
    maint = _sf(signals, "maintenance_ratio")

    # Undervaluation is the central signal
    if value is not None:
        if value > 0.60:
            score += 0.30
            drivers.append(f"value_score={value:.3f} — undervalued")
        elif value > 0.30:
            score += 0.15
            drivers.append(f"value_score={value:.3f} — moderate value")

    # Improving momentum (not strongly negative, not overextended)
    if momentum is not None:
        if 0.0 < momentum <= 0.40:
            score += 0.15
            drivers.append(f"momentum_score={momentum:.3f} — improving")
        elif momentum < -0.20:
            score -= 0.10

    # Moderate quality (not distressed, not compounder)
    if quality is not None:
        if 0.15 <= quality <= 0.55:
            score += 0.12
            drivers.append(f"quality_score={quality:.3f} — moderate quality / recovery profile")

    # High maintenance ratio = distress, reduces recovery confidence
    if maint is not None and maint >= 1.0:
        score -= 0.15
        drivers.append(f"maintenance_ratio={maint:.2f} — distress risk reduces recovery conviction (−)")

    # Legacy terms can co-exist with value recovery
    if desc_feats["legacy"] and value is not None and value > 0.30:
        score += 0.08

    return max(score, 0.0), drivers


def _score_defensive_income(signals: dict, desc_feats: dict, desc_terms: dict) -> tuple[float, list[str]]:
    score = 0.0
    drivers: list[str] = []

    income = _sf(signals, "income_score", 0.0)
    quality = _sf(signals, "quality_score", 0.0)
    momentum = _sf(signals, "momentum_score", 0.0)
    yield_trap = bool(signals.get("yield_trap_flag", False))
    sector = str(signals.get("sector", "") or "")
    industry = str(signals.get("industry", "") or "")

    # Income is the central signal
    if yield_trap:
        score -= 0.25
        drivers.append("yield_trap_flag=True — income not safe (−)")
    elif income is not None:
        if income > 0.80:
            score += 0.35
            drivers.append(f"income_score={income:.3f} — high dividend income")
        elif income > 0.50:
            score += 0.20
            drivers.append(f"income_score={income:.3f} — moderate income")
        elif income <= 0.05:
            score -= 0.15
            drivers.append(f"income_score={income:.3f} — no income (−)")

    # Sector classification
    if sector in _DEFENSIVE_SECTORS:
        score += 0.20
        drivers.append(f"sector={sector} — defensive sector")
    if industry in _DEFENSIVE_INDUSTRIES:
        score += 0.15
        drivers.append(f"industry={industry} — regulated/utility industry")

    # Description terms
    if desc_feats["defensive"] and not yield_trap:
        score += 0.12
        terms = desc_terms["defensive"]
        drivers.append(f"description: {', '.join(terms)}")

    # Moderate quality (stability, not growth)
    if quality is not None and quality > 0.25:
        score += 0.08

    # Low/negative momentum is OK for income; not a penalty
    if momentum is not None and momentum > 0.50:
        score -= 0.08  # strong momentum → not a defensive income play

    return max(score, 0.0), drivers


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

def classify_archetype(signals: dict, archetype_cfg: dict | None = None) -> ArchetypeResult:
    """
    Classify a position into a behavioral archetype.

    Parameters
    ----------
    signals : dict
        Any combination of available signals. All keys are optional; the
        classifier degrades gracefully when signals are missing.

        Common keys used:
          quality_score, momentum_score, value_score, income_score
          market_cap, maintenance_ratio, day_trade_ratio
          buy_to_sell_ratio, analyst_buy_pct
          sector, industry, description, num_employees
          yield_trap_flag, instrument_type, country

    archetype_cfg : dict, optional
        Parsed archetype_management config section for policy overrides.

    Returns
    -------
    ArchetypeResult with .archetype, .confidence, .scores, .drivers, .policy
    """
    desc = signals.get("description", "")
    desc_feats = _desc_features(desc)
    desc_terms = _desc_matched_terms(desc)

    s_comp,  d_comp  = _score_quality_compounder   (signals, desc_feats, desc_terms)
    s_leg,   d_leg   = _score_legacy_turnaround     (signals, desc_feats, desc_terms)
    s_spec,  d_spec  = _score_speculative_momentum  (signals, desc_feats, desc_terms)
    s_val,   d_val   = _score_value_recovery        (signals, desc_feats, desc_terms)
    s_def,   d_def   = _score_defensive_income      (signals, desc_feats, desc_terms)
    s_fallback       = 0.05  # core_default always possible but weakly

    raw_scores: dict[str, float] = {
        "quality_compounder":   s_comp,
        "legacy_turnaround":    s_leg,
        "speculative_momentum": s_spec,
        "value_recovery":       s_val,
        "defensive_income":     s_def,
        "core_default":         s_fallback,
    }
    driver_map: dict[str, list[str]] = {
        "quality_compounder":   d_comp,
        "legacy_turnaround":    d_leg,
        "speculative_momentum": d_spec,
        "value_recovery":       d_val,
        "defensive_income":     d_def,
        "core_default":         ["insufficient signals for confident classification"],
    }

    total = sum(raw_scores.values())
    winner = max(raw_scores, key=lambda k: raw_scores[k])

    # If no archetype scored meaningfully, fall back to core_default
    if raw_scores[winner] < 0.05:
        winner = "core_default"

    confidence: float
    if total > 0.0:
        confidence = min(raw_scores[winner] / total, 0.99)
    else:
        confidence = 0.50

    # Minimum confidence floor — we are never more than 99% sure
    confidence = max(confidence, 0.30)

    policy = get_archetype_policy(winner, archetype_cfg)

    return ArchetypeResult(
        archetype=winner,
        confidence=round(confidence, 3),
        scores={k: round(v, 4) for k, v in raw_scores.items()},
        drivers=driver_map[winner],
        policy=policy,
    )


# ---------------------------------------------------------------------------
# Backtest lightweight classifier (no API calls, precomp-only signals)
# ---------------------------------------------------------------------------

def classify_archetype_from_scores(
    quality_score: float,
    momentum_score: float,
    income_score: float,
    yield_trap: bool = False,
    archetype_cfg: dict | None = None,
) -> ArchetypePolicy:
    """
    Lightweight archetype classification for backtesting — uses only precomputed
    factor scores (no market structure API calls). Returns the policy directly.

    Used by backtesting/simulator.py to assign per-position exit thresholds
    without requiring external data fetches.
    """
    signals = {
        "quality_score":  quality_score,
        "momentum_score": momentum_score,
        "income_score":   income_score,
        "yield_trap_flag": yield_trap,
    }
    result = classify_archetype(signals, archetype_cfg)
    return result.policy
