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

from dataclasses import dataclass, field

from core.instruments import is_fund_asset_value, is_fund_instrument_type

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
    "fund",
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
    # Behavioral controls — defaults are no-ops (live/backtest behave as today).
    enabled: bool = True
    score_multiplier: float = 1.0
    max_position_multiplier: float = 1.0
    max_active_weight: float | None = None
    min_score_to_buy: float | None = None


@dataclass
class ArchetypeResult:
    """Full classification result with evidence trail."""
    archetype: str
    confidence: float                           # 0.0–1.0
    scores: dict[str, float]                    # raw score per archetype
    drivers: list[str]                          # human-readable evidence (winner only)
    policy: ArchetypePolicy
    # ── Extended diagnostics (populated by classify_archetype v2) ──────────
    confidence_bucket: str = "medium"           # "high" | "medium" | "low"
    runner_up: str | None = None
    runner_up_score: float = 0.0
    reason_codes: dict[str, list[str]] = field(default_factory=dict)
    missing_signals: list[str] = field(default_factory=list)
    features_used: list[str] = field(default_factory=list)


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
    "fund": {
        # Pooled vehicle (ETF/CEF/MLP/ETN), not a single-company bet. Manage like a
        # diversified holding: wider stops, longer hold, no aggressive single-name
        # trim/harvest. Stock factor scorecards do not apply to funds.
        "trim_profit_threshold":            0.30,
        "harvest_profit_threshold":         0.45,
        "trailing_stop_pct":               -0.15,
        "minimum_hold_days":                30,
        "thesis_exit_requires_confirmation":True,
        "allow_deeper_drawdown":            True,
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

    def _opt(key: str, default):
        v = overrides.get(key, default)
        if v is None:
            return None
        return v

    max_active_weight = _opt("max_active_weight", None)
    min_score_to_buy = _opt("min_score_to_buy", None)

    return ArchetypePolicy(
        archetype=archetype,
        trim_profit_threshold=float(_get("trim_profit_threshold")),
        harvest_profit_threshold=float(_get("harvest_profit_threshold")),
        trailing_stop_pct=float(_get("trailing_stop_pct")),
        minimum_hold_days=int(_get("minimum_hold_days")),
        thesis_exit_requires_confirmation=bool(_get("thesis_exit_requires_confirmation")),
        allow_deeper_drawdown=bool(_get("allow_deeper_drawdown")),
        enabled=bool(overrides.get("enabled", True)),
        score_multiplier=float(overrides.get("score_multiplier", 1.0)),
        max_position_multiplier=float(overrides.get("max_position_multiplier", 1.0)),
        max_active_weight=None if max_active_weight is None else float(max_active_weight),
        min_score_to_buy=None if min_score_to_buy is None else float(min_score_to_buy),
    )


# ---------------------------------------------------------------------------
# Description feature extraction
# ---------------------------------------------------------------------------

def _is_fund_signal(signals: dict) -> bool:
    """True when the signals indicate a pooled fund (ETF/CEF/MLP/ETN), not a stock.

    Reuses the shared core predicate (single source of truth) and honours an
    explicit ``is_etf`` flag or an ``asset_type``/``security_type`` field when
    present. Never infers fund status from sector or missing fundamentals.
    """
    if bool(signals.get("is_etf", False)):
        return True
    if is_fund_instrument_type(signals.get("instrument_type")):
        return True
    for _k in ("asset_type", "security_type"):
        if is_fund_asset_value(signals.get(_k)):
            return True
    return False


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


def _company_age(signals: dict) -> float | None:
    """Years since founding (proxy for old-economy vs young/speculative)."""
    yf = _sf(signals, "year_founded")
    if yf is None or yf < 1700 or yf > 2100:
        return None
    import datetime as _dt
    return float(_dt.date.today().year - int(yf))


def _valuation_character(signals: dict) -> dict:
    """Derive valuation-character flags from raw PE/PB (no fair-value API exists).

    Robinhood get_fundamentals exposes pe_ratio / pb_ratio but NOT an intrinsic
    fair value, so we read raw multiples as a *character* proxy, not a target price:
      - cheap:     low PE and/or low PB  -> value_recovery / defensive lean
      - expensive: very high PE/PB       -> growth / speculative lean
      - negative_pe: loss-making          -> speculative / turnaround lean
    Thresholds are deliberately wide (regime-agnostic character buckets), and
    missing data yields all-False so behavior is unchanged when unavailable.
    """
    pe = _sf(signals, "pe_ratio")
    pb = _sf(signals, "pb_ratio")
    out = {"cheap": False, "expensive": False, "negative_pe": False}
    if pe is not None:
        if pe < 0:
            out["negative_pe"] = True
        elif pe <= 15.0:
            out["cheap"] = True
        elif pe >= 50.0:
            out["expensive"] = True
    if pb is not None:
        if 0 < pb <= 1.5:
            out["cheap"] = True
        elif pb >= 10.0:
            out["expensive"] = True
    return out


# ---------------------------------------------------------------------------
# Per-archetype scorecards
# ---------------------------------------------------------------------------
def _score_quality_compounder(
    signals: dict, desc_feats: dict, desc_terms: dict, *, cfg: dict | None = None,
) -> tuple[float, list[str], list[str]]:
    score = 0.0
    drivers: list[str] = []
    reason_codes: list[str] = []
    thr = (cfg or {}).get("quality_compounder", {}) or {}
    _mega = float(thr.get("market_cap_mega",  _MEGA_CAP))
    _large = float(thr.get("market_cap_large", _LARGE_CAP))
    _small = float(thr.get("market_cap_small", _SMALL_CAP))
    _maint_low  = float(thr.get("maintenance_low",  0.25))
    _maint_high = float(thr.get("maintenance_high", 0.27))
    _maint_spec = float(thr.get("maintenance_speculative", 1.0))
    _dt_normal  = float(thr.get("day_trade_normal_max", 0.25))
    _ab_strong  = float(thr.get("analyst_buy_strong",   0.80))
    _ab_mod     = float(thr.get("analyst_buy_moderate", 0.65))
    _ab_weak    = float(thr.get("analyst_buy_weak",     0.40))
    _q_high     = float(thr.get("quality_high",     0.60))
    _q_mod      = float(thr.get("quality_moderate", 0.35))
    _q_low      = float(thr.get("quality_low",      0.10))
    _emp_scaled = float(thr.get("employees_scaled", 50_000))
    _emp_small  = float(thr.get("employees_small",  2_000))

    maint = _sf(signals, "maintenance_ratio")
    day_trade = _sf(signals, "day_trade_ratio")
    market_cap = _sf(signals, "market_cap")
    quality = _sf(signals, "quality_score", 0.0)
    buy_pct = _analyst_buy_pct(signals)
    employees = _sf(signals, "num_employees")

    # Market cap — strongest size signal
    if market_cap is not None:
        if market_cap >= _mega:
            score += 0.30
            drivers.append(f"mega-cap (${market_cap/1e9:.0f}B)")
            reason_codes.append("mega_cap")
        elif market_cap >= _large:
            score += 0.15
            drivers.append(f"large-cap (${market_cap/1e9:.1f}B)")
            reason_codes.append("large_cap")
        elif market_cap < _small:
            score -= 0.10
            drivers.append(f"small-cap (${market_cap/1e6:.0f}M) — weak compounder signal")
            reason_codes.append("small_cap_penalty")

    if maint is not None:
        if maint <= _maint_low:
            score += 0.25
            drivers.append(f"maintenance_ratio={maint:.2f} — institution-trusted")
            reason_codes.append("low_maintenance")
        elif maint <= _maint_high:
            score += 0.08
            drivers.append(f"maintenance_ratio={maint:.2f} — low-risk margin profile")
        elif maint >= _maint_spec:
            score -= 0.30
            drivers.append(f"maintenance_ratio={maint:.2f} — speculative flag (−)")
            reason_codes.append("high_maintenance_penalty")

    if day_trade is not None and day_trade <= _dt_normal:
        score += 0.08
        drivers.append(f"day_trade_ratio={day_trade:.2f} — normal")

    if buy_pct is not None:
        if buy_pct > _ab_strong:
            score += 0.22
            drivers.append(f"analyst buy%={buy_pct:.0%} — very strong consensus")
            reason_codes.append("analyst_strong")
        elif buy_pct > _ab_mod:
            score += 0.14
            drivers.append(f"analyst buy%={buy_pct:.0%} — strong consensus")
            reason_codes.append("analyst_moderate")
        elif buy_pct < _ab_weak:
            score -= 0.20
            drivers.append(f"analyst buy%={buy_pct:.0%} — weak consensus (−)")
            reason_codes.append("analyst_weak_penalty")

    if quality >= _q_high:
        score += 0.15
        drivers.append(f"quality_score={quality:.3f} — high")
        reason_codes.append("quality_high")
    elif quality >= _q_mod:
        score += 0.07
        reason_codes.append("quality_moderate")
    elif quality < _q_low:
        score -= 0.10
        drivers.append(f"quality_score={quality:.3f} — low (−)")
        reason_codes.append("quality_low_penalty")

    if employees is not None:
        if employees >= _emp_scaled:
            score += 0.10
            drivers.append(f"employees={employees:,} — scaled organization")
            reason_codes.append("scaled_organization")
        elif employees < _emp_small:
            score -= 0.05

    if desc_feats["compounder"]:
        score += 0.12
        terms = desc_terms["compounder"]
        drivers.append(f"description: {', '.join(terms)}")
        reason_codes.append("compounder_terms")
    if desc_feats["legacy"]:
        score -= 0.12
        drivers.append("description: legacy/patent/restructuring language (−)")
        reason_codes.append("legacy_terms_penalty")

    return max(score, 0.0), drivers, reason_codes


def _score_legacy_turnaround(
    signals: dict, desc_feats: dict, desc_terms: dict, *, cfg: dict | None = None,
) -> tuple[float, list[str], list[str]]:
    score = 0.0
    drivers: list[str] = []
    reason_codes: list[str] = []
    thr = (cfg or {}).get("legacy_turnaround", {}) or {}
    _maint_spec  = float(thr.get("maintenance_speculative",   1.0))
    _maint_elev  = float(thr.get("maintenance_elevated",      0.40))
    _maint_above = float(thr.get("maintenance_above_standard", 0.27))
    _dt_elev     = float(thr.get("day_trade_elevated",        0.25))
    _mc_mid      = float(thr.get("market_cap_mid",   _MID_CAP))
    _mc_large    = float(thr.get("market_cap_large", _LARGE_CAP))
    _mc_mega     = float(thr.get("market_cap_mega",  _MEGA_CAP))
    _ab_weak     = float(thr.get("analyst_buy_weak",     0.35))
    _ab_mod      = float(thr.get("analyst_buy_moderate", 0.55))
    _ab_strong   = float(thr.get("analyst_buy_strong",   0.80))
    _mom_strong  = float(thr.get("momentum_strong",      0.30))

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
        if maint >= _maint_spec:
            score += 0.38
            drivers.append(f"maintenance_ratio={maint:.2f} — strong speculative/risk flag")
        elif maint > _maint_elev:
            score += 0.20
            drivers.append(f"maintenance_ratio={maint:.2f} — elevated risk ratio")
        elif maint > _maint_above:
            score += 0.12
            drivers.append(f"maintenance_ratio={maint:.2f} — above-standard margin requirement")

    if day_trade is not None and day_trade > _dt_elev:
        score += 0.12
        drivers.append(f"day_trade_ratio={day_trade:.2f} — elevated day-trade requirement")

    # Market cap
    if market_cap is not None:
        if market_cap < _mc_mid:
            score += 0.15
            drivers.append(f"small/mid cap (${market_cap/1e6:.0f}M)")
        elif market_cap < _mc_large:
            score += 0.08
        elif market_cap >= _mc_mega:
            score -= 0.30
            drivers.append("mega-cap disqualifies legacy archetype (−)")

    # Analyst consensus
    if buy_pct is not None:
        if buy_pct < _ab_weak:
            score += 0.18
            drivers.append(f"analyst buy%={buy_pct:.0%} — weak analyst conviction")
        elif buy_pct < _ab_mod:
            score += 0.10
            drivers.append(f"analyst buy%={buy_pct:.0%} — moderate analyst support")
        elif buy_pct > _ab_strong:
            score -= 0.20
            drivers.append(f"analyst buy%={buy_pct:.0%} — too strong for legacy archetype (−)")

    # Legacy description terms
    if desc_feats["legacy"]:
        score += 0.18
        terms = desc_terms["legacy"]
        drivers.append(f"description: {', '.join(terms)}")

    # Strong recent momentum on a legacy name = rally pattern
    if momentum is not None and momentum > _mom_strong:
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

    if desc_feats["compounder"]:
        score -= 0.12
        reason_codes.append("compounder_terms_penalty")

    # Company age: long-established firms lean old-economy / legacy character.
    _age = _company_age(signals)
    if _age is not None and _age >= 50:
        score += 0.10
        drivers.append(f"long-established (~{int(_age)}y old) — old-economy character")
        reason_codes.append("old_company")

    if maint is not None and maint >= 1.0:
        reason_codes.append("high_maintenance")
    if desc_feats["legacy"]:
        reason_codes.append("legacy_terms")
    if buy_pct is not None and buy_pct < 0.35:
        reason_codes.append("weak_analyst")
    if market_cap is not None and market_cap >= _MEGA_CAP:
        reason_codes.append("mega_cap_disqualifies")

    return max(score, 0.0), drivers, reason_codes


def _score_speculative_momentum(
    signals: dict, desc_feats: dict, desc_terms: dict, *, cfg: dict | None = None,
) -> tuple[float, list[str], list[str]]:
    score = 0.0
    reason_codes: list[str] = []
    thr = (cfg or {}).get("speculative_momentum", {}) or {}
    _mom_vstrong = float(thr.get("momentum_very_strong", 0.60))
    _mom_strong  = float(thr.get("momentum_strong",      0.35))
    _q_vlow      = float(thr.get("quality_very_low",     0.10))
    _q_low       = float(thr.get("quality_low",          0.25))
    _q_toohigh   = float(thr.get("quality_too_high",     0.60))
    _maint_high  = float(thr.get("maintenance_high",     1.0))
    _maint_elev  = float(thr.get("maintenance_elevated", 0.40))
    _dt_high     = float(thr.get("day_trade_high",       0.40))
    _mc_small    = float(thr.get("market_cap_small",     _SMALL_CAP))
    _mc_mega     = float(thr.get("market_cap_mega",      _MEGA_CAP))
    _income_min  = float(thr.get("income_minimal",       0.05))
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
        if momentum > _mom_vstrong:
            score += 0.28
            drivers.append(f"momentum_score={momentum:.3f} — very strong")
        elif momentum > _mom_strong:
            score += 0.18
            drivers.append(f"momentum_score={momentum:.3f} — strong")
        elif momentum < 0.0:
            score -= 0.15
            drivers.append(f"momentum_score={momentum:.3f} — negative (−)")

    # High margin/risk ratios
    if maint is not None:
        if maint >= _maint_high:
            score += 0.25
            drivers.append(f"maintenance_ratio={maint:.2f} — high risk flag")
        elif maint > _maint_elev:
            score += 0.15
    if day_trade is not None and day_trade > _dt_high:
        score += 0.12
        drivers.append(f"day_trade_ratio={day_trade:.2f} — high speculative flag")

    # Low quality = speculative
    if quality is not None:
        if quality < _q_vlow:
            score += 0.20
            drivers.append(f"quality_score={quality:.3f} — very low quality")
        elif quality < _q_low:
            score += 0.10
            drivers.append(f"quality_score={quality:.3f} — low quality")
        elif quality > _q_toohigh:
            score -= 0.25
            drivers.append(f"quality_score={quality:.3f} — too high for speculative archetype (−)")

    # Small cap
    if market_cap is not None:
        if market_cap < _mc_small:
            score += 0.15
            drivers.append(f"small-cap (${market_cap/1e6:.0f}M) — high speculative risk")
        elif market_cap >= _mc_mega:
            score -= 0.25

    # No income / no dividend
    if income is not None and income <= _income_min and not yield_trap:
        score += 0.08
        drivers.append("no income/dividend — pure price return play")

    if desc_feats["compounder"]:
        score -= 0.15
        reason_codes.append("compounder_terms_penalty")
    if momentum is not None and momentum > _mom_vstrong:
        reason_codes.append("momentum_very_strong")
    if quality is not None and quality < _q_vlow:
        reason_codes.append("quality_very_low")

    # Raw-multiple character: rich/expensive or loss-making (negative PE) names
    # lean speculative-momentum (priced for growth, not value).
    _val_char = _valuation_character(signals)
    if _val_char["negative_pe"]:
        score += 0.12
        drivers.append(f"negative PE (PE={_sf(signals, 'pe_ratio')}) — loss-making / speculative")
        reason_codes.append("negative_pe")
    elif _val_char["expensive"]:
        score += 0.10
        drivers.append(f"expensive multiples (PE={_sf(signals, 'pe_ratio')}) — growth/speculative pricing")
        reason_codes.append("expensive_multiples")

    return max(score, 0.0), drivers, reason_codes


def _score_value_recovery(
    signals: dict, desc_feats: dict, desc_terms: dict, *, cfg: dict | None = None,
) -> tuple[float, list[str], list[str]]:
    score = 0.0
    drivers: list[str] = []
    reason_codes: list[str] = []
    thr = (cfg or {}).get("value_recovery", {}) or {}
    _v_under  = float(thr.get("value_undervalued",     0.60))
    _v_mod    = float(thr.get("value_moderate",        0.30))
    _mom_imp  = float(thr.get("momentum_improving_max", 0.40))
    _mom_fall = float(thr.get("momentum_falling_min",  -0.20))
    _q_min    = float(thr.get("quality_min",           0.15))
    _q_max    = float(thr.get("quality_max",           0.55))
    _maint_distress = float(thr.get("maintenance_distress", 1.0))

    value = _sf(signals, "value_score", 0.0)
    quality = _sf(signals, "quality_score", 0.0)
    momentum = _sf(signals, "momentum_score", 0.0)
    maint = _sf(signals, "maintenance_ratio")

    # Undervaluation is the central signal
    if value is not None:
        if value > _v_under:
            score += 0.30
            drivers.append(f"value_score={value:.3f} — undervalued")
        elif value > _v_mod:
            score += 0.15
            drivers.append(f"value_score={value:.3f} — moderate value")

    # Improving momentum (not strongly negative, not overextended)
    if momentum is not None:
        if 0.0 < momentum <= _mom_imp:
            score += 0.15
            drivers.append(f"momentum_score={momentum:.3f} — improving")
        elif momentum < _mom_fall:
            score -= 0.10

    # Moderate quality (not distressed, not compounder)
    if quality is not None:
        if _q_min <= quality <= _q_max:
            score += 0.12
            drivers.append(f"quality_score={quality:.3f} — moderate quality / recovery profile")

    # High maintenance ratio = distress, reduces recovery confidence
    if maint is not None and maint >= _maint_distress:
        score -= 0.15
        drivers.append(f"maintenance_ratio={maint:.2f} — distress risk reduces recovery conviction (−)")

    if desc_feats["legacy"] and value is not None and value > _v_mod:
        score += 0.08
        reason_codes.append("legacy_value_overlap")
    if value is not None and value > _v_under:
        reason_codes.append("value_undervalued")
    if momentum is not None and 0.0 < momentum <= _mom_imp:
        reason_codes.append("momentum_improving")
    if maint is not None and maint >= _maint_distress:
        reason_codes.append("distress_penalty")

    # Raw-multiple valuation character (PE/PB) — cheap reinforces value_recovery.
    _val_char = _valuation_character(signals)
    if _val_char["cheap"]:
        score += 0.12
        pe = _sf(signals, "pe_ratio")
        pb = _sf(signals, "pb_ratio")
        drivers.append(f"cheap multiples (PE={pe}, PB={pb}) — value character")
        reason_codes.append("cheap_multiples")
    elif _val_char["expensive"]:
        score -= 0.10
        reason_codes.append("expensive_multiples_penalty")

    return max(score, 0.0), drivers, reason_codes


def _score_defensive_income(
    signals: dict,
    desc_feats: dict,
    desc_terms: dict,
    *,
    cfg: dict | None = None,
) -> tuple[float, list[str], list[str]]:
    """Score defensive_income. When cfg["defensive_income"]["require_yield"]=true,
    apply strict eligibility gates that disqualify the label (returns score=0)."""
    score = 0.0
    drivers: list[str] = []
    reason_codes: list[str] = []

    income = _sf(signals, "income_score", 0.0)
    quality = _sf(signals, "quality_score", 0.0)
    momentum = _sf(signals, "momentum_score", 0.0)
    yield_trap = bool(signals.get("yield_trap_flag", False))
    sector = str(signals.get("sector", "") or "")
    industry = str(signals.get("industry", "") or "")

    di_cfg = (cfg or {}).get("defensive_income", {}) or {}
    # Per-config thresholds (defaults preserve original behavior)
    yield_high     = float(di_cfg.get("yield_high",     0.80))
    yield_moderate = float(di_cfg.get("yield_moderate", 0.50))
    yield_minimal  = float(di_cfg.get("yield_minimal",  0.05))
    quality_min    = float(di_cfg.get("quality_min_label", 0.25))
    momentum_disq  = float(di_cfg.get("momentum_disqualify_above", 0.50))
    sec_defensive  = di_cfg.get("sector_defensive") or list(_DEFENSIVE_SECTORS)
    ind_defensive  = di_cfg.get("industry_defensive") or list(_DEFENSIVE_INDUSTRIES)

    # ── Strict eligibility gate (config-gated; default-off) ──────────────
    if di_cfg.get("require_yield", False):
        if income is None or income < float(di_cfg.get("min_income_score", 0.30)):
            reason_codes.append("gate_disqualified:income_below_min")
            drivers.append(f"defensive_income disqualified: income_score={income} < min")
            return 0.0, drivers, reason_codes
        if quality is None or quality < float(di_cfg.get("min_quality_score", 0.40)):
            reason_codes.append("gate_disqualified:quality_below_min")
            drivers.append(f"defensive_income disqualified: quality_score={quality} < min")
            return 0.0, drivers, reason_codes
        if momentum is None or momentum < float(di_cfg.get("min_momentum_score", -0.10)):
            reason_codes.append("gate_disqualified:momentum_below_min")
            drivers.append(f"defensive_income disqualified: momentum_score={momentum} < min")
            return 0.0, drivers, reason_codes
        if bool(di_cfg.get("reject_falling_knife", True)) and momentum is not None and momentum < -0.20:
            reason_codes.append("gate_disqualified:falling_knife")
            drivers.append("defensive_income disqualified: falling knife (momentum < -0.20)")
            return 0.0, drivers, reason_codes
        if yield_trap:
            reason_codes.append("gate_disqualified:yield_trap")
            drivers.append("defensive_income disqualified: yield_trap_flag=True")
            return 0.0, drivers, reason_codes
        reason_codes.append("gate_passed")

    # Income is the central signal
    if yield_trap:
        score -= 0.25
        drivers.append("yield_trap_flag=True — income not safe (−)")
        reason_codes.append("yield_trap_penalty")
    elif income is not None:
        if income > yield_high:
            score += 0.35
            drivers.append(f"income_score={income:.3f} — high dividend income")
            reason_codes.append("income_high")
        elif income > yield_moderate:
            score += 0.20
            drivers.append(f"income_score={income:.3f} — moderate income")
            reason_codes.append("income_moderate")
        elif income <= yield_minimal:
            score -= 0.15
            drivers.append(f"income_score={income:.3f} — no income (−)")
            reason_codes.append("no_income_penalty")

    if sector in sec_defensive:
        score += 0.20
        drivers.append(f"sector={sector} — defensive sector")
        reason_codes.append("defensive_sector")
    if industry in ind_defensive:
        score += 0.15
        drivers.append(f"industry={industry} — regulated/utility industry")
        reason_codes.append("defensive_industry")

    if desc_feats["defensive"] and not yield_trap:
        score += 0.12
        terms = desc_terms["defensive"]
        drivers.append(f"description: {', '.join(terms)}")
        reason_codes.append("defensive_description")

    if quality is not None and quality > quality_min:
        score += 0.08
        reason_codes.append("quality_acceptable")

    if momentum is not None and momentum > momentum_disq:
        score -= 0.08
        reason_codes.append("momentum_too_strong")

    return max(score, 0.0), drivers, reason_codes


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------

_EXPECTED_SIGNALS: tuple[str, ...] = (
    "quality_score", "momentum_score", "value_score", "income_score",
    "market_cap", "maintenance_ratio", "day_trade_ratio",
    "buy_to_sell_ratio", "analyst_buy_pct",
    "sector", "industry", "description", "num_employees",
    "yield_trap_flag", "instrument_type", "country",
    "year_founded", "pe_ratio", "pb_ratio",
)


def _bucket_confidence(confidence: float, winner_score: float, runner_up_score: float,
                       cfg: dict | None) -> str:
    """Compute the confidence bucket: high / medium / low."""
    cb = (cfg or {}).get("confidence_buckets", {}) or {}
    high_min = float(cb.get("high_min", 0.65))
    med_min  = float(cb.get("medium_min", 0.45))
    margin_ok = winner_score >= runner_up_score * 1.5
    if confidence >= high_min and margin_ok:
        return "high"
    if confidence < med_min:
        return "low"
    return "medium"


def classify_archetype(signals: dict, archetype_cfg: dict | None = None) -> ArchetypeResult:
    """Classify a position into a behavioral archetype.

    Returns an `ArchetypeResult` carrying:
      - winner archetype + policy
      - confidence (winner_score / total)
      - confidence_bucket (high / medium / low) tunable via archetype_classifier.confidence_buckets
      - runner_up + runner_up_score (second-best archetype)
      - scores (all 6 raw)
      - drivers (winner's human-readable evidence)
      - reason_codes (concise codes per archetype — for diagnostics)
      - missing_signals (signals the classifier wanted but didn't get)
      - features_used (signal keys actually consumed)

    When the new archetype_classifier config block is present (and `enabled=true`),
    config-driven thresholds + the strict defensive_income gate take effect.
    Otherwise the existing hardcoded thresholds run unchanged.
    """
    # Fund short-circuit: ETFs / CEFs / MLPs / ETNs are pooled vehicles, not
    # single companies. Running them through the stock factor scorecards produces
    # nonsense labels (e.g. a leveraged muni-bond CEF scored "speculative_momentum"
    # off its day-trade margin ratio). Detect via the shared core predicate
    # (Robinhood instrument_type or an explicit is_etf / asset_type flag) and route
    # straight to the dedicated `fund` archetype + policy.
    if _is_fund_signal(signals):
        policy = get_archetype_policy("fund", archetype_cfg)
        return ArchetypeResult(
            archetype="fund",
            confidence=1.0,
            scores={"fund": 1.0},
            drivers=[f"instrument_type={signals.get('instrument_type', 'fund')} — pooled fund (not a single stock)"],
            policy=policy,
            confidence_bucket="high",
            runner_up=None,
            runner_up_score=0.0,
            reason_codes={"fund": ["pooled_fund"]},
            features_used=["instrument_type"],
        )

    desc = signals.get("description", "")
    desc_feats = _desc_features(desc)
    desc_terms = _desc_matched_terms(desc)

    # Pull the archetype_classifier config (new); fall back to None if absent.
    try:
        from util import ARCHETYPE_CLASSIFIER_PARAMS as _ACP
        classifier_cfg: dict | None = _ACP if _ACP.get("enabled", False) else None
    except Exception:
        classifier_cfg = None

    s_comp, d_comp, rc_comp = _score_quality_compounder  (signals, desc_feats, desc_terms, cfg=classifier_cfg)
    s_leg,  d_leg,  rc_leg  = _score_legacy_turnaround   (signals, desc_feats, desc_terms, cfg=classifier_cfg)
    s_spec, d_spec, rc_spec = _score_speculative_momentum(signals, desc_feats, desc_terms, cfg=classifier_cfg)
    s_val,  d_val,  rc_val  = _score_value_recovery      (signals, desc_feats, desc_terms, cfg=classifier_cfg)
    s_def,  d_def,  rc_def  = _score_defensive_income    (signals, desc_feats, desc_terms, cfg=classifier_cfg)
    s_fallback = 0.05

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
    reason_code_map: dict[str, list[str]] = {
        "quality_compounder":   rc_comp,
        "legacy_turnaround":    rc_leg,
        "speculative_momentum": rc_spec,
        "value_recovery":       rc_val,
        "defensive_income":     rc_def,
        "core_default":         ["fallback"],
    }

    total = sum(raw_scores.values())
    sorted_archetypes = sorted(raw_scores.items(), key=lambda kv: -kv[1])
    winner, winner_score = sorted_archetypes[0]
    runner_up, runner_up_score = (sorted_archetypes[1] if len(sorted_archetypes) > 1
                                  else (None, 0.0))

    # If no archetype scored meaningfully, fall back to core_default
    if winner_score < 0.05:
        winner = "core_default"
        winner_score = s_fallback

    if total > 0.0:
        confidence = min(winner_score / total, 0.99)
    else:
        confidence = 0.50
    confidence = max(confidence, 0.30)

    bucket = _bucket_confidence(confidence, winner_score, runner_up_score, classifier_cfg)

    # Diagnostic: which signals were consumed vs missing
    features_used = [k for k in _EXPECTED_SIGNALS if k in signals and signals.get(k) is not None]
    missing_signals = [k for k in _EXPECTED_SIGNALS if k not in features_used]

    policy = get_archetype_policy(winner, archetype_cfg)

    return ArchetypeResult(
        archetype=winner,
        confidence=round(confidence, 3),
        scores={k: round(v, 4) for k, v in raw_scores.items()},
        drivers=driver_map[winner],
        policy=policy,
        confidence_bucket=bucket,
        runner_up=runner_up if runner_up != winner else None,
        runner_up_score=round(float(runner_up_score), 4),
        reason_codes=reason_code_map,
        missing_signals=missing_signals,
        features_used=features_used,
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
    *,
    sector: str | None = None,
    industry: str | None = None,
    market_cap: float | None = None,
    realized_vol: float | None = None,
    value_score: float | None = None,
) -> ArchetypePolicy:
    """Lightweight archetype classification for backtesting.

    Beyond the 4 base factor scores, accepts sector/industry/market_cap/realized_vol
    when the simulator has them in precomp — substantially closing the live/backtest
    label-disagreement gap. Returns the policy directly.

    Used by backtesting/simulator.py to assign per-position exit thresholds without
    requiring external data fetches.
    """
    signals: dict = {
        "quality_score":  quality_score,
        "momentum_score": momentum_score,
        "income_score":   income_score,
        "yield_trap_flag": yield_trap,
    }
    if sector is not None:
        signals["sector"] = sector
    if industry is not None:
        signals["industry"] = industry
    if market_cap is not None:
        signals["market_cap"] = market_cap
    if realized_vol is not None:
        signals["realized_vol_3m"] = realized_vol
    if value_score is not None:
        signals["value_score"] = value_score
    result = classify_archetype(signals, archetype_cfg)
    return result.policy


def classify_archetype_full_from_scores(
    quality_score: float,
    momentum_score: float,
    income_score: float,
    yield_trap: bool = False,
    archetype_cfg: dict | None = None,
    **extra,
) -> ArchetypeResult:
    """Same signal-augmented entry point but returns the full ArchetypeResult
    (with confidence, bucket, runner_up, reason_codes) instead of just the policy.

    The simulator can use this when it needs to populate confidence-bucket
    attribution rollups.
    """
    signals: dict = {
        "quality_score":  quality_score,
        "momentum_score": momentum_score,
        "income_score":   income_score,
        "yield_trap_flag": yield_trap,
    }
    for k in ("sector", "industry", "market_cap", "realized_vol_3m", "value_score"):
        if extra.get(k) is not None:
            signals[k] = extra[k]
    return classify_archetype(signals, archetype_cfg)
