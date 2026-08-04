"""0DTE tunable knobs, read once from the `odte:` section of cfg/config.yaml.

Leaf module: imports only core/stdlib so every odte_* module (and its tests) can
bind constants from here without layering violations. Every key has a code
default — a missing/partial `odte:` section changes nothing. Values that arrive
with the wrong type fall back to the default (fail-safe, never fail-open).

These knobs deliberately exclude the execution-safety invariants that must NOT
be config-tunable: the 60s lease hard cap, the 1-contract PARTIAL_ACCOUNT limit,
single-use lease semantics, and the FULL_ACCOUNT_A_PLUS proof requirements all
stay hard-coded in odte_execution_policy.py.
"""
from __future__ import annotations

from core.utils import load_config_raw, safe_float


def _section() -> dict:
    raw = load_config_raw()
    sec = raw.get("odte") if isinstance(raw, dict) else None
    return sec if isinstance(sec, dict) else {}


_CFG = _section()


def _num(key: str, default: float) -> float:
    val = safe_float(_CFG.get(key), default)
    return default if val is None else float(val)


def _int(key: str, default: int) -> int:
    return int(_num(key, float(default)))


def _symbols(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    val = _CFG.get(key)
    if not isinstance(val, (list, tuple)) or not val:
        return default
    syms = tuple(str(s).strip().upper() for s in val if str(s).strip())
    return syms or default


# ── Universes ──────────────────────────────────────────────────────────────────
# scan universe: symbols whose tape counts for confirmation/dissent.
# executable universe: symbols a candidate may actually convert on (XSP stays
# scan-only: thin broker liquidity for this account, but its tape still confirms).
SCAN_UNIVERSE: tuple[str, ...] = _symbols("scan_universe", ("SPY", "QQQ", "XSP", "IWM"))
EXECUTABLE_UNIVERSE: tuple[str, ...] = _symbols("executable_universe", ("SPY", "QQQ", "IWM"))

# ── Entry pricing rails ────────────────────────────────────────────────────────
# Chase band: max fill above the anchor quote stamped at CONFIRM_ENTRY. The old
# rail anchored at precompute and tolerated 0% — it vetoed every working
# momentum entry (premium rises BECAUSE the thesis confirms).
CHASE_BAND_FRACTION: float = _num("chase_band_fraction", 0.15)

# BP-proportional debit caps (replace the flat $120 agent-side rail).
MAX_DEBIT_FRACTION: float = _num("max_debit_fraction", 0.60)
B_PLUS_DEBIT_FRACTION: float = _num("b_plus_debit_fraction", 0.30)

# ── Confirmation tiers ─────────────────────────────────────────────────────────
A_PLUS_MIN_CONFIRMATIONS: int = _int("a_plus_min_confirmations", 3)
B_PLUS_MIN_CONFIRMATIONS: int = _int("b_plus_min_confirmations", 2)
B_PLUS_MAX_DISSENTERS: int = _int("b_plus_max_dissenters", 1)
B_PLUS_MIN_VEHICLE_SCORE: int = _int("b_plus_min_vehicle_score", 3)
GOOD_DAY_MIN_SCORE: int = _int("good_day_min_score", 4)

# ── Trade cadence ──────────────────────────────────────────────────────────────
DAILY_TRADE_BUDGET: int = _int("daily_trade_budget", 2)
REENTRY_COOLDOWN_MINUTES: int = _int("reentry_cooldown_minutes", 20)
GREEN_REENTRY_MIN_BP_MULTIPLE: float = _num("green_reentry_min_bp_multiple", 1.5)

# ── Freshness ──────────────────────────────────────────────────────────────────
# Lease default TTL (still clamped by the un-tunable 60s hard cap in
# odte_execution_policy). Snapshot TTL covers the observed fetch→authorize
# latency; price risk inside it is bounded by the chase band + lease ceilings.
LEASE_TTL_SECONDS: float = _num("lease_ttl_seconds", 60.0)
SNAPSHOT_TTL_SECONDS: float = _num("snapshot_ttl_seconds", 120.0)
CONFIRM_CONVERSION_SLA_SECONDS: float = _num("confirm_conversion_sla_seconds", 180.0)

# ── Computed final confirmations (odte-convert) ────────────────────────────────
SPREAD_CAP_FRACTION: float = _num("spread_cap_fraction", 0.25)

# ── Watchdog ───────────────────────────────────────────────────────────────────
WATCHDOG_REALERT_MINUTES: int = _int("watchdog_realert_minutes", 10)

# ── Weekly trade-rate telemetry ────────────────────────────────────────────────
WEEKLY_TRADE_TARGET_MIN: int = _int("weekly_trade_target_min", 3)
WEEKLY_TRADE_TARGET_MAX: int = _int("weekly_trade_target_max", 4)
# 0=Mon .. 6=Sun: tripwire arms when the ET weekday reaches this with 0 trades.
ZERO_TRADE_TRIPWIRE_WEEKDAY: int = _int("zero_trade_tripwire_weekday", 2)
