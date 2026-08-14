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


def _bool(key: str, default: bool) -> bool:
    val = _CFG.get(key)
    return val if isinstance(val, bool) else default


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
# Breadth score (2026-08-07): confirmation is graded, not counted. Each index contributes its VWAP
# side plus its opening-range side (odte_tape.FULL_ALIGNMENT == 2 for a fully aligned index), so an
# index above VWAP but still inside its opening range finally counts for something instead of
# nothing. The defaults are exactly 2x the confirmer counts above, so "two full confirmers" keeps
# its meaning and "one full plus two halves" becomes its equal.
#
# Why: on 2026-08-07 a SPY-led breakout sat un-convertible at confirmations=1 for 45 minutes while
# day_score graded the same tape GOOD_DAY off "3 indices trend-aligned" — the two modules scored
# breadth by different rules. Replay over the snapshot history (scripts/odte_breadth_replay.py):
# 51 tick-level opens, 0 closes, but only 2 of 10 replayable sessions gain a confirm they did not
# already have, so the daily budget bounds this at ~4 added trades across the replayed history.
BREADTH_MIN_SCORE: int = _int("breadth_min_score", B_PLUS_MIN_CONFIRMATIONS * 2)
A_PLUS_MIN_BREADTH: int = _int("a_plus_min_breadth", A_PLUS_MIN_CONFIRMATIONS * 2)
B_PLUS_MIN_VEHICLE_SCORE: int = _int("b_plus_min_vehicle_score", 3)
GOOD_DAY_MIN_SCORE: int = _int("good_day_min_score", 4)

# ── Trade cadence ──────────────────────────────────────────────────────────────
DAILY_TRADE_BUDGET: int = _int("daily_trade_budget", 2)
# A+ UNCAPPED (2026-08-06, user policy): an a_plus-tier setup (>=3 confirmers, zero dissenters)
# is exempt from the daily budget cap WHILE the day is net-green — winners fund more attempts,
# any net-red day hard-stops at the base cap (the anti-tilt half of the rail stays). Every other
# rail applies per trade: tier-fraction BP sizing, cooldown, chase band, green re-entry tier bar.
DAILY_BUDGET_APLUS_UNCAPPED: bool = _bool("daily_budget_aplus_uncapped", True)
REENTRY_COOLDOWN_MINUTES: int = _int("reentry_cooldown_minutes", 20)
GREEN_REENTRY_MIN_BP_MULTIPLE: float = _num("green_reentry_min_bp_multiple", 1.5)
# 2026-08-03: after a banked green, re-entry auto-arms when the daily budget has a slot, the
# cooldown passed, the new candidate's tape tier ranks >= the winning trade's tier, and BP covers
# the multiple above. On 2026-08-03 the manual-override-only rule made trade #2 structurally
# impossible while SPY ran 2.6 points. False restores the manual-override-only behavior.
GREEN_REENTRY_AUTO_ARM: bool = _bool("green_reentry_auto_arm", True)
# STRICT RE-ENTRY BAR (2026-08-14, operator decision). The same-or-better bar is what failed:
# both tiered-era post-green losers (08-10 -$17, 08-11 -$34) were SAME-tier b_plus re-entries
# after b_plus wins — a same-grade setup re-taken after variance already paid. With this on, a
# re-entry must be STRICTLY better than the winning tier: after b_plus only full/a_plus arms,
# after full only a_plus, and an a_plus win ends the day. n=0 on strictly-better outcomes — no
# such re-entry has ever occurred — so this is pre-registered upside, not proven edge.
GREEN_REENTRY_REQUIRE_BETTER_TIER: bool = _bool("green_reentry_require_better_tier", False)

# ── Freshness ──────────────────────────────────────────────────────────────────
# Lease default TTL (still clamped by the un-tunable 60s hard cap in
# odte_execution_policy). Snapshot TTL covers the observed fetch→authorize
# latency; price risk inside it is bounded by the chase band + lease ceilings.
LEASE_TTL_SECONDS: float = _num("lease_ttl_seconds", 60.0)
SNAPSHOT_TTL_SECONDS: float = _num("snapshot_ttl_seconds", 120.0)
CONFIRM_CONVERSION_SLA_SECONDS: float = _num("confirm_conversion_sla_seconds", 180.0)

# ── Computed final confirmations (odte-convert) ────────────────────────────────
SPREAD_CAP_FRACTION: float = _num("spread_cap_fraction", 0.25)

# ── W33 entry fences (2026-08-14) ──────────────────────────────────────────────
# Pre-registered in the W33 post-mortem plan; each code default is OFF so deleting the YAML key is
# the rollback. Evidence (all 20 closed trades, forensic audit 2026-08-13):
#   min_entry_premium      — every recent loser entered at $0.45-0.50 premium; every winner
#                            >= $0.64. At DEFAULT_MAX_CONTRACTS=1 the tier debit fraction halves
#                            nothing, so premium selection is the only real size knob, and cheap
#                            contracts carry the worst tick-% and spread-% economics.
#   midday_full_tier_after_et_hour — the 13:00-15:30 entry bucket is 1 win in 6, -$49; after this
#                            ET hour a NEW entry must be tier >= full (b_plus blocked).
#   daily_loss_floor_dollars — the daily budget caps COUNT (2), not dollars; 08-11 stacked a -$34
#                            stop on a +$22 day. Once day net P/L <= -N, no new entries.
MIN_ENTRY_PREMIUM: float = _num("min_entry_premium", 0.0)
MIDDAY_FULL_TIER_AFTER_ET_HOUR: float = _num("midday_full_tier_after_et_hour", 0.0)
DAILY_LOSS_FLOOR_DOLLARS: float = _num("daily_loss_floor_dollars", 0.0)

# ── Tier-scaled contract sizing (2026-08-14, operator decision) ────────────────
# "A +20% trade moving the account +3.3% is the cap working against us." At 1 contract the tier
# debit fractions (0.30/0.60) never bind, so tier quality changes nothing about size. These raise
# the CONTRACT ceiling for the stronger tiers only — b_plus (the tier that carried every recent
# loss) stays at DEFAULT_MAX_CONTRACTS=1. Quantity is still capped by the tier debit fraction and
# buying power at authorize time; code default 1 = feature off, YAML arms it.
TIER_MAX_CONTRACTS_FULL: int = _int("tier_max_contracts_full", 1)
TIER_MAX_CONTRACTS_APLUS: int = _int("tier_max_contracts_aplus", 1)

# ── Watchdog ───────────────────────────────────────────────────────────────────
WATCHDOG_REALERT_MINUTES: int = _int("watchdog_realert_minutes", 10)

# ── Weekly trade-rate telemetry ────────────────────────────────────────────────
WEEKLY_TRADE_TARGET_MIN: int = _int("weekly_trade_target_min", 3)
WEEKLY_TRADE_TARGET_MAX: int = _int("weekly_trade_target_max", 4)
# 0=Mon .. 6=Sun: tripwire arms when the ET weekday reaches this with 0 trades.
ZERO_TRADE_TRIPWIRE_WEEKDAY: int = _int("zero_trade_tripwire_weekday", 2)
