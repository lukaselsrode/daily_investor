"""0DTE atomic conversion — CONFIRM_ENTRY → entry gate → execution lease in ONE process.

PURE/OFFLINE composition of the existing decision tiers, places NO orders, makes NO broker/network/
LLM calls. Born from the 2026-07-27..31 zero-trade week: the controller orchestrated candidate-watch
→ entry-gate → authorize as separate commands across a ~5-minute LLM tick, so the 60-second
freshness bounds between them were structurally unmeetable — on 2026-07-31 the gate PASSED and the
lease was refused 11 seconds later on stale snapshots built earlier in the same tick. This command
runs the whole conversion under ONE clock in one process (milliseconds between stages), so the
tight in-process TTLs pass by construction while remaining fully enforced.

The caller (Hermes) supplies FRESH market/broker/contract snapshots fetched immediately before the
call; the preflight refuses — naming the exact stale input — when they exceed the snapshot TTL.
The three final confirmations (`live_chain_recheck`, `spread_cap_check`, `budget_check`) are
COMPUTED here from the supplied artifacts instead of being asserted by the agent — deterministic on
this path, exactly as journaled.

Every non-converting stage journals an IDENTITY-BOUND terminal `no_trade_decision` (option_id +
candidate fingerprint + failed stage), closing the silent scan-only dead end: `odte-loop-status`'s
conversion accountability then resolves the confirm instead of raising a conversion failure. A
journal-append error withholds the lease (fail closed), same rule as the entry-gate CLI.

Execution-safety invariants are UNCHANGED: the lease is still minted only by
`odte_execution_policy.authorize_entry` (single-use, exact identity, 60s hard cap, chase band,
tiered BP-proportional sizing) and the broker order still runs the pending-order guard.

----------------------------------------------------------------------------------------------------
EVERY PATH THAT CAN REACH AN ORDER  (enumerated 2026-08-07 — keep this current)
----------------------------------------------------------------------------------------------------
1. CONTROLLER SCAN LANE — the only one that has ever traded.
     odte-watchdog -> triggers.json -> controller -> odte-candidate-watch -> CONFIRM_ENTRY
     -> odte-convert (preflight -> day_score -> vehicle_score -> candidate re-check -> entry gate
        -> authorize_entry mints the lease)
     -> agent: ONE review_option_order + place_option_order inside the 60s lease
     -> odte-order-guard poll -> entry_fill -> odte-position HAWK -> exit.
2. CONTROLLER POST-GREEN RE-ENTRY — same path, re-armed by `green_reentry` when the budget has a
   slot, the cooldown passed, tier ranks >= the winner's and BP covers the multiple.
3. WATCHDOG LANE — CANNOT reach an order. `scan_only=True` / `execution_allowed=False` at every
   return, no order/broker/lease imports at all, and a `market_scorecard` candidate is refused by
   candidate-watch even on a perfect tape. Verified by assertion, not by reading.
4. FAST LANE (`execution/odte_fast_lane.py`) — armed intents -> in-process convert -> direct MCP
   place. Dormant (stage `shadow`, no process/plist/cron). It SHARES `consumed_leases.json` with
   this lane, which is why the ledger claim is an atomic test-and-set.
5. EOD / close-bell jobs — read-only reporting; they place nothing.

STATE OWNERSHIP AND THE ONE INTENTIONAL DUAL WRITER
  `active_candidate.json` and `candidate_decision.json` are written by BOTH `odte_candidate_watch`
  and this module, sequentially within one controller tick. That is the atomic re-check working as
  designed, not a race: on 2026-08-07 candidate-watch wrote CONFIRM_ENTRY at 15:29:14 and convert
  correctly overwrote it with KEEP_WATCHING at 15:30:03 when the tape decayed 49 seconds later.
  TREAT BOTH FILES AS PER-TICK SNAPSHOTS, NEVER AS STABLE STATE — a reader that assumes stability
  is the bug. Everything else is single-owner: the watchdog owns triggers/watchdog_state, position
  owns position_state/position_decision/active_trade, order_guard owns order_guard_decision.
  `decision_journal.jsonl` has many writers and is safe: appends take an advisory file lock.

RESIDUAL, ACCEPTED
  Controller ticks are serialised by the cron lock, so a long tick SKIPS the next rather than
  overlapping (2026-08-07: the 14:00 tick ran to 14:10:57 and 14:05/14:10 never fired — that tick
  was the position management, not a gap). If a tick dies while a position is open, management
  resumes at the next tick, so the unmanaged window is bounded by the 5-minute schedule.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from core.paths import ODTE_DATA_DIR, atomic_write_text
from data.odte_config import SNAPSHOT_TTL_SECONDS, SPREAD_CAP_FRACTION

_ET = ZoneInfo("America/New_York")

SCHEMA_VERSION = 1
DEFAULT_STATE_DIR = ODTE_DATA_DIR

# Session constants persisted per ET trade date (currently just the overnight gap) so a snapshot
# that loses a constant key mid-session can be backfilled instead of silently re-tiering the day.
SESSION_CONSTANTS_FILENAME = "session_constants.json"

# Conversion stages, in order. `stage` in the payload names the FIRST stage that refused.
STAGES = ("preflight", "candidate_watch", "entry_gate", "journal", "authorize")


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _num(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _first_num(d: dict, *keys: str) -> float | None:
    """First key whose value PARSES as a number — a legitimate 0.0 wins (2026-08-04 hardening:
    `or`-chains let a real zero fall through to the next key, silently sizing off the wrong field)."""
    for key in keys:
        val = _num(d.get(key))
        if val is not None:
            return val
    return None


def _load_json(path: str | None, raw_json: str | dict | None,
               default: dict | None = None) -> dict:
    # Callers holding the artifact in memory (the fast-lane daemon) pass the dict itself — a
    # json.dumps round trip would be pure waste on the latency-critical path.
    if isinstance(raw_json, dict):
        return raw_json
    if raw_json:
        obj = json.loads(raw_json)
    elif path:
        obj = json.loads(Path(os.path.expanduser(path)).read_text())
    else:
        return dict(default or {})
    if not isinstance(obj, dict):
        raise ValueError("payload must be a JSON object")
    return obj


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _payload_ts(payload: dict) -> datetime | None:
    for key in ("generated_at", "as_of", "ts", "updated_at", "created_at"):
        dt = _parse_ts(payload.get(key))
        if dt is not None:
            return dt
    return None


def _identity(candidate: dict) -> dict:
    """Identity fields for the terminal no-trade event — the same keys loop-status matches on."""
    from data.odte_execution_policy import candidate_fingerprint
    cand = _dict(candidate)
    return {
        "symbol": (cand.get("ticker") or cand.get("underlying") or cand.get("symbol")),
        "option_id": cand.get("option_id") or cand.get("id") or cand.get("occ_symbol"),
        "candidate_fingerprint": (cand.get("candidate_fingerprint")
                                  or (candidate_fingerprint(cand) if cand else None)),
    }


def _computed_confirmations(candidate: dict, contract: dict, broker: dict,
                            now: datetime) -> tuple[dict, dict]:
    """Deterministically compute the three final confirmations from the supplied artifacts.

    Returns (confirmations, detail). Missing inputs compute False — fail closed, never None —
    because on THIS path the caller is contractually feeding fresh artifacts.
    """
    detail: dict[str, Any] = {}

    quote_ts = _payload_ts(contract)
    quote_fresh = bool(quote_ts is not None
                       and 0 <= (now - quote_ts).total_seconds() <= SNAPSHOT_TTL_SECONDS)
    cand_option = str(candidate.get("option_id") or candidate.get("id")
                      or candidate.get("occ_symbol") or "").strip()
    contract_option = str(contract.get("option_id") or contract.get("id")
                          or contract.get("occ_symbol") or "").strip()
    identity_match = bool(cand_option and contract_option and cand_option == contract_option)
    live_chain_recheck = quote_fresh and identity_match
    detail["live_chain_recheck"] = {"quote_fresh": quote_fresh, "identity_match": identity_match,
                                    "quote_age_seconds": (round((now - quote_ts).total_seconds(), 1)
                                                          if quote_ts else None)}

    bid = _first_num(contract, "bid", "bid_price")
    ask = _first_num(contract, "ask", "ask_price")
    spread = None
    if bid is not None and ask is not None and (bid + ask) > 0:
        spread = (ask - bid) / ((ask + bid) / 2.0)
    spread_cap_check = bool(spread is not None and 0 <= spread <= SPREAD_CAP_FRACTION)
    detail["spread_cap_check"] = {"spread": (round(spread, 4) if spread is not None else None),
                                   "cap": SPREAD_CAP_FRACTION}

    from data.odte_config import B_PLUS_DEBIT_FRACTION, MAX_DEBIT_FRACTION
    bp = _first_num(broker, "buying_power", "options_buying_power", "account_buying_power")
    tier = str(candidate.get("tier") or "").lower() or None
    frac = B_PLUS_DEBIT_FRACTION if tier == "b_plus" else MAX_DEBIT_FRACTION
    debit = round(ask * 100.0, 2) if ask is not None else None
    budget_check = bool(debit is not None and bp is not None and debit <= frac * bp)
    detail["budget_check"] = {"debit": debit, "buying_power": bp, "tier": tier,
                               "max_debit_fraction": frac}

    return ({"live_chain_recheck": live_chain_recheck,
             "spread_cap_check": spread_cap_check,
             "budget_check": budget_check}, detail)


def _journal_no_trade(stage: str, reason_codes: list[str], candidate: dict,
                      journal_path: str | None, now: datetime,
                      extra: dict | None = None) -> dict:
    """Journal the identity-bound terminal no_trade_decision for a non-converting stage."""
    from data.odte_journal import append_decision_journal
    ident = _identity(candidate)
    event = {
        "event_type": "no_trade_decision",
        "decision": "NO_TRADE",
        "ts": now.isoformat(timespec="seconds"),
        "stage": stage,
        "reason_codes": list(reason_codes),
        **(extra or {}),
        **ident,
        "candidate": {k: candidate.get(k) for k in
                      ("ticker", "direction", "option_id", "expiration_date", "strike_price",
                       "option_type", "candidate_fingerprint", "tier", "anchor_quote")
                      if candidate.get(k) is not None},
        "basis": "odte-convert terminal disposition — no silent scan-only dead ends",
    }
    return append_decision_journal(event, source="odte_convert", event_type="no_trade_decision",
                                   journal_path=journal_path, now=now)


# Only artifacts a builder can actually produce appear here. `broker_snapshot` is deliberately
# absent: `odte_build_snapshot` has no broker mode yet, and emitting a command that does not exist
# would be worse guidance than none.
_PREFLIGHT_REMEDY = {
    "market_snapshot": (".venv/bin/python scripts/odte_build_snapshot.py "
                        "--historicals <get_equity_historicals> --quotes <get_equity_quotes> "
                        "--gap-pct <session gap_pct> --out data/odte/market.json"),
    "contract_quote": (".venv/bin/python scripts/odte_build_snapshot.py "
                       "--instruments <get_option_instruments> "
                       "--option-quotes <get_option_quotes> "
                       "--contract-option-id <candidate option_id> "
                       "--out-contract data/odte/contract.json"),
}


def run_convert(candidate_json: str | dict | None = None, candidate_path: str | None = None,
                market_json: str | dict | None = None, market_path: str | None = None,
                broker_json: str | dict | None = None, broker_path: str | None = None,
                contract_json: str | dict | None = None, contract_path: str | None = None,
                gamma_json: str | dict | None = None, gamma_path: str | None = None,
                policy_json: str | dict | None = None, policy_path: str | None = None,
                journal_path: str | None = None, state_dir: str | None = None,
                write: bool = True, journal: bool = True,
                journal_events: list[dict] | None = None,
                now: datetime | None = None) -> dict:
    """Run the full CONFIRM_ENTRY→gate→lease conversion atomically under one clock.

    Inputs: an active/confirmed candidate (default `data/odte/active_candidate.json`), fresh market
    + broker snapshots, the exact locked contract quote, optional gamma map and lease policy.
    The `*_json` inputs accept a JSON string OR the already-parsed dict (fast-lane callers skip
    the serialize/parse round trip). Returns `{converted, stage, reason_codes, day_score,
    vehicle_score, candidate_decision, entry_gate, authorization, ...}`. `write=True` persists
    candidate_decision/active_candidate and the lease artifact through the same atomic writers the
    individual CLIs use; `journal=True` appends the entry_decision / execution_lease_issued /
    terminal no_trade events. `journal_events` overrides the budget/green/cooldown READ without
    touching append behavior — shadow mode reads the REAL journal (so those gates evaluate as they
    would live) while appending nowhere (`journal=False`) or to a shadow journal.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    from data.odte_candidate_watch import (
        ACTIVE_CANDIDATE_FILENAME,
        CANDIDATE_DECISION_FILENAME,
        CONFIRM_ENTRY,
        evaluate_candidate_watch,
    )
    from data.odte_day_score import score_day
    from data.odte_entry_gate import build_entry_gate_decision
    from data.odte_execution_policy import (
        CONSUMED_LEASES_FILENAME,
        LEASE_FILENAME,
        authorize_entry,
        seed_consumed_ledger,
    )
    from data.odte_journal import (
        DEFAULT_JOURNAL_PATH,
        append_decision_journal,
        event_from_candidate_evaluation,
        event_from_day_score,
        event_from_entry_gate,
        event_from_execution_lease,
        event_from_vehicle_score,
        read_events,
    )
    from data.odte_vehicle_score import score_vehicle

    default_candidate = (str(Path(os.path.expanduser(state_dir or DEFAULT_STATE_DIR))
                             / ACTIVE_CANDIDATE_FILENAME)
                         if not candidate_json and not candidate_path else None)
    candidate = _load_json(candidate_path or default_candidate, candidate_json)
    market = _load_json(market_path, market_json)
    broker = _load_json(broker_path, broker_json)
    contract = _load_json(contract_path, contract_json)
    gamma = _load_json(gamma_path, gamma_json, default={})
    policy = _load_json(policy_path, policy_json, default={})
    journal_path = journal_path or DEFAULT_JOURNAL_PATH
    if journal_events is None:
        try:
            journal_events = read_events(journal_path) if journal else []
        except OSError:
            # An unreadable journal is not fatal here — the append below fails loudly and
            # withholds the lease, which is the honest fail-closed surface for a broken journal.
            journal_events = []
    else:
        journal_events = list(journal_events)

    base_dir = Path(os.path.expanduser(state_dir or DEFAULT_STATE_DIR))

    # ── SESSION-CONSTANT GAP BACKFILL (2026-08-04 hardening) ─────────────────────────────────
    # gap_pct is the OVERNIGHT gap — constant for the whole session. On 2026-08-04 the controller
    # dropped the key mid-day, which silently flipped GOOD_DAY→CHOP → tier full→b_plus → cap
    # halved → a $214.00 debit refused against $214.24. Persist the first sighting per ET day and
    # backfill later snapshots that lost it, loudly flagged.
    gap_backfilled = False
    session_constants: dict[str, Any] = {}
    sc_path = base_dir / SESSION_CONSTANTS_FILENAME
    et_today = now.astimezone(_ET).date().isoformat()
    try:
        session_constants = json.loads(sc_path.read_text()) if sc_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        session_constants = {}
    gap_now = _num(market.get("gap_pct"))
    if gap_now is not None:
        if write and (session_constants.get("trade_date") != et_today
                      or session_constants.get("gap_pct") != gap_now):
            session_constants = {"trade_date": et_today, "gap_pct": gap_now}
            base_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(sc_path, json.dumps(session_constants, indent=2))
    elif (session_constants.get("trade_date") == et_today
          and _num(session_constants.get("gap_pct")) is not None):
        market = {**market, "gap_pct": _num(session_constants["gap_pct"]),
                  "gap_pct_backfilled": True}
        gap_backfilled = True

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "converted": False,
        "stage": None,
        "reason_codes": [],
        "places_orders": False,
        "execution_allowed": False,
        "scan_only": True,
        "basis": ("atomic CONFIRM_ENTRY→gate→lease conversion under one clock; composes the "
                  "existing deterministic tiers, weakens none of them, places NO orders"),
    }

    # Authorize-stage reasons that mean THE CONTRACT didn't fit the account, not the day — the
    # correct next step is a FRESH candidate cycle on a BP-fitting vehicle, not a stand-down.
    _AUTHORIZE_BP_REASONS = {"debit_exceeds_policy", "debit_exceeds_buying_power",
                             "insufficient_buying_power"}

    def _refuse(stage: str, reasons: list[str], **extra: Any) -> dict:
        payload.update({"stage": stage, "reason_codes": reasons, **extra})
        # GUIDANCE PASS-THROUGH (2026-08-04): the generic "refresh and re-run" prose here used to
        # OVERWRITE the entry gate's specific guidance — the rotate-to-a-BP-fitting-vehicle
        # instruction never reached the controller, which re-tried one unaffordable SPY contract
        # 12x while an $80 IWM GOOD_BET sat on disk. Stage-specific guidance wins.
        gate_payload = _dict(payload.get("entry_gate"))
        if stage == "entry_gate" and gate_payload.get("next_action"):
            payload["next_action"] = gate_payload["next_action"]
            payload["next_command"] = gate_payload.get("next_command") or "odte-convert"
        elif stage == "authorize" and _AUTHORIZE_BP_REASONS.intersection(reasons):
            payload["next_action"] = (
                "the CONTRACT doesn't fit the account, not the day — start a FRESH candidate "
                "cycle on a BP-fitting vehicle (QQQ/SPY/IWM; the vehicle lock forbids silent "
                "substitution), then re-run odte-convert; reject only if every candidate vehicle "
                "fails")
            payload["next_command"] = ("make odte-candidate-watch CANDIDATE=<other-vehicle "
                                       "candidate.json> ... JSON=1 → odte-convert")
        elif stage == "preflight":
            # NAME THE BUILDER, DON'T JUST SAY "REFRESH" (2026-08-12). `scripts/odte_build_snapshot.py`
            # has gained contract mode but was invoked 0 times across every conversion — the
            # controller hand-authors contract.json instead, which is what produced
            # contract_quote_missing, contract_quote_undated (the raw `{"result": "<json>"}` MCP
            # envelope saved verbatim) and a live_chain_recheck failure. Telling the agent to use
            # the tool in prose has a poor record; `next_command` is followed. So a preflight
            # refusal now answers "refresh it HOW" with the exact invocation.
            arts = list(dict.fromkeys(r.rsplit("_", 1)[0] for r in reasons))
            cmds = [_PREFLIGHT_REMEDY[a] for a in arts if a in _PREFLIGHT_REMEDY]
            payload["next_action"] = (
                f"conversion refused at preflight ({', '.join(reasons[:4])}) — REBUILD the named "
                "artifact with the builder below rather than hand-writing it; pipe the MCP "
                "response in as-is (the tool unwraps the envelope), then re-run odte-convert")
            payload["next_command"] = (" ; ".join(cmds) if cmds else
                                       "odte-convert  # after refreshing the failing inputs")
        else:
            payload["next_action"] = (f"conversion refused at {stage} ({', '.join(reasons[:4])}) "
                                      "— refresh the named inputs and re-run odte-convert, or let "
                                      "the candidate degrade; the refusal is journaled terminally")
            payload["next_command"] = "odte-convert  # after refreshing the failing inputs"
        if journal:
            result = _journal_no_trade(stage, reasons, _dict(payload.get("candidate")) or candidate,
                                       journal_path, now,
                                       extra={"gap_pct_backfilled": True} if gap_backfilled else None)
            payload["journal_append"] = {"status": result.get("status"),
                                         "event_id": result.get("event_id")}
        return payload

    # ── preflight: snapshots must be fresh NOW, named individually ────────────────────────────
    preflight: list[str] = []
    for name, artifact in (("market_snapshot", market), ("broker_snapshot", broker),
                           ("contract_quote", contract)):
        ts = _payload_ts(artifact)
        if not artifact:
            preflight.append(f"{name}_missing")
        elif ts is None:
            preflight.append(f"{name}_undated")
        elif (now - ts).total_seconds() > SNAPSHOT_TTL_SECONDS:
            preflight.append(f"{name}_stale")
    if not candidate:
        preflight.append("candidate_missing")
    if preflight:
        return _refuse("preflight", preflight)

    # ── deterministic scores under the same clock ─────────────────────────────────────────────
    day_score = score_day(market=market, gamma=gamma)
    bp = _first_num(broker, "buying_power", "options_buying_power", "account_buying_power")
    direction = str(candidate.get("direction") or candidate.get("option_type") or "") or None
    vehicle_score = score_vehicle(contract, direction=direction, market=market, gamma=gamma,
                                  buying_power=bp,
                                  tier=str(candidate.get("tier") or "") or None)
    # The scores are computed in-process AT `now` — stamp them with the conversion clock so the
    # lease freshness checks measure this run's clock, not the module wall clock.
    day_score["generated_at"] = now.isoformat(timespec="seconds")
    vehicle_score["generated_at"] = now.isoformat(timespec="seconds")
    # Components ride every payload so a single-component flip (the 2026-08-04 gap 1→0) is
    # VISIBLE, never a silent re-tiering.
    payload["day_score"] = {"verdict": day_score.get("verdict"), "score": day_score.get("score"),
                            "components": day_score.get("components")}
    # Journal the score where it is COMPUTED. The artifact under reports/ is overwritten every
    # tick, so before this the only day_score events in the journal were whatever the EOD artifact
    # sweep happened to catch — 2 events on one day, which cannot show whether GOOD_DAY was ever
    # structurally reachable (max_possible_score vs the threshold) as the session progressed.
    if journal:
        append_decision_journal(event_from_day_score(day_score), source="odte_convert",
                                event_type="day_score", journal_path=journal_path, now=now)
    payload["gap_pct_backfilled"] = gap_backfilled
    if gap_backfilled:
        payload["session_constants"] = session_constants
    payload["vehicle_score"] = {"verdict": vehicle_score.get("verdict"),
                                 "score": vehicle_score.get("score")}
    # Journal the vehicle score where it is COMPUTED. This is the only live caller of
    # score_vehicle, and until now it kept two keys and dropped the rest — so `bp_fit` (shipped
    # 2026-08-05 to explain the 2026-08-04 refusals: a $288 contract certified GOOD_BET against
    # raw BP while the gate refused it against the 60%-tier cap 12x) recorded ZERO events. On
    # 2026-08-06 this line ran on 14 gate ticks and journaled none of them. Telemetry only:
    # scan_only, no trade_id, never a gate input.
    if journal:
        append_decision_journal(event_from_vehicle_score(vehicle_score), source="odte_convert",
                                event_type="vehicle_score", journal_path=journal_path, now=now)

    # ── fresh tape re-check + re-anchor (candidate watch) ─────────────────────────────────────
    cd = evaluate_candidate_watch(candidate, market=market, day_score=day_score,
                                  vehicle_score=vehicle_score, gamma_map=gamma,
                                  broker_health=broker, now=now)
    payload["candidate_decision"] = cd
    payload["candidate"] = _dict(cd.get("candidate"))
    # Journal the evaluation on EVERY tick, before the refusal branch below — the near-miss
    # population (tape confirmed, one clause short) lives entirely in the ticks that do NOT
    # convert, and until now those checks reached only an overwritten artifact. Telemetry only:
    # scan_only, never gate input. append_decision_journal never raises.
    if journal:
        append_decision_journal(event_from_candidate_evaluation(cd),
                                source="odte_convert", event_type="candidate_evaluation",
                                journal_path=journal_path, now=now)
        # CLOSE OUT THE SUPERSEDED FINGERPRINT (2026-08-12). Confirms arrive in pairs: the tape
        # lane confirms without a contract, then convert re-binds one here. `candidate_fingerprint`
        # hashes the option_id/expiry/strike/type, so the two are DIFFERENT IDENTITIES BY
        # CONSTRUCTION — that rebind is the safety property that stops a stale confirm converting
        # into a changed tape, so it stays. What was missing is a terminal event for the id being
        # replaced: on 2026-08-12 both of the day's confirms orphaned their predecessor
        # (7016c855 -> 8e6cb1a5, 43705ae7 -> 70539262), leaving 2 of 4 unauditable.
        #
        # This belongs HERE, not in run_candidate_watch: by the watch tick after a rebind the
        # candidate file already holds the new fingerprint, so watch compares it to itself and
        # never sees a difference. Convert is the only place holding both ids at once.
        #
        # Telemetry only — scan_only, never a gate input, no trade_id, and weekly_telemetry
        # deliberately ignores candidate_evaluation when moving funnel counters.
        prior = _dict(candidate)
        prior_fp = prior.get("candidate_fingerprint")
        new_fp = _dict(cd.get("candidate")).get("candidate_fingerprint")
        if prior_fp and new_fp and prior_fp != new_fp and prior.get("state") == "CONFIRMED_ENTRY":
            append_decision_journal(
                {**event_from_candidate_evaluation(cd),
                 "decision": "superseded_by_rebind",
                 "candidate_fingerprint": prior_fp,
                 "superseded_by": new_fp,
                 "scan_only": True, "execution_allowed": False, "trade_id": None,
                 "reason_codes": ["contract rebind replaced the contractless tape confirm"]},
                source="odte_convert", event_type="candidate_evaluation",
                journal_path=journal_path, now=now)
    if write:
        base_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(base_dir / CANDIDATE_DECISION_FILENAME,
                          json.dumps(cd, indent=2, default=str))
        active = dict(_dict(cd.get("candidate")))
        active.update({"state": cd.get("state"), "updated_at": cd.get("generated_at"),
                       "decision": cd.get("decision"), "scan_only": True,
                       "execution_allowed": False})
        atomic_write_text(base_dir / ACTIVE_CANDIDATE_FILENAME,
                          json.dumps(active, indent=2, default=str))
    if cd.get("decision") != CONFIRM_ENTRY:
        return _refuse("candidate_watch",
                       [f"candidate_watch:{cd.get('decision')}", *map(str, cd.get("reasons") or [])])

    locked = _dict(cd.get("candidate"))

    # ── computed final confirmations (deterministic on this path) ─────────────────────────────
    confirmations, confirmation_detail = _computed_confirmations(locked, contract, broker, now)
    payload["confirmations"] = confirmations
    payload["confirmation_detail"] = confirmation_detail

    # ── entry gate under the same clock ───────────────────────────────────────────────────────
    gate = build_entry_gate_decision(candidate=locked, candidate_decision=cd,
                                     day_score=day_score, vehicle_score=vehicle_score,
                                     gamma_map=gamma, broker_snapshot=broker,
                                     confirmations=confirmations,
                                     journal_events=journal_events, now=now)
    payload["entry_gate"] = gate
    if journal:
        gate_event = event_from_entry_gate(gate)
        if gap_backfilled:
            gate_event["gap_pct_backfilled"] = True
        gate_result = append_decision_journal(gate_event, source="odte_convert",
                                              event_type="entry_decision",
                                              journal_path=journal_path, now=now)
        payload["journal_append"] = {"status": gate_result.get("status"),
                                     "event_id": gate_result.get("event_id")}
        if gate_result.get("status") == "error":
            # Fail closed: an unjournaled gate must not lease (same rule as the entry-gate CLI).
            payload.update({"stage": "journal",
                            "reason_codes": ["journal_append_failed_execution_withheld"]})
            payload["next_action"] = ("journal append FAILED — the gate decision could not be "
                                      "recorded, so no lease is minted; repair the journal path "
                                      "and re-run")
            payload["next_command"] = "odte-convert  # after repairing the journal"
            return payload
    if not gate.get("execution_allowed"):
        reasons = [f"gate:{gate.get('decision')}",
                   *[str(r) for r in (gate.get("veto_reasons") or [])],
                   *[str(r) for r in (gate.get("reason_codes") or [])
                     if str(r).endswith((":fail", ":unknown"))]]
        return _refuse("entry_gate", reasons)

    # ── lease authorization under the same clock ──────────────────────────────────────────────
    auth = authorize_entry(gate=gate, candidate_decision=cd, vehicle_score=vehicle_score,
                           broker_snapshot=broker, market_snapshot=market, now=now, policy=policy)
    payload["authorization"] = auth
    if journal:
        lease = _dict(auth.get("lease"))
        # Use the canonical builder rather than a hand-rolled subset: the atomic path was dropping
        # issued_at/expires_at/max_limit_price/max_debit/max_premium_loss/quantity/risk_mode, all of
        # which the direct-CLI path recorded and all of which are already in `auth` here. Without
        # them a lease's TTL and its price/debit ceilings are unreadable after the fact, so "expired"
        # has to be inferred from the absence of a fill instead of read.
        lease_event = event_from_execution_lease(auth, extra={
            "tier": lease.get("tier") or locked.get("tier"),
            "anchor_quote": lease.get("anchor_quote"),
            "ts": now.isoformat(timespec="seconds"),
            **_identity(locked),
        })
        append_decision_journal(lease_event, source="odte_convert",
                                event_type="execution_lease_issued",
                                journal_path=journal_path, now=now)
    if not auth.get("authorized"):
        return _refuse("authorize", list(auth.get("reason_codes") or []))

    if write:
        base_dir.mkdir(parents=True, exist_ok=True)
        lease_path = base_dir / LEASE_FILENAME
        atomic_write_text(lease_path, json.dumps(auth, indent=2, default=str))
        payload["lease_artifact"] = str(lease_path)
        # The ledger must EXIST before any place so no reader can skip the replay check.
        seed_consumed_ledger(base_dir / CONSUMED_LEASES_FILENAME)

    lease_expires = _parse_ts(_dict(auth.get("lease")).get("expires_at"))
    payload.update({
        "converted": True,
        "stage": "authorize",
        "reason_codes": [],
        "execution_allowed": True,
        "scan_only": False,
        "lease": auth.get("lease"),
        # Machine-readable race deadline (2026-08-04: the expiry lived only in prose and a nested
        # field; the controller lost a lease by 2.2s). Place BEFORE this moment or the cycle dies.
        "place_deadline": _dict(auth.get("lease")).get("expires_at"),
        "lease_seconds_remaining": (round((lease_expires - now).total_seconds(), 1)
                                    if lease_expires else None),
        "next_action": ("CONVERTED — consume the single-use lease at broker review/place via the "
                        "Hermes MCP lane IMMEDIATELY (it expires at "
                        f"{_dict(auth.get('lease')).get('expires_at')}), then run the "
                        "pending-order guard every ~10s until filled-fresh or cancelled"),
        "next_command": ("broker-review-place (Hermes MCP lane) → "
                         "odte-order-guard --order <broker order.json>"),
    })
    return payload


def render_markdown(payload: dict) -> str:
    p = payload or {}
    cand = _dict(p.get("candidate"))
    lines = [f"# 0DTE convert: **{'CONVERTED' if p.get('converted') else 'REFUSED'}**",
             "",
             f"Candidate: **{cand.get('ticker') or '?'}** {cand.get('direction') or ''} "
             f"{cand.get('option_id') or ''}  ·  tier: {cand.get('tier') or '?'}  ·  "
             f"anchor: {cand.get('anchor_quote') or '?'}  ",
             f"Day: {_dict(p.get('day_score')).get('verdict')}  ·  "
             f"vehicle: {_dict(p.get('vehicle_score')).get('verdict')}  ·  "
             f"stage: {p.get('stage')}  ",
             f"Execution allowed: {p.get('execution_allowed')} · places orders: "
             f"{p.get('places_orders')}  ",
             f"Next: {p.get('next_action')}", ""]
    if p.get("reason_codes"):
        lines += ["## Refusal reasons (fail closed, journaled terminally)",
                  *[f"- {r}" for r in p["reason_codes"]], ""]
    lease = _dict(p.get("lease"))
    if lease:
        lines += ["## Lease (single use)",
                  *[f"- {k}: {lease.get(k)}" for k in
                    ("lease_id", "expires_at", "symbol", "option_id", "quantity",
                     "max_limit_price", "max_debit", "tier", "anchor_quote")]]
    return "\n".join(lines)
