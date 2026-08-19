"""Shadow validation of the pre-registered freshness entry spec (2026-08-18 halt).

While live entries are halted (daily_trade_budget: 0), every clean confirm still journals an
`entry_decision` (`confirmed_candidate_transition`, all checks ok) immediately followed by the
gate's `no_trade_decision` veto (`daily_trade_budget_exhausted`). Each such pair is a would-be
entry. This tracker:

  1. detects the pair (plus the adjacent `vehicle_score` that carries the exact contract),
  2. stamps its SIGNAL AGE — minutes from the first same-direction candidate evaluation of the
     move to the confirm (the autopsy's lateness clock, computed the same way),
  3. polls the contract's real quote through a would-be hold and records the full price path
     (MFE / MAE / final), so ANY exit rule can be scored offline afterwards.

Read-only by construction: no orders, no leases, no writes outside data/odte/shadow_validation/.
Every public entry point swallows its own errors — validation must never break a daemon tick.
Evidence target (pre-registered in data/odte/reports/entry_signal_autopsy_2026-08-18.md): >=10
qualifying shadow entries scored before any un-halt decision.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from core.paths import atomic_write_text

logger = logging.getLogger("odte_shadow_validation")

BUDGET_VETO = "daily_trade_budget_exhausted"
# 2026-08-19: the freshness gate's stale vetoes are the OTHER counterfactual the spec's forward
# validation needs — a vetoed-stale confirm whose move kept running is evidence AGAINST the
# 20-minute threshold; one that snapped back is evidence FOR it. Track both veto kinds.
STALE_VETO = "entry_signal_stale"
TRACKED_VETOES = (BUDGET_VETO, STALE_VETO)
CONFIRM_MARK = "confirmed_candidate_transition"
PAIR_WINDOW_SECONDS = 10.0        # entry_decision <-> veto / vehicle_score adjacency
SIGNAL_LOOKBACK_MINUTES = 90.0    # same clock as the autopsy extraction
MAX_CONCURRENT = 3


def _ts(e: dict) -> datetime | None:
    raw = e.get("ts") or e.get("generated_at") or ""
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def find_qualifying_confirms(events: list[dict], now: datetime) -> list[dict]:
    """Vetoed-but-clean confirms with their contract and signal age. Pure — no IO."""
    out = []
    for i, e in enumerate(events):
        if e.get("event_type") != "entry_decision":
            continue
        reasons = [str(r) for r in (e.get("reasons") or e.get("reason_codes") or [])]
        if CONFIRM_MARK not in reasons:
            continue
        t0 = _ts(e)
        if t0 is None or (now - t0) > timedelta(minutes=SIGNAL_LOOKBACK_MINUTES):
            continue

        def near(kind: str, idx: int = i, anchor: datetime = t0) -> list[dict]:
            hits = []
            for other in events[max(0, idx - 6):idx + 7]:
                if other.get("event_type") != kind:
                    continue
                t1 = _ts(other)
                if t1 is not None and abs((t1 - anchor).total_seconds()) <= PAIR_WINDOW_SECONDS:
                    hits.append(other)
            return hits

        veto_kind = next((kind for v in near("no_trade_decision") for kind in TRACKED_VETOES
                          if kind in str(v.get("reason_codes") or "")), None)
        if veto_kind is None:
            continue
        contract = next((v for v in near("vehicle_score") if v.get("option_id")), None)
        if contract is None:
            continue

        direction = (contract.get("vehicle_score") or {}).get("direction") or contract.get("direction")
        underlying = contract.get("underlying") or e.get("symbol")
        # SHARED CLOCK: the same function the entry gate's freshness veto uses — the harness
        # scores exactly the lateness the gate will enforce, by construction.
        from data.odte_journal import first_signal_age_minutes
        out.append({
            "confirm_ts": t0.isoformat(timespec="seconds"),
            "underlying": underlying,
            "direction": direction,
            "tier": e.get("tier"),
            "veto_kind": veto_kind,
            "option_id": str(contract["option_id"]),
            "strike": contract.get("strike"),
            "option_type": contract.get("option_type"),
            "signal_age_minutes": first_signal_age_minutes(events, underlying, direction, now=t0),
        })
    return out


class ShadowValidationTracker:
    """Owns data/odte/shadow_validation/: tracking.json (live state, survives daemon restarts)
    and outcomes.jsonl (append-only results)."""

    def __init__(self, out_dir: str | Path, *, hold_minutes: float, poll_seconds: float) -> None:
        self.out_dir = Path(out_dir)
        self.hold = timedelta(minutes=hold_minutes)
        self.poll_seconds = poll_seconds
        self.tracking_path = self.out_dir / "tracking.json"
        self.outcomes_path = self.out_dir / "outcomes.jsonl"
        self.tracking: dict[str, dict] = {}
        self.errors = 0
        try:
            self.tracking = json.loads(self.tracking_path.read_text())
        except (OSError, ValueError):
            self.tracking = {}

    # ── state ────────────────────────────────────────────────────────────────────────────
    def _persist(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.tracking_path, json.dumps(self.tracking, indent=2))

    def _finalize(self, key: str, rec: dict, reason: str, now: datetime) -> None:
        ref = _num(rec.get("ref_ask"))
        bids = [b for _, b in rec.get("samples", [])]
        outcome = {
            **{k: rec.get(k) for k in ("confirm_ts", "underlying", "direction", "tier",
                                       "veto_kind", "option_id", "strike", "option_type",
                                       "signal_age_minutes", "ref_ask")},
            "finalized_at": now.isoformat(timespec="seconds"),
            "final_reason": reason,
            "n_samples": len(bids),
        }
        if ref and ref > 0 and bids:
            outcome["mfe_pct"] = round(max(bids) / ref - 1.0, 4)
            outcome["mae_pct"] = round(min(bids) / ref - 1.0, 4)
            outcome["final_pnl_pct"] = round(bids[-1] / ref - 1.0, 4)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with self.outcomes_path.open("a") as fh:
            fh.write(json.dumps(outcome) + "\n")
        self.tracking.pop(key, None)
        self._persist()

    def summary(self) -> dict:
        """Heartbeat surface — 2026-08-19: three empty holds finalized before anyone could see
        the tracker was erroring/parsing nothing."""
        return {"tracking": len(self.tracking), "errors": self.errors,
                "sampled": sum(1 for r in self.tracking.values() if r.get("samples"))}

    # ── per-tick entry point (never raises) ──────────────────────────────────────────────
    async def step(self, events: list[dict],
                   quote_fn: Callable[[str], Any], now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        try:
            await self._step(events, quote_fn, now)
        except Exception as exc:                               # advisory lane: fail open, count it
            self.errors += 1
            logger.exception("shadow-validation step error: %s", exc)

    async def _step(self, events, quote_fn, now: datetime) -> None:
        for cand in find_qualifying_confirms(events, now):
            # Keyed by CONTRACT (2026-08-19): keying by confirm_ts let one stale move's
            # re-confirms fill every tracking slot with the same option — three slots, one
            # contract, zero new information.
            key = str(cand["option_id"])
            if key in self.tracking or len(self.tracking) >= MAX_CONCURRENT:
                continue
            self.tracking[key] = {**cand, "started_at": now.isoformat(timespec="seconds"),
                                  "samples": [], "ref_ask": None, "last_poll": None}
            self._persist()

        for key, rec in list(self.tracking.items()):
            started = datetime.fromisoformat(rec["started_at"])
            if now - started > self.hold:
                self._finalize(key, rec, "hold_expired", now)
                continue
            last = rec.get("last_poll")
            if last and (now - datetime.fromisoformat(last)).total_seconds() < self.poll_seconds:
                continue
            quote = await quote_fn(rec["option_id"])
            rec["last_poll"] = now.isoformat(timespec="seconds")
            q = quote if isinstance(quote, dict) else getattr(quote, "__dict__", {}) or {}
            # The live payload NESTS under "quote" ({'quote': {'bid_price': ...}}) — same unwrap
            # as the daemon's _option_quote_to_contract. 2026-08-19: the flat read parsed None on
            # every poll and three 45-min holds finalized with zero samples, silently.
            if isinstance(q.get("quote"), dict):
                q = q["quote"]
            bid = _num(q.get("bid_price") or q.get("bid"))
            ask = _num(q.get("ask_price") or q.get("ask"))
            if rec.get("ref_ask") is None and ask and ask > 0:
                rec["ref_ask"] = ask                          # would-be fill = ask at first poll
            if bid is not None:
                rec["samples"].append([now.isoformat(timespec="seconds"), bid])
            self._persist()
